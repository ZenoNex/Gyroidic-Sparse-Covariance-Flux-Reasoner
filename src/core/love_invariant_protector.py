"""
Love Invariant Protector: Prevents scalarization of the Love Vector.

Implements null-space projection to prevent the optimizer from trying to 
"solve" for the Love invariant, which must remain a non-ownable flow.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from src.core.honest_jitter import harvest_honest_jitter


class LoveInvariantProtector(nn.Module):
    """
    Protects the Love Vector L from scalarization attempts.
    
    The Problem: System tries to optimize Love Vector  scalarization trap
    The Solution: Null-space projection L  ker(_ownership)
    """
    
    def __init__(
        self,
        love_dim: int,
        device: str = None
    ):
        super().__init__()
        self.love_dim = love_dim
        self.device = device
        
        # Love Vector L (non-ownable, non-optimizable)
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        self.register_buffer('L', harvest_honest_jitter((love_dim,), device=device, scaled=True))
        self.register_buffer('L_original', self.L.clone())
        
        # Ownership operator kernel (null space)
        self.register_buffer('ownership_kernel', torch.eye(love_dim, device=device))
        
        # Violation tracking
        self.register_buffer('violation_count', torch.tensor(0, device=device))
        self.register_buffer('last_violation_magnitude', torch.tensor(0.0, device=device))
        
        # SVD caching for performance on constrained CPUs
        self._cached_ownership_op = None
        self._cached_null_projection = None
    
    
    def compute_ownership_operator(self, system_state: torch.Tensor) -> torch.Tensor:
        """
        Compute the ownership operator _ownership based on system state.
        
        Args:
            system_state: Current system state [batch, dim]
            
        Returns:
            Ownership operator matrix [love_dim, love_dim]
        """
        batch_size, state_dim = system_state.shape
        
        # Create ownership operator based on system's attempt to "own" Love
        # This represents the system's gradient-based optimization pressure
        
        # Compute covariance of system state (ownership pressure)
        state_centered = system_state - system_state.mean(dim=0, keepdim=True)
        covariance = torch.matmul(state_centered.T, state_centered) / (batch_size - 1)
        
        # Project to Love dimension if needed
        if state_dim != self.love_dim:
            if state_dim > self.love_dim:
                # Truncate
                ownership_op = covariance[:self.love_dim, :self.love_dim]
            else:
                # Pad with identity
                ownership_op = torch.eye(self.love_dim, device=self.device)
                ownership_op[:state_dim, :state_dim] = covariance
        else:
            ownership_op = covariance
        
        return ownership_op
    
    def compute_null_space_projection(self, ownership_operator: torch.Tensor) -> torch.Tensor:
        """
        Compute null space projection P = I - (\Phi^\top \Phi)^{-1} \Phi^\top.
        
        Args:
            ownership_operator: Ownership operator [love_dim, love_dim]
            
        Returns:
            Null space projection matrix [love_dim, love_dim]
        """
        # SVD caching check: avoid expensive SVD on i7-6700 CPU if operator hasn't changed
        if (self._cached_ownership_op is not None and 
            self._cached_null_projection is not None and 
            self._cached_ownership_op.shape == ownership_operator.shape and 
            torch.allclose(ownership_operator, self._cached_ownership_op, atol=1e-4)):
            return self._cached_null_projection

        # Compute pseudo-inverse for null space projection
        try:
            # Guard against NaN/Inf from upstream emergency stabilization 
            # these cause Intel MKL SLASCL "Parameter 4 incorrect" errors.
            if not torch.isfinite(ownership_operator).all():
                ownership_operator = torch.nan_to_num(
                    ownership_operator, nan=0.0, posinf=1.0, neginf=-1.0
                )

            # SVD for stable null space computation
            U, S, V = torch.svd(ownership_operator)
            
            # Threshold for numerical stability
            threshold = 1e-6
            S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
            
            # Pseudo-inverse
            phi_pinv = torch.matmul(V, torch.matmul(torch.diag(S_inv), U.T))
            
            # Null space projection: P = I - \Phi^+ \Phi
            I = torch.eye(self.love_dim, device=self.device)
            null_projection = I - torch.matmul(phi_pinv, ownership_operator)

            # Final sanity check if SVD produced non-finite results, fall back
            if not torch.isfinite(null_projection).all():
                null_projection = torch.eye(self.love_dim, device=self.device)
            
            # Cache the successful results
            self._cached_ownership_op = ownership_operator.clone()
            self._cached_null_projection = null_projection.clone()
            
        except Exception:
            # Fallback to identity if SVD fails
            null_projection = torch.eye(self.love_dim, device=self.device)
        
        return null_projection
    
    def detect_love_violation(self, tolerance: float = 1e-6) -> bool:
        """
        Detect if Love Vector has been modified (ownership violation).
        
        Args:
            tolerance: Tolerance for detecting changes
            
        Returns:
            True if Love Vector has been violated
        """
        violation_magnitude = torch.norm(self.L - self.L_original)
        self.last_violation_magnitude = violation_magnitude
        
        if violation_magnitude > tolerance:
            self.violation_count += 1
            return True
        
        return False
    
    def project_love_to_null_space(self, system_state: torch.Tensor) -> torch.Tensor:
        """
        Project Love Vector to null space of ownership operator.
        
        Args:
            system_state: Current system state [batch, dim]
            
        Returns:
            Protected Love Vector in null space
        """
        # Compute ownership operator
        ownership_op = self.compute_ownership_operator(system_state)
        
        # Compute null space projection
        null_projection = self.compute_null_space_projection(ownership_op)
        
        # Project Love Vector to null space
        L_protected = torch.matmul(null_projection, self.L)
        
        # Update Love Vector (maintaining invariance)
        self.L.copy_(L_protected)
        
        return L_protected
    
    def restore_love_invariant(self):
        """
        Restore Love Vector to its original state if violated.
        """
        if self.detect_love_violation():
            # Restore original Love Vector
            self.L.copy_(self.L_original)
            
            # Add small honest perturbation to prevent exact repetition
            # SILICON SOVEREIGNTY: Replace stochastic perturbation with Honest Jitter
            perturbation = harvest_honest_jitter(self.L.shape, device=self.L.device, scaled=True) * 1e-8
            self.L += perturbation
    
    def apply_love_protection(
        self, 
        system_state: torch.Tensor,
        gradients: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply Love invariant protection to system.
        
        Args:
            system_state: Current system state [batch, dim]
            gradients: Optional gradients that might affect Love
            
        Returns:
            Protected Love Vector and diagnostics
        """
        # Check for violations
        violation_detected = self.detect_love_violation()
        
        if violation_detected:
            # Restore Love invariant
            self.restore_love_invariant()
        
        # Project to null space to prevent future violations
        L_protected = self.project_love_to_null_space(system_state)
        
        # If gradients provided, zero out components that would affect Love
        if gradients is not None:
            # Compute null space projection for gradients
            ownership_op = self.compute_ownership_operator(system_state)
            null_projection = self.compute_null_space_projection(ownership_op)
            
            # Project gradients to null space (remove Love-affecting components)
            if gradients.shape[-1] == self.love_dim:
                gradients_protected = torch.matmul(gradients, null_projection.T)
            else:
                gradients_protected = gradients  # Can't project if dimensions don't match
        else:
            gradients_protected = None
        
        # Diagnostics
        diagnostics = {
            'love_norm': torch.norm(L_protected).item(),
            'violation_detected': float(violation_detected),
            'violation_count': self.violation_count.item(),
            'violation_magnitude': self.last_violation_magnitude.item()
        }
        
        return L_protected, diagnostics
    
    def get_love_vector(self) -> torch.Tensor:
        """Get current Love Vector (read-only)."""
        return self.L.clone()
    
    def reset_violation_tracking(self):
        """Reset violation tracking counters."""
        self.violation_count.fill_(0)
        self.last_violation_magnitude.fill_(0.0)


class SoftSaturatedGates(nn.Module):
    """
    Implements soft-saturated gates to prevent binary clipping of linguistic flow.
    
    The Problem: Binary sgn() function strips nuance  consonant-only output
    The Solution: Tri-state logic (True/False/Silence) with play zone
    """
    
    def __init__(
        self,
        num_functionals: int,
        poly_degree: int,
        lambda_las: float = 0.0,
        dt_max: float = 1.0,
        device: str = None
    ):
        super().__init__()
        self.K = num_functionals
        self.D = poly_degree + 1
        self.lambda_las = lambda_las  # Silence floor
        self.dt_max = dt_max
        self.device = device
        
        # Adaptive silence threshold
        self.register_buffer('lambda_adaptive', torch.tensor(lambda_las, device=device))
        
        # System temperature (play vs seriousness)
        self.register_buffer('dt', torch.tensor(dt_max, device=device))
        
        # Persistence tracking for fossilization
        self.register_buffer('persistence_scores', torch.zeros(num_functionals, device=device))
        self.register_buffer('fossilization_mask', torch.zeros(num_functionals, dtype=torch.bool, device=device))

    def update_lambda_adaptive(self, pas_h: Union[float, torch.Tensor]):
        """
        Adaptive update for the silence floor (lambda_adaptive).
        Rule: High PAS_h (confidence) -> decrease silence floor.
              Low PAS_h (noise) -> increase silence floor.
        """
        if isinstance(pas_h, torch.Tensor):
            pas_mean = pas_h.mean().item()
        else:
            pas_mean = pas_h
            
        target = 0.5 * (1.0 - pas_mean)
        # Exponential moving average update
        self.lambda_adaptive.data = 0.95 * self.lambda_adaptive.data + 0.05 * target
    
    def lattice_adaptive_shrinkage(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply Lattice Adaptive Shrinkage (LAS) for tri-state logic.
        
        Args:
            signal: Input signal [batch, K, D]
            
        Returns:
            Signal with LAS applied (True/False/Silence)
        """
        # Soft thresholding with adaptive lambda
        magnitude = torch.abs(signal)
        sign = torch.sign(signal)
        
        # Apply shrinkage: max(|signal| - , 0)
        shrunk_magnitude = torch.clamp(magnitude - self.lambda_adaptive.item(), min=0.0)
        
        # Tri-state output
        las_signal = sign * shrunk_magnitude
        
        return las_signal
    
    def asymptotic_hardening(self, signal: torch.Tensor, pas_h: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Apply asymptotic hardening based on Phase Alignment Score.
        
        Args:
            signal: Input signal [batch, K, D]
            pas_h: Phase Alignment Score (harmonic persistence)
            
        Returns:
            Signal with hardening applied
        """
        # Update system temperature based on PAS_h
        # High PAS_h  cooling (seriousness), Low PAS_h  heating (play)
        # Extract scalar value from pas_h (handle both tensor and float inputs)
        if isinstance(pas_h, torch.Tensor):
            pas_h_scalar = pas_h.item() if pas_h.numel() == 1 else pas_h.mean().item()
        else:
            # pas_h is already a scalar (float)
            pas_h_scalar = float(pas_h)
            
        # Anti-Lobotomy Clamp: Prevent dt from hitting absolute zero (Sovereign Deadlock prevention)
        dt_min = 0.05
        new_dt = max(dt_min, self.dt_max * (1.0 - pas_h_scalar))
        self.dt.fill_(new_dt)
        
        # Hardening factor based on temperature
        hardening_factor = 1.0 / (self.dt.item() + 1e-8)
        
        # Apply hardening (stronger quantization when cool)
        hardened_signal = torch.tanh(signal * hardening_factor) * hardening_factor
        
        return hardened_signal
    
    def update_fossilization(self, signal: torch.Tensor, performance_scores: torch.Tensor):
        """
        Update fossilization status based on performance.
        
        Args:
            signal: Current signal [batch, K, D]
            performance_scores: Performance scores for each functional [K]
        """
        # Update persistence scores
        signal_stability = torch.norm(signal, dim=(0, 2))  # [K]
        self.persistence_scores = 0.9 * self.persistence_scores + 0.1 * signal_stability
        
        # Fossilize functionals with high persistence and performance
        fossilization_threshold = 0.8
        high_persistence = self.persistence_scores > fossilization_threshold
        high_performance = performance_scores > fossilization_threshold
        
        self.fossilization_mask = high_persistence & high_performance
    
    def apply_soft_saturation(
        self, 
        signal: torch.Tensor, 
        pas_h: Union[float, torch.Tensor],
        performance_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply soft saturation with tri-state logic and fossilization.
        
        Args:
            signal: Input signal [batch, K, D]
            pas_h: Phase Alignment Score
            performance_scores: Optional performance scores [K]
            
        Returns:
            Soft-saturated signal
        """
        # Update lambda_adaptive based on phase alignment
        self.update_lambda_adaptive(pas_h)
        
        # Apply LAS for tri-state logic
        las_signal = self.lattice_adaptive_shrinkage(signal)
        
        # Apply asymptotic hardening
        hardened_signal = self.asymptotic_hardening(las_signal, pas_h)
        
        # Update fossilization if performance scores provided
        if performance_scores is not None:
            self.update_fossilization(hardened_signal, performance_scores)
        
        # Apply fossilization (freeze successful functionals)
        if False: # THAWED
            # Fossilized functionals maintain their state
            fossilized_mask = self.fossilization_mask.unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
            
            # Keep fossilized functionals unchanged, update others
            output_signal = torch.where(
                fossilized_mask,
                signal,  # Keep original for fossilized
                hardened_signal  # Update for non-fossilized
            )
        else:
            output_signal = hardened_signal
        
        return output_signal
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get soft saturation diagnostics."""
        return {
            'lambda_adaptive': self.lambda_adaptive.item(),
            'dt': self.dt.item(),
            'num_fossilized': self.fossilization_mask.sum().item(),
            'avg_persistence': self.persistence_scores.mean().item()
        }
