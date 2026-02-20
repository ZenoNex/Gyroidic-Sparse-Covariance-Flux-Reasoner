"""
Chern-Simons Gasket: Topological Twist Repair for Logic Leaks.

Implements the Chern-Simons gasket to plug logic leaks at the boundary
where discrete symbolic data transitions to continuous geometric reasoning.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import math


class ChernSimonsGasket(nn.Module):
    """
    Implements the Chern-Simons gasket to prevent logic leaks.
    
    The Problem: Data leaks through holes in the manifold at the boundary Σ
    The Solution: Chern-Simons term provides topological twist (chirality)
    """
    
    def __init__(
        self,
        manifold_dim: int = 3,
        level_k: int = 1,
        device: str = None
    ):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.level_k = level_k
        self.device = device
        
        # Gauge field A (connection 1-form)
        self.register_buffer('gauge_field', torch.zeros(manifold_dim, manifold_dim, device=device))
        
        # Holonomy tracking for closed loops
        self.register_buffer('holonomy_cache', torch.zeros(8, device=device))  # Cache for 8 loops
        
        # Twist detection
        self.register_buffer('twist_energy', torch.tensor(0.0, device=device))
    
    def initialize_gauge_field(self, polynomial_coeffs: torch.Tensor, winding_numbers: torch.Tensor):
        """
        Initialize gauge field based on polynomial coefficients and winding numbers.
        
        Args:
            polynomial_coeffs: Polynomial coefficients [K, D] from PolynomialCoprimeConfig
            winding_numbers: Winding numbers around gyroid throat [K]
        """
        K, D = polynomial_coeffs.shape
        
        # Convert polynomial coefficients to scalar indices for GCD computation
        # Use the dominant coefficient (largest magnitude) as the representative value
        dominant_coeffs = torch.argmax(torch.abs(polynomial_coeffs), dim=1)  # [K]
        
        # Scale to integer-like values for GCD computation
        scaled_coeffs = (dominant_coeffs + 1) * 2  # Ensure positive integers
        
        # Compute GCD for each functional
        gcd_values = torch.gcd(scaled_coeffs.long(), winding_numbers.long()).float()
        
        # Initialize gauge field with holonomy condition
        for i in range(min(self.manifold_dim, K)):
            holonomy_value = 2 * math.pi * gcd_values[i] / self.level_k
            
            # Set gauge field components (antisymmetric)
            if i + 1 < self.manifold_dim:
                self.gauge_field[i, i + 1] = holonomy_value
                self.gauge_field[i + 1, i] = -holonomy_value
    
    def compute_field_strength(self) -> torch.Tensor:
        """
        Compute field strength F = dA + A ∧ A.
        
        Returns:
            Field strength tensor [dim, dim]
        """
        A = self.gauge_field
        
        # Curvature F = dA + [A, A] (simplified for discrete case)
        # Using commutator [A, A] = AA - AA = 0, so F ≈ dA
        # In discrete setting, approximate dA as finite differences
        
        F = torch.zeros_like(A)
        
        # Compute discrete exterior derivative (simplified)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                if i != j:
                    # Discrete curl-like operation
                    F[i, j] = A[i, j] - A[j, i]
        
        return F
    
    def chern_simons_action(self, loop_path: torch.Tensor) -> torch.Tensor:
        """
        Compute Chern-Simons action along a loop path.
        
        S_CS = (k/4π) ∫_Σ Tr(A ∧ dA + (2/3) A ∧ A ∧ A)
        
        Args:
            loop_path: Path coordinates [path_length, dim]
            
        Returns:
            Chern-Simons action value
        """
        path_length = loop_path.shape[0]
        
        # Compute line integral of gauge field along path
        line_integral = torch.tensor(0.0, device=self.device)
        
        for i in range(path_length - 1):
            # Current and next points
            x_curr = loop_path[i]
            x_next = loop_path[i + 1]
            dx = x_next - x_curr
            
            # Gauge field at current point (simplified evaluation)
            A_curr = self.gauge_field.mean(dim=0)  # Average over components
            
            # Add contribution to line integral
            line_integral += torch.dot(A_curr[:len(dx)], dx)
        
        # Chern-Simons action (simplified 3D case)
        cs_action = (self.level_k / (4 * math.pi)) * line_integral
        
        return cs_action
    
    def detect_logic_leak(self, residues: torch.Tensor, threshold: float = 1e-6) -> bool:
        """
        Detect if there's a logic leak (non-trivial topology).
        
        Args:
            residues: Current residues [batch, K, D]
            threshold: Threshold for non-triviality
            
        Returns:
            True if logic leak detected
        """
        batch_size, K, D = residues.shape
        
        # Create a simple closed loop in residue space
        loop_points = []
        for k in range(min(4, K)):  # Use first 4 functionals
            point = residues[0, k, :min(3, D)].clone()  # Take first 3 dimensions
            if len(point) < 3:
                # Pad to 3D
                point = torch.cat([point, torch.zeros(3 - len(point), device=point.device)])
            loop_points.append(point)
        
        # Close the loop
        if loop_points:
            loop_points.append(loop_points[0])
            loop_path = torch.stack(loop_points)
            
            # Compute Chern-Simons action
            cs_action = self.chern_simons_action(loop_path)
            
            # Update twist energy
            self.twist_energy = torch.abs(cs_action)
            
            # Leak detected if action is too small (trivial topology) OR residues show high variance
            residue_variance = residues.var().item()
            topology_trivial = self.twist_energy < threshold
            high_variance = residue_variance > 2.0  # Indicates chaos/fracture
            
            return topology_trivial or high_variance
        
        return True  # Default to leak detected if no loop can be formed
    
    def apply_chiral_torsion_shift(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Apply 90° chiral torsion shift to rotate consonants out of collapsed state.
        
        Args:
            residues: Input residues [batch, K, D]
            
        Returns:
            Residues with chiral torsion applied
        """
        batch_size, K, D = residues.shape
        
        # Create rotation matrix for 90° twist
        if D >= 2:
            # 2D rotation matrix for 90° (π/2)
            cos_theta = torch.cos(torch.tensor(math.pi / 2, device=self.device))
            sin_theta = torch.sin(torch.tensor(math.pi / 2, device=self.device))
            
            rotation_2d = torch.tensor([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ], device=self.device)
            
            # Apply rotation to first two dimensions of each functional
            rotated_residues = residues.clone()
            for k in range(K):
                if D >= 2:
                    # Extract first two dimensions
                    xy = residues[:, k, :2]  # [batch, 2]
                    
                    # Apply rotation
                    xy_rotated = torch.matmul(xy, rotation_2d.T)
                    
                    # Put back
                    rotated_residues[:, k, :2] = xy_rotated
            
            return rotated_residues
        
        return residues
    
    def plug_logic_leak(self, residues: torch.Tensor, polynomial_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Main method to plug logic leaks using Chern-Simons gasket.
        
        Args:
            residues: Input residues [batch, K, D]
            polynomial_coeffs: Polynomial coefficients [K, D] from PolynomialCoprimeConfig
            
        Returns:
            Repaired residues with plugged leaks
        """
        K = residues.shape[1]
        
        # Initialize gauge field if needed
        if torch.allclose(self.gauge_field, torch.zeros_like(self.gauge_field)):
            winding_numbers = torch.arange(1, K + 1, device=self.device)  # Simple winding
            self.initialize_gauge_field(polynomial_coeffs, winding_numbers)
        
        # Detect logic leak
        leak_detected = self.detect_logic_leak(residues)
        
        if leak_detected:
            # Apply chiral torsion shift to repair
            repaired_residues = self.apply_chiral_torsion_shift(residues)
            
            # Verify repair
            repair_successful = not self.detect_logic_leak(repaired_residues)
            
            if repair_successful:
                return repaired_residues
            else:
                # If repair failed, apply stronger correction
                # Increase the level k for stronger twist
                self.level_k = min(self.level_k + 1, 5)
                return self.apply_chiral_torsion_shift(residues)
        
        return residues
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get Chern-Simons diagnostics."""
        return {
            'twist_energy': self.twist_energy.item(),
            'level_k': float(self.level_k),
            'gauge_field_norm': torch.norm(self.gauge_field).item()
        }


class SolitonStabilityHealer(nn.Module):
    """
    Heals fractured solitons using Drucker-Prager global plastic flow.
    
    The Problem: MC-rupture sites (local fractures) without global healing
    The Solution: DP global envelope with ranging signal to heat manifold
    """
    
    def __init__(
        self,
        alpha_0: float = 1.0,
        gamma: float = 0.5,
        healing_iterations: int = 400,
        device: str = None
    ):
        super().__init__()
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.healing_iterations = healing_iterations
        self.device = device
        
        # Current alpha (adaptive)
        self.register_buffer('alpha', torch.tensor(alpha_0, device=device))
        
        # Healing progress tracking
        self.register_buffer('healing_progress', torch.tensor(0.0, device=device))
        self.register_buffer('iteration_count', torch.tensor(0, device=device))
    
    def detect_fractured_soliton(self, output_text: str) -> bool:
        """
        Detect if output represents a fractured soliton (garbled text).
        
        Args:
            output_text: Generated text to analyze
            
        Returns:
            True if fractured soliton detected
        """
        if not output_text or len(output_text) < 5:
            return False
        
        # Check for characteristics of fractured soliton:
        # 1. High consonant density
        # 2. Lack of recognizable words
        # 3. Repetitive patterns
        
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
        vowel_count = sum(1 for c in output_text if c in vowels)
        consonant_count = sum(1 for c in output_text if c in consonants)
        
        if consonant_count == 0:
            return False
        
        # Fractured if very low vowel ratio and high repetition
        vowel_ratio = vowel_count / (vowel_count + consonant_count)
        
        # Check for repetitive patterns (sign of collapse)
        unique_chars = len(set(output_text))
        repetition_ratio = unique_chars / len(output_text)
        
        # Check for known garbled patterns
        garbled_patterns = ['nccmts', 'mnelt', 'clrcl', 'tncsec']
        has_garbled_pattern = any(pattern in output_text.lower() for pattern in garbled_patterns)
        
        return (vowel_ratio < 0.15 and repetition_ratio < 0.0) or has_garbled_pattern
    
    def apply_ranging_signal(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Apply ranging signal: α → α₀ + γ for manifold heating.
        
        Args:
            residues: Input residues [batch, K, D]
            
        Returns:
            Heated residues with ranging applied
        """
        # Update alpha with ranging
        self.alpha = self.alpha_0 + self.gamma * (self.iteration_count / self.healing_iterations)
        
        # Apply topological free energy heating
        # Heat manifold by adding controlled noise scaled by alpha
        heating_noise = torch.randn_like(residues) * (self.alpha * 0.1)
        heated_residues = residues + heating_noise
        
        # Increment iteration count
        self.iteration_count = torch.clamp(self.iteration_count + 1, max=self.healing_iterations)
        
        return heated_residues
    
    def drucker_prager_healing(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Apply Drucker-Prager global plastic flow for healing.
        
        Args:
            residues: Input residues [batch, K, D]
            
        Returns:
            Healed residues with DP global envelope
        """
        batch_size, K, D = residues.shape
        
        # Compute stress invariants
        # I1 = trace (first invariant)
        I1 = residues.sum(dim=-1)  # [batch, K]
        
        # J2 = second deviatoric invariant (simplified)
        residue_mean = residues.mean(dim=-1, keepdim=True)  # [batch, K, 1]
        deviatoric = residues - residue_mean  # [batch, K, D]
        J2 = 0.5 * (deviatoric ** 2).sum(dim=-1)  # [batch, K]
        
        # Drucker-Prager yield criterion: α*I1 + sqrt(J2) - k = 0
        # We use this to identify regions needing healing
        dp_stress = self.alpha * I1 + torch.sqrt(J2 + 1e-8)
        
        # Apply healing where stress is high
        stress_threshold = 2.0
        healing_mask = (dp_stress > stress_threshold).float().unsqueeze(-1)  # [batch, K, 1]
        
        # Healing: smooth toward mean (global plastic flow)
        global_mean = residues.mean(dim=1, keepdim=True)  # [batch, 1, D]
        healing_target = 0.8 * residues + 0.2 * global_mean
        
        healed_residues = (1 - healing_mask) * residues + healing_mask * healing_target
        
        return healed_residues
    
    def heal_fractured_soliton(
        self, 
        residues: torch.Tensor, 
        output_text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Main healing method for fractured solitons.
        
        Args:
            residues: Input residues [batch, K, D]
            output_text: Optional output text for fracture detection
            
        Returns:
            Healed residues
        """
        # Check if healing is needed
        fracture_detected = False
        if output_text:
            fracture_detected = self.detect_fractured_soliton(output_text)
        
        if fracture_detected or self.iteration_count < self.healing_iterations:
            # Apply ranging signal (heating)
            heated_residues = self.apply_ranging_signal(residues)
            
            # Apply Drucker-Prager healing
            healed_residues = self.drucker_prager_healing(heated_residues)
            
            # Update healing progress
            self.healing_progress = self.iteration_count / self.healing_iterations
            
            return healed_residues
        
        return residues
    
    def reset_healing(self):
        """Reset healing process for new sequence."""
        self.iteration_count.fill_(0)
        self.healing_progress.fill_(0.0)
        self.alpha.fill_(self.alpha_0)
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get healing diagnostics."""
        return {
            'alpha': self.alpha.item(),
            'healing_progress': self.healing_progress.item(),
            'iteration_count': self.iteration_count.item()
        }
