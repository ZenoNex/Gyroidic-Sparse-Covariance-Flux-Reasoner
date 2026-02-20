"""
Speculative Coprime Chiral Coherence Gating (SCCCG).

Implements recovery of structure from "converged" states using:
1. Coprime Parity Tracking (winding numbers around functional heads)
2. Chiral Coherence Estimation (gyroidic tensor covariance)
3. Wasserstein Optimal Transport (recovering structure via optimal mass transport)
4. Dimensional Reduction Gating (gate by coprime structure, not scalar thresholds)

Key Insight: "Non-Convergence is Data" - what appears as convergence to noise
may contain recoverable chiral structure via optimal transport.

References:
- MATHEMATICAL_DETAILS.md §18.2 Coprime Parity
- INVARIANT_OPTIMIZATION.md §7.3 Chiral Drift Optimizer
- EFFICIENCY_BY_NON_SCALAR_REWARD.md Speculative Exit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class WassersteinOptimalTransport(nn.Module):
    """
    Wasserstein-2 Optimal Transport for State Recovery.
    
    Given a "converged" (low-entropy) source distribution and a target
    coprime structure, computes the optimal transport plan that recovers
    the lost chiral structure.
    
    The Wasserstein-2 distance between distributions P and Q:
        W_2(P, Q)^2 = inf_{γ ∈ Γ(P,Q)} ∫ ||x - y||^2 dγ(x,y)
    
    We approximate this using the Sinkhorn algorithm for entropic regularization:
        W_ε(P, Q) = min_{T} <T, C> - ε H(T)
    
    where C is the cost matrix and H is the entropy regularizer.
    """
    
    def __init__(self, dim: int, sinkhorn_iters: int = 20, epsilon: float = 0.1):
        super().__init__()
        self.dim = dim
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon
        
        # Learnable transport cost modulation (chiral-aware)
        self.cost_modulator = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.cost_modulator.weight)
        
    def compute_cost_matrix(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the cost matrix C[i,j] = ||source[i] - target[j]||^2
        with chiral modulation.
        
        Args:
            source: [n_source, dim] source distribution samples
            target: [n_target, dim] target distribution samples
            
        Returns:
            C: [n_source, n_target] cost matrix
        """
        # Apply chiral modulation to emphasize asymmetric dimensions
        source_mod = self.cost_modulator(source)
        target_mod = self.cost_modulator(target)
        
        # Squared Euclidean distance
        # C[i,j] = ||source[i] - target[j]||^2
        C = torch.cdist(source_mod, target_mod, p=2) ** 2
        
        return C
    
    def sinkhorn(self, C: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn algorithm for optimal transport with entropic regularization.
        
        Args:
            C: [n, m] cost matrix
            a: [n] source marginal (must sum to 1)
            b: [m] target marginal (must sum to 1)
            
        Returns:
            T: [n, m] optimal transport plan
        """
        n, m = C.shape
        
        # Initialize dual variables
        u = torch.ones(n, device=C.device) / n
        v = torch.ones(m, device=C.device) / m
        
        # Gibbs kernel
        K = torch.exp(-C / self.epsilon)
        
        for _ in range(self.sinkhorn_iters):
            # Row normalization
            u = a / (K @ v + 1e-8)
            # Column normalization
            v = b / (K.T @ u + 1e-8)
        
        # Transport plan
        T = torch.diag(u) @ K @ torch.diag(v)
        
        return T
    
    def transport(
        self, 
        source: torch.Tensor, 
        target: torch.Tensor,
        source_weights: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optimal transport from source to target.
        
        Args:
            source: [n, dim] source samples
            target: [m, dim] target samples
            source_weights: [n] weights (default: uniform)
            target_weights: [m] weights (default: uniform)
            
        Returns:
            transported: [n, dim] source transported toward target
            wasserstein_dist: scalar W_2 distance
        """
        n = source.shape[0]
        m = target.shape[0]
        
        if source_weights is None:
            source_weights = torch.ones(n, device=source.device) / n
        if target_weights is None:
            target_weights = torch.ones(m, device=target.device) / m
            
        # Normalize weights
        source_weights = source_weights / source_weights.sum()
        target_weights = target_weights / target_weights.sum()
        
        # Compute cost matrix
        C = self.compute_cost_matrix(source, target)
        
        # Compute optimal transport plan
        T = self.sinkhorn(C, source_weights, target_weights)
        
        # Transport source toward target using barycentric projection
        # transported[i] = Σ_j T[i,j] * target[j] / Σ_j T[i,j]
        T_normalized = T / (T.sum(dim=1, keepdim=True) + 1e-8)
        transported = T_normalized @ target
        
        # Wasserstein distance
        wasserstein_dist = (T * C).sum()
        
        return transported, wasserstein_dist


class CoprimeWindingTracker(nn.Module):
    """
    Tracks winding numbers around functional heads for coprime parity.
    
    The coprime parity condition (MATHEMATICAL_DETAILS.md §18.2):
        gcd(w_k, p_k) = 1
    
    Where w_k is the winding number around homology group H_k
    and p_k is the prime index of functional head k.
    
    This prevents "bubbly equations" from collapsing into singular orientation.
    """
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Polynomial coefficients for each head instead of primes (anti-lobotomy)
        poly_coeffs = self._generate_polynomial_coefficients(num_heads)
        self.register_buffer('polynomial_indices', torch.tensor(poly_coeffs, dtype=torch.float))
        
        # Projection to compute winding contribution per head
        # Each head needs 2 components (sine/cosine) for phase tracking
        self.head_proj = nn.Linear(dim, 2 * num_heads)
        
        # Running estimate of winding numbers (updated each call)
        self.register_buffer('winding_numbers', torch.zeros(num_heads))
        self.register_buffer('winding_history', torch.zeros(16, num_heads))  # History buffer
        self.register_buffer('history_idx', torch.tensor(0))
        
    def _generate_polynomial_coefficients(self, n: int) -> list:
        """Generate polynomial coefficients instead of primes (anti-lobotomy)."""
        coeffs = []
        for k in range(n):
            # Use Legendre polynomial P_k evaluated at x=0.7
            x = 0.7
            if k == 0:
                p_k = 1.0
            elif k == 1:
                p_k = x
            else:
                # P_k(x) = ((2k-1)*x*P_{k-1}(x) - (k-1)*P_{k-2}(x)) / k
                p_prev2 = 1.0
                p_prev1 = x
                for j in range(2, k + 1):
                    p_curr = ((2*j - 1) * x * p_prev1 - (j - 1) * p_prev2) / j
                    p_prev2 = p_prev1
                    p_prev1 = p_curr
                p_k = p_prev1
            
            # Scale to positive values suitable for indexing
            coeff = abs(p_k * 10) + 2
            coeffs.append(coeff)
        
        return coeffs
    
    def _gcd(self, a: int, b: int) -> int:
        """Euclidean GCD."""
        while b:
            a, b = b, a % b
        return a
    
    def compute_winding(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute instantaneous winding contribution from state.
        
        The winding number is computed as the phase angle in each head's
        projected subspace, accumulated over time.
        
        Args:
            state: [batch, dim] current state
            
        Returns:
            winding: [num_heads] winding contributions
        """
        # Project state to head space
        head_activations = self.head_proj(state)  # [batch, 2 * num_heads]
        
        # Compute phase angle (using arctan2 with adjacent dimensions)
        # We interpret pairs of dimensions as complex phase per head
        phases = torch.atan2(
            head_activations[:, :self.num_heads], # Sines
            head_activations[:, self.num_heads:] + 1e-8 # Cosines
        )
        
        # Winding = phase / 2π (normalized to integer revolutions)
        winding = phases.mean(dim=0) / (2 * math.pi)
        
        return winding
    
    def update_and_check(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update winding numbers and check coprime parity condition.
        
        Args:
            state: [batch, dim] current state
            
        Returns:
            dict with:
                - winding_numbers: [num_heads] current winding estimates
                - coprime_lock: bool, True if gcd(w_k, p_k) = 1 for all k
                - parity_violations: [num_heads] bool mask of violations
        """
        # Compute new winding contribution
        new_winding = self.compute_winding(state)
        
        # Update running sum (accumulate winding)
        self.winding_numbers = self.winding_numbers + new_winding
        
        # Store in history
        idx = self.history_idx.item() % 16
        self.winding_history[idx] = new_winding
        self.history_idx += 1
        
        # Quantize to nearest integer for GCD check
        winding_int = torch.round(self.winding_numbers).long().abs()
        winding_int = torch.clamp(winding_int, min=1)  # Avoid gcd(0, p) = p
        
        # Check coprime parity using polynomial coefficients instead of primes
        coprime_checks = torch.zeros(self.num_heads, dtype=torch.bool, device=state.device)
        for k in range(self.num_heads):
            w_k = winding_int[k].item()
            p_k = int(self.polynomial_indices[k].item())
            coprime_checks[k] = (self._gcd(w_k, p_k) == 1)
        
        coprime_lock = coprime_checks.all()
        parity_violations = ~coprime_checks
        
        return {
            'winding_numbers': self.winding_numbers.clone(),
            'coprime_lock': coprime_lock,
            'parity_violations': parity_violations
        }


class ChiralCoherenceEstimator(nn.Module):
    """
    Estimates chiral coherence using spectral asymmetry.
    
    Chiral Score (INVARIANT_OPTIMIZATION.md §7.3):
        C = Σ_i (λ_i^+ - λ_i^-) / (λ_i^+ + λ_i^-)
    
    Where λ+ and λ- are eigenvalues from the positive and negative
    chiral sectors of the covariance matrix.
    
    High chiral score = strong asymmetric structure (good)
    Low chiral score = symmetric/collapsed structure (bad)
    """
    
    def __init__(self, dim: int, sample_size: int = 16):
        super().__init__()
        self.dim = dim
        self.sample_size = sample_size
        
        # Sample buffer for covariance estimation
        self.register_buffer('sample_buffer', torch.zeros(sample_size, dim))
        self.register_buffer('buffer_idx', torch.tensor(0))
        
        # Chiral projection (splits dim into positive/negative sectors)
        self.chiral_proj = nn.Linear(dim, dim, bias=False)
        # Initialization with a chiral skew
        with torch.no_grad():
            nn.init.orthogonal_(self.chiral_proj.weight)
            # Inject a small prime-based asymmetry into the projection to ensure sector divergence
            for i in range(dim):
                self.chiral_proj.weight.data[i] *= (1.0 + 0.05 * math.sin(i * 3.14159 / 7.0))
        
    def update_buffer(self, state: torch.Tensor):
        """Add state to sample buffer."""
        if state.dim() == 2:
            state = state[0]  # Take first batch element
            
        idx = self.buffer_idx.item() % self.sample_size
        self.sample_buffer[idx] = state.detach()
        self.buffer_idx += 1
        
    def compute_chiral_score(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute chiral score from buffer samples.
        
        Args:
            state: Optional new state to add first
            
        Returns:
            chiral_score: Scalar in [-1, 1], higher = more asymmetric
        """
        if state is not None:
            self.update_buffer(state)
        
        # Get filled portion of buffer
        n_filled = min(self.buffer_idx.item(), self.sample_size)
        
        # FALLBACK: If low samples, use spectral flux of the current state
        if n_filled < 4:
            if state is None:
                return torch.tensor(0.01, device=self.sample_buffer.device)
            # Spectral flux asymmetry
            freq = torch.fft.rfft(state)
            power = torch.abs(freq)
            half = power.shape[-1] // 2
            p_low = power[..., :half].mean()
            p_high = power[..., half:].mean()
            asym = (p_low - p_high).abs() / (p_low + p_high + 1e-8)
            return asym.clamp(min=0.01)
            
        samples = self.sample_buffer[:n_filled]
        
        # Apply chiral projection
        chiral_samples = self.chiral_proj(samples)
        
        # Split into positive and negative sectors
        half = self.dim // 2
        positive_sector = chiral_samples[:, :half]
        negative_sector = chiral_samples[:, half:2*half]
        
        # Compute covariance eigenvalues for each sector
        def get_eigenvalues(x):
            centered = x - x.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / (x.shape[0] - 1)
            try:
                eigvals = torch.linalg.eigvalsh(cov)
                return eigvals.clamp(min=1e-8)
            except:
                return torch.ones(x.shape[1], device=x.device) * 1e-8
        
        lambda_plus = get_eigenvalues(positive_sector)
        lambda_minus = get_eigenvalues(negative_sector)
        
        # Align dimensions if unequal
        min_len = min(len(lambda_plus), len(lambda_minus))
        lambda_plus = lambda_plus[:min_len]
        lambda_minus = lambda_minus[:min_len]
        
        # Chiral score: spectral asymmetry
        numerator = (lambda_plus - lambda_minus).abs().sum()
        denominator = (lambda_plus + lambda_minus).sum().clamp(min=1e-8)
        
        chiral_score = numerator / denominator
        
        return chiral_score


class SpeculativeCoprimeGate(nn.Module):
    """
    Main Speculative Coprime Chiral Coherence Gating module.
    
    Combines:
    1. Coprime Winding Tracker - tracks gcd(w_k, p_k) = 1 condition
    2. Chiral Coherence Estimator - spectral asymmetry score
    3. Wasserstein Optimal Transport - recovers structure from convergence
    4. Dimensional Reduction Gating - gates by coprime structure
    
    When the system detects potential collapse (low chiral score, broken coprime
    parity), it attempts speculative recovery using Wasserstein optimal transport
    to move the converged state back toward a coprime-coherent manifold.
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8,
        recovery_threshold: float = 0.1,
        sinkhorn_iters: int = 20,
        wasserstein_epsilon: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.recovery_threshold = recovery_threshold
        
        # Sub-modules
        self.winding_tracker = CoprimeWindingTracker(dim=dim, num_heads=num_heads)
        self.chiral_estimator = ChiralCoherenceEstimator(dim=dim)
        self.wasserstein = WassersteinOptimalTransport(
            dim=dim, 
            sinkhorn_iters=sinkhorn_iters, 
            epsilon=wasserstein_epsilon
        )
        
        # Coprime reference manifold (learned target for recovery)
        self.register_buffer('coprime_manifold', torch.randn(16, dim))
        self.manifold_proj = nn.Linear(dim, dim)
        
        # Dimensional gating weights (learned per-dimension importance)
        self.dim_gate = nn.Parameter(torch.ones(dim))
        
        # --- Monstrous Equation Parameters ---
        # Near-Far Coupling (epsilon * K)
        self.far_coupling = nn.Parameter(torch.tensor(0.1))
        
        # Stress-Yield Thresholds (Mohr-Coulomb / Drucker-Prager)
        # Yield = |shear| - mu * normal
        self.yield_mu = nn.Parameter(torch.tensor(0.5)) # Friction coefficient
        self.yield_cohesion = nn.Parameter(torch.tensor(0.1)) # Cohesion c
        
        # Recovery MLP (Now handles Near-Far concatenation)
        self.recovery_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), # Original + Transported + Far-Coupled
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
    def get_yield_pressure(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes Mohr-Coulomb yield pressure (|\Sigma_shear| - mu * \Sigma_normal).
        """
        # Distinguish shear (off-diagonal) and normal (diagonal) via projection
        half = self.dim // 2
        normal_part = state[:, :half]
        shear_part = state[:, half:]
        
        yield_val = shear_part.abs().mean(dim=-1) - self.yield_mu * normal_part.abs().mean(dim=-1) - self.yield_cohesion
        return F.relu(yield_val) # Success iff yield > 0 (generative rupture)

    def update_manifold(self, state: torch.Tensor):
        """
        Update the coprime reference manifold with new samples when in good state.
        """
        if state.dim() == 2:
            state = state[0]
            
        # Shift and update (FIFO)
        self.coprime_manifold = torch.roll(self.coprime_manifold, -1, dims=0)
        self.coprime_manifold[-1] = state.detach()
        
    def gated_output(self, state: torch.Tensor, parity_violations: torch.Tensor) -> torch.Tensor:
        """
        Apply dimensional gating based on coprime parity.
        
        Dimensions associated with violated parity are suppressed.
        """
        # Map head-level violations to dim-level gates
        num_heads = parity_violations.shape[0]
        dims_per_head = self.dim // num_heads
        
        gate_mask = torch.ones(self.dim, device=state.device)
        for h in range(num_heads):
            if parity_violations[h]:
                start = h * dims_per_head
                end = start + dims_per_head
                gate_mask[start:end] *= 0.1  # Suppress violated dimensions
        
        # Apply learned gate and mask
        gate = torch.sigmoid(self.dim_gate) * gate_mask
        
        return state * gate
        
    def speculative_recovery(
        self, 
        converged_state: torch.Tensor,
        residues: Optional[torch.Tensor] = None,
        chirality_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Attempt speculative recovery of structure from converged state.
        
        Uses Wasserstein optimal transport to move the converged state
        toward the coprime reference manifold.
        
        Args:
            converged_state: [batch, dim] potentially collapsed state
            residues: [batch, k] CRT residues (if available)
            chirality_target: [batch, dim] target chirality alignment (e.g., input)
            
        Returns:
            recovered_state: [batch, dim] recovered state
            metrics: dict with recovery diagnostics
        """
        batch = converged_state.shape[0]
        
        # Project coprime manifold through learnable transform
        target_manifold = self.manifold_proj(self.coprime_manifold)
        
        # If chirality target provided, bias manifold toward it
        if chirality_target is not None:
            # Add chirality target as additional transport targets
            target_manifold = torch.cat([
                target_manifold,
                chirality_target.expand(4, -1)  # Repeat target
            ], dim=0)
        
        # Flatten batch for transport
        source = converged_state  # [batch, dim]
        
        # Compute optimal transport toward coprime manifold
        transported, wasserstein_dist = self.wasserstein.transport(source, target_manifold)
        
        # --- Near-Far Coupling ---
        # "far" component is a global manifold summary (mean of target)
        far_state = target_manifold.mean(dim=0, keepdim=True).expand(batch, -1)
        far_coupled = self.far_coupling * far_state
        
        # Blend transported with original using recovery MLP
        combined = torch.cat([converged_state, transported, far_coupled], dim=-1)
        recovery_delta = self.recovery_mlp(combined)
        
        # Recovered state = original + learned recovery
        recovered_state = converged_state + recovery_delta
        
        # Check Yield Condition (is this rupture generative?)
        yield_pressure = self.get_yield_pressure(recovered_state)
        
        # Check if recovery achieved coprime lock
        winding_result = self.winding_tracker.update_and_check(recovered_state)
        chiral_score = self.chiral_estimator.compute_chiral_score(recovered_state)
        
        # Apply dimensional gating
        recovered_state = self.gated_output(recovered_state, winding_result['parity_violations'])
        
        # Success criteria for "Generative Rupture"
        is_generative = yield_pressure.mean() > 0.0 or winding_result['coprime_lock']
        
        # If generative lock achieved, update reference manifold
        if is_generative and chiral_score > self.recovery_threshold:
            self.update_manifold(recovered_state)
        
        metrics = {
            'coprime_lock': winding_result['coprime_lock'],
            'chiral_score': chiral_score.item(),
            'wasserstein_distance': wasserstein_dist.item(),
            'yield_pressure': yield_pressure.mean().item(),
            'winding_numbers': winding_result['winding_numbers'],
            'parity_violations': winding_result['parity_violations'].sum().item(),
            'recovery_attempted': True,
            'is_generative': bool(is_generative)
        }
        
        return recovered_state, metrics
    
    def forward(
        self, 
        state: torch.Tensor,
        abort_score: Optional[torch.Tensor] = None,
        residues: Optional[torch.Tensor] = None,
        chirality_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Main forward pass with conditional speculative recovery.
        
        Args:
            state: [batch, dim] current state
            abort_score: [batch, 1] CALM abort score (triggers recovery if > 0.5)
            residues: [batch, k] CRT residues
            chirality_target: [batch, dim] target for chirality alignment
            
        Returns:
            output_state: [batch, dim] possibly recovered state
            metrics: dict with diagnostics
        """
        # Track winding and chiral coherence
        winding_result = self.winding_tracker.update_and_check(state)
        chiral_score = self.chiral_estimator.compute_chiral_score(state)
        
        # Determine if recovery needed
        needs_recovery = False
        
        if abort_score is not None and abort_score.mean().item() > 0.5:
            needs_recovery = True
        elif chiral_score < self.recovery_threshold:
            needs_recovery = True
        elif not winding_result['coprime_lock']:
            needs_recovery = True
            
        # Attempt recovery or pass through
        if needs_recovery:
            output_state, recovery_metrics = self.speculative_recovery(
                converged_state=state,
                residues=residues,
                chirality_target=chirality_target
            )
            metrics = recovery_metrics
        else:
            # Good state - update reference manifold and pass through
            self.update_manifold(state)
            output_state = self.gated_output(state, winding_result['parity_violations'])
            metrics = {
                'coprime_lock': winding_result['coprime_lock'],
                'chiral_score': chiral_score.item(),
                'wasserstein_distance': 0.0,
                'winding_numbers': winding_result['winding_numbers'],
                'parity_violations': winding_result['parity_violations'].sum().item(),
                'recovery_attempted': False
            }
        
        return output_state, metrics
