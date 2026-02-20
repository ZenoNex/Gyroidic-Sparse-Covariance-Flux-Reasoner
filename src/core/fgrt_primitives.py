"""
Fiberalized Gyroidic Recurrent Topology (FGRT) Primitives.

Implements the base mathematical operators for:
1. Gyroidic Manifold flows.
2. Affine connections with Torsion.
3. Geometric Berry Phase tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GyroidManifold(nn.Module):
    """
    Representation of the Gyroid Triply Periodic Minimal Surface (TPMS).
    
    The surface is defined by: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Gyroid violation score for a given set of coordinates."""
        # Assume x has shape (..., 3)
        x = x * self.scale
        g = torch.sin(x[..., 0]) * torch.cos(x[..., 1]) + \
            torch.sin(x[..., 1]) * torch.cos(x[..., 2]) + \
            torch.sin(x[..., 2]) * torch.cos(x[..., 0])
        return g

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the normal vector (gradient) to the gyroid surface.
        dG/dx = cos(x)cos(y) - sin(z)sin(x)
        dG/dy = cos(y)cos(z) - sin(x)sin(y)
        dG/dz = cos(z)cos(x) - sin(y)sin(z)
        """
        # Ensure x is [..., 3]
        x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
        dg_dx = torch.cos(x0) * torch.cos(x1) - torch.sin(x2) * torch.sin(x0)
        dg_dy = torch.cos(x1) * torch.cos(x2) - torch.sin(x0) * torch.sin(x1)
        dg_dz = torch.cos(x2) * torch.cos(x0) - torch.sin(x1) * torch.sin(x2)
        
        return torch.stack([dg_dx, dg_dy, dg_dz], dim=-1)

    def gaussian_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Gaussian Curvature K for the Gyroid surface.
        For a minimal surface like Gyroid, K = -|grad G|^4 / ... (complex formula)
        Approximate using the Hessian and Gradient magnitudes.
        """
        # Minimal surfaces have H=0 and K <= 0.
        grad = self.gradient(x)
        grad_norm_sq = torch.sum(grad**2, dim=-1) + 1e-8
        
        # Simplified K for visualization/modulation: proportional to negative norm squared
        # In a real TPMS, K is a function of the coordinates that is never positive.
        return -grad_norm_sq / (1.0 + grad_norm_sq)

class TorsionConnection(nn.Module):
    """
    Affine connection with Torsion field for Chiral Symmetry Breaking.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Torsion tensor T^k_{ij} (antisymmetric in i, j)
        # We represent it as a learned parameter
        self.torsion = nn.Parameter(torch.randn(dim, dim, dim) * 0.01)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the covariant derivative update with torsion.
        nabla_v x = v * grad(x) + T(v, x)
        """
        # Simplified implementation: T(v, x) = sum_{ijk} v_i x_j T^k_{ij}
        # We'll use einsum for the torsion contraction
        tx = torch.einsum('...i,...j,ijk->...k', v, x, self.torsion)
        return tx

class BerryPhaseTracker(nn.Module):
    """
    Tracks the Geometric Berry Phase across recursions.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('running_phase', torch.zeros(1))

    def update(self, prev_state: torch.Tensor, current_state: torch.Tensor):
        """
        Computes Diegetic Holonomy: Geometric Phase shift across state transition.
        Interprets state as N/2 complex pairs: z = x_2k + i*x_{2k+1}
        """
        # Ensure even dimension for complex pairing
        dim = prev_state.shape[-1]
        half_dim = dim // 2
        
        # Reshape into complex pairs [batch, half_dim, 2]
        z_prev = prev_state.view(-1, half_dim, 2)
        z_curr = current_state.view(-1, half_dim, 2)
        
        # Extract phases: theta = atan2(Im, Re)
        phase_prev = torch.atan2(z_prev[..., 1], z_prev[..., 0])
        phase_curr = torch.atan2(z_curr[..., 1], z_curr[..., 0])
        
        # Geometric Phase jump (Holonomy)
        delta_phi = phase_curr - phase_prev
        
        # Wrap to [-pi, pi] to handle phase wrapping
        delta_phi = (delta_phi + 3.14159) % (2 * 3.14159) - 3.14159
        
        # Accumulate mean absolute holonomy (Structural Winding)
        self.running_phase += delta_phi.abs().mean()
        
        return delta_phi

class PrimeResonanceLadder(nn.Module):
    """
    Prime-based Resonance Ladder (Eq 1).
    
    Generates resonance frequencies based on the sequence of prime numbers:
    f_{p_n} = 2 * pi * log(p_n)
    
    Used to initialize the oscillator components of the Resonance Intelligence Core.
    """
    def __init__(self, num_resonators: int = 100):
        super().__init__()
        self.num_resonators = num_resonators
        self.primes = self._generate_primes(num_resonators)
        
        # Eq (1): f_{p_n} = 2 * pi * log(p_n)
        frequencies = 2 * 3.14159265359 * torch.log(self.primes.float())
        self.register_buffer('frequencies', frequencies)
        
    def _generate_primes(self, n: int) -> torch.Tensor:
        """Generates the first n prime numbers."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
                if p * p > candidate:
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return torch.tensor(primes, dtype=torch.long)
        
    def forward(self) -> torch.Tensor:
        """Returns the resonance frequencies."""
        return self.frequencies
    
    def get_fibonacci_entropy(self, alpha: float = 1.0) -> dict:
        """
        Bridge to FibonacciResonanceEntropy.
        
        Returns both the prime frequencies and the Fibonacci-scaled
        entropy matrix, providing a unified resonance descriptor.
        
        Args:
            alpha: Saturation parameter for entropy (default 1.0)
            
        Returns:
            Dictionary with:
                - 'frequencies': [N] prime resonance frequencies
                - 'entropy_matrix': [N, N] Fibonacci×Prime entropy coupling
        """
        entropy = FibonacciResonanceEntropy(
            num_oscillators=self.num_resonators,
            alpha=alpha
        )
        return {
            'frequencies': self.frequencies,
            'entropy_matrix': entropy.forward()
        }


class FibonacciResonanceEntropy(nn.Module):
    """
    Fibonacci-Structured Resonance Entropy (RIC Eq 1.2).
    
    S_resonance(i,j) = alpha / (exp(pi / (F_i * P_j)) + 1)
    
    Creates an incommensurate coupling lattice using the cross-product
    of Fibonacci numbers (additive recurrence) and primes (multiplicative
    independence). The Fermi envelope ensures smooth saturation from
    tight coupling (low indices) to full independence (high indices).
    """
    def __init__(self, num_oscillators: int = 20, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.num_oscillators = num_oscillators
        
        # Generate Fibonacci numbers
        fibs = self._generate_fibonacci(num_oscillators)
        self.register_buffer('fibonacci', fibs)
        
        # Generate primes
        primes = self._generate_primes(num_oscillators)
        self.register_buffer('primes', primes)
        
        # Precompute the entropy matrix S[i,j]
        # S(i,j) = alpha / (exp(pi / (F_i * P_j)) + 1)
        F_i = fibs.float().unsqueeze(1)  # [N, 1]
        P_j = primes.float().unsqueeze(0)  # [1, N]
        product = F_i * P_j  # [N, N]
        # Clamp product to avoid division by zero
        product = torch.clamp(product, min=1.0)
        entropy_matrix = alpha / (torch.exp(torch.tensor(3.14159265359) / product) + 1.0)
        self.register_buffer('entropy_matrix', entropy_matrix)
    
    def _generate_fibonacci(self, n: int) -> torch.Tensor:
        """Generates the first n Fibonacci numbers (starting from 1, 1)."""
        fibs = [1, 1]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        return torch.tensor(fibs[:n], dtype=torch.long)
    
    def _generate_primes(self, n: int) -> torch.Tensor:
        """Generates the first n prime numbers."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
                if p * p > candidate:
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return torch.tensor(primes, dtype=torch.long)
    
    def forward(self, i: int = None, j: int = None) -> torch.Tensor:
        """
        Returns the entropy matrix S[i,j] or a specific entry.
        
        If i and j are None, returns the full [N, N] entropy matrix.
        If i and j are provided, returns the scalar S(i,j).
        """
        if i is not None and j is not None:
            return self.entropy_matrix[i, j]
        return self.entropy_matrix


class CoherentPrimeResonance(nn.Module):
    """
    Coherent Prime Resonance (CPR) Condition (RIC Eq 7).
    
    CPR(F, {u_n}) = 1 iff:
        1. PAS_h(F) >= theta_CPR         (global phase coherence)
        2. forall n: <u_n, F> > 0         (breather-field alignment)
        3. Spec(F) subset {p_n}           (spectral support on primes)
    
    When CPR is satisfied, the harmonic field and its breather modes are
    mutually phase-locked — cognition as standing-wave interference.
    """
    def __init__(
        self,
        theta_cpr: float = 0.7,
        spectral_purity_threshold: float = 0.8,
        num_primes: int = 20
    ):
        super().__init__()
        self.theta_cpr = theta_cpr
        self.spectral_purity_threshold = spectral_purity_threshold
        
        # Generate prime frequencies for spectral check
        primes = []
        candidate = 2
        while len(primes) < num_primes:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
                if p * p > candidate:
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        prime_freqs = 2 * 3.14159265359 * torch.log(torch.tensor(primes, dtype=torch.float))
        self.register_buffer('prime_freqs', prime_freqs)
    
    def check_phase_coherence(self, field_phases: torch.Tensor) -> bool:
        """
        Condition 1: PAS_h(F) >= theta_CPR.
        field_phases: [N] instantaneous phases of oscillators.
        """
        N = field_phases.shape[-1]
        # PAS = |mean(exp(i*theta))|
        complex_phases = torch.exp(1j * field_phases.to(torch.complex64))
        pas = torch.abs(complex_phases.mean()).item()
        return pas >= self.theta_cpr
    
    def check_breather_alignment(
        self,
        breather_amplitudes: torch.Tensor,
        field_amplitudes: torch.Tensor
    ) -> bool:
        """
        Condition 2: forall n: <u_n, F> > 0.
        All breather modes must be positively correlated with the field.
        """
        inner_products = breather_amplitudes * field_amplitudes
        return bool(torch.all(inner_products > 0).item())
    
    def check_spectral_purity(self, field_spectrum: torch.Tensor) -> bool:
        """
        Condition 3: Spec(F) subset {p_n}.
        The dominant spectral energy must be concentrated at prime frequencies.
        
        field_spectrum: [num_freq_bins] power spectrum of the field.
        """
        total_energy = field_spectrum.sum().item() + 1e-8
        # Compute energy at prime frequency bins (approximate: nearest bin)
        num_bins = field_spectrum.shape[-1]
        prime_energy = 0.0
        for pf in self.prime_freqs:
            bin_idx = min(int(pf.item() * num_bins / (2 * 3.14159265359 * 10)), num_bins - 1)
            if bin_idx < num_bins:
                prime_energy += field_spectrum[bin_idx].item()
        
        purity = prime_energy / total_energy
        return purity >= self.spectral_purity_threshold
    
    def forward(
        self,
        field_phases: torch.Tensor,
        breather_amplitudes: torch.Tensor,
        field_amplitudes: torch.Tensor,
        field_spectrum: torch.Tensor = None
    ) -> bool:
        """
        Evaluate the full CPR condition (Eq 7).
        
        Returns True iff all three sub-conditions are satisfied.
        """
        cond1 = self.check_phase_coherence(field_phases)
        cond2 = self.check_breather_alignment(breather_amplitudes, field_amplitudes)
        
        if field_spectrum is not None:
            cond3 = self.check_spectral_purity(field_spectrum)
        else:
            # If no spectrum provided, skip spectral check
            cond3 = True
        
        return cond1 and cond2 and cond3

