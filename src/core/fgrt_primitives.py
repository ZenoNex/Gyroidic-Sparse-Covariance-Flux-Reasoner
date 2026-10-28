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
from src.core.invariants import (
    PI,
    is_prime,
    PrimeResonanceLadder,
    get_prime_ladder,
    catalan_number,
    get_catalan_ladder,
    segner_catalan_weights,
    CatalanBranchingInvariant
)
from typing import Tuple, Optional, Dict, Any, Union
from src.core.honest_jitter import harvest_honest_jitter
import time

def expand_bits(v: torch.Tensor) -> torch.Tensor:
    """
    Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    Used for 3D Morton encoding (Z-order curve) on GPU/CPU.
    """
    v = v.to(torch.int32)
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8))  & 0x0300F00F
    v = (v | (v << 4))  & 0x030C30C3
    v = (v | (v << 2))  & 0x09249249
    return v

def morton_encode_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Computes the 3D Morton code by interleaving the bits of x, y, and z.
    Maintains spatial locality when flattening 3D coordinates to 1D indices,
    which is critical for preventing cache misses during topological ADMM repair.
    """
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)

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

class GyroidicAdmissibilityFilter(nn.Module):
    """
    Enforces the Speculative Exit threshold (H_{spec} < epsilon).
    Rejects topologically invalid thoughts before they are serialized.
    """
    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon
        self.gyroid = GyroidManifold()
        
    def forward(self, state: torch.Tensor) -> Tuple[bool, float]:
        """
        Evaluates Gyroidic Admissibility.
        
        Args:
            state: The manifold state [batch, dim]
            
        Returns:
            is_admissible: True if H_spec < epsilon
            h_spec: The computed speculative constraint violation
        """
        # H_{spec} is the mean absolute violation of the Gyroid constraint
        # Expand state to 3D groups for Gyroid if needed
        # Assume state represents coordinates or can be reshaped
        dim = state.shape[-1]
        if dim < 3:
            padded_state = F.pad(state, (0, 3 - dim))
            g_violation = self.gyroid(padded_state)
        else:
            # Group into triplets and average
            triplets = dim // 3
            if triplets * 3 < dim:
                state_trunc = state[..., :triplets*3]
            else:
                state_trunc = state
            
            state_reshaped = state_trunc.view(*state_trunc.shape[:-1], triplets, 3)
            g_violation = self.gyroid(state_reshaped).mean(dim=-1)
            
        h_spec = g_violation.abs().mean().item()
        is_admissible = h_spec < self.epsilon
        
        return is_admissible, h_spec

class TorsionConnection(nn.Module):
    """
    Affine connection with Torsion field for Chiral Symmetry Breaking,
    parameterized via Rashba-Dresselhaus Spin-Orbit Coupling.
    """
    def __init__(self, dim: int, alpha: float = 0.05, beta: float = 0.05):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        
        # Base antisymmetric torsion tensor initialized with honest jitter
        init_torsion = harvest_honest_jitter((dim, dim, dim), scaled=True) * 0.01
        
        # Embed Rashba-Dresselhaus SU(2) Pauli blocks across adjacent coordinate pairs
        # A_x = -alpha * sigma_y + beta * sigma_x
        # A_y =  alpha * sigma_x - beta * sigma_y
        half = dim // 2
        for j in range(half):
            idx0 = 2 * j
            idx1 = 2 * j + 1
            # sigma_x: anti-diagonal, sigma_y: real antisymmetric representation
            init_torsion[idx0, idx0, idx1] += float(-alpha)
            init_torsion[idx0, idx1, idx0] += float(alpha)
            init_torsion[idx0, idx0, idx0] += float(beta)
            init_torsion[idx1, idx0, idx0] += float(alpha)
            init_torsion[idx1, idx0, idx1] += float(-beta)
            init_torsion[idx1, idx1, idx0] += float(beta)
            
        self.torsion = nn.Parameter(init_torsion)

    def get_psh_delta(self) -> torch.Tensor:
        """Returns |alpha^2 - beta^2|, measuring deviation from Persistent Spin Helix."""
        return torch.abs(self.alpha ** 2 - self.beta ** 2)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the covariant derivative update with torsion.
        nabla_v x = v * grad(x) + T(v, x)
        """
        tx = torch.einsum('...i,...j,ijk->...k', v, x, self.torsion)
        return tx

class BerryPhaseTracker(nn.Module):
    """
    Tracks the Geometric Berry Phase across recursions.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('running_phase', torch.zeros(1))

    def update(self, prev_state: torch.Tensor, current_state: torch.Tensor, torsion: Optional['TorsionConnection'] = None):
        """
        Computes Diegetic Holonomy: Geometric Phase shift across state transition.
        Interprets state as N/2 complex pairs: z = x_2k + i*x_{2k+1}.
        If torsion is provided, aligns computed Berry holonomies Delta phi with the
        physical gauge curvature solid angle swept by the state vector.
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
        
        if torsion is not None:
            # Physical solid angle swept by state vector under (alpha, beta) gauge curvature
            psh_delta = torsion.get_psh_delta()
            area_element = (z_prev[..., 0] * z_curr[..., 1] - z_prev[..., 1] * z_curr[..., 0])
            delta_phi = delta_phi + 0.5 * psh_delta * area_element

        # Wrap to [-pi, pi] to handle phase wrapping
        delta_phi = (delta_phi + PI) % (2 * PI) - PI
        
        # Accumulate mean absolute holonomy (Structural Winding)
        self.running_phase += delta_phi.abs().mean()
        
        return delta_phi

# PrimeResonanceLadder and CatalanBranchingInvariant are canonically defined upstream in src.core.invariants.
# Re-exported here for backwards compatibility.
__all_primitives__ = ["PrimeResonanceLadder", "CatalanBranchingInvariant", "catalan_number", "get_catalan_ladder"]


class RepunitHasher(nn.Module):
    """
    Uses Repunit sequences to map continuous topological states to discrete 
    hash-like cyclic markers. These markers act as digital congruence checks
    before resorting to continuous Wasserstein Optimal Transport.
    """
    def __init__(self, base: int = 2, sequence_length: int = 10, device: str = None):
        super().__init__()
        self.base = base
        self.sequence_length = sequence_length
        self.register_buffer('repunits', self._generate_repunits(base, sequence_length, device))
        
    def _generate_repunits(self, base: int, length: int, device: str = None) -> torch.Tensor:
        """R_n^(b) = (b^n - 1) / (b - 1)"""
        repunits = []
        for n in range(1, length + 1):
            r = (base**n - 1) // (base - 1)
            repunits.append(r)
        return torch.tensor(repunits, dtype=torch.float32, device=device)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Maps continuous state to repunit-based cyclic markers.
        """
        dim = state.shape[-1]
        repunits_expanded = self.repunits[:dim] if dim <= self.sequence_length else torch.nn.functional.pad(self.repunits, (0, dim - self.sequence_length), value=1)
        
        # Scale and extract fractional phase mapped against the cyclic repunits
        scaled_state = state * repunits_expanded
        
        # Stability Guard: Hardware-anchored jitter grounding to prevent diagnostic lock (0.4390 attractor)
        # Follows 45.2 (Silicon Sovereignty).
        jitter = harvest_honest_jitter(scaled_state.shape, device=state.device, scaled=True) * 0.001
        
        # We derive a discrete cyclic marker:
        cyclic_markers = torch.fmod(torch.abs(scaled_state + jitter), repunits_expanded)
        return cyclic_markers


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
        entropy_matrix = alpha / (torch.exp(torch.tensor(PI) / product) + 1.0)
        self.register_buffer('entropy_matrix', entropy_matrix)
    
    def _generate_fibonacci(self, n: int) -> torch.Tensor:
        """Generates the first n Fibonacci numbers (starting from 1, 1)."""
        fibs = [1, 1]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        return torch.tensor(fibs[:n], dtype=torch.long)
    
    def _generate_primes(self, n: int) -> torch.Tensor:
        """Generates the first n prime numbers via PrimeResonanceLadder."""
        return PrimeResonanceLadder(num_resonators=n).primes
    
    def forward(self, i: int = None, j: int = None) -> torch.Tensor:
        """
        Returns the entropy matrix S[i,j] or a specific entry.
        
        If i and j are None, returns the full [N, N] entropy matrix.
        If i and j are provided, returns the scalar S(i,j).
        """
        if i is not None and j is not None:
            return self.entropy_matrix[i, j]
        return self.entropy_matrix

    def hunger_modulated_entropy(self, hunger: float = 0.0) -> torch.Tensor:
        """
        Recomputes the entropy matrix with Manifold Hunger (H) expanding
        the Fermi envelope.

        S_resonance(i,j) = alpha / (exp(pi / (F_i * P_j * (1 + H))) + 1)

        When H = 0: standard tight coupling (strict legibility).
        When H rises: the denominator's exponential argument shrinks,
        the Fermi envelope expands, and more oscillator pairs decouple.
        This organically widens the allowable resonance entropy.

        The CALM predictor monitors spectral entropy. By routing hunger
        through the coprime Fibonacci-Prime basis rather than injecting
        random noise, the widened entropy retains prime-harmonic structure.
        CALM reads this as "Generative Rupture" (valid starvation search)
        rather than "Terminal Degeneration" (random drift).

        Args:
            hunger: Scalar Manifold Hunger in [0, inf).
                    Typically from ValenceFunctional.forward().

        Returns:
            [N, N] hunger-modulated entropy matrix.
        """
        F_i = self.fibonacci.float().unsqueeze(1)  # [N, 1]
        P_j = self.primes.float().unsqueeze(0)     # [1, N]
        product = F_i * P_j * (1.0 + hunger)       # Hunger widens the lattice
        product = torch.clamp(product, min=1.0)
        modulated = self.alpha / (torch.exp(torch.tensor(PI) / product) + 1.0)
        return modulated

    def get_triadic_catalan_coupling(self, hunger: float = 0.0) -> torch.Tensor:
        """
        Computes the Triadic Prime-Fibonacci-Catalan Resonance Entropy matrix:
            S_triad(i, j) = alpha / (exp(pi / (F_i * P_j * log(C_i + e) * (1 + H))) + 1)

        Unifies additive recurrence (Fibonacci F_i), multiplicative irreducibility (Primes P_j),
        and non-associative binary tree convolution (Catalan C_i).
        """
        import math
        F_i = self.fibonacci.float().unsqueeze(1)
        P_j = self.primes.float().unsqueeze(0)
        c_logs = torch.tensor(
            [math.log(float(catalan_number(k)) + math.e) for k in range(self.num_oscillators)],
            dtype=torch.float32,
            device=self.fibonacci.device
        ).unsqueeze(1)
        product = F_i * P_j * c_logs * (1.0 + hunger)
        product = torch.clamp(product, min=1.0)
        return self.alpha / (torch.exp(torch.tensor(PI, device=product.device) / product) + 1.0)


class CatalanAssociahedronEntropy(nn.Module):
    """
    Catalan Associahedron Non-Associative Coupling & Bracket Entropy (RIC Eq 1.3).

    Governs non-associative operator parenthesizations F(x) = B_k G_m E_n(x) across
    the Stasheff Associahedron K_{n+1}, whose vertices correspond to the C_n binary
    bracketings of n + 1 non-associative operators or dyad fusions.

    Coupling Lattice:
        S_catalan(i, j) = alpha / (exp(pi / (log(C_i + e) * P_j)) + 1)
    """
    def __init__(self, num_oscillators: int = 20, alpha: float = 1.0):
        super().__init__()
        import math
        self.num_oscillators = num_oscillators
        self.alpha = alpha

        catalan_vals = [catalan_number(k) for k in range(num_oscillators)]
        catalan_logs = [math.log(float(c) + math.e) for c in catalan_vals]
        # Clamp int64 representation for buffer safety while keeping exact float logs
        catalan_clamped = [min(c, (1 << 62) - 1) for c in catalan_vals]
        self.register_buffer("catalan", torch.tensor(catalan_clamped, dtype=torch.long))
        self.register_buffer("catalan_log", torch.tensor(catalan_logs, dtype=torch.float32))

        primes = PrimeResonanceLadder(num_resonators=num_oscillators).primes
        self.register_buffer("primes", primes)

        C_i = self.catalan_log.unsqueeze(1)        # [N, 1]
        P_j = primes.float().unsqueeze(0)          # [1, N]
        product = torch.clamp(C_i * P_j, min=1.0)  # [N, N]
        associahedron_matrix = alpha / (torch.exp(torch.tensor(PI) / product) + 1.0)
        self.register_buffer("associahedron_matrix", associahedron_matrix)

    def forward(self, i: Optional[int] = None, j: Optional[int] = None) -> torch.Tensor:
        """Returns the full [N, N] Catalan-Prime Associahedron matrix or entry (i, j)."""
        if i is not None and j is not None:
            return self.associahedron_matrix[i, j]
        return self.associahedron_matrix

    def non_associative_bracket_convolution(self, left_state: torch.Tensor, right_state: torch.Tensor) -> torch.Tensor:
        """
        Computes the Segner-Catalan non-associative binary tree convolution of two latent states:
            (u *_C v)_n = sum_{k=0}^{n} w_{n,k} * u_k * v_{n-k}
        where w_{n,k} = (C_k * C_{n-k}) / C_{n+1} are the normalized Segner split weights.
        """
        orig_shape = left_state.shape
        u = left_state.reshape(-1, orig_shape[-1])
        v = right_state.reshape(-1, orig_shape[-1])
        D = u.shape[-1]
        out = torch.zeros_like(u)
        for n in range(D):
            w = segner_catalan_weights(n + 1, device=u.device, dtype=u.dtype)  # [n + 1]
            u_slice = u[:, :n + 1]
            v_slice = torch.flip(v[:, :n + 1], dims=[-1])
            out[:, n] = torch.sum(w.unsqueeze(0) * u_slice * v_slice, dim=-1)
        return out.reshape(orig_shape)


class CoherentPrimeResonance(nn.Module):
    """
    Coherent Prime Resonance (CPR) Condition (RIC Eq 7).
    
    CPR(F, {u_n}) = 1 iff:
        1. PAS_h(F) >= theta_CPR         (global phase coherence)
        2. forall n: <u_n, F> > 0         (breather-field alignment)
        3. Spec(F) subset {p_n}           (spectral support on primes)
    
    When CPR is satisfied, the harmonic field and its breather modes are
    mutually phase-locked  cognition as standing-wave interference.
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
        ladder = PrimeResonanceLadder(num_resonators=num_primes)
        prime_freqs = 2 * PI * torch.log(ladder.primes.float())
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
            bin_idx = min(int(pf.item() * num_bins / (2 * PI * 10)), num_bins - 1)
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


class CALMCollapseDetector(nn.Module):
    """
    CALM Collapse Detector.
    Uses bounded Martinova correlation and Collatz limit-cycle monitoring
    to track structural collapse, stagnation, or non-terminating cycle debt (BB(5) Collatz coral)
    in trajectory history of states. When cycle_debt > 0.5 or Martinova correlation >= warning_threshold,
    triggers an entropic veto, diverting execution into the SpeculativeCoprimeGate (SCCCG) rather than stalling.
    """
    def __init__(self, warning_threshold: float = 0.8, cycle_debt_threshold: float = 0.5):
        super().__init__()
        self.warning_threshold = warning_threshold
        self.cycle_debt_threshold = cycle_debt_threshold

    def forward(
        self,
        history: torch.Tensor,
        cycle_debt: Optional[Union[float, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history: [batch, history_len, dim] trajectory history
            cycle_debt: Optional cycle debt measure monitoring Collatz-like modular traps
            
        Returns:
            is_collapsed: [batch] boolean indicator of collapse or entropic veto
            correlation: [batch] local Martinova correlation score in [-1, 1]
        """
        from src.core.martinova_correlation import compute_bounded_correlation
        # Compute bounded correlation of history. Treat history_len as N points of dim-dimensional space.
        corr = compute_bounded_correlation(history)
        
        # Stagnation/collapse is characterized by maximal clustering/fossilization (corr -> 1.0)
        is_collapsed = corr >= self.warning_threshold
        
        # Monitor non-terminating Collatz limit cycle debt (cycle_debt > 0.5)
        if cycle_debt is not None:
            if isinstance(cycle_debt, torch.Tensor):
                c_val = cycle_debt.squeeze()
                collatz_trap = c_val > self.cycle_debt_threshold
            else:
                collatz_trap = torch.tensor(cycle_debt > self.cycle_debt_threshold, device=history.device)
            is_collapsed = is_collapsed | collatz_trap
            
        return is_collapsed, corr



class KleinThroatTransition(nn.Module):
    """
    Klein Throat Transition.
    Implements orientation flipping for normal vectors traversing the non-orientable
    Klein-bottle throat, tracks the Contorsion Tensor, and handles Geometric Berry Phase backpropagation.
    """
    def __init__(self, dim: int = 3):
        super().__init__()
        self.dim = dim
        # Contorsion tensor: K^i_{jk} = T^i_{jk} + T_{j k}^i + T_{k j}^i (where T is torsion)
        self.contorsion = nn.Parameter(harvest_honest_jitter((dim, dim, dim), scaled=True) * 0.02)
        # Berry Phase Tracker for holonomy
        self.berry_tracker = BerryPhaseTracker()

    def forward(self, state: torch.Tensor, normal_vector: torch.Tensor, traverse_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [batch, dim] representing coordinate state
            normal_vector: [batch, dim] representing normal vectors (e.g. from GyroidManifold.gradient)
            traverse_threshold: threshold to trigger crossing the throat
            
        Returns:
            flipped_normal: [batch, dim] orientation-flipped normal vector if crossing the throat
            contorsion_effect: [batch, dim] contorsion tensor tracking contribution
            berry_phase: [batch] geometric phase shift
        """
        # 1. Orientation Flipping: If traversing the Klein-bottle throat, the normal vector's orientation flips.
        traverse_factor = torch.sin(state[:, 0])  # Proxy for throat position
        cross_throat = traverse_factor.abs() > traverse_threshold
        
        # Flip orientation along the traverse boundary (non-orientability)
        flipped_normal = normal_vector.clone()
        flipped_normal[cross_throat] = -flipped_normal[cross_throat]
        
        # 2. Contorsion Tensor Tracking: K(v, w) = v_j * w_k * K^i_{jk}
        contorsion_effect = torch.einsum('...j,...k,ijk->...i', state, flipped_normal, self.contorsion)
        
        # 3. Geometric Berry Phase Tracking
        berry_phase = self.berry_tracker.update(state, state + contorsion_effect)
        
        return flipped_normal, contorsion_effect, berry_phase


# =========================================================================
# FRACTAL L'HÔPITAL REGULARIZATION & VALLEY VORTICITY SWEEPER
# =========================================================================

class FractalLHopitalRegularizer(nn.Module):
    """
    Non-Differentiable Fractal L'Hôpital Regularization:
    lim_{eps -> 0} S(eps) / V(eps) -[invalid L'Hopital]-> undefined
    ===> R_{reg} = lim_{eps -> eps_min^+} C * eps^{2-D}, where eps_{min} = lambda_{min} / kappa_{accel}.

    As proven in docs/Recursive Fractals and L'Hôpital.txt, boundary differentiation
    breaks down across recursive Weierstrass fractures.
    This module enforces the spectral/hardware regularized limit R_reg.
    """
    def __init__(self, fractal_dim_D: float = 2.73, C_constant: float = 1.0):
        super().__init__()
        self.fractal_dim_D = fractal_dim_D
        self.C_constant = C_constant

    def compute_cutoff(self, lambda_min: float = 1e-3, kappa_accel: float = 1.0) -> float:
        """eps_{min} = lambda_{min} / kappa_{accel}."""
        return max(1e-8, float(lambda_min) / max(1e-8, float(kappa_accel)))

    def evaluate_regularized_ratio(
        self,
        lambda_min: float = 1e-3,
        kappa_accel: float = 1.0,
        fractal_dim_D: Optional[float] = None,
        C_constant: Optional[float] = None
    ) -> float:
        """Computes R_{reg} = C * eps_{min}^{2 - D}."""
        D = fractal_dim_D if fractal_dim_D is not None else self.fractal_dim_D
        C = C_constant if C_constant is not None else self.C_constant
        eps_min = self.compute_cutoff(lambda_min, kappa_accel)
        return float(C * (eps_min ** (2.0 - D)))

    def regularize_complexity(
        self,
        raw_complexity: torch.Tensor,
        lambda_min: float = 1e-3,
        kappa_accel: float = 1.0
    ) -> torch.Tensor:
        """Applies R_{reg} scaling factor to regularize non-differentiable fractal complexity."""
        r_reg = self.evaluate_regularized_ratio(lambda_min, kappa_accel)
        return raw_complexity * r_reg


class ValleyVorticitySweeper(nn.Module):
    """
    Valley Circulation Flux Threshold & Sweep Operator:
    Upsilon_{valley} = sum_j oint_{partial gamma_j} (nabla x S) . dA > tau_{crit} ===> P_sweep

    Monitors the vorticity/circulation flux Upsilon_{valley} of the entropy/stress vector
    field S across valley boundary loops partial gamma_j. When Upsilon exceeds critical threshold
    tau_{crit}, triggers the projection sweep operator P_sweep to flush vortex traps and
    choked topological debris.
    """
    def __init__(self, tau_crit: float = 2.5, sweep_damping: float = 0.85):
        super().__init__()
        self.tau_crit = tau_crit
        self.sweep_damping = sweep_damping

    def compute_vorticity_flux(
        self,
        curl_S_fields: torch.Tensor,
        area_normals: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes Upsilon_{valley} = sum_j (nabla x S)_j . dA_j.
        curl_S_fields: [..., J, 3] or [J, 3]
        area_normals: [..., J, 3] or [J, 3]
        """
        dot_flux = torch.sum(curl_S_fields * area_normals, dim=-1)
        upsilon = torch.sum(dot_flux, dim=-1)
        return upsilon

    def apply_sweep_operator(
        self,
        state_tensor: torch.Tensor,
        upsilon_val: float,
        tau_crit: Optional[float] = None
    ) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
        """
        Checks if Upsilon_{valley} > tau_{crit}. If triggered, applies P_sweep
        projection operator: P_sweep(S) = S - (1 - sweep_damping) * S_{vortex}.
        """
        crit = tau_crit if tau_crit is not None else self.tau_crit
        triggered = float(upsilon_val) > float(crit)
        
        if triggered:
            # P_sweep projection: projects out divergent vortex accumulation
            swept_state = state_tensor * self.sweep_damping
        else:
            swept_state = state_tensor
            
        diagnostics = {
            "upsilon_valley": float(upsilon_val),
            "tau_crit": float(crit),
            "sweep_triggered": triggered,
            "damping_applied": self.sweep_damping if triggered else 1.0
        }
        return swept_state, triggered, diagnostics



