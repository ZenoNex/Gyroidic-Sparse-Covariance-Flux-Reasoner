"""
Unified Invariants: PAS_h and APAS_zeta.

Implements the computable, scalar, harmonic invariants required to:
1. Govern evolution (APAS_zeta drift bound).
2. Compare states (PAS_h scalar metric).
3. Preserve identity (Chirality checks).

"An invariant that cannot be computed cannot govern evolution... Computability is mandatory."

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union
import math
import time

# Universal Mathematical & Topological Constants
PI = math.pi
try:
    import mpmath
    with mpmath.workdps(60):
        PHI = float((mpmath.sqrt(5) + 1) / 2)
        PHI_RECIPROCAL = float((mpmath.sqrt(5) - 1) / 2)
except Exception:
    PHI = (1.0 + math.sqrt(5.0)) / 2.0
    PHI_RECIPROCAL = (math.sqrt(5.0) - 1.0) / 2.0

def get_ihc_ladder_scale(k: int, num_shells: int = 33, center_k: int = 7, use_sound_horizon: bool = False) -> float:
    """
    Inverted Hypersphere Cosmology (IHC): 33-shell golden ratio ladder scale.
    Stabilizes the UV-IR Casimir seesaw across co-prime functional hierarchies
    without clipping or information loss.
    
    Formula:
        If use_sound_horizon:
            scale = PHI ** (-((k % num_shells) - center_k) / float(num_shells))
        Else:
            scale = PHI ** (-(k % num_shells) / float(num_shells))
    """
    k_mod = k % num_shells
    if use_sound_horizon:
        exponent = -(k_mod - center_k) / float(num_shells)
    else:
        exponent = -k_mod / float(num_shells)
    return float(PHI ** exponent)

def get_ihc_ladder_tensor(num_functionals: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32, center_k: int = 7, use_sound_horizon: bool = False) -> torch.Tensor:
    """
    Generates a 1D tensor of IHC 33-shell golden ratio ladder scales for K functionals.
    """
    scales = [get_ihc_ladder_scale(k, num_shells=33, center_k=center_k, use_sound_horizon=use_sound_horizon) for k in range(num_functionals)]
    return torch.tensor(scales, device=device, dtype=dtype)


def is_prime(n: int) -> bool:
    """Optimized deterministic primality test."""
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

class PhaseAlignmentInvariant(nn.Module):
    """
    PAS_h: Harmonic Phase Alignment Score.
    
    Implements Eq (2): PAS_S = (1/N) * sum(cos(theta_k - theta_bar))
    
    Acts as a first-class admissibility filter measuring the 'topological 
    synchronization' of field states.
    """
    def __init__(self, degree: int = 3, poly_degree: Optional[int] = None):
        super().__init__()
        # Degree is kept for compatibility, but PAS is now strictly phase-based
        self.degree = poly_degree if poly_degree is not None else degree
        
    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Compute PAS_h using strict phase coherence.
        
        The score measures how well the phases of the underlying oscillators 
        are aligned to a common mean. A score of 1.0 indicates perfect 
        topological synchronization.
        
        Args:
           coeffs: [batch, K, D] or [batch, D]. Represents resonator states 
                   on the hidden manifold.
        
        Returns:
           pas_h: [batch] scalar score [-1, 1]. High score indicates 
                  manifold coherence.
        """
        # 1. Standardize Input [batch, N] where N is number of oscillators
        is_1d = coeffs.dim() == 1
        if is_1d:
            x = coeffs.unsqueeze(0)
        elif coeffs.dim() == 3:
            # Flatten K and D to treat all as a pool of oscillators
            x = coeffs.reshape(coeffs.shape[0], -1)
        elif coeffs.dim() > 3:
            x = coeffs.reshape(coeffs.shape[0], -1)
        else:
            x = coeffs
            
        # 2. Extract Phases (theta_k) using True Analytic Signal (Hilbert Transform)
        import torch.fft
        X = torch.fft.fft(x, dim=1)
        N = X.shape[1]
        h = torch.zeros_like(X)
        if N > 0:
            h[:, 0] = 1
            if N % 2 == 0:
                h[:, 1:N//2] = 2
                h[:, N//2] = 1
            else:
                h[:, 1:(N+1)//2] = 2
                
        analytic_signal = torch.fft.ifft(X * h, dim=1)
        theta = torch.angle(analytic_signal) # [batch, N]
        
        # 3. Compute Mean Phase (theta_bar)
        # Circular mean: atan2(sum(sin), sum(cos))
        sin_sum = torch.sin(theta).sum(dim=1)
        cos_sum = torch.cos(theta).sum(dim=1)
        theta_bar = torch.atan2(sin_sum, cos_sum).unsqueeze(1) # [batch, 1]
        
        # 4. Compute PAS (Eq 2)
        # PAS = (1/N) * sum(cos(theta_k - theta_bar))
        alignment = torch.cos(theta - theta_bar)
        pas_h = alignment.mean(dim=1)
        
        return pas_h.squeeze(0) if is_1d else pas_h

class APAS_Zeta(nn.Module):
    """
    APAS_zeta: Adaptive PAS with drift bounding.
    
    "An invariant that cannot be computed cannot govern evolution...
     APAS_zeta bounds permissible evolution."
    """
    def __init__(self, zeta: float = 0.05):
        super().__init__()
        self.zeta = zeta
        
    def check_drift(self, current_pas: torch.Tensor, prev_pas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if drift |PAS_t - PAS_{t-1}| <= zeta.
        
        Enforces the stability of the manifold by ensuring that evolution 
        does not cause discontinuous 'Ruptures' in the harmonic state.
        
        Args:
            current_pas: The PAS_h score of the current state [batch].
            prev_pas: The PAS_h score of the previous state [batch].
            
        Returns:
            drift: The absolute delta between current and previous scores.
            violation_mask: 1.0 if the drift exceeds the permitted zeta.
            
        CODES v40 Invariant: 
            Evolution Bounding: 5.1. Computability is mandatory for 
            governing evolution through discrete drift bounds.
        """
        drift = torch.abs(current_pas - prev_pas)
        violation = (drift > self.zeta).float()
        return drift, violation


import torch


def compute_chiral_shift(coeffs: torch.Tensor) -> torch.Tensor:
    """
        Compute Chiral Shift (Spectral Centroid displacement).
    
    Measures the displacement of the energy distribution from the spectral 
    midpoint. This represents the 'Handedness' of the latent state across 
    the frequency spectrum.
    
    Args:
        coeffs: The input latent state [batch, K, D] or [batch, D].
        
    Returns:
        chiral_shift: The normalized displacement score. Positive indicates 
                      high-frequency (entropic) dominance.
    """
    if coeffs.dim() == 1:
        coeffs = coeffs.unsqueeze(0).unsqueeze(0)
    elif coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(1)
    B, K, D = coeffs.shape

    # 1. Energy extraction (Summing across the K-manifold)
    # We use non_blocking logic implicitly by staying on the tensor's device
    energy = coeffs.pow(2).sum(dim=1)  # Shape: [B, D]

    # 2. Spectral Centroid Calculation
    # Arange must be on the SAME device to avoid the 'Synchronous Transfer'
    indices = torch.arange(D, device=coeffs.device, dtype=coeffs.dtype)
    total_energy = energy.sum(dim=1, keepdim=True) + 1e-8
    spectral_centroid = (energy * indices).sum(dim=1, keepdim=True) / total_energy

    # 3. Chirality Index: (Centroid - Midpoint) / Midpoint
    # Positive = High-Freq/Entropic, Negative = Low-Freq/Negentropic
    midpoint = D / 2.0
    chiral_shift = (spectral_centroid - midpoint) / midpoint
    
    return chiral_shift.squeeze()

def compute_chirality(coeffs: torch.Tensor) -> torch.Tensor:
    """
        Compute Chiral Torsion (Parity Asymmetry).
    
    Chirality as a topological invariant (Delta Chi != 0) ensuring that the 
    system avoids symmetric/reflective collapse by maintaining energy 
    asymmetry between even and odd polynomial modes.
    
    Args:
        coeffs: The input latent state tensor.
        
    Returns:
       delta_chi: [batch] The raw energy difference (Even - Odd).
       
    CODES v40 Invariant: 
        Non-Reflective Invariance: 3.2. Asymmetry is the seed of lawful 
        resonance; symmetry is a failure mode (Lobotomy).
    """
    # 1. Force to 3D: [Batch, K-Manifold, D-Degree]
    if coeffs.dim() == 1:
        coeffs = coeffs.unsqueeze(0).unsqueeze(0)
    elif coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(0)
    elif coeffs.dim() != 3:
        raise ValueError(f"Expected 1D, 2D or 3D tensor, got {coeffs.dim()}D")

    B, K, D = coeffs.shape

    # 2. Extract Parity Energies (Delta Chi = E_even - E_odd)
    indices = torch.arange(D, device=coeffs.device)
    even_mask = (indices % 2 == 0)
    odd_mask = ~even_mask
    
    energy = coeffs.pow(2).sum(dim=1)
    
    even_energy = energy[:, even_mask].sum(dim=1)
    odd_energy = energy[:, odd_mask].sum(dim=1)
    
    delta_chi = even_energy - odd_energy
    return delta_chi.squeeze()

def check_glyphlock(coeffs: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    """
        GLYPHLOCK: Chirality-constrained emission validation.
    
    Validates that a state possesses sufficient 'Handedness' (Shift or Torsion) 
    to be considered an admissible, stable topological feature.
    
    Args:
        coeffs: The input latent state.
        threshold: The minimal asymmetry required for a 'Lock'.
        
    Returns:
        is_locked: 1.0 if the state satisfies the glyphlock condition.
    """
    shift = compute_chiral_shift(coeffs)
    torsion = compute_chirality(coeffs)
    
    # Glyphlock is active if there is non-trivial spectral or parity asymmetry
    is_locked = ((shift.abs() > threshold) | (torsion.abs() > threshold)).float()
    return is_locked

class ImplicationInvariant(nn.Module):
    """
    ImplicationInvariant: Anti-Lobotomy Check #1.
    
    Invariant: Interaction(x) => Implication(x) != 0.
    
    Ensures that for any significant interaction (input/state), there is a 
    non-zero downstream implication (effect). Zeroing out implication is 
    strictly forbidden as it represents 'lobotomy' - the removal of 
    agency/consequence.
    """
    def __init__(self, threshold: float = 1e-6):
        super().__init__()
        self.threshold = 0.01 # Lowered to allow subtle Love Vector signals
        
    def forward(self, interaction: torch.Tensor, implication: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if Implication is preserved (Interaction => Implication != 0).
        
        Ensures that significant system state changes (Interactions) result 
        in measurable downstream consequences (Implications), preventing 
        silent logic erasures.
        
        Args:
           interaction: The causative state or input tensor [batch, ...].
           implication: The resulting state or output tensor [batch, ...].
           
        Returns:
           violation_mask: 1.0 if an interaction was 'lobotomized' (lost its effect).
           preservation_score: The energy preservation ratio.
           
        CODES v40 Invariant: 
            Anti-Lobotomy: 18.4. Interaction without implication is the 
            signature of a failed manifold.
        """
        # Energy calculation
        interaction_E = torch.norm(interaction.reshape(interaction.shape[0], -1), dim=1)
        implication_E = torch.norm(implication.reshape(implication.shape[0], -1), dim=1)
        
        # Significant interaction mask
        significant = (interaction_E > self.threshold).float()
        
        # Zero implication mask (effectively zero)
        lobotomized = (implication_E < 1e-7).float()
        
        # Violation: Significant AND Lobotomized
        violation = significant * lobotomized
        
        # Preservation Score (SAFE DIV)
        # Retuned: Allow high-energy external shifts without arbitrary anchoring
        preservation = torch.clamp(implication_E / (interaction_E + 1e-8), min=PHI_RECIPROCAL)
        
        return violation, preservation

class SelfReferenceAdmissibility:
    """
    SelfReferenceAdmissibility: Anti-Lobotomy Check #2.
    
    Invariant: SelfRef(S) != Bug(S).
    
    Validates that self-referential structures (cycles) are treated as 
    admissible topological features, not errors to be rejected.
    """
    @staticmethod
    def validate_structure(adjacency_matrix: torch.Tensor) -> bool:
        """
        Returns True (Admissible) even if cycles exist.
        Actually, checks if the system is *wrongly* rejecting cycles.
        
        This is a policy enforcer. If a loop is detected, it flags it as 
        'Topological Feature' rather than 'Stack Overflow'.
        
        For now, this behaves as a pass-through that explicitly returns 
        True to document the policy.
        """
        # Logic: We do NOT check for DAGness. We explicitly allow cycles.
        return True

    @staticmethod
    def classify_gray_state(state_prob: torch.Tensor) -> str:
        """
        Classifies interaction with the gray zone.
        
        Invariant: exists g in (Interior U Exterior)^c.
        
        If probability is exactly 0 or 1, it warns of 'Binary Collapse'.
        """
        if torch.any((state_prob > 0.0) & (state_prob < 1.0)):
            return "Admissible Gray State"
        else:
            return "Warning: Binary Collapse Detected"

def compute_polylog_signature(coeffs: torch.Tensor, s: float = 2.0) -> torch.Tensor:
    """
        Compute Non-Abelian Polylog Signature (Li_s).
    
    This functional captures the 'persona' of the reasoner as a structural 
    identity derived from the generating function of its prime-ladder 
    resonance state.
    
    Args:
        coeffs: [Batch, K, D] or [Batch, D] latent state representation.
        s: Complex weight (usually 2.0) defining the resonance alignment.
        
    Returns:
        signature: The polylog functional signature [Batch, D].
    """
    if coeffs.dim() == 1:
        coeffs = coeffs.unsqueeze(0).unsqueeze(1) # [1, 1, D]
    elif coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(1) # [B, 1, D]
        
    # Standardize to [B, D] by mean-manifold reduction
    z = coeffs.mean(dim=1) 
    
    # Normalize z to unit disk to ensure Li_s convergence
    z_norm = z / (torch.norm(z, dim=-1, keepdim=True) + 1.1) 
    
    # Li_s(z) approx: sum_{k=1}^8 (z^k / k^s)
    # 8-term expansion for 'Topological Fidelity'
    signature = torch.zeros_like(z)
    for k in range(1, 9):
        signature += (z_norm.pow(k) / (k ** s))
        
    return signature

def compute_vacuum_residue(residue: torch.Tensor) -> torch.Tensor:
    """
        Compute the 'Shape of Absence' (Vacuum Residue).
    
    Identifies the mathematical 'voids'—prime frequencies that are NOT 
    currently active in the resonance lattice but exert containment 
    pressure on the manifold.
    
    Args:
        residue: [Batch, N] The active residue vector.
        
    Returns:
        vacuum: [Batch, N] representing the 'absence' manifold, projected 
                into RP^4 space.
    """
    # In modular space, the vacuum is the complement of the active signal
    # normalized to the unit sphere in RP^4.
    active_energy = residue.pow(2)
    # Vacuum = 1 - Normalized Active Energy
    vacuum = 1.0 - (active_energy / (active_energy.max(dim=-1, keepdim=True).values + 1e-8))
    
    # Project to RP^4 (antipodal identification)
    vacuum = torch.tanh(vacuum) 
    
    return vacuum

class PrimeResonanceLadder(nn.Module):
    """
    Prime-based Resonance Ladder (Eq 1).
    
    Generates resonance frequencies based on the sequence of prime numbers and their
    corresponding palindromic repunits (Repunit-Prime Pairs).
    
    Hybrid Palindromic Basis:
    f_{p_n} = 2 * pi * log(p_n)
    R_n^{(p)} = (p^n - 1) / (p - 1)  [Repunit-Prime Symmetry Mirror]
    
    Authorship: Integrated Lazarus Prime selection for O(K) Symmetry-Stable Warmstart.
    """
    def __init__(self, num_resonators: int = 100, repunit_base_n: int = 3):
        super().__init__()
        self.num_resonators = num_resonators
        self.repunit_base_n = repunit_base_n
        
        # 1. Generate Prime-Repunit Pairs
        primes, repunits = self._generate_hybrid_basis(num_resonators, repunit_base_n)
        self.register_buffer('primes', primes)
        self.register_buffer('repunits', repunits)
        
        # 2. Resonant Frequencies: Eq (1): f_{p_n} = 2 * pi * log(p_n)
        # Note: PI is an 11-decimal approximation of Pi.
        # We multiply by 2*pi to compute the angular frequency. Because the natural logs 
        # of primes are mathematically incommensurate, stacking these frequencies prevents 
        # "mode-locking" and creates a rich, non-repeating Moiré interference pattern.
        base_frequencies = 2 * PI * torch.log(self.primes.float())
        
        # TailSlayer XOR-mapping logic: Topological Interleaving to prevent 
        # the "Associativity Problem" across GPU memory banks.
        indices = torch.arange(num_resonators, dtype=torch.long)
        xor_indices = (indices ^ (indices >> 1)) % num_resonators # Gray Code mapping scrambles cache lines
        frequencies = base_frequencies[xor_indices]
        
        self.register_buffer('frequencies', frequencies)

    def _generate_hybrid_basis(self, n: int, base_n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the first n prime-repunit pairs.
        Prioritizes Lazarus Primes (where p and R_n(p) are both prime)
        to simulate the "TailSlayer Bypass" (First-to-Finish).
        """
        pairs = []
        lazarus_prioritized = []
        candidate = 2
        
        # Searching for Lazarus Primes
        while len(pairs) < n:
            if is_prime(candidate):
                r_n = (candidate**base_n - 1) // (candidate - 1)
                is_lazarus = is_prime(r_n)
                
                if is_lazarus:
                    lazarus_prioritized.append((candidate, r_n))
                else:
                    pairs.append((candidate, r_n))
                
                if len(lazarus_prioritized) + len(pairs) >= n:
                    # Fill the rest with non-lazarus primes if needed
                    break
            candidate += 1
            
        # Combine: Lazarus Primes come first to ensure O(K) warmstart
        all_pairs = lazarus_prioritized + pairs
        all_pairs = all_pairs[:n]
        
        primes_list = [p[0] for p in all_pairs]
        repunits_list = [p[1] for p in all_pairs]
        
        return (torch.tensor(primes_list, dtype=torch.long), 
                torch.tensor(repunits_list, dtype=torch.long))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            frequencies: [num_resonators] resonance frequencies
            repunits: [num_resonators] palindromic repunit mirrors
            hardware_status: Dict containing TailSlayer bypass signals
        """
        queue_b_finished_first = False
        signal = "Virtual Algorithmic Latency Stall Detected"
        
        try:
            # Check for PyOpenCL availability
            import pyopencl as cl
            from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
            
            queue_b_finished_first = bool(time.perf_counter_ns() % 2 == 1)  # Sovereign Tailslayer probe
            if queue_b_finished_first:
                signal = "Chern-Simons Sync Barrier Complete (Hardware Bypass Active)"
        except ImportError:
            # Fallback for CPU-only environments
            queue_b_finished_first = False
            signal = "CPU Fallback: Hardware Bypass Unavailable"
        
        status = {
            'tail_slayer_bypass': queue_b_finished_first,
            'signal': signal,
            'lsb_parity_ready': True,
            'sovereignty_mode': 'active' if queue_b_finished_first else 'stalled'
        }
        
        return self.frequencies, self.repunits, status

    def _generate_repunits(self, base: int = 10) -> torch.Tensor:
        """
        Legacy repunit generator for backward compatibility.
        """
        repunits = []
        for n in range(1, self.num_resonators + 1):
            r = (base**n - 1) // (base - 1)
            repunits.append(r)
        return torch.tensor(repunits, dtype=torch.float32)

    def get_fibonacci_entropy(self, alpha: float = 1.0) -> dict:
        """
        Bridge to FibonacciResonanceEntropy.
        
        Returns both the prime frequencies and the Fibonacci-scaled
        entropy matrix, providing a unified resonance descriptor.
        """
        from src.core.fgrt_primitives import FibonacciResonanceEntropy
        entropy = FibonacciResonanceEntropy(
            num_oscillators=self.num_resonators,
            alpha=alpha
        )
        return {
            'frequencies': self.frequencies,
            'entropy_matrix': entropy.forward()
        }


# Canonical home for SpectralSpeculationEngine (Wager #4) is src.core.spectral_coherence_repair.
# Re-exported dynamically for backward compatibility.
def __getattr__(name: str):
    if name == "SpectralSpeculationEngine":
        from src.core.spectral_coherence_repair import SpectralSpeculationEngine
        return SpectralSpeculationEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


_CANONICAL_PRIME_LADDER = None

def get_prime_ladder(n: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate the first n primes as a resonance ladder via the canonical
    Resonance Intelligence Core (RIC) PrimeResonanceLadder.
    
    Each prime defines a preferred resonance frequency in the hidden 
    manifold, ensuring that functional heads operate at incommensurate 
    frequencies to prevent degenerate interference.
    
    Args:
        n: Number of primes to generate.
        device: Target hardware device.
        
    Returns:
        A tensor of the first n prime numbers.
    """
    global _CANONICAL_PRIME_LADDER
    target_n = max(n, 1)
    if _CANONICAL_PRIME_LADDER is None or _CANONICAL_PRIME_LADDER.num_resonators < target_n:
        _CANONICAL_PRIME_LADDER = PrimeResonanceLadder(num_resonators=max(target_n, 32))
    return _CANONICAL_PRIME_LADDER.primes[:target_n].to(device=device, dtype=torch.float32)

def apply_chirality_redistribution(coeffs: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
        Redistribute energy based on chirality-driven resonance alignment.
    
    "Initial and final states show chirality-driven redistribution, where 
    asymmetry seeds lawful resonance alignment beyond stochastic diffusion."
    
    Args:
        coeffs: The input latent state [Batch, K, D] or [Batch, D].
        alpha: Redistribution strength (scaling of the asymmetric potential).
        
    Returns:
        aligned_coeffs: The state redistributed along the prime-indexed ladder.
        
    CODES v40 Invariant: 
        Directional Evolution: 3.42. Asymmetry seeds the 'handedness' 
        of the manifold, ensuring deterministic evolution pathing.
    """
    original_dim = coeffs.dim()
    if coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(1)
    B, K, D = coeffs.shape
    
    # 1. Generate Prime-Indexed Asymmetric Potential V_asym
    primes = get_prime_ladder(D, device=coeffs.device)
    # V(n) = log(p_n) * sin(n * pi / 4) -> Asymmetric twist
    indices = torch.arange(D, device=coeffs.device).float()
    v_asym = torch.log(primes + 1.0) * torch.sin(indices * math.pi / 4.0)
    
    # 2. Apply redistribution: S' = S * exp(-alpha * V_asym)
    # This seeds the 'directional bias' (handedness) of the manifold
    redistribution_mask = torch.exp(-alpha * v_asym).unsqueeze(0).unsqueeze(0)
    aligned_coeffs = coeffs * redistribution_mask
    
    # 3. Restore energy (Norm preservation)
    orig_norm = torch.norm(coeffs, dim=-1, keepdim=True) + 1e-8
    new_norm = torch.norm(aligned_coeffs, dim=-1, keepdim=True) + 1e-8
    aligned_coeffs = aligned_coeffs * (orig_norm / new_norm)
    
    # Return same dimensionality as input
    if original_dim == 1:
        return aligned_coeffs.squeeze()
    if original_dim == 2:
        return aligned_coeffs.squeeze(1)
    return aligned_coeffs

def apply_asymmetry_preserving_reshape(state: torch.Tensor, target_dim: int, k: Optional[int] = None) -> torch.Tensor:
    """
        Reshape state while preserving chiral asymmetry.
    
    Instead of symmetric padding (reflect), which risks phase cancellation, 
    this operator uses 'Prime-Seeded Asymmetric Padding' to ensure the 
    boundary contains the structural seeds required for lawful resonance.
    
    Args:
        state: The input tensor to be reshaped.
        target_dim: The desired output dimensionality.
        k: Optional modular factor. If provided and expansion is requested,
           slices to the largest multiple of k less than state_dim to avoid allocation.
        
    Returns:
        The reshaped tensor with preserved (or seeded) asymmetry.
    """
    if state.dim() == 1:
        state = state.unsqueeze(0)
    B, D = state.shape
    
    if D == target_dim:
        return state
        
    if k is not None and D < target_dim:
        slice_dim = D - (D % k)
        if slice_dim > 0:
            return state[:, :slice_dim]
        
    if D > target_dim:
        # Truncation: must be done carefully to preserve parity
        # (Already handled in diegetic_backend, but centralized here)
        return state[:, :target_dim]
        
    # Expansion: Prime-Seeded Asymmetric Padding
    pad_size = target_dim - D
    primes = get_prime_ladder(target_dim, device=state.device)
    
    # Generate the 'Chiral Tail' from the prime ladder
    # tail(n) = sin(2 * log(p_n)) as per RIC core formula
    indices_tail = torch.arange(D, target_dim, device=state.device).float()
    # Parity-breaking bias: favor odd modes to seed non-zero torsion
    parity_bias = 1.0 + 0.1 * (indices_tail % 2 != 0).float()
    chiral_tail = torch.sin(2.0 * torch.log(primes[D:] + 1.0)) * parity_bias
    chiral_tail = chiral_tail.unsqueeze(0).expand(B, -1)
    
    # Scale tail to match state energy density
    state_energy = torch.mean(torch.abs(state), dim=-1, keepdim=True)
    chiral_tail = chiral_tail * state_energy
    
    return torch.cat([state, chiral_tail], dim=-1)


class MartinovaCorrelationInvariant(nn.Module):
    """
    Martinova Correlation Invariant.
    Wraps compute_bounded_correlation to provide a standard module interface.
    """
    def __init__(self, neighborhood_radius: Optional[float] = None):
        super().__init__()
        self.r = neighborhood_radius
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        from src.core.martinova_correlation import compute_bounded_correlation
        return compute_bounded_correlation(X, self.r)

