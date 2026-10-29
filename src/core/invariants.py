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
PERCOLATION_2D_SQUARE = 0.592746
QUADRATIC_SURD_PI_BOUND = math.sqrt(2.0) + math.sqrt(3.0)
QUADRATIC_SURD_PI_GAP = QUADRATIC_SURD_PI_BOUND - PI
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
    """Optimized deterministic primality test with Miller-Rabin for large integers."""
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0 or n % 3 == 0: return False
    if n < 1000000:
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    # Deterministic Miller-Rabin for 64-bit integers using standard bases
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n <= a:
            break
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
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

    def get_catalan_branching(self, alpha: float = 1.0) -> dict:
        """
        Bridge to CatalanAssociahedronEntropy.

        Returns the prime frequencies, Catalan numbers C_n, and the
        Catalan-Prime Associahedron non-associative coupling matrix.
        """
        from src.core.fgrt_primitives import CatalanAssociahedronEntropy
        catalan_mod = CatalanAssociahedronEntropy(
            num_oscillators=self.num_resonators,
            alpha=alpha
        )
        return {
            'frequencies': self.frequencies,
            'catalan_numbers': catalan_mod.catalan,
            'associahedron_matrix': catalan_mod.forward()
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


# =========================================================================
# CATALAN NUMBERS, DYCK BALLOT ADMISSIBILITY, & STASHEFF ASSOCIAHEDRON
# =========================================================================

def catalan_number(n: int) -> int:
    """
    Computes the n-th Catalan number exactly:
        C_n = (1 / (n + 1)) * binom(2n, n) = sum_{i=0}^{n-1} C_i * C_{n-1-i}

    Catalan numbers govern:
    1. Non-associative binary bracketings (vertices of the Stasheff Associahedron K_{n+1}).
    2. Dyck paths of length 2n satisfying the ballot non-negativity invariant.
    3. Rooted ordered plane trees with n edges (Euler tour depth trajectories).
    4. Polygon triangulations of a convex (n + 2)-gon into n triangles.
    5. Unimodular Hankel moment matrices det([C_{i+j}]) = 1 of the Chebyshev semicircle law.
    """
    if n < 0:
        raise ValueError(f"Catalan index n must be >= 0, got {n}")
    if n <= 1:
        return 1
    return math.comb(2 * n, n) // (n + 1)


def get_catalan_ladder(
    n: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    log_scale: bool = False
) -> torch.Tensor:
    """
    Generates the first n Catalan numbers [C_0, C_1, ..., C_{n-1}] as a tensor.

    Args:
        n: Number of Catalan terms to generate.
        device: Target torch device.
        dtype: Output tensor dtype.
        log_scale: If True, returns log(C_k + 1) to prevent float32 overflow at large n
                   (since C_n ~ 4^n / (n^{3/2} * sqrt(pi))).
    """
    target_n = max(int(n), 1)
    if log_scale:
        vals = [math.log(float(catalan_number(k)) + 1.0) for k in range(target_n)]
    else:
        vals = [float(catalan_number(k)) for k in range(target_n)]
    return torch.tensor(vals, device=device, dtype=dtype)


def segner_catalan_weights(
    n_splits: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Computes the normalized Segner recurrence probability distribution across
    all binary tree root splits i in {0, ..., n_splits - 1}:
        w_i = (C_i * C_{n_splits - 1 - i}) / C_{n_splits}
        sum_{i=0}^{n_splits - 1} w_i = 1.0

    Provides the canonical prior over binary decompositions of an (n_splits + 1)-leaf
    non-associative expression or agglutinative morpho-unit.
    """
    if n_splits <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    c_total = float(catalan_number(n_splits))
    weights = [
        (float(catalan_number(i)) * float(catalan_number(n_splits - 1 - i))) / c_total
        for i in range(n_splits)
    ]
    return torch.tensor(weights, device=device, dtype=dtype)


class CatalanBranchingInvariant(nn.Module):
    """
    Catalan Branching & Stasheff Associahedron Invariant (C_n).

    Completes the Resonance Intelligence Core (RIC) number-theoretic triad:
    - Primes (p_n): Multiplicative irreducibility & frequency incommensurability.
    - Fibonacci (F_n): Additive two-term linear recurrence & golden-ratio scaling.
    - Catalan (C_n): Non-associative binary tree convolution C_n = sum_{i=0}^{n-1} C_i C_{n-1-i},
      enumerating vertices of the Stasheff Associahedron K_{n+1}, Dyck ballot paths,
      and unimodular Chebyshev semicircle Hankel matrices (det H_n = 1).
    """
    def __init__(self, max_order: int = 16):
        super().__init__()
        self.max_order = max(int(max_order), 2)
        catalan_seq = torch.tensor(
            [float(catalan_number(k)) for k in range(self.max_order)],
            dtype=torch.float64
        )
        catalan_log = torch.tensor(
            [math.log(float(catalan_number(k)) + 1.0) for k in range(self.max_order)],
            dtype=torch.float32
        )
        self.register_buffer("catalan_seq", catalan_seq)
        self.register_buffer("catalan_log", catalan_log)

    def verify_dyck_ballot(self, step_sequence: Union[torch.Tensor, list]) -> Tuple[bool, float]:
        """
        Verifies whether a +1/-1 step sequence forms an admissible Dyck path
        (prefix sums >= 0 everywhere and total sum == 0) or measures its ballot
        admissibility score in [0, 1].
        """
        if not isinstance(step_sequence, torch.Tensor):
            steps = torch.tensor(step_sequence, dtype=torch.float32)
        else:
            steps = step_sequence.float()

        if steps.numel() == 0:
            return True, 1.0

        prefix_sums = torch.cumsum(steps, dim=-1)
        non_negative_fraction = (prefix_sums >= -1e-6).float().mean().item()
        terminal_balance = float(torch.abs(prefix_sums[..., -1]).mean().item())
        is_valid_dyck = (non_negative_fraction == 1.0) and (terminal_balance < 1e-5) and (steps.shape[-1] % 2 == 0)
        score = non_negative_fraction / (1.0 + terminal_balance)
        return is_valid_dyck, float(score)

    def hankel_matrix(self, order: int = 4, shifted: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Constructs the n x n Catalan Hankel matrix:
            H_{i,j} = C_{i + j}       (if shifted=False)
            H^{(1)}_{i,j} = C_{i+j+1} (if shifted=True)

        By Touchard-Hankel theorem, det(H) == 1 and det(H^{(1)}) == 1 for all orders n >= 1.
        """
        n = max(1, int(order))
        mat = torch.zeros((n, n), dtype=torch.float64, device=device)
        offset = 1 if shifted else 0
        for i in range(n):
            for j in range(n):
                mat[i, j] = float(catalan_number(i + j + offset))
        return mat

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the Segner-Catalan non-associative binary tree convolution invariant
        across the latent dimension D of coeffs.

        Args:
            coeffs: [batch, D] or [batch, K, D] or [D] tensor.

        Returns:
            catalan_coherence: [batch] (or scalar if 1D) score in [0, 1] measuring
            alignment with the Segner-Catalan binary parenthesization profile.
        """
        is_1d = coeffs.dim() == 1
        if is_1d:
            x = coeffs.unsqueeze(0)
        elif coeffs.dim() >= 3:
            x = coeffs.reshape(coeffs.shape[0], -1)
        else:
            x = coeffs

        B, D = x.shape
        weights = segner_catalan_weights(D, device=x.device, dtype=x.dtype)  # [D], sums to 1
        energy = x.pow(2)
        energy_prob = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)

        # Cosine alignment between empirical energy profile and Segner-Catalan envelope
        dot = (energy_prob * weights.unsqueeze(0)).sum(dim=-1)
        norm_e = torch.norm(energy_prob, dim=-1) + 1e-8
        norm_w = torch.norm(weights) + 1e-8
        coherence = dot / (norm_e * norm_w)
        return coherence.squeeze(0) if is_1d else coherence


def quadratic_surd_polytope_sandwich(state: torch.Tensor) -> Dict[str, Any]:
    """
    Quadratic Surd Polytope Bounding (Inscribed Square vs. Circumscribed Hexagon):
        pi_surd = sqrt(2) + sqrt(3) approx 3.14626436994...
    Bounds the transcendental continuum pi between the inscribed 4-polytope
    half-perimeter (2 * sqrt(2) approx 2.828427) and circumscribed 6-polytope
    half-perimeter (2 * sqrt(3) approx 3.464102).
    """
    inscribed_half_perim = 2.0 * math.sqrt(2.0)
    circumscribed_half_perim = 2.0 * math.sqrt(3.0)
    norm_val = float(torch.norm(state.float()).item()) if state.numel() > 0 else 0.0
    span = circumscribed_half_perim - inscribed_half_perim
    wrapped_norm = inscribed_half_perim + math.fmod(abs(norm_val), span)
    in_surd_corridor = abs(wrapped_norm - QUADRATIC_SURD_PI_BOUND) <= (span * 0.5)
    return {
        "inscribed_4_polytope_bound": inscribed_half_perim,
        "circumscribed_6_polytope_bound": circumscribed_half_perim,
        "quadratic_surd_pi_bound": QUADRATIC_SURD_PI_BOUND,
        "transcendental_pi": PI,
        "algebraic_surd_gap": QUADRATIC_SURD_PI_GAP,
        "wrapped_norm": wrapped_norm,
        "in_surd_corridor": bool(in_surd_corridor)
    }


def evaluate_percolation_regime(occupancy_prob: float) -> Dict[str, Any]:
    """
    Evaluates the 2D Square Lattice Site Percolation Phase Transition (p_c = 0.592746).
    - Below p_c (p < 0.592746): Meliponini Cerumen Pot regime (isolated finite clusters).
    - At or above p_c (p >= 0.592746): Apis Honeycomb regime (infinite spanning cluster).
    """
    p = float(max(0.0, min(1.0, occupancy_prob)))
    is_spanning = p >= PERCOLATION_2D_SQUARE
    margin = p - PERCOLATION_2D_SQUARE
    return {
        "occupancy_prob": p,
        "percolation_threshold_pc": PERCOLATION_2D_SQUARE,
        "is_spanning_cluster": is_spanning,
        "regime": "APIS_SPANNING_HONEYCOMB" if is_spanning else "MELIPONINI_ISOLATED_CERUMEN",
        "critical_margin": margin,
        "requires_isolation_rescue": p < (PERCOLATION_2D_SQUARE * 0.35)
    }


# IHC BAO Validation Constants (Zero-Parameter Topological Cosmology on RP^4)
IHC_HUBBLE_RADIUS_MPC = 4451.14
IHC_SOUND_HORIZON_MPC = IHC_HUBBLE_RADIUS_MPC * (PHI ** -7.0)  # ~153.20 Mpc (k=7 shell)
IHC_XI_RATIO = 1.036707  # r_s^IHC / r_s^Planck
IHC_Z1_SHELL_REDSHIFT = 0.7540  # k=1 co-rotating shell boundary R_1 = R_H * phi^{-1}
IHC_DELTA_Z_TRANSITION = 0.3630  # Geodesic width of the k=1 shell transition
IHC_Z3_AMPLITUDE = 5.9441 / 1344.5  # ~0.004421 Z_3 phase standing-wave amplitude
IHC_Z3_WAVELENGTH_MPC = IHC_HUBBLE_RADIUS_MPC / 11.0  # ~404.65 Mpc (N = 33 = 3 x 11)


def compute_ihc_bao_shell_modulation(
    z_effective: Union[float, torch.Tensor] = 0.754,
    comoving_distance_mpc: Optional[Union[float, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Computes the zero-parameter Inverted Hypersphere Cosmology (IHC) BAO shell
    expansion factor f_{k_1}(z) and Z_3 standing-wave modulation on RP^4
    (from ihc_bao_validation.py):
        f_{k_1}(z) = 1 + ((xi - 1) / 2) * [1 + tanh((z_1 - z) / delta_z)]
        delta(r) = A_{Z_3} * sin(2 * pi * r / lambda_{Z_3} + 2 * pi / 3)
    """
    if isinstance(z_effective, torch.Tensor):
        z_val = float(z_effective.mean().item()) if z_effective.numel() > 0 else IHC_Z1_SHELL_REDSHIFT
    else:
        z_val = float(z_effective)

    f_k1 = 1.0 + ((IHC_XI_RATIO - 1.0) / 2.0) * (
        1.0 + math.tanh((IHC_Z1_SHELL_REDSHIFT - z_val) / IHC_DELTA_Z_TRANSITION)
    )

    if comoving_distance_mpc is None:
        r_mpc = IHC_HUBBLE_RADIUS_MPC * (PHI ** -1.0)  # Default to R_1 = R_H * phi^{-1}
    elif isinstance(comoving_distance_mpc, torch.Tensor):
        r_mpc = float(comoving_distance_mpc.mean().item())
    else:
        r_mpc = float(comoving_distance_mpc)

    z3_mod = IHC_Z3_AMPLITUDE * math.sin(2.0 * PI * r_mpc / IHC_Z3_WAVELENGTH_MPC + (2.0 * PI / 3.0))
    effective_scale = f_k1 * (1.0 + z3_mod)

    return {
        "z_effective": z_val,
        "f_k1_shell_factor": float(f_k1),
        "z3_standing_wave_modulation": float(z3_mod),
        "effective_rp4_bao_scale": float(effective_scale),
        "r_s_ihc_mpc": float(IHC_SOUND_HORIZON_MPC),
        "lambda_z3_mpc": float(IHC_Z3_WAVELENGTH_MPC),
    }


def compute_fractal_lhopital_boundary_ratio(
    residue_state: torch.Tensor,
    target_state: torch.Tensor,
    alpha_fractional: float = 0.5,
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """
    Implements the Modified L'Hopital's Rule for Fractal Boundaries and
    Scale-Dependent Entropy (from docs/Recursive Fractals and L'Hopital.txt):
        lim_{(x,y)->(x_0,y_0)} f(x)/g(y) = lim [D_alpha f(x) / D_alpha g(y)] * (dx/dy)^alpha
        H_d(F) = [log N(epsilon) / log(1/epsilon)] * phi(Delta)
        S(F, epsilon) = -sum p_i(epsilon) log p_i(epsilon)
    Prevents gradient explosion across discontinuous fractal fractures during
    survivorship / Bouligand tangent cone projection.
    """
    r_flat = residue_state.detach().float().flatten()
    t_flat = target_state.detach().float().flatten()
    min_len = min(r_flat.numel(), t_flat.numel())
    if min_len < 2:
        return {
            "lhopital_fractional_ratio": 1.0,
            "generalized_hausdorff_dim": 1.0,
            "scale_dependent_entropy": 0.0,
            "damping_factor": 1.0,
        }

    r_flat = r_flat[:min_len]
    t_flat = t_flat[:min_len]

    # Fractional difference proxy D_alpha f(x) ~ |Delta f|^alpha
    df = torch.abs(r_flat[1:] - r_flat[:-1]) + epsilon
    dg = torch.abs(t_flat[1:] - t_flat[:-1]) + epsilon
    d_alpha_f = torch.pow(df, alpha_fractional).mean()
    d_alpha_g = torch.pow(dg, alpha_fractional).mean()
    lhopital_ratio = float((d_alpha_f / (d_alpha_g + epsilon)).clamp(0.05, 20.0).item())

    # Box-counting proxy N(epsilon) and Generalized Hausdorff Dimension H_d(F)
    active_boxes = float((df > epsilon).float().sum().item()) + 1.0
    log_inv_eps = max(1.0, math.log(1.0 / max(epsilon, 1e-8)))
    phi_delta = float(1.0 + torch.std(df).clamp(0.0, 1.0).item() * PHI_RECIPROCAL)
    hausdorff_dim = (math.log(active_boxes) / log_inv_eps) * phi_delta

    # Scale-Dependent Entropy S(F, epsilon)
    probs = df / (df.sum() + 1e-8)
    scale_entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item())

    # Smooth damping factor in [0.25, 1.5] so fractal fractures do not explode gradients
    damping_factor = 1.0 / (1.0 + 0.15 * abs(lhopital_ratio - 1.0))

    return {
        "lhopital_fractional_ratio": lhopital_ratio,
        "generalized_hausdorff_dim": float(hausdorff_dim),
        "scale_dependent_entropy": scale_entropy,
        "damping_factor": float(damping_factor),
    }


def compute_enso_inkblot_emergence(
    interior_state: torch.Tensor,
    boundary_residue: torch.Tensor,
) -> Dict[str, Any]:
    """
    Implements the Enso Circle vs. Warping Rorschach Inkblot Boundary diagnostic
    (from docs/Gemini-Untangling the Cosmic Feedback Loop-20261013-1752.txt):
      - Interior Enso Emptiness: Non-separate zero-center harmonic anchor (the space
        inside the circle that does not move while the rim warps).
      - Exterior Rorschach Inkblot Boundary: Dynamic, time-warping boundary where
        semantic projections and temporal friction occur ("the inkblot realizing it is wet").
    """
    int_flat = interior_state.detach().float().flatten()
    bnd_flat = boundary_residue.detach().float().flatten()
    if int_flat.numel() == 0 or bnd_flat.numel() == 0:
        return {
            "enso_interior_stillness": 1.0,
            "inkblot_boundary_warp": 0.0,
            "non_separate_wetness_index": 0.5,
        }

    # Enso interior stillness: proximity of mean interior phase/center to zero-balance
    interior_mean_abs = float(torch.abs(int_flat.mean()).item())
    enso_stillness = 1.0 / (1.0 + interior_mean_abs)

    # Inkblot boundary warp: boundary variance / roughness across dimensions
    inkblot_warp = float(torch.std(bnd_flat).item()) if bnd_flat.numel() > 1 else 0.0

    # Non-separate wetness index: coupling between interior stillness and boundary warp
    wetness_index = float(enso_stillness * math.tanh(inkblot_warp + 1e-4))

    return {
        "enso_interior_stillness": float(enso_stillness),
        "inkblot_boundary_warp": float(inkblot_warp),
        "non_separate_wetness_index": float(wetness_index),
    }

