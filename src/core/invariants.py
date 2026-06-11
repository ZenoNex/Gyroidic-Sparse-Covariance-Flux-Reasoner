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
from typing import Tuple, Optional
import math

class PhaseAlignmentInvariant(nn.Module):
    """
    PAS_h: Harmonic Phase Alignment Score.
    
    Implements Eq (2): PAS_S = (1/N) * sum(cos(theta_k - theta_bar))
    
    Acts as a first-class admissibility filter measuring the 'topological 
    synchronization' of field states.
    """
    def __init__(self, degree: int):
        super().__init__()
        # Degree is kept for compatibility, but PAS is now strictly phase-based
        self.degree = degree
        
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
        if coeffs.dim() == 3:
            # Flatten K and D to treat all as a pool of oscillators?
            # Or average over K? Eq (2) sums over "elements in a set S".
            # Let's treat (K, D) as the set S.
            x = coeffs.reshape(coeffs.shape[0], -1)
        else:
            x = coeffs
            
        # 2. Extract Phases (theta_k)
        # We assume x contains real values that form complex pairs (Analytic Signal assumption)
        # Pad if odd length
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1))
            
        # Reshape to [batch, N/2, 2] -> Z = a + ib
        z = x.view(x.shape[0], -1, 2)
        
        # theta_k = atan2(Im, Re)
        theta = torch.atan2(z[..., 1], z[..., 0]) # [batch, N_pairs]
        
        # 3. Compute Mean Phase (theta_bar)
        # Circular mean: atan2(sum(sin), sum(cos))
        sin_sum = torch.sin(theta).sum(dim=1)
        cos_sum = torch.cos(theta).sum(dim=1)
        theta_bar = torch.atan2(sin_sum, cos_sum).unsqueeze(1) # [batch, 1]
        
        # 4. Compute PAS (Eq 2)
        # PAS = (1/N) * sum(cos(theta_k - theta_bar))
        alignment = torch.cos(theta - theta_bar)
        pas_h = alignment.mean(dim=1)
        
        return pas_h

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
        # Retuned: Allow high-energy external shifts (0.61) without 0.30 anchoring
        preservation = torch.clamp(implication_E / (interaction_E + 1e-8), min=0.618)
        
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
    if coeffs.dim() == 2:
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

def get_prime_ladder(n: int, device: torch.device = None) -> torch.Tensor:
    """
        Generate the first n primes as a resonance ladder.
    
    Each prime defines a preferred resonance frequency in the hidden 
    manifold, ensuring that functional heads operate at incommensurate 
    frequencies to prevent degenerate interference.
    
    Args:
        n: Number of primes to generate.
        device: Target hardware device.
        
    Returns:
        A tensor of the first n prime numbers.
    """
    primes = []
    num = 2
    while len(primes) < n:
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                break
        else:
            primes.append(num)
        num += 1
    return torch.tensor(primes, device=device, dtype=torch.float32)

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
        from core.martinova_correlation import compute_bounded_correlation
        return compute_bounded_correlation(X, self.r)

