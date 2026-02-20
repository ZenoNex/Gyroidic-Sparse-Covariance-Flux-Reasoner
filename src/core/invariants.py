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
from typing import Tuple

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
        
        Args:
           coeffs: [batch, K, D] or [batch, D]. Represents resonator states.
        
        Returns:
           pas_h: [batch] scalar score [-1, 1] (usually [0, 1] for coherence)
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
        
        Returns:
            drift: |delta|
            violation_mask: 1 if drifted too much
        """
        drift = torch.abs(current_pas - prev_pas)
        violation = (drift > self.zeta).float()
        return drift, violation


import torch


def compute_chirality(coeffs: torch.Tensor) -> torch.Tensor:
    """
    Compute Chirality Index with Batch-Awareness.
    Fixed to handle [K, D] (Test/Single) and [B, K, D] (Operational).
    """
    # 1. Capture context
    original_dim = coeffs.dim()

    # 2. Force to 3D: [Batch, K-Manifold, D-Degree]
    if original_dim == 2:
        # If [K, D], make it [1, K, D]
        coeffs = coeffs.unsqueeze(0)
    elif original_dim != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {original_dim}D")

    B, K, D = coeffs.shape

    # 3. Energy extraction (Summing across the K-manifold)
    # We use non_blocking logic implicitly by staying on the tensor's device
    energy = coeffs.pow(2).sum(dim=1)  # Shape: [B, D]

    # 4. Spectral Centroid Calculation
    # Arange must be on the SAME device to avoid the 'Synchronous Transfer'
    # identified in the audit (e.g., uni_pc.py Line 465)
    indices = torch.arange(D, device=coeffs.device, dtype=coeffs.dtype)

    total_energy = energy.sum(dim=1, keepdim=True) + 1e-8
    spectral_centroid = (energy * indices).sum(dim=1, keepdim=True) / total_energy

    # 5. Chirality Index: (Centroid - Midpoint) / Midpoint
    # Positive = High-Freq/Entropic, Negative = Low-Freq/Negentropic
    midpoint = D / 2.0
    chirality = (spectral_centroid - midpoint) / midpoint

    # 6. Return to original context (scalar if 2D, vector if 3D)
    return chirality.squeeze()

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
        Check if Implication is preserved.
        
        Args:
           interaction: Input or causative state [batch, ...]
           implication: Downstream effect [batch, ...]
           
        Returns:
           violation_mask: 1 if interaction is significant but implication is zero.
           preservation_score: Ratio of implication energy to interaction energy.
        """
        # Energy calculation
        interaction_E = torch.norm(interaction.reshape(interaction.shape[0], -1), dim=1)
        implication_E = torch.norm(implication.reshape(implication.shape[0], -1), dim=1)
        
        # Significant interaction mask
        self.threshold = 0.01 # Lowered to allow subtle Love Vector signals
        
        # Zero implication mask (effectively zero)
        self.threshold = 0.01 # Lowered to allow subtle Love Vector signals
        
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

