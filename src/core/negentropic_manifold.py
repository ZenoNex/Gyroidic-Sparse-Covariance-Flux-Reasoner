"""
Negentropic Trigonometric Manifold (NTM).

Implements the developmental scaling operator that evolves polynomial 
basis configurations using trigonometric oscillators governed by negentropy.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class NTMOperator(nn.Module):
    """
    NTM: Developmental Scaffolding over Asymptotic Time.
    
    Governs the 'Negentropic Trunk' by modulating the frequency and scale 
    of polynomial basis functions based on the manifold's informational density.
    """
    
    def __init__(
        self,
        dim: int = 64,
        degree: int = 4,
        device: str = None
    ):
        super().__init__()
        self.dim = dim
        self.degree = degree
        self.device = device
        
        # 1. Asymptotic Clock τ
        self.register_buffer('tau', torch.tensor(0.0, device=device))
        
        # 2. Basis Warping Frequencies
        # Use a distribution of frequencies for harmonic diversity
        self.register_buffer('frequencies', torch.linspace(0.5, 2.5, degree + 1, device=device))
        
        # 3. Phase offsets φ
        self.register_buffer('phases', torch.randn(degree + 1, device=device))

    def forward(self, negentropy: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Compute the current basis warping vector w(t).
        w_j(τ) = cos(ω_j * τ + φ_j) / (negentropy + 1)
        """
        self.tau += dt
        
        # Negentropy dampens the oscillation amplitude to stabilize the scaffold
        # as the system matures (Saturation).
        damping = 1.0 / (1.0 + negentropy.mean())
        
        # Basis warping factors [degree + 1]
        w = torch.cos(self.frequencies * self.tau + self.phases) * damping
        
        return w

    def get_asymptotic_state(self) -> Dict[str, float]:
        """Returns the current state of the negentropic manifold."""
        # Evolution is a property of the record (tau), not just a value.
        return {
            'asymptotic_time_tau': self.tau.item(),
            'structural_heat': torch.exp(-self.tau * 0.05).item() # Entropy dissipation rate
        }

