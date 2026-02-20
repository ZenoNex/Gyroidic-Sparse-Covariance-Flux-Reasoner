"""
Yield Criteria Primitives: Mohr-Coulomb (MC) and Drucker-Prager (DP).

Integrates dual-regime plasticity into the information flow:
1. Mohr-Coulomb: Sharp situational failure planes (brittle/local).
2. Drucker-Prager: Smooth global adaptation envelope (isotropic/global).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MohrCoulombProjection(nn.Module):
    """
    Situational Yield Projection (Sharp/Local).
    
    Ensures that local implications are not smoothed away.
    When pressure hits a 'shear plane', the system ruptures locally.
    """
    def __init__(self, friction_angle: float = 30.0, cohesion: float = 0.5):
        super().__init__()
        self.phi = torch.tensor(friction_angle * (torch.pi / 180.0))
        self.cohesion = cohesion

    def forward(self, pressure: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """
        Projects pressure onto the MC yield surface.
        τ = c + σ tan φ
        """
        # proxy for normal stress σ and shear stress τ
        sigma = pressure.mean(dim=-1, keepdim=True)
        tau = pressure - sigma
        
        strength = self.cohesion + sigma * torch.tan(self.phi.to(pressure.device))
        
        # MC rupture: if tau > strength, we project it sharply
        tau_norm = torch.norm(tau, dim=-1, keepdim=True)
        scale = torch.min(torch.ones_like(tau_norm), strength / (tau_norm + 1e-8))
        
        # In MC, we don't 'smooth' the rupture, we preserve the brittle edge
        yielded_tau = tau * scale
        
        return sigma + yielded_tau

class DruckerPragerProjection(nn.Module):
    """
    Global Adaptation Projection (Smooth/Global).
    
    Provides a convex envelope over incompatible MC rupture sites.
    Allows for navigability without erasing local sharpness.
    """
    def __init__(self, alpha: float = 0.1, k: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.k = k

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        """
        Projects pressure onto the DP yield surface.
        α I1 + sqrt(J2) - k = 0
        """
        # I1: First invariant of stress (sum of diagonal)
        i1 = pressure.sum(dim=-1, keepdim=True)
        
        # J2: Second invariant of deviatoric stress
        mean_p = pressure.mean(dim=-1, keepdim=True)
        s = pressure - mean_p
        j2 = 0.5 * torch.sum(s * s, dim=-1, keepdim=True)
        
        yield_val = self.alpha * i1 + torch.sqrt(j2 + 1e-8)
        
        # DP projection: Smoothly scale back if exceeding k
        scale = torch.clamp(self.k / (yield_val + 1e-8), max=1.0)
        
        return pressure * scale
