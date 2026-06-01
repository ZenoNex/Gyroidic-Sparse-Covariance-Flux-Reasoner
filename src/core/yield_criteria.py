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
         = c +  tan 
        """
        # proxy for normal stress  and shear stress 
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
         I1 + sqrt(J2) - k = 0
        """
        # I1: First invariant of stress (sum of diagonal)
        i1 = pressure.sum(dim=-1, keepdim=True)
        
        # J2: Second invariant of deviatoric stress
        mean_p = pressure.mean(dim=-1, keepdim=True)
        s = pressure - mean_p
        j2 = 0.5 * torch.sum(s * s, dim=-1, keepdim=True)
        
        # Drucker-Prager yield criterion with improved stability
        # Increased k = 1.0 (default was 0.5) to prevent manifold flattening
        yield_val = self.alpha * i1 + torch.sqrt(j2 + 1e-8)
        
        # DP projection: Smoothly scale back if exceeding k
        # Stabilization: Ensure k is at least 0.1 to avoid total collapse
        effective_k = max(self.k, 0.1)
        scale = torch.clamp(torch.tensor(effective_k, device=pressure.device) / (yield_val + 1e-8), max=1.0)
        
        return pressure * scale


class BouligandMohrCoulombProjection(MohrCoulombProjection):
    """
    Nonsmooth Bouligand Tangent Cone Projection for Mohr-Coulomb yield surfaces.
    
    Inherits from MohrCoulombProjection to maintain full backward compatibility.
    Provides direction projection onto the contingent (tangent) cone of the yield surface
    to allow plastic flow along the shear planes without structural rupture.
    """
    def project_direction(self, pressure: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Projects an update direction vector onto the Bouligand tangent cone of the MC surface
        at the current pressure state.
        
        Args:
            pressure: [batch, dim] current pressure tensor state
            direction: [batch, dim] proposed update direction tensor
        """
        device = pressure.device
        sigma = pressure.mean(dim=-1, keepdim=True)
        tau = pressure - sigma
        tau_norm = torch.norm(tau, dim=-1, keepdim=True)
        
        strength = self.cohesion + sigma * torch.tan(self.phi.to(device))
        yield_val = tau_norm - strength
        
        # Determine if state is on the yield boundary (epsilon guard matching precision)
        eps = torch.finfo(pressure.dtype).eps * 1000.0  # safe scaling
        on_boundary = yield_val.abs() < max(eps, 1e-4)
        
        if not on_boundary.any():
            return direction
            
        # Outward normal vector components
        # normal direction in normal stress space is -tan(phi), in shear space is tau / ||tau||
        normal_sigma = -torch.tan(self.phi.to(device))
        normal_tau = tau / (tau_norm + 1e-8)
        
        # Calculate normal component of direction
        dir_sigma = direction.mean(dim=-1, keepdim=True)
        dir_tau = direction - dir_sigma
        
        inner_product = dir_sigma * normal_sigma + torch.sum(dir_tau * normal_tau, dim=-1, keepdim=True)
        
        # Direction points outward if inner product is positive and state is on the boundary
        is_outward = (inner_product > 0) & on_boundary
        
        projected_dir = direction.clone()
        if is_outward.any():
            # Project onto the tangent cone: v_proj = v - <v, n> * n / ||n||^2
            n_norm_sq = normal_sigma**2 + torch.sum(normal_tau**2, dim=-1, keepdim=True)
            scale = inner_product / (n_norm_sq + 1e-8)
            
            # Reconstruct normal vector direction
            n_vector = normal_sigma + normal_tau
            correction = scale * n_vector
            
            # Print console reporting for debugging and transparency
            print(f"[BOULIGAND_MC] Outward update detected on yield surface. Projecting direction onto contingent cone (norm correction: {torch.norm(correction).item():.4f})", flush=True)
            projected_dir = torch.where(is_outward, direction - correction, direction)
            
        return projected_dir

