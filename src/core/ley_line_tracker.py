"""
Ley Line Tracker: Emergent Resonance Streamlines.

Formalizes preferred-flow vectors as the gradient of resonance potential 
along the non-Euclidean gyroidic manifold.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple


class LeyLineTracker(nn.Module):
    """
    Computes streamlines of the preferred-flow vector l_i = _M V(x_i).
    
    Resonance Potential V(x) combines relational adjacency, love 
    oscillation, and defect amplification.
    """
    
    def __init__(
        self,
        num_samples: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.2,
        device: str = None
    ):
        super().__init__()
        self.num_samples = num_samples
        self.alpha = alpha # Relational adjacency weight
        self.beta = beta   # Love tensor weight
        self.gamma = gamma # Defect amplification weight
        self.device = device
        
        # 1. Resonance Potential Field V [num_samples]
        self.register_buffer('V', torch.zeros(num_samples, device=device))
        
        # 2. Preferred-flow vectors l_i [num_samples, latent_dim]
        # In this implementation, we treat them as discrete flow probabilities 
        # or manifold gradients.
        self.register_buffer('ley_energies', torch.zeros(num_samples, device=device))

    def update_potential(
        self, 
        adjacency: torch.Tensor, 
        love_magnitudes: torch.Tensor, 
        defects: torch.Tensor
    ):
        """
        V(x_i) =   * R_ij * ||_j - _i||^2 +  ||L_i||^2 +  D_i
        
        Args:
            adjacency: Sparse or dense adjacency R_ij
            love_magnitudes: Magnitude of love tensors ||L_i||^2
            defects: Sparse defect signals D_i
        """
        # Adjacency term (Influence flow)
        # Assuming adjacency contains pre-calculated squared differences
        influence = self.alpha * torch.sum(adjacency, dim=1) 
        
        # Love & Defect terms
        v_new = influence + self.beta * love_magnitudes + self.gamma * defects
        self.V.copy_(v_new.detach())
        
        # Accumulate ley energy for streamline tracking
        self.ley_energies.add_(v_new.detach())

    def detect_shear_planes(self, pressure: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Identifies MC failure planes where pressure gradients are non-smooth.
        These become 'corridors of rupture' or preferred ley lines.
        """
        if pressure.shape[0] < 2:
            return torch.zeros_like(pressure)
        grad = torch.gradient(pressure)[0]
        shear_magnitude = torch.norm(grad, dim=-1) if grad.dim() > 1 else torch.abs(grad)
        return (shear_magnitude > threshold).float()

    def get_preferred_flow(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the resonance gradient (Ley Line direction) for given indices.
        
        In a discrete graph, this is approximated by the softmax over 
        neighbor potentials.
        """
        return torch.softmax(self.V[indices] * 5.0, dim=0)

    def get_metrics(self) -> Dict[str, float]:
        return {
            'max_resonance': self.V.max().item(),
            'mean_ley_energy': self.ley_energies.mean().item(),
            'resonance_sparsity': (self.V == 0).float().mean().item()
        }

    def compute_secant_shear_wheel_transversality(
        self,
        states: torch.Tensor,
        pressure: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Computes Secant Plane slopes (Delta P / Delta S) and Mohr-Coulomb Shear Planes,
        resolving zero-chord secant singularities (Delta S -> 0, Delta P -> 0) via
        Carlstrom's Wheel Algebra Hub (0/0 = bot) and L'Hopital boundary scaling
        (`docs/How To Divide By Zero, For Real.txt` & `docs/Recursive Fractals and L'Hôpital.txt`).
        """
        from src.core.invariants import compute_wheel_algebra_division

        shear_mask = self.detect_shear_planes(pressure, threshold=threshold)
        s_flat = states.detach().float()
        if s_flat.dim() == 1:
            s_flat = s_flat.unsqueeze(0)
        p_flat = pressure.detach().float().view(-1)

        if s_flat.shape[0] >= 2 and p_flat.numel() >= 2:
            n_pairs = min(s_flat.shape[0], p_flat.numel())
            ds_chord = torch.norm(s_flat[1:n_pairs] - s_flat[: n_pairs - 1], dim=-1)
            dp_secant = p_flat[1:n_pairs] - p_flat[: n_pairs - 1]
        else:
            ds_chord = torch.norm(s_flat, dim=-1)
            dp_secant = p_flat[: ds_chord.numel()] if p_flat.numel() >= ds_chord.numel() else torch.zeros_like(ds_chord)

        wheel_secant = compute_wheel_algebra_division(dp_secant, ds_chord)
        secant_slopes = wheel_secant["regularized_quotient"]
        transversal_ratio = float((secant_slopes.abs() > 1e-5).float().mean().item()) if secant_slopes.numel() > 0 else 1.0

        return {
            "shear_plane_mask": shear_mask,
            "active_shear_plane_count": int(shear_mask.sum().item()),
            "secant_slope_norm": float(secant_slopes.norm().item()),
            "wheel_hub_0_over_0_count": wheel_secant["hub_bot_count"],
            "wheel_projective_infinity_count": wheel_secant["projective_infinity_count"],
            "secant_shear_transversality_ratio": round(transversal_ratio, 6),
            "wheel_identities_verified": (
                wheel_secant["wheel_identity_x_minus_x_eq_0x2"]
                and wheel_secant["wheel_identity_x_div_x_eq_1_plus_0xx"]
            ),
        }


