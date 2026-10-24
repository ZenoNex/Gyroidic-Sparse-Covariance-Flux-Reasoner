"""
Deflagration Scout: Omipedial Interstitiality.

Implements "defect scouting" to amplify sparse anomalies and 
enable "jumps" across manifold holes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from src.core.noncommutativity_curvature import NonCommutativityCurvature


class OmipedialDeflagrator(nn.Module):
    """
    Omipedial Flash and Defect Scout.
    
    Prevents "Goo" (phase flattening) by rewarding rare anomalies 
    and enabling signal propagation across topological gaps.
    """
    
    def __init__(
        self,
        dim: int = 96,
        threshold_jump: float = 0.8,
        amplification: float = 2.0,
        device: str = None
    ):
        super().__init__()
        self.dim = dim
        self.threshold_jump = threshold_jump
        self.amplification = amplification
        self.device = device
        
        self.curvature_engine = NonCommutativityCurvature(dim=dim)
        
        # 1. Defect density tracker
        self.register_buffer('defect_count', torch.tensor(0.0, device=device))

    def scout_defects(self, predicted_flux: torch.Tensor, actual_flux: torch.Tensor) -> torch.Tensor:
        """
        D_i =  (R_ij - R_hat_ij)
        
        Identifies deviations from the expected resonance pattern.
        """
        # Difference between expected (mean/predicted) and actual flux
        diff = torch.abs(actual_flux - predicted_flux)
        
        # Defect signal: Anomaly amplification
        # Rare anomalies (low predicted flux, high difference) are rewarded
        defects = diff * self.amplification
        
        self.defect_count.add_(defects.mean().detach())
        
        # Highlight points with correlation -1.0 as structural gaps/topological holes
        from src.core.martinova_correlation import compute_bounded_correlation
        corr = compute_bounded_correlation(actual_flux)
        self.structural_gaps = (corr <= -0.8).float()
        
        return defects

    def omipedial_jump(self, ley_potential: torch.Tensor) -> torch.Tensor:
        """
        Enables "jumps" across holes where resonance potential is high 
        but adjacency is sparse.
        """
        # Threshold-based trigger for topological shortcuts
        jumps = (ley_potential > self.threshold_jump).float()
        return jumps

    def get_metrics(self) -> Dict[str, float]:
        return {
            'defect_density': self.defect_count.item(),
            'jump_readiness': (self.defect_count > 10.0).float().item()
        }

    def compute_transversality(self, archetype_state: torch.Tensor, voynich_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculates Symbolic Transversality using Non-Commutativity Curvature (kappa)
        and Folium of Descartes implicit tangency (reciprocal exchange slope -1.0).
        
        To measure path dependence, we project states into rank-1 operators
        and measure their commutator: [A, B] = A@B - B@A.
        Additionally, the interaction is evaluated against the Folium of Descartes
        tangent line (slope -1.0 at (3/2, 3/2)), indicating perfect reciprocal exchange
        without topological rupture.
        """
        # Ensure 2D [1, dim]
        arc = archetype_state.view(1, -1)
        voy = voynich_state.view(1, -1)
        
        # Rank-1 Projectors
        OpA = arc.T @ arc
        OpB = voy.T @ voy
        
        # Calculate Curvature
        metrics = self.curvature_engine.compute_curvature(OpA, OpB, update_ema=True)
        
        # Folium of Descartes Transversal Evaluation
        # Take scalar projections or first coordinates as (x, y)
        x = arc[0, 0] if arc.numel() > 0 else torch.tensor(1.5, device=arc.device)
        y = voy[0, 0] if voy.numel() > 0 else torch.tensor(1.5, device=voy.device)
        
        num = y - x**2
        denom = y**2 - x
        safe_denom = torch.where(denom.abs() < 1e-5, torch.sign(denom + 1e-6) * 1e-5, denom)
        slope = num / safe_denom
        
        dist_smooch = torch.sqrt((x - 1.5)**2 + (y - 1.5)**2)
        slope_error = (slope - (-1.0)).abs()
        reciprocal_score = 1.0 / (1.0 + slope_error)
        is_smooch = (dist_smooch < 0.3) & (slope_error < 0.3)
        
        metrics['folium_slope'] = slope
        metrics['reciprocal_exchange_score'] = reciprocal_score
        metrics['is_folium_smooch'] = bool(is_smooch.item())
        
        # Expose combined state for VoynichExemptionToken issuance
        # Normalize/pad to at least 3 dims if needed or 2 dims
        min_dim = min(arc.shape[-1], voy.shape[-1])
        combined_state = torch.stack([x, y], dim=-1).unsqueeze(0)
        metrics['state'] = combined_state
        
        return metrics


