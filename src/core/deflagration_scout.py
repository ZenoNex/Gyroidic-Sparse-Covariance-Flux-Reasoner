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
        dim: int = 137,
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
        ΔD_i = Σ (R_ij - R_hat_ij)
        
        Identifies deviations from the expected resonance pattern.
        """
        # Difference between expected (mean/predicted) and actual flux
        diff = torch.abs(actual_flux - predicted_flux)
        
        # Defect signal: Anomaly amplification
        # Rare anomalies (low predicted flux, high difference) are rewarded
        defects = diff * self.amplification
        
        self.defect_count.add_(defects.mean().detach())
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
        Calculates Symbolic Transversality using Non-Commutativity Curvature (kappa).
        To measure path dependence, we project states into rank-1 operators
        and measure their commutator: [A, B] = A@B - B@A.
        """
        # Ensure 2D [1, dim]
        arc = archetype_state.view(1, -1)
        voy = voynich_state.view(1, -1)
        
        # Rank-1 Projectors
        OpA = arc.T @ arc
        OpB = voy.T @ voy
        
        # Calculate Curvature
        metrics = self.curvature_engine.compute_curvature(OpA, OpB, update_ema=True)
        return metrics

