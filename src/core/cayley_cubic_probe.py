import torch
import torch.nn as nn
from typing import Optional, Tuple
from .birkhoff_projection import sinkhorn_knopp

class CayleyCubicProbe(nn.Module):
    """
    Cayley-Birkhoff Hybridization Probe.
    
    This replaces/hybridizes the basic BirkhoffPolytope.
    System 2 usually operates on the Birkhoff faces (smooth manifold).
    However, when 'Cycle Debt' (or Mischief/Contorsion) is high, it forces
    the state toward one of the four A_1 singularities of the Cayley surface.

    The Cayley Constraint V(C): x^2 + y^2 + z^2 - xyz = 4
    The Sovereign Loci / Parabolic Singularities: (+-2, +-2, +-2)
    """

    def __init__(self, high_mischief_threshold: float = 0.5):
        super().__init__()
        self.high_mischief_threshold = high_mischief_threshold
        # Four A_1 singularities on the Cayley cubic
        self.register_buffer('sovereign_loci', torch.tensor([
            [2.0, 2.0, 2.0],
            [2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, -2.0, 2.0]
        ], dtype=torch.float32))

    def _apply_cayley_constraint(self, T: torch.Tensor) -> torch.Tensor:
        """
        Projects points towards the closest Sovereign Locus on the Cayley Cubic surface.
        Because these are discrete isolated points, we map the local coordinate state
        towards the nearest parabolic singularity.
        """
        original_shape = T.shape
        # Flatten to (-1, 3) for distance calculation if possible, or just treat as embeddings
        # We will pad or pool to size 3 for the projection.
        # This is simplified: in practice the state must match the physical representation.
        if T.shape[-1] >= 3:
            coords = T[..., :3]
            # Find closest locus
            # coords: [..., 3], sovereign_loci: [4, 3]
            dist = torch.cdist(coords.view(-1, 3), self.sovereign_loci) # [..., 4]
            closest_idx = torch.argmin(dist, dim=-1) # [...]
            closest_loci = self.sovereign_loci[closest_idx].view(coords.shape) # [..., 3]
            
            # Snap the first 3 dimensions to the singularity, keeping the rest intact 
            # (as the "windings" or non-diagonalizable components)
            T_projected = T.clone()
            T_projected[..., :3] = closest_loci
            return T_projected
        return T

    def forward(self, T: torch.Tensor, cycle_debt: float = 0.0) -> torch.Tensor:
        """
        Hybrid projection: 
        If cycle_debt < threshold, enforce Birkhoff (Sinkhorn local numerical stability).
        If cycle_debt >= threshold, enforce Cayley A_1 singularity (Topological repair / Neglecton anchor).
        """
        if cycle_debt >= self.high_mischief_threshold:
            # High Mischief: Force Phase Transition to Sovereign Loci
            return self._apply_cayley_constraint(T)
        else:
            # Low Mischief: Proceed with Doubly Stochastic matrix smoothing
            return sinkhorn_knopp(T)

