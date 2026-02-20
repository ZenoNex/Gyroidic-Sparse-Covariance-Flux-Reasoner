"""
Symplectic Gluing Diffeomorphism.

Handles the transition between the orientable Gyroid manifold and the 
non-orientable Klein-bottle throat.
"""

import torch
import torch.nn as nn
from src.core.fgrt_primitives import GyroidManifold

class GluingOperator(nn.Module):
    """
    Operator for 'Symplectic Gluing' across the boundary manifold M.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gyroid = GyroidManifold()
        # Rotation matrix for spatial manifold reversal
        self.reversal_matrix = nn.Parameter(torch.eye(dim))
        # Initialize as a reflection/reversal
        with torch.no_grad():
            self.reversal_matrix[0, 0] = -1.0 

    def chern_simons_constraint(self, connection: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Chern-Simons gasket penalty for the connection A.
        S_CS = int tr(A ^ dA + 2/3 A ^ A ^ A)
        """
        # Simplified symbolic constraint
        # tr(A * rot(A)) or similar proxy
        # connection: (Batch, Dim, Dim)
        rot_a = torch.rot90(connection, k=1, dims=(-2, -1))
        cs_term = torch.sum(connection * rot_a)
        return cs_term

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Passes the state through the gluing map.
        state: (Batch, Dim)
        """
        # 1. Check Gyroid alignment
        g_violation = self.gyroid(state[..., :3])
        
        # 2. Apply Spacial Manifold Reversal
        # This doubles the representation by flipping orientation
        reversed_state = torch.matmul(state, self.reversal_matrix)
        
        # 3. Symplectic Gluing: Blend original and reversed state based on proximity to boundary
        # boundary is defined where g_violation is low (on the surface)
        weight = torch.exp(-torch.abs(g_violation)).unsqueeze(-1)
        
        glued_state = (1 - weight) * state + weight * reversed_state
        return glued_state
