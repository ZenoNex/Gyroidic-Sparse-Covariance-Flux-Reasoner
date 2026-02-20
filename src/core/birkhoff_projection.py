"""
Stochastic Matrix Manifold (formerly Birkhoff Projection).

Role: Ensures that the DyadicTransferMap remains a valid stochastic matrix
(probabilistic flow constraints).

Uses the Sinkhorn-Knopp algorithm to project any transfer matrix T onto
the Birkhoff polytope (Doubly Stochastic) or Stochastic polytope.

Author: William Matthew Bryant
Created: January 2026
Refactored: January 2026 (Anti-Lobotomy)
"""

import torch
import torch.nn as nn
from typing import Optional


class ObscuredBirkhoffManifold(nn.Module):
    """
    Obscured Birkhoff Polytope B_N^o.
    
    Ensures that T_{ij} satisfies conservation of probability with 
    partial visibility (obstruction):
        sum_j T_{ij} = 1 - delta_o
        sum_i T_{ij} = 1 - delta_o
        
    delta_o evolves via genome g (Obsc(g)).
    """
    
    def __init__(
        self, 
        max_iterations: int = 50,
        epsilon: float = 1e-4,
        temperature: float = 1.0,
        delta_o: float = 0.0  # Initial obstruction level (0 = full visibility)
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.register_buffer('delta_o', torch.tensor(delta_o))
    
    def evolve_obstruction(self, genome: torch.Tensor, decay: float = 0.99):
        """
        Evolve obstruction level delta_o based on genome g.
        delta_o = Obsc(g)
        """
        # Simple mapping: normalized genome energy -> obstruction
        target_obsc = torch.sigmoid(torch.mean(genome)) * 0.5 # Max 0.5 obstruction
        self.delta_o = decay * self.delta_o + (1 - decay) * target_obsc
        
    def project(self, T: torch.Tensor) -> torch.Tensor:
        """
        Project matrix T to be doubly stochastic (Sinkhorn) with obstruction.
        Target sum = 1.0 - delta_o.
        """
        target_sum = 1.0 - self.delta_o
        
        # Ensure positive
        T_soft = torch.exp(T / self.temperature)
        
        # Sinkhorn Iterations
        for _ in range(self.max_iterations):
            # Row Norm
            row_sums = T_soft.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            T_soft = T_soft / row_sums * target_sum
            
            # Col Norm (for Doubly Stochastic)
            col_sums = T_soft.sum(dim=-2, keepdim=True).clamp(min=1e-8)
            T_soft = T_soft / col_sums * target_sum
            
        return T_soft
    
    def forward(self, T: torch.Tensor) -> torch.Tensor:
        return self.project(T)

BirkhoffProjection = ObscuredBirkhoffManifold

def sinkhorn_knopp(
    T: torch.Tensor, 
    max_iterations: int = 50, 
    temperature: float = 1.0,
    delta_o: float = 0.0
) -> torch.Tensor:
    """Functional wrapper for Sinkhorn-Knopp projection."""
    manifold = ObscuredBirkhoffManifold(
        max_iterations=max_iterations, 
        temperature=temperature,
        delta_o=delta_o
    )
    return manifold.project(T)

def project_to_birkhoff(T: torch.Tensor, max_iterations: int = 50) -> torch.Tensor:
    """Alternative name for sinkhorn_knopp."""
    return sinkhorn_knopp(T, max_iterations=max_iterations)
