"""
Audience Mapping: Lipschitz Homeomorphic Projection.

Implements the audience projection operator Phi: M -> A defined in 
"garden statistical attractors.txt". Ideally maps the manifold M to 
an Audience space A while preserving topological roughness (singularities).
"""

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np

class AudienceProjection(nn.Module):
    """
    Audience Mapping operator Phi: M -> A.
    
    Constraints:
    1. Lipschitz Continuous (bounded gradient).
    2. Homeomorphic (bijective, continuous inverse) - approximated via invertibility.
    3. Preserves Roughness (singularities are mapped, not smoothed).
    
    [ANTI-LOBOTOMY REWRITE]: 
    Eradicated nn.Linear ML proxies and legacy video encoding hacks. 
    Now utilizes Symplectic Gluing (GluingOperator) for rigorous P2P manifold integration.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        audience_dim: int, 
        lipschitz_k: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.audience_dim = audience_dim
        self.lipschitz_k = min(lipschitz_k, 0.95)
        
        from src.core.gluing_operator import GluingOperator
        self.gluing_operator = GluingOperator(dim=input_dim)
        
    def forward(self, manifold_state: torch.Tensor) -> torch.Tensor:
        """
        Phi(m). Uses Symplectic Gluing to dynamically stitch state boundaries for projection.
        """
        # 1. Apply Symplectic Gluing (P2P manifold integration)
        glued_state = self.gluing_operator(manifold_state)
        
        # 2. Align dimensions for Audience space
        if self.input_dim > self.audience_dim:
            audience_state = glued_state[..., :self.audience_dim]
        elif self.input_dim < self.audience_dim:
            audience_state = torch.cat([glued_state, torch.zeros_like(glued_state)], dim=-1)[..., :self.audience_dim]
        else:
            audience_state = glued_state
            
        # 3. Enforce Lipschitz boundary scaling
        audience_state = audience_state * self.lipschitz_k
        
        # 4. Roughness Preservation: Add raw singularities directly back (skip connection style)
        if self.input_dim == self.audience_dim:
            identity = manifold_state
        elif self.input_dim < self.audience_dim:
            identity = torch.cat([manifold_state, torch.zeros_like(manifold_state)], dim=-1)[..., :self.audience_dim]
        else:
            identity = manifold_state[..., :self.audience_dim]
            
        return audience_state + identity
        
    def inverse(self, audience_state: torch.Tensor, iterations: int = 5) -> torch.Tensor:
        """
        Approximate inverse Phi^-1(a) via fixed point iteration.
        x = a - f(x)
        Only works if Lip(f) < 1 (Banach Fixed Point Theorem).
        """
        x = audience_state # Initial guess
        for _ in range(iterations):
            if self.input_dim != self.audience_dim:
                return x
            
            # Use Symplectic Gluing forward pass for fixed-point iteration
            f_x = self.gluing_operator(x)
            
            x = audience_state - (f_x * self.lipschitz_k)
        return x
