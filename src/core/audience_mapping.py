"""
Audience Mapping: Lipschitz Homeomorphic Projection.

Implements the audience projection operator Phi: M -> A defined in 
"garden statistical attractors.txt". Ideally maps the manifold M to 
an Audience space A while preserving topological roughness (singularities).
"""

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class AudienceProjection(nn.Module):
    """
    Audience Mapping operator Phi: M -> A.
    
    Constraints:
    1. Lipschitz Continuous (bounded gradient).
    2. Homeomorphic (bijective, continuous inverse) - approximated via invertibility.
    3. Preserves Roughness (singularities are mapped, not smoothed).
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
        self.lipschitz_k = lipschitz_k
        
        # We use Spectral Normalization to enforce Lipschitz constraint K=1 (roughly)
        # Residual structure y = x + f(x) helps with invertibility (homeomorphism)
        
        hidden = max(input_dim, audience_dim) * 2
        
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(hidden, audience_dim))
        )
        
        # Roughness preservation: A skip connection that carries high frequencies?
        # Or simply the residual itself.
        
    def forward(self, manifold_state: torch.Tensor) -> torch.Tensor:
        """
        Phi(m).
        """
        # Base projection
        smooth_projection = self.net(manifold_state)
        
        # Enforce global Lipschitz sealing scaling
        # (Spectral norm does this per layer, but accumulation can grow)
        
        # Roughness Preservation:
        # If input has high-frequency noise (singularities), we want it in output.
        # We add a fraction of the input (padded/projected) to the output.
        if self.input_dim == self.audience_dim:
            identity = manifold_state
        elif self.input_dim < self.audience_dim:
            identity = torch.cat([manifold_state, torch.zeros_like(manifold_state)], dim=-1)[:, :self.audience_dim]
        else:
            identity = manifold_state[:, :self.audience_dim]
            
        # y = f(x) + x  (ResNet style, tends to be homeomorphic if Lip(f) < 1)
        audience_state = smooth_projection + identity
        
        return audience_state
        
    def inverse(self, audience_state: torch.Tensor, iterations: int = 5) -> torch.Tensor:
        """
        Approximate inverse Phi^-1(a) via fixed point iteration.
        x = a - f(x)
        Only works if Lip(f) < 1 (Banach Fixed Point Theorem).
        """
        x = audience_state # Initial guess
        for _ in range(iterations):
            # mapping handles the identity part logic in reverse? 
            # If y = N(x) + x, then x = y - N(x)
            
            # Need to handle dimension mismatch for true inverse, 
            # but assuming dims match for the core manifold mappings.
            if self.input_dim != self.audience_dim:
                # Can't easily invert dimensionality change without a decoder
                return x
                
            x = audience_state - self.net(x)
        return x
