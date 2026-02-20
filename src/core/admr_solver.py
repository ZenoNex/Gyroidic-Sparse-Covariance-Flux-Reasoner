"""
Polynomial ADMR: Alternating Direction of Multiplicative Remainders.

Implements the number-theoretic analogue of ADMM using continuous 
polynomial functionals instead of discrete prime moduli.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from .polynomial_coprime import PolynomialCoprimeConfig


class PolynomialADMRSolver(nn.Module):
    """
    Implements the Polynomial ADMR update:
    S^{(n+1)} = Proj_{Poly} [ S^{(n)} * Σ w_ik S_k ]
    
    Uses co-prime polynomial functionals to ensure discrete-like 
    information separation in a differentiable, continuous field.
    """
    
    def __init__(
        self,
        poly_config: PolynomialCoprimeConfig,
        state_dim: int,
        eta_scaffold: float = 0.01,
        device: str = None
    ):
        """
        Args:
            poly_config: Configuration for co-prime polynomial functionals.
            state_dim: Dimension of the state being optimized.
            eta_scaffold: Rate of scaffold adaptation.
            device: Computing device.
        """
        super().__init__()
        self.config = poly_config
        self.state_dim = state_dim
        self.eta_scaffold = eta_scaffold
        self.device = device
        
        # 1. Manifold State (Asymptotic Time)
        self.register_buffer('tau', torch.tensor(0.0, device=device))
        
        # 2. Non-selfadjoint Transition Operators (A_i)
        # We initialize non-selfadjoint matrices for facet-wise dynamics
        num_facet_channels = poly_config.k
        self.A = nn.Parameter(torch.randn(num_facet_channels, state_dim, state_dim, device=device) * 0.01)
        
        # 3. Stochastic Forcing Buffer
        self.register_buffer('eta', torch.randn(state_dim, device=device) * 0.005)

    def update_scaffold(self, negentropy_flux: torch.Tensor, dt: torch.Tensor):
        """
        Update the polynomial coefficients based on negadaptive dynamics.
        """
        self.tau += dt
        # Negentropy modulates the 'breathing' of the polynomial grid
        with torch.no_grad():
            self.config.mutate()

    def forward(
        self, 
        states: torch.Tensor, 
        neighbor_states: torch.Tensor, 
        adjacency_weight: torch.Tensor,
        valence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multiplicative update using relational graph adjacency and polynomial projection.
        
        Args:
            states: [batch, state_dim] S_i
            neighbor_states: [batch, neighbors, state_dim] S_k
            adjacency_weight: [batch, neighbors] R_ik from Relational Graph
            valence: [batch] training hunger / valency drive
        """
        # 1. Weighted sum of neighbors from Relational Graph
        # Σ R_ik S_k
        weighted_neighbors = torch.einsum('bn,bnd->bd', adjacency_weight, neighbor_states)
        
        # 2. Multiplicative interaction: S_i * (Σ S_k + V)
        v_drive = valence.unsqueeze(-1) if valence is not None else 1.0
        interaction = states * (weighted_neighbors + v_drive)
        
        # 3. Polynomial Projection (Functional Co-primality)
        # Instead of 'remainder', we evaluate the interaction through the co-prime basis.
        # This acts as a 'soft modulus' that preserves symbolic structure.
        projected = self.config.evaluate(interaction)
        
        # If k-functionals are used as the 'modulus', we aggregate their response
        # to get the new state. This preserves state_dim while incorporating 
        # the non-linear co-prime filtering.
        if projected.dim() > interaction.dim():
             projected = projected.mean(dim=-1)
             
        # Matching safety
        if projected.shape[-1] != self.state_dim:
            if projected.shape[-1] > self.state_dim:
                projected = projected[..., :self.state_dim]
            else:
                projected = torch.nn.functional.pad(projected, (0, self.state_dim - projected.shape[-1]))
                
        return projected

    def stochastic_differential_step(
        self, 
        states: torch.Tensor, 
        neighbor_states: torch.Tensor, 
        adjacency_weight: torch.Tensor,
        dt: float = 0.1,
        sigma: float = 0.01
    ) -> torch.Tensor:
        """
        Continuous-time Stochastic Differential Update:
        dx(t) = [ Σ A_i x_i(t) - ρ Σ (x - r(x_k)) ] dt + σ dW
        """
        batch_size = states.shape[0]
        
        # 1. Non-selfadjoint Drifts (Σ A_i x_i)
        # We treat the co-prime evaluation as the 'decomposition' into facets
        facets = self.config.evaluate(states) # [batch, state_dim, num_functionals]
        
        # A @ facets: [num_functionals, state_dim, state_dim] @ [batch, state_dim, num_functionals]
        # We sum over facets
        drift = torch.zeros_like(states)
        for i in range(self.config.k):
            # facet_i: [batch, state_dim]
            facet_i = facets[..., i]
            # A_i: [state_dim, state_dim]
            drift += torch.matmul(facet_i, self.A[i])
            
        # 2. Survival Pressure (ADMR Negotiation)
        weighted_neighbors = torch.einsum('bn,bnd->bd', adjacency_weight, neighbor_states)
        # Negotiation term: states - weighted_neighbors
        negotiation = states - weighted_neighbors
        
        # 3. Stochastic Forcing (dW)
        noise = torch.randn_like(states) * sigma * (dt**0.5)
        
        # 4. Update Step (Continuous Approximation)
        # dx = (drift - negotiation) * dt + noise
        dx = (drift - negotiation) * dt + noise
        new_state = states + dx
        
        # 5. Polynomial Projection (Structural Lock)
        # Ensure the new state adheres to the co-prime manifold
        locked_state = self.config.evaluate(new_state)
        if locked_state.dim() > new_state.dim():
            locked_state = locked_state.mean(dim=-1)
            
        return locked_state

    def get_coherence_metrics(self, states: torch.Tensor) -> Dict[str, float]:
        """Measures how well states align with the co-prime polynomial scaffold."""
        # Orthogonality pressure measures functional separation
        pressures = self.config.orthogonality_pressure()
        
        # Scalarize for logging
        local_h = pressures['local_entropy'].mean().item()
        global_h = pressures['global_entropy'].item()
        
        return {
            'polynomial_coherence': 1.0 / (1.0 + global_h),
            'local_functional_entropy': local_h,
            'global_functional_entropy': global_h
        }

