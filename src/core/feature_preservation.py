"""
Feature Preservation via Sparse Polytope Projection.

Implements F^(d)_active: d-th order quantized derivatives along active
polytope facets. This extends the shell-based tensor dynamics
(SparseHigherOrderTensorDynamics) with directional derivative quantization.

The key insight: instead of computing derivatives in the standard basis,
project onto polytope facet normals first, then quantize. This preserves
structural features that live on the polytope boundary while discarding
noise in non-structural directions.

References:
    - ai project report_2-2-2026.txt §"Feature Preservation"
    - VETO_SUBSPACE_ARCHITECTURE.md §5 (BoundaryState)
    - SYSTEM_ARCHITECTURE.md (Matrioshka shell dynamics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FeaturePreservationProjection(nn.Module):
    """
    F^(d)_active = Q_Δ(∂^d x / ∂f_i^d)  for i ∈ active_facets
    
    Computes quantized directional derivatives along active polytope facets.
    
    Pipeline:
        1. Compute facet normal directions from learnable facet embeddings
        2. Project state onto each active facet normal
        3. Compute d-th order finite differences along projected directions
        4. Quantize: round(deriv / Δ) * Δ with context-dependent step sizes
        
    Features on active facets are preserved (high resolution / small Δ).
    Features on inactive facets are coarsened (large Δ) or dropped entirely.
    """
    
    def __init__(
        self, 
        dim: int, 
        num_facets: int = 16, 
        max_order: int = 3,
        delta_min: float = 0.01,
        delta_max: float = 0.5
    ):
        """
        Args:
            dim: State space dimensionality.
            num_facets: Number of polytope facets (learnable normals).
            max_order: Maximum derivative order to compute.
            delta_min: Quantization step for fossilized/trusted facets.
            delta_max: Quantization step for volatile/non-commutative facets.
        """
        super().__init__()
        self.dim = dim
        self.num_facets = num_facets
        self.max_order = max_order
        self.delta_min = delta_min
        self.delta_max = delta_max
        
        # Learnable facet normal directions (on the unit sphere after normalization)
        self.facet_normals = nn.Parameter(
            F.normalize(torch.randn(num_facets, dim), dim=-1)
        )
        
        # Per-facet trust scores: high trust → small delta → preserved features
        self.register_buffer('facet_trust', torch.ones(num_facets) * 0.5)
        
        # Derivative kernels per order (finite difference stencils)
        # Order 1: [-1, 1],  Order 2: [1, -2, 1],  Order 3: [-1, 3, -3, 1]
        self._stencils = {
            1: torch.tensor([-1.0, 1.0]),
            2: torch.tensor([1.0, -2.0, 1.0]),
            3: torch.tensor([-1.0, 3.0, -3.0, 1.0])
        }
    
    def _get_delta(self, facet_idx: int) -> float:
        """
        Context-dependent quantization step size.
        
        Δ_i = Δ_min + (Δ_max - Δ_min) * (1 - trust_i)
        
        High trust → small Δ → fine resolution (features preserved)
        Low trust → large Δ → coarse resolution (noise suppressed)
        """
        trust = self.facet_trust[facet_idx].item()
        return self.delta_min + (self.delta_max - self.delta_min) * (1.0 - trust)
    
    def project_onto_facet(
        self, 
        x: torch.Tensor, 
        facet_idx: int
    ) -> torch.Tensor:
        """
        Project state onto a facet normal direction.
        
        Args:
            x: [batch, dim] state vectors
            facet_idx: Index of the facet normal
            
        Returns:
            projection: [batch, 1] scalar projection onto facet normal
        """
        normal = F.normalize(self.facet_normals[facet_idx], dim=0)
        return torch.matmul(x, normal).unsqueeze(-1)
    
    def compute_directional_derivative(
        self, 
        x: torch.Tensor, 
        facet_idx: int,
        order: int = 1,
        h: float = 0.01
    ) -> torch.Tensor:
        """
        Compute d-th order directional derivative along facet normal.
        
        Uses central finite differences:
            ∂^d f / ∂n^d ≈ Σ_k stencil[k] * f(x + k*h*n) / h^d
        
        Args:
            x: [batch, dim] state vectors
            facet_idx: Facet index for direction
            order: Derivative order (1, 2, or 3)
            h: Finite difference step size
            
        Returns:
            derivative: [batch, 1] directional derivative values
        """
        normal = F.normalize(self.facet_normals[facet_idx], dim=0)
        stencil = self._stencils[min(order, 3)].to(x.device)
        
        # Evaluate at stencil points
        values = []
        for k in range(len(stencil)):
            offset = (k - len(stencil) // 2) * h
            x_shifted = x + offset * normal.unsqueeze(0)
            proj = torch.matmul(x_shifted, normal)
            values.append(proj)
        
        values = torch.stack(values, dim=-1)  # [batch, stencil_len]
        
        # Apply stencil and normalize by h^order
        derivative = torch.sum(values * stencil.unsqueeze(0), dim=-1) / (h ** order)
        
        return derivative.unsqueeze(-1)
    
    def quantize(self, value: torch.Tensor, delta: float) -> torch.Tensor:
        """
        Context-aware quantization: Q_Δ(v) = round(v / Δ) * Δ
        """
        return torch.round(value / delta) * delta
    
    def forward(
        self, 
        x: torch.Tensor, 
        active_facets: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute preserved features along active facets.
        
        Args:
            x: [batch, dim] state vectors
            active_facets: Indices of active facets. If None, auto-detect
                          top-k facets by projection magnitude.
                          
        Returns:
            Dict with:
                'preserved_features': Dict[int, Dict[int, Tensor]]
                    {facet_idx: {order: quantized_derivative}}
                'active_facets': List[int] facet indices used
                'total_preserved_energy': scalar (sum of |F^(d)|)
                'compression_ratio': fraction of facets active
        """
        batch_size = x.shape[0]
        
        # Auto-detect active facets if not specified
        if active_facets is None:
            # Project onto all facets, keep top 25%
            normals = F.normalize(self.facet_normals, dim=-1)
            projections = torch.matmul(x, normals.T)  # [batch, num_facets]
            proj_magnitudes = projections.abs().mean(dim=0)  # [num_facets]
            k = max(1, self.num_facets // 4)
            _, top_indices = torch.topk(proj_magnitudes, k)
            active_facets = top_indices.tolist()
        
        preserved = {}
        total_energy = torch.tensor(0.0, device=x.device)
        
        for facet_idx in active_facets:
            delta = self._get_delta(facet_idx)
            facet_features = {}
            
            for order in range(1, self.max_order + 1):
                # Compute directional derivative
                deriv = self.compute_directional_derivative(
                    x, facet_idx, order=order
                )
                
                # Quantize with context-dependent step
                q_deriv = self.quantize(deriv, delta)
                
                facet_features[order] = q_deriv
                total_energy = total_energy + q_deriv.abs().sum()
            
            preserved[facet_idx] = facet_features
        
        return {
            'preserved_features': preserved,
            'active_facets': active_facets,
            'total_preserved_energy': total_energy,
            'compression_ratio': len(active_facets) / self.num_facets
        }
    
    def update_trust(
        self, 
        facet_idx: int, 
        increment: float = 0.01,
        decay: float = 0.001
    ):
        """
        Update facet trust based on stability feedback.
        
        Stable facets gain trust (→ finer quantization, better preservation).
        All facets decay slowly (→ trust must be continuously earned).
        """
        with torch.no_grad():
            # Decay all
            self.facet_trust *= (1.0 - decay)
            # Boost specified facet
            self.facet_trust[facet_idx] = torch.clamp(
                self.facet_trust[facet_idx] + increment, 0.0, 1.0
            )
    
    def compute_savings(self, active_facets: List[int]) -> Dict[str, float]:
        """Estimate computational savings from sparse facet selection."""
        total_ops = self.num_facets * self.max_order
        active_ops = len(active_facets) * self.max_order
        return {
            'sparsity_ratio': 1.0 - (len(active_facets) / self.num_facets),
            'theoretical_speedup': total_ops / max(active_ops, 1),
            'active_facets': len(active_facets),
            'total_facets': self.num_facets
        }
