"""
Gyroidic Differentiation Constraint

Implements gyroid implicit surface and flow constraints to ensure
embedding flow is perpendicular to gyroid gradient.

Mathematical Foundation:
    G(x) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
    
    Constraint: ∇_flow Φ(r) ⟂ ∇G
    
    Forbidden smoothing: NOT exists gamma: [0,1] -> C s.t.
    Phi(r1) ~ Phi(r2) and gamma subset G^perp
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np
from src.core.fgrt_primitives import GyroidManifold


class GyroidFlowConstraint(nn.Module):
    """
    Gyroid Flow Constraint: ∇_flow Φ(r) ⟂ ∇G
    
    Ensures embedding flow is perpendicular to gyroid gradient.
    """
    
    def __init__(self, tolerance: float = 1e-4):
        """
        Args:
            tolerance: Tolerance for perpendicularity check
        """
        super().__init__()
        self.gyroid = GyroidManifold() # Use consolidated primitive
        self.tolerance = tolerance
    
    def compute_flow_gradient(
        self,
        residue: torch.Tensor,
        embedding: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Compute flow gradient ∇_flow Φ(r).
        
        Args:
            residue: [batch, ...] residue tensor
            embedding: [batch, dim] or [batch, 3] embedding
            embedding_fn: Optional function to compute embedding from residue
            
        Returns:
            grad_flow: [batch, dim] or [batch, 3] flow gradient
        """
        if embedding_fn is not None:
            # Compute embedding from residue
            embedding = embedding_fn(residue)
        
        # Flow gradient: gradient of embedding with respect to constraint flow
        # Simplified: use finite differences or autograd
        if embedding.requires_grad:
            # Use autograd
            grad_flow = torch.autograd.grad(
                embedding.sum(), embedding, create_graph=True, retain_graph=True
            )[0]
        else:
            # Use finite differences (approximate)
            delta = 1e-5
            embedding_perturbed = embedding + delta
            grad_flow = (embedding_perturbed - embedding) / delta
        
        return grad_flow
    
    def check_constraint(
        self,
        residue: torch.Tensor,
        embedding: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check flow constraint: <∇_flow Φ(r), ∇G> = 0
        
        Args:
            residue: [batch, ...] residue tensor
            embedding: [batch, dim] or [batch, 3] embedding (must be 3D for gyroid)
            embedding_fn: Optional embedding function
            
        Returns:
            dot_product: [batch] dot product (should be ~0)
            is_satisfied: [batch] boolean tensor
        """
        # Ensure embedding is 3D for gyroid
        if embedding.shape[-1] != 3:
            # Project to 3D or pad
            if embedding.shape[-1] < 3:
                # Pad with zeros
                padding = torch.zeros(
                    *embedding.shape[:-1], 3 - embedding.shape[-1],
                    device=embedding.device, dtype=embedding.dtype
                )
                embedding_3d = torch.cat([embedding, padding], dim=-1)
            else:
                # Truncate to first 3 dimensions
                embedding_3d = embedding[..., :3]
        else:
            embedding_3d = embedding
        
        # Compute flow gradient
        grad_flow = self.compute_flow_gradient(residue, embedding_3d, embedding_fn)
        
        # Compute gyroid gradient
        grad_G = self.gyroid.gradient(embedding_3d)
        
        # Dot product: <grad_flow, grad_G>
        if grad_flow.dim() == 2 and grad_G.dim() == 2:
            # [batch, 3] and [batch, 3]
            dot_product = torch.sum(grad_flow * grad_G, dim=-1)  # [batch]
        elif grad_flow.dim() == 3 and grad_G.dim() == 3:
            # [batch, seq_len, 3] and [batch, seq_len, 3]
            dot_product = torch.sum(grad_flow * grad_G, dim=-1)  # [batch, seq_len]
            dot_product = dot_product.mean(dim=-1)  # Average over sequence
        else:
            # Allow broadcasting or reshaping if needed, but for now raise error if strict mismatch
            # Attempting simple broadcast if one is [1, 3]
            dot_product = torch.sum(grad_flow * grad_G, dim=-1)
        
        # Check if constraint is satisfied (perpendicular)
        is_satisfied = torch.abs(dot_product) < self.tolerance
        
        return dot_product, is_satisfied
    
    def forward(
        self,
        residue: torch.Tensor,
        embedding: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: check flow constraint.
        
        Returns:
            Dictionary with:
            - 'dot_product': [batch] dot product
            - 'is_satisfied': [batch] boolean
            - 'grad_flow': [batch, 3] flow gradient
            - 'grad_gyroid': [batch, 3] gyroid gradient
        """
        # Ensure 3D embedding
        if embedding.shape[-1] != 3:
            if embedding.shape[-1] < 3:
                padding = torch.zeros(
                    *embedding.shape[:-1], 3 - embedding.shape[-1],
                    device=embedding.device, dtype=embedding.dtype
                )
                embedding_3d = torch.cat([embedding, padding], dim=-1)
            else:
                embedding_3d = embedding[..., :3]
        else:
            embedding_3d = embedding
        
        grad_flow = self.compute_flow_gradient(residue, embedding_3d, embedding_fn)
        grad_G = self.gyroid.gradient(embedding_3d)
        
        dot_product, is_satisfied = self.check_constraint(residue, embedding_3d, embedding_fn)
        
        return {
            'dot_product': dot_product,
            'is_satisfied': is_satisfied,
            'grad_flow': grad_flow,
            'grad_gyroid': grad_G
        }


class ForbiddenSmoothingChecker(nn.Module):
    """
    Checks forbidden smoothing condition.
    
    Forbidden: NOT exists gamma: [0,1] -> C s.t.
    Phi(r1) ~ Phi(r2) and gamma subset G^perp
    """
    
    def __init__(self, num_path_samples: int = 20, distance_tolerance: float = 0.1):
        """
        Args:
            num_path_samples: Number of samples along path for checking
            distance_tolerance: Tolerance for "similar" embeddings
        """
        super().__init__()
        self.num_path_samples = num_path_samples
        self.distance_tolerance = distance_tolerance
        self.gyroid = GyroidManifold()
        self.flow_constraint = GyroidFlowConstraint()
    
    def check_forbidden_smoothing(
        self,
        residue_1: torch.Tensor,
        residue_2: torch.Tensor,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Check if smooth path exists in gyroid-orthogonal space.
        
        Args:
            residue_1: [batch, ...] first residue
            residue_2: [batch, ...] second residue
            embedding_1: [batch, dim] first embedding
            embedding_2: [batch, dim] second embedding
            embedding_fn: Optional embedding function
            
        Returns:
            is_forbidden: [batch] boolean (True = forbidden, path exists)
            path_exists: Whether a smooth path was found
        """
        # Check if embeddings are similar
        if embedding_1.shape != embedding_2.shape:
            return torch.zeros(embedding_1.shape[0], dtype=torch.bool, device=embedding_1.device), False
        
        distance = torch.norm(embedding_1 - embedding_2, dim=-1)
        is_similar = distance < self.distance_tolerance
        
        if not is_similar.any():
            # Not similar, no path needed
            return torch.zeros(embedding_1.shape[0], dtype=torch.bool, device=embedding_1.device), False
        
        # Check if smooth path exists in G^perp
        # Sample points along linear interpolation
        batch_size = embedding_1.shape[0]
        is_forbidden = torch.zeros(batch_size, dtype=torch.bool, device=embedding_1.device)
        
        for b in range(batch_size):
            if not is_similar[b]:
                continue
            
            # Linear interpolation path
            t_values = torch.linspace(0, 1, self.num_path_samples, device=embedding_1.device)
            path_points = embedding_1[b:b+1] * (1 - t_values.unsqueeze(-1)) + \
                         embedding_2[b:b+1] * t_values.unsqueeze(-1)
            
            # Check if all points on path satisfy flow constraint (in G^perp)
            # Ensure 3D for gyroid
            if path_points.shape[-1] != 3:
                if path_points.shape[-1] < 3:
                    padding = torch.zeros(
                        path_points.shape[0], 3 - path_points.shape[-1],
                        device=path_points.device, dtype=path_points.dtype
                    )
                    path_points_3d = torch.cat([path_points, padding], dim=-1)
                else:
                    path_points_3d = path_points[..., :3]
            else:
                path_points_3d = path_points
            
            # Check gyroid gradient at each point
            grad_G = self.gyroid.gradient(path_points_3d)  # [num_samples, 3]
            
            # Check if path tangent is perpendicular to grad_G
            # Path tangent: difference between consecutive points
            tangents = path_points_3d[1:] - path_points_3d[:-1]  # [num_samples-1, 3]
            grad_G_mid = (grad_G[1:] + grad_G[:-1]) / 2.0  # [num_samples-1, 3]
            
            # Dot product: should be ~0 for path in G^perp
            dot_products = torch.sum(tangents * grad_G_mid, dim=-1)  # [num_samples-1]
            
            # Path is in G^perp if all dot products are small
            path_in_G_perp = torch.all(torch.abs(dot_products) < self.flow_constraint.tolerance)
            
            if path_in_G_perp:
                is_forbidden[b] = True
        
        return is_forbidden, is_forbidden.any().item()
    
    def forward(
        self,
        residue_1: torch.Tensor,
        residue_2: torch.Tensor,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: check forbidden smoothing.
        
        Returns:
            Dictionary with:
            - 'is_forbidden': [batch] boolean
            - 'path_exists': boolean
        """
        is_forbidden, path_exists = self.check_forbidden_smoothing(
            residue_1, residue_2, embedding_1, embedding_2, embedding_fn
        )
        
        return {
            'is_forbidden': is_forbidden,
            'path_exists': path_exists
        }
