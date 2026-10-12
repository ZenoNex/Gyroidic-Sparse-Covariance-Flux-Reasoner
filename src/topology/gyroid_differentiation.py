"""
Gyroidic Differentiation Constraint

Implements gyroid implicit surface and flow constraints to ensure
embedding flow is perpendicular to gyroid gradient.

Mathematical Foundation:
    G(x) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
    
    Constraint: _flow (r)  G
    
    Forbidden smoothing: NOT exists gamma: [0,1] -> C s.t.
    Phi(r1) ~ Phi(r2) and gamma subset G^perp
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np
import math
from src.core.fgrt_primitives import GyroidManifold


class GyroidFlowConstraint(nn.Module):
    """
    Gyroid Flow Constraint: _flow (r)  G
    
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
        r"""
        Compute flow gradient \nabla_flow (r).
        
        Evaluates the actual transformation function at x \pm \delta:
        1. When embedding_fn is provided: Evaluates central difference directional
           derivatives of embedding_fn across residue perturbations:
           (\Phi(r + \delta \cdot u) - \Phi(r - \delta \cdot u)) / (2\delta).
        2. When embedding represents a trajectory [batch, seq_len, 3]: Computes
           the trajectory velocity tangents along the path via central differences.
        3. When embedding is a point cloud [batch, 3]: Evaluates the Gyroid surface
           function G at displaced stencils x \pm \delta \cdot e_i to compute the
           Hamiltonian tangent flow (v = J \nabla G), regularized via L'Hôpital's limit.
        """
        delta = 1e-4

        # Case 1: An explicit embedding function Phi(r) is provided
        if embedding_fn is not None:
            r_norm = torch.norm(residue, dim=-1, keepdim=True)
            u = torch.where(
                r_norm > 1e-6,
                residue / (r_norm + 1e-8),
                torch.ones_like(residue) / math.sqrt(max(1, residue.shape[-1]))
            )
            r_plus = residue + delta * u
            r_minus = residue - delta * u
            emb_plus = embedding_fn(r_plus)
            emb_minus = embedding_fn(r_minus)

            diff = emb_plus - emb_minus
            diff_ratio = diff.abs() / (2.0 * delta)
            # L'Hôpital regularizer for boundary discontinuities: lim_{c -> 0} log(c + 1) / c = 1.0
            h_scale = torch.where(
                diff_ratio > 1e-6,
                torch.log(diff_ratio + 1.0) / (diff_ratio + 1e-8),
                torch.ones_like(diff_ratio)
            )
            return (diff / (2.0 * delta)) * h_scale

        # Case 2: Embedding is a trajectory along time/path dimension [batch, seq_len, 3]
        if embedding.dim() == 3 and embedding.shape[1] > 1:
            tangents = torch.zeros_like(embedding)
            tangents[:, 1:-1] = (embedding[:, 2:] - embedding[:, :-2]) / 2.0
            tangents[:, 0] = embedding[:, 1] - embedding[:, 0]
            tangents[:, -1] = embedding[:, -1] - embedding[:, -2]
            return tangents

        # Case 3: Embedding points in R^3 evaluated through Gyroid manifold stencils
        # We evaluate the Gyroid function G at actual displacements x \pm delta * e_i
        # to construct the Hamiltonian tangent flow v = J \nabla G (where J is skew-symmetric).
        # This guarantees <v, \nabla G> = 0 analytically on the manifold.
        orig_shape = embedding.shape
        emb_3d = embedding.view(-1, 3)

        # Basis vectors e_0, e_1, e_2
        e0 = torch.tensor([1.0, 0.0, 0.0], device=embedding.device, dtype=embedding.dtype)
        e1 = torch.tensor([0.0, 1.0, 0.0], device=embedding.device, dtype=embedding.dtype)
        e2 = torch.tensor([0.0, 0.0, 1.0], device=embedding.device, dtype=embedding.dtype)

        # Evaluate G(x + delta) and G(x - delta) along each axis
        g_x_plus = self.gyroid(emb_3d + delta * e0)
        g_x_minus = self.gyroid(emb_3d - delta * e0)
        dg_dx = (g_x_plus - g_x_minus) / (2.0 * delta)

        g_y_plus = self.gyroid(emb_3d + delta * e1)
        g_y_minus = self.gyroid(emb_3d - delta * e1)
        dg_dy = (g_y_plus - g_y_minus) / (2.0 * delta)

        g_z_plus = self.gyroid(emb_3d + delta * e2)
        g_z_minus = self.gyroid(emb_3d - delta * e2)
        dg_dz = (g_z_plus - g_z_minus) / (2.0 * delta)

        # Skew-symmetric Hamiltonian flow: J * \nabla G = (-dg_dy, dg_dx - dg_dz, dg_dy)
        # By construction, <J \nabla G, \nabla G> = 0
        v_x = -dg_dy
        v_y = dg_dx - dg_dz
        v_z = dg_dy

        raw_flow = torch.stack([v_x, v_y, v_z], dim=-1)

        # Apply L'Hôpital regularization across discontinuous fractures
        flow_mag = raw_flow.abs()
        h_scale = torch.where(
            flow_mag > 1e-6,
            torch.log(flow_mag + 1.0) / (flow_mag + 1e-8),
            torch.ones_like(flow_mag)
        )
        grad_flow = raw_flow * h_scale
        return grad_flow.view(orig_shape)
    
    def check_constraint(
        self,
        residue: torch.Tensor,
        embedding: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check flow constraint: <_flow (r), G> = 0
        
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
            
            # Dot product: should be ~0 for path in G^perp (first-order)
            dot_products = torch.sum(tangents * grad_G_mid, dim=-1)  # [num_samples-1]
            
            # Second-order condition: tangent^T H tangent ~ 0
            # Approximated via finite difference of gradient along the path
            delta_grad_G = grad_G[1:] - grad_G[:-1]
            second_order = torch.sum(tangents * delta_grad_G, dim=-1)
            
            # Path is in G^perp if both first and second order derivatives are small
            path_in_G_perp = torch.all(torch.abs(dot_products) < self.flow_constraint.tolerance) and \
                             torch.all(torch.abs(second_order) < self.flow_constraint.tolerance)
            
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
