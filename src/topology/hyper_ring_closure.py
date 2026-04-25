"""
Hyper-Ring Closure Condition: Topological Closure Check

Implements the hyper-ring operator H(r) = ∮_C ∇_top Φ(r) and checks
closure conditions for soliton stability.

Mathematical Foundation:
    H(r) = ∮_C ∇_top Φ(r)
    
    Closure iff:
    - H(r) in Z_1(C) (closed)
    - [H(r)] != 0 in H_1(C) (non-trivial)
    
    Interpretation:
    - trivial loop => collapse
    - non-closed => fracture
    - non-trivial cycle => survivable soliton
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import numpy as np


class HyperRingOperator(nn.Module):
    """
    Hyper-Ring Operator: H(r) = ∮_C ∇_top Φ(r)
    
    Computes the line integral of the topological gradient around
    the constraint boundary.
    """
    
    def __init__(self, num_integration_points: int = 32):
        """
        Args:
            num_integration_points: Number of points for numerical integration
        """
        super().__init__()
        self.num_integration_points = num_integration_points
    
    def compute_topological_gradient(
        self,
        residue: torch.Tensor,
        constraint_manifold: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Compute topological gradient ∇_top Φ(r).
        
        The topological gradient measures how the embedding changes
        along the constraint manifold.
        
        Args:
            residue: [batch, ...] residue tensor
            constraint_manifold: [batch, dim] constraint manifold points
            embedding_fn: Optional function Phi: r -> constraint space
            
        Returns:
            grad_top: [batch, dim] topological gradient
        """
        if embedding_fn is None:
            # Default: identity embedding
            embedding_fn = lambda r: r.reshape(-1, constraint_manifold.shape[-1])
        
        # Embed residue
        phi_r = embedding_fn(residue)
        
        # Ensure same shape
        if phi_r.shape != constraint_manifold.shape:
            if phi_r.numel() == constraint_manifold.numel():
                phi_r = phi_r.reshape(constraint_manifold.shape)
            else:
                # Project to same dimension
                if phi_r.shape[-1] != constraint_manifold.shape[-1]:
                    proj = nn.Linear(phi_r.shape[-1], constraint_manifold.shape[-1], 
                                   device=phi_r.device)
                    phi_r = proj(phi_r)
        
        # Compute gradient: difference from constraint manifold
        # This approximates the topological gradient
        grad_top = phi_r - constraint_manifold
        
        return grad_top
    
    def line_integral(
        self,
        gradient: torch.Tensor,
        constraint_manifold: torch.Tensor,
        boundary_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute line integral ∮_C ∇_top Φ(r) around constraint boundary.
        
        Uses numerical integration along a closed path.
        
        Args:
            gradient: [batch, dim] topological gradient
            constraint_manifold: [batch, dim] constraint manifold
            boundary_points: Optional [num_points, dim] explicit boundary points
            
        Returns:
            hyper_ring: [batch] hyper-ring values
        """
        batch_size = gradient.shape[0]
        dim = gradient.shape[-1]
        device = gradient.device
        
        if boundary_points is None:
            # Generate boundary points on a hypersphere around constraint manifold
            angles = torch.linspace(0, 2 * np.pi, self.num_integration_points, device=device)
            
            # center: [batch, dim]
            center = constraint_manifold
            
            if dim >= 2:
                # path_points: [batch, num_points, dim]
                path_points = center.unsqueeze(1).repeat(1, self.num_integration_points, 1)
                
                # radius: [batch]
                radius = torch.norm(gradient[:, :2], dim=-1) + 1e-6
                
                # Broadcast radius and angles to [batch, num_points]
                # angles: [num_points] -> [1, num_points]
                # radius: [batch] -> [batch, 1]
                path_points[:, :, 0] += radius.unsqueeze(1) * torch.cos(angles).unsqueeze(0)
                path_points[:, :, 1] += radius.unsqueeze(1) * torch.sin(angles).unsqueeze(0)
            else:
                # 1D: [batch, num_points, 1]
                # center: [batch, 1]
                # linspace: [num_points] -> [1, num_points, 1]
                # gradient: [batch, 1] -> [batch, 1, 1]
                path_points = center.unsqueeze(1) + torch.linspace(
                    -1, 1, self.num_integration_points, device=device
                ).unsqueeze(0).unsqueeze(-1) * gradient.unsqueeze(1)
            
            # Compute tangents: [batch, num_points, dim]
            # tangents = p[1:] - p[:-1]
            tangents = path_points[:, 1:] - path_points[:, :-1]
            # closing the loop: [batch, 1, dim]
            closing_tangent = path_points[:, 0:1] - path_points[:, -1:]
            tangents = torch.cat([tangents, closing_tangent], dim=1)
            
            # Evaluate gradient at path points (simplified: use constant gradient)
            # gradient: [batch, dim] -> [batch, num_points, dim]
            grad_at_points = gradient.unsqueeze(1).expand(-1, self.num_integration_points, -1)
            
            # Dot product: grad · tangent -> [batch, num_points]
            integrand = torch.sum(grad_at_points * tangents, dim=-1)
            
            # Integrate: [batch]
            hyper_rings = torch.sum(integrand, dim=-1) / self.num_integration_points
        else:
            # boundary_points: [num_points, dim]
            # Compute tangents: [num_points, dim]
            tangents = boundary_points[1:] - boundary_points[:-1]
            closing_tangent = boundary_points[0:1] - boundary_points[-1:]
            tangents = torch.cat([tangents, closing_tangent], dim=0)
            
            # gradient: [batch, dim]
            # tangents: [num_points, dim]
            # dot: [batch, num_points]
            integrand = torch.matmul(gradient, tangents.t())
            
            # Integrate: [batch]
            hyper_rings = torch.sum(integrand, dim=-1) / len(boundary_points)
        
        return hyper_rings
    
    def forward(
        self,
        residue: torch.Tensor,
        constraint_manifold: torch.Tensor,
        embedding_fn: Optional[callable] = None,
        boundary_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hyper-ring: H(r) = ∮_C ∇_top Φ(r)
        
        Args:
            residue: [batch, ...] residue tensor
            constraint_manifold: [batch, dim] constraint manifold
            embedding_fn: Optional embedding function
            boundary_points: Optional explicit boundary points
            
        Returns:
            hyper_ring: [batch] hyper-ring values
        """
        grad_top = self.compute_topological_gradient(residue, constraint_manifold, embedding_fn)
        hyper_ring = self.line_integral(grad_top, constraint_manifold, boundary_points)
        return hyper_ring


class HyperRingClosureChecker(nn.Module):
    """
    Checks closure conditions for hyper-ring operator.
    
    Closure iff:
    - H(r) in Z_1(C) (closed)
    - [H(r)] != 0 in H_1(C) (non-trivial)
    """
    
    def __init__(self, closure_tolerance: float = 1e-4, trivial_threshold: float = 1e-3):
        """
        Args:
            closure_tolerance: Tolerance for checking if loop is closed
            trivial_threshold: Threshold for detecting trivial cycles
        """
        super().__init__()
        self.closure_tolerance = closure_tolerance
        self.trivial_threshold = trivial_threshold
    
    def is_in_cycle_group(
        self,
        hyper_ring: torch.Tensor,
        constraint_manifold: torch.Tensor
    ) -> torch.Tensor:
        """
        Check if H(r) is in Z_1(C) (closed cycle group).
        
        A cycle is closed if the boundary is zero (or within tolerance).
        
        Args:
            hyper_ring: [batch] hyper-ring values
            constraint_manifold: [batch, dim] constraint manifold
            
        Returns:
            is_closed: [batch] boolean tensor
        """
        # For a closed cycle, the hyper-ring should be approximately zero
        # (or the boundary should vanish)
        # Simplified check: hyper-ring magnitude should be small relative to constraint scale
        constraint_scale = torch.norm(constraint_manifold, dim=-1)
        relative_magnitude = torch.abs(hyper_ring) / (constraint_scale + 1e-8)
        
        # Closed if relative magnitude is below tolerance
        is_closed = relative_magnitude < self.closure_tolerance
        
        return is_closed
    
    def is_trivial_cycle(
        self,
        hyper_ring: torch.Tensor,
        constraint_manifold: torch.Tensor
    ) -> torch.Tensor:
        """
        Check if [H(r)] is trivial in H_1(C).
        
        A cycle is trivial if it bounds a disk (contractible to a point).
        
        Args:
            hyper_ring: [batch] hyper-ring values
            constraint_manifold: [batch, dim] constraint manifold
            
        Returns:
            is_trivial: [batch] boolean tensor
        """
        # Trivial cycles have very small hyper-ring values
        # (they can be continuously deformed to a point)
        constraint_scale = torch.norm(constraint_manifold, dim=-1)
        relative_magnitude = torch.abs(hyper_ring) / (constraint_scale + 1e-8)
        
        # Trivial if relative magnitude is below threshold
        is_trivial = relative_magnitude < self.trivial_threshold
        
        return is_trivial
    
    def check_closure(
        self,
        hyper_ring: torch.Tensor,
        constraint_manifold: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check closure conditions.
        
        Args:
            hyper_ring: [batch] hyper-ring values
            constraint_manifold: [batch, dim] constraint manifold
            
        Returns:
            is_valid: [batch] boolean tensor (True if survivable soliton)
            status: [batch] string tensor ("survivable_soliton", "fracture", "collapse")
        """
        is_closed = self.is_in_cycle_group(hyper_ring, constraint_manifold)
        is_trivial = self.is_trivial_cycle(hyper_ring, constraint_manifold)
        
        # Valid if closed and non-trivial
        is_valid = is_closed & (~is_trivial)
        
        # Determine status
        status = []
        for b in range(hyper_ring.shape[0]):
            if not is_closed[b]:
                status.append("fracture")
            elif is_trivial[b]:
                status.append("collapse")
            else:
                status.append("survivable_soliton")
        
        return is_valid, status
    
    def forward(
        self,
        hyper_ring: torch.Tensor,
        constraint_manifold: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: check closure and return diagnostics.
        
        Returns:
            Dictionary with:
            - 'is_valid': [batch] boolean
            - 'status': List of status strings
            - 'is_closed': [batch] boolean
            - 'is_trivial': [batch] boolean
        """
        is_valid, status = self.check_closure(hyper_ring, constraint_manifold)
        is_closed = self.is_in_cycle_group(hyper_ring, constraint_manifold)
        is_trivial = self.is_trivial_cycle(hyper_ring, constraint_manifold)
        
        return {
            'is_valid': is_valid,
            'status': status,
            'is_closed': is_closed,
            'is_trivial': is_trivial,
            'hyper_ring': hyper_ring
        }
