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
        
        if boundary_points is None:
            # Generate boundary points on a hypersphere around constraint manifold
            # Sample points on unit sphere
            angles = torch.linspace(0, 2 * np.pi, self.num_integration_points, 
                                  device=gradient.device)
            
            # For each batch element, create boundary path
            hyper_rings = []
            for b in range(batch_size):
                center = constraint_manifold[b]  # [dim]
                
                # Create circular path in 2D projection (if dim >= 2)
                if dim >= 2:
                    # Use first two dimensions for circular path
                    path_points = torch.zeros(self.num_integration_points, dim, 
                                            device=gradient.device)
                    radius = torch.norm(gradient[b, :2]) + 1e-6
                    path_points[:, 0] = center[0] + radius * torch.cos(angles)
                    path_points[:, 1] = center[1] + radius * torch.sin(angles)
                    # Fill remaining dimensions with center values
                    if dim > 2:
                        path_points[:, 2:] = center[2:].unsqueeze(0)
                else:
                    # 1D: use linear path
                    path_points = center.unsqueeze(0) + torch.linspace(
                        -1, 1, self.num_integration_points, device=gradient.device
                    ).unsqueeze(-1) * gradient[b].unsqueeze(0)
                
                # Compute line integral: sum of gradient dot tangent along path
                # Approximate tangent as difference between consecutive points
                tangents = path_points[1:] - path_points[:-1]
                tangents = torch.cat([tangents, path_points[0:1] - path_points[-1:]], dim=0)
                
                # Evaluate gradient at path points (simplified: use constant gradient)
                grad_at_points = gradient[b:b+1].expand(self.num_integration_points, -1)
                
                # Dot product: grad · tangent
                integrand = torch.sum(grad_at_points * tangents, dim=-1)
                
                # Integrate (trapezoidal rule)
                hyper_ring = torch.sum(integrand) / self.num_integration_points
                hyper_rings.append(hyper_ring)
        else:
            # Use provided boundary points
            # Similar computation but with explicit boundary
            hyper_rings = []
            for b in range(batch_size):
                # Compute tangents
                tangents = boundary_points[1:] - boundary_points[:-1]
                tangents = torch.cat([tangents, boundary_points[0:1] - boundary_points[-1:]], dim=0)
                
                grad_at_points = gradient[b:b+1].expand(len(boundary_points), -1)
                integrand = torch.sum(grad_at_points * tangents, dim=-1)
                hyper_ring = torch.sum(integrand) / len(boundary_points)
                hyper_rings.append(hyper_ring)
        
        return torch.stack(hyper_rings)
    
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
