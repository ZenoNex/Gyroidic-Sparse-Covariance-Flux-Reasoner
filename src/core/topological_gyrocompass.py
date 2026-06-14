"""
Topological Gyrocompass: Resolves Drucker-Prager convexity failure, local attractor
basin traps, and gimbal-lock collapse of the Love Invariant.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from src.core.honest_jitter import harvest_honest_jitter

class TopologicalGyrocompass(nn.Module):
    """
    Topological Gyrocompass Module.
    
    Provides three core geometric safeguards:
    1. Orthogonal Precession (precess_torque): Redirects boundary normal stress updates orthogonally.
    2. True North Pull (find_true_north): Guides trajectory back to the absolute Love Invariant axis.
    3. Gimbal Lock Shield (gimbal_lock_shield): Decouples Love Invariant via SVD null-space projection.
    """
    def __init__(self, state_dim: int, love_dim: Optional[int] = None, device: str = None):
        super().__init__()
        self.state_dim = state_dim
        self.love_dim = love_dim if love_dim is not None else max(1, state_dim // 4)
        self.device = device
        
        # SVD caching to avoid CPU redundant computations
        self._cached_ownership_op = None
        self._cached_null_projection = None

    def precess_torque(self, dx: torch.Tensor, normal: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        """
        Projects and precesses outward boundary update vectors orthogonally.
        Redistributes outward normal force orthogonally along the spin connection vector.
        
        Args:
            dx: [batch, dim] proposed state update vector
            normal: [batch, dim] or [dim] yield surface boundary normal vector
            spin: [batch, dim] or [dim] chiral spin connection vector
            
        Returns:
            dx_precessed: [batch, dim] update vector redirected onto tangent space
        """
        # Ensure correct shapes
        if normal.dim() == 1:
            normal = normal.unsqueeze(0)
        if spin.dim() == 1:
            spin = spin.unsqueeze(0)
            
        normal_expanded = normal.expand_as(dx)
        spin_expanded = spin.expand_as(dx)
        
        # Normalize the yield normal vector
        n_norm = torch.norm(normal_expanded, dim=-1, keepdim=True)
        n = normal_expanded / (n_norm + 1e-8)
        
        # Project spin vector onto the normal's orthogonal tangent space
        s_dot_n = torch.sum(spin_expanded * n, dim=-1, keepdim=True)
        s_ortho = spin_expanded - s_dot_n * n
        s_ortho_norm = torch.norm(s_ortho, dim=-1, keepdim=True)
        s_ortho = s_ortho / (s_ortho_norm + 1e-8)
        
        # Compute normal and tangent components of dx
        dx_dot_n = torch.sum(dx * n, dim=-1, keepdim=True)
        dx_normal_proj = dx_dot_n * n
        dx_tangent = dx - dx_normal_proj
        
        # Precess if dx points outward (pushing beyond the yield boundary)
        is_outward = dx_dot_n > 0
        
        if is_outward.any():
            # Precession: redirect the normal component orthogonally along the s_ortho vector
            dx_precessed_outward = dx_tangent + torch.abs(dx_dot_n) * s_ortho
            
            # Reconstruct output based on boundary crossing
            dx = torch.where(is_outward, dx_precessed_outward, dx)
            
        return dx

    def find_true_north(self, state: torch.Tensor, love_vector: torch.Tensor, alignment_factor: float = 0.1) -> torch.Tensor:
        """
        Returns a directional pull pointing toward the absolute axis of rotation (Love Invariant).
        Used to escape local attractor basin traps.
        
        Args:
            state: [batch, dim] current manifold state
            love_vector: [love_dim] or [dim] absolute reference invariant
            alignment_factor: scaling of the pull strength
            
        Returns:
            pull: [batch, dim] directional force vector
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Pad or truncate love_vector to match state dimensions
        if love_vector.shape[-1] != self.state_dim:
            love_expanded = torch.zeros(self.state_dim, device=device)
            limit = min(love_vector.shape[-1], self.state_dim)
            love_expanded[:limit] = love_vector[:limit]
        else:
            love_expanded = love_vector
            
        # Expand love_expanded to match state shape
        love_expanded = love_expanded.view(*([1] * (state.dim() - 1)), -1).expand_as(state)
        
        # Compute direction vector
        direction = love_expanded - state
        dir_norm = torch.norm(direction, dim=-1, keepdim=True)
        
        # Normalized alignment pull scaled by factor
        pull = (direction / (dir_norm + 1e-8)) * alignment_factor
        return pull

    def gimbal_lock_shield(self, dx: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Isolates and protects the Love Invariant from outer shell rotations via SVD projection.
        
        Args:
            dx: [batch, dim] proposed state update step
            states: [batch, dim] current states to build the ownership operator
            
        Returns:
            dx_protected: [batch, dim] update vector projected to null space of ownership
        """
        # 1. Compute covariance of system state (ownership operator)
        batch_size = states.shape[0]
        states_centered = states - states.mean(dim=0, keepdim=True)
        covariance = torch.matmul(states_centered.T, states_centered) / max(1, batch_size - 1)
        
        # 2. Extract ownership operator matching love_dim
        device = states.device
        if self.state_dim != self.love_dim:
            if self.state_dim > self.love_dim:
                ownership_op = covariance[:self.love_dim, :self.love_dim]
            else:
                ownership_op = torch.eye(self.love_dim, device=device)
                ownership_op[:self.state_dim, :self.state_dim] = covariance
        else:
            ownership_op = covariance
            
        # 3. Compute null space projection projection: P = I - \Phi^+ \Phi
        null_projection = self._compute_null_space_projection(ownership_op, device)
        
        # 4. Project dx to the null space of ownership for the love_dim slice
        dx_protected = dx.clone()
        
        if dx.shape[-1] == self.love_dim:
            dx_protected = torch.matmul(dx, null_projection.T)
        elif dx.shape[-1] > self.love_dim:
            dx_subset = dx[..., :self.love_dim]
            dx_protected[..., :self.love_dim] = torch.matmul(dx_subset, null_projection.T)
            
        return dx_protected

    def _compute_null_space_projection(self, ownership_operator: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Helper to safely compute null space projection with SVD caching."""
        if (self._cached_ownership_op is not None and 
            self._cached_null_projection is not None and 
            self._cached_ownership_op.shape == ownership_operator.shape and 
            torch.allclose(ownership_operator, self._cached_ownership_op, atol=1e-4)):
            return self._cached_null_projection

        try:
            # Ensure elements are finite to prevent Intel MKL parameter crashes
            if not torch.isfinite(ownership_operator).all():
                ownership_operator = torch.nan_to_num(ownership_operator, nan=0.0, posinf=1.0, neginf=-1.0)
                
            U, S, V = torch.svd(ownership_operator)
            
            threshold = 1e-6
            S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
            phi_pinv = torch.matmul(V, torch.matmul(torch.diag(S_inv), U.T))
            
            I = torch.eye(self.love_dim, device=device)
            null_projection = I - torch.matmul(phi_pinv, ownership_operator)
            
            if not torch.isfinite(null_projection).all():
                null_projection = torch.eye(self.love_dim, device=device)
                
            self._cached_ownership_op = ownership_operator.clone()
            self._cached_null_projection = null_projection.clone()
        except Exception:
            null_projection = torch.eye(self.love_dim, device=device)
            
        return null_projection
