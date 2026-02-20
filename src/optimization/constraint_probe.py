"""
Constraint Probe Operator: System 2 as Local Feasibility Probe

Implements the probe operator P_k: r -> argmin_{c in C_k} L_k(r, c)
with NO global objective, only local feasibility.

Mathematical Foundation:
    L_k(r, c) = |Phi_k(r) - c|_{Sigma_k} + psi_k(c)
    
    where:
    - Sigma_k: sparse covariance (anisotropic)
    - psi_k: gyroid violation (admissibility filter, not truth metric)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Callable

# Phase 3: Import gyroid flow constraint
try:
    from src.topology.gyroid_differentiation import GyroidFlowConstraint
    HAS_GYROID_FLOW = True
except ImportError:
    HAS_GYROID_FLOW = False


class ConstraintProbeOperator(nn.Module):
    """
    Probe Operator P_k: r -> argmin_{c in C_k} L_k(r, c)
    
    No global objective, only local feasibility per constraint.
    """
    
    def __init__(
        self,
        constraint_index: int,
        sparse_covariance: torch.Tensor,
        embedding_fn: Optional[Callable] = None,
        gyroid_violation_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            constraint_index: Index k of this constraint
            sparse_covariance: [dim, dim] sparse covariance matrix Sigma_k (anisotropic)
            embedding_fn: Function Phi_k: r -> constraint space
            gyroid_violation_fn: Function psi_k: c -> violation score
            device: Device for tensors
        """
        super().__init__()
        self.k = constraint_index
        self.device = device or sparse_covariance.device
        
        # Store sparse covariance (anisotropic)
        # Ensure it's positive semi-definite
        self.register_buffer('Sigma_k', self._ensure_psd(sparse_covariance))
        
        # Embedding function (default: identity if not provided)
        self.embedding_fn = embedding_fn or (lambda x: x)
        
        # Gyroid violation function (default: zero if not provided)
        self.gyroid_violation_fn = gyroid_violation_fn or (lambda x: torch.tensor(0.0, device=self.device))
    
    def _ensure_psd(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Ensure covariance matrix is positive semi-definite.
        """
        # Symmetrize
        cov = (cov + cov.t()) / 2.0
        
        # Add small diagonal to ensure positive definiteness
        cov = cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6
        
        return cov
    
    def compute_local_strain(
        self,
        residue: torch.Tensor,
        constraint_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute local strain: |Phi_k(r) - c|_{Sigma_k}
        
        Args:
            residue: [batch, ...] residue tensor
            constraint_state: [batch, dim] constraint state c
            
        Returns:
            strain: [batch] local strain values
        """
        # Embed residue into constraint space
        phi_r = self.embedding_fn(residue)  # [batch, dim]
        
        # Ensure same shape
        if phi_r.shape != constraint_state.shape:
            # Try to reshape or project
            if phi_r.numel() == constraint_state.numel():
                phi_r = phi_r.reshape(constraint_state.shape)
            else:
                # Project to same dimension
                if phi_r.dim() == 1:
                    phi_r = phi_r.unsqueeze(0)
                if constraint_state.dim() == 1:
                    constraint_state = constraint_state.unsqueeze(0)
                
                # Use linear projection if dimensions don't match
                if phi_r.shape[-1] != constraint_state.shape[-1]:
                    proj = nn.Linear(phi_r.shape[-1], constraint_state.shape[-1], device=self.device)
                    phi_r = proj(phi_r)
        
        # Compute difference
        diff = phi_r - constraint_state  # [batch, dim]
        
        # Weighted norm using sparse covariance
        # |diff|_{Sigma_k} = sqrt(diff^T Sigma_k diff)
        # For efficiency, use diagonal approximation if Sigma_k is large
        if self.Sigma_k.shape[0] > 100:
            # Use diagonal approximation
            diag = torch.diag(self.Sigma_k)
            strain = torch.sqrt(torch.sum(diff.pow(2) * diag.unsqueeze(0), dim=-1))
        else:
            # Full quadratic form
            strain = torch.sqrt(torch.sum(diff.unsqueeze(-1) * (self.Sigma_k @ diff.unsqueeze(-1)), dim=-2).squeeze(-1))
        
        return strain
    
    def compute_gyroid_violation(
        self,
        constraint_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gyroid violation: psi_k(c)
        
        This is an admissibility filter, NOT a truth metric.
        
        Phase 3: Can include gyroid flow constraint if enabled.
        
        Args:
            constraint_state: [batch, dim] constraint state c
            
        Returns:
            violation: [batch] violation scores
        """
        base_violation = self.gyroid_violation_fn(constraint_state)
        
        # Phase 3: Add gyroid flow constraint violation if enabled
        if self.use_gyroid_flow and constraint_state.shape[-1] >= 3:
            try:
                # Check flow constraint (requires 3D embedding)
                flow_result = self.gyroid_flow_constraint(
                    residue=constraint_state,
                    embedding=constraint_state[..., :3],  # Use first 3 dims
                    embedding_fn=None
                )
                # Add flow violation to base violation
                flow_violation = (~flow_result['is_satisfied']).float() * 0.1
                base_violation = base_violation + flow_violation
            except Exception:
                # If flow constraint fails, use base violation only
                pass
        
        return base_violation
    
    def forward(
        self,
        residue: torch.Tensor,
        lambda_k: Optional[torch.Tensor] = None,
        initial_constraint: Optional[torch.Tensor] = None,
        max_probe_iters: int = 10,
        probe_step_size: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Probe for local feasibility: argmin_{c in C_k} L_k(r, c)
        
        Args:
            residue: [batch, ...] residue tensor
            lambda_k: [batch, dim] optional Lagrange multiplier
            initial_constraint: [batch, dim] optional initial constraint state
            max_probe_iters: Maximum iterations for local probe
            probe_step_size: Step size for local optimization
            
        Returns:
            constraint_state: [batch, dim] optimal constraint state c_k
            loss: [batch] local loss L_k(r, c_k)
        """
        batch_size = residue.shape[0] if residue.dim() > 0 else 1
        
        # Initialize constraint state
        if initial_constraint is not None:
            c = initial_constraint.clone()
        else:
            # Initialize from embedded residue
            phi_r = self.embedding_fn(residue)
            if phi_r.dim() == 1:
                phi_r = phi_r.unsqueeze(0)
            c = phi_r.clone()
            
            # Ensure correct dimension
            if c.shape[-1] != self.Sigma_k.shape[0]:
                # Project to correct dimension
                if c.shape[-1] < self.Sigma_k.shape[0]:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, self.Sigma_k.shape[0] - c.shape[-1], device=self.device)
                    c = torch.cat([c, padding], dim=-1)
                else:
                    # Truncate
                    c = c[..., :self.Sigma_k.shape[0]]
        
        # Local feasibility probe (no global objective)
        c.requires_grad_(True)
        
        for _ in range(max_probe_iters):
            # Compute local loss
            local_strain = self.compute_local_strain(residue, c)
            gyroid_violation = self.compute_gyroid_violation(c)
            
            # Total local loss (no global terms)
            loss_k = local_strain + gyroid_violation
            
            # Add Lagrange multiplier term if provided
            if lambda_k is not None:
                if lambda_k.shape == c.shape:
                    lagrange_term = torch.sum(lambda_k * (self.embedding_fn(residue) - c), dim=-1)
                    loss_k = loss_k + lagrange_term
            
            # Compute gradient (local only)
            grad = torch.autograd.grad(
                loss_k.sum(),
                c,
                create_graph=False,
                retain_graph=True
            )[0]
            
            # Update constraint state (local step)
            with torch.no_grad():
                c = c - probe_step_size * grad
                c.requires_grad_(True)
        
        # Final loss computation
        with torch.no_grad():
            local_strain = self.compute_local_strain(residue, c)
            gyroid_violation = self.compute_gyroid_violation(c)
            final_loss = local_strain + gyroid_violation
        
        return c.detach(), final_loss.detach()


class ConstraintManifold:
    """
    Represents the constraint manifold C = C_sym × C_phys × C_ext
    """
    
    def __init__(
        self,
        symbolic_dim: int,
        physical_dim: int,
        external_dim: int = 0
    ):
        """
        Args:
            symbolic_dim: Dimension of symbolic constraints C_sym
            physical_dim: Dimension of physical constraints C_phys
            external_dim: Dimension of external/evidence constraints C_ext
        """
        self.symbolic_dim = symbolic_dim
        self.physical_dim = physical_dim
        self.external_dim = external_dim
        self.total_dim = symbolic_dim + physical_dim + external_dim
    
    def split(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split constraint state into components.
        
        Args:
            state: [batch, total_dim] constraint state
            
        Returns:
            Dictionary with 'symbolic', 'physical', 'external' components
        """
        return {
            'symbolic': state[..., :self.symbolic_dim],
            'physical': state[..., self.symbolic_dim:self.symbolic_dim + self.physical_dim],
            'external': state[..., self.symbolic_dim + self.physical_dim:] if self.external_dim > 0 else None
        }
    
    def combine(
        self,
        symbolic: torch.Tensor,
        physical: torch.Tensor,
        external: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine constraint components into full state.
        
        Args:
            symbolic: [batch, symbolic_dim]
            physical: [batch, physical_dim]
            external: [batch, external_dim] optional
            
        Returns:
            state: [batch, total_dim]
        """
        components = [symbolic, physical]
        if external is not None:
            components.append(external)
        return torch.cat(components, dim=-1)
