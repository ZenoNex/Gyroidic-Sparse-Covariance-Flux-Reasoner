"""
Structural Irreducibility: Orthogonality Criterion

Implements evidence module projections and checks structural irreducibility
to ensure no single-face embedding exists.

Mathematical Foundation:
    For evidence modules E_alpha, projections pi_alpha: C -> R^{d_alpha}
    
    Residue r is structurally irreducible iff:
    - <pi_alpha Phi(r), pi_beta Phi(r)> = 0 for alpha != beta
    - rank(⊕_alpha pi_alpha Phi(r)) > 1

Phase 3: Advanced Constraints Implementation

Author: Implementation Documentation
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict


class EvidenceModuleProjection(nn.Module):
    """
    Evidence Module Projection: pi_alpha: C -> R^{d_alpha}
    
    Projects constraint manifold onto evidence cluster alpha.
    """
    
    def __init__(
        self,
        evidence_cluster: torch.Tensor,
        projection_dim: int,
        constraint_dim: Optional[int] = None
    ):
        """
        Args:
            evidence_cluster: [cluster_size, feature_dim] evidence cluster E_alpha
            projection_dim: Dimension d_alpha of projection space
            constraint_dim: Dimension of constraint space (inferred if None)
        """
        super().__init__()
        self.E_alpha = evidence_cluster
        self.d_alpha = projection_dim
        
        # Infer constraint dimension from evidence cluster or use provided
        if constraint_dim is None:
            constraint_dim = evidence_cluster.shape[-1]
        
        # Learnable projection matrix
        self.proj_matrix = nn.Linear(constraint_dim, projection_dim, bias=False)
        
        # Initialize to preserve evidence cluster structure
        with torch.no_grad():
            # Use PCA-like initialization from evidence cluster
            if evidence_cluster.shape[0] > 1:
                # Center the cluster
                centered = evidence_cluster - evidence_cluster.mean(dim=0, keepdim=True)
                # SVD for principal directions
                try:
                    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
                    # Use top d_alpha directions
                    if projection_dim <= Vt.shape[0]:
                        self.proj_matrix.weight.data = Vt[:projection_dim, :].clone()
                except Exception:
                    # Fallback: random orthogonal initialization
                    nn.init.orthogonal_(self.proj_matrix.weight)
            else:
                nn.init.orthogonal_(self.proj_matrix.weight)
    
    def forward(self, constraint_state: torch.Tensor) -> torch.Tensor:
        """
        Project constraint state onto evidence module space.
        
        Args:
            constraint_state: [batch, constraint_dim] constraint state
            
        Returns:
            projection: [batch, d_alpha] projected state
        """
        return self.proj_matrix(constraint_state)


class StructuralIrreducibilityChecker(nn.Module):
    """
    Checks structural irreducibility of residues.
    
    Ensures no single-face embedding exists by checking orthogonality
    across evidence modules and rank condition.
    """
    
    def __init__(self, orthogonality_tolerance: float = 1e-6, min_rank: int = 2):
        """
        Args:
            orthogonality_tolerance: Tolerance for orthogonality check
            min_rank: Minimum rank required (default: 2)
        """
        super().__init__()
        self.orthogonality_tolerance = orthogonality_tolerance
        self.min_rank = min_rank
    
    def is_structurally_irreducible(
        self,
        residue: torch.Tensor,
        embedding_fn: callable,
        evidence_modules: List[EvidenceModuleProjection]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Check structural irreducibility.
        
        Args:
            residue: [batch, ...] residue tensor
            embedding_fn: Function Phi: r -> constraint space
            evidence_modules: List of evidence module projections
            
        Returns:
            is_irreducible: [batch] boolean tensor
            diagnostics: Dictionary with orthogonality and rank information
        """
        if len(evidence_modules) < 2:
            # Need at least 2 modules for irreducibility check
            batch_size = residue.shape[0] if residue.dim() > 0 else 1
            return torch.zeros(batch_size, dtype=torch.bool, device=residue.device), {}
        
        # Embed residue
        phi_r = embedding_fn(residue)  # [batch, constraint_dim]
        
        # Project onto each evidence module
        projections = []
        for pi_alpha in evidence_modules:
            proj = pi_alpha(phi_r)  # [batch, d_alpha]
            projections.append(proj)
        
        batch_size = projections[0].shape[0]
        is_irreducible = torch.ones(batch_size, dtype=torch.bool, device=residue.device)
        
        # Check orthogonality: <pi_alpha Phi(r), pi_beta Phi(r)> = 0 for alpha != beta
        orthogonality_violations = []
        for i, proj_i in enumerate(projections):
            for j, proj_j in enumerate(projections):
                if i != j:
                    # Compute dot product (batch-wise)
                    # proj_i: [batch, d_i], proj_j: [batch, d_j]
                    # Need to handle different dimensions
                    if proj_i.shape[-1] == proj_j.shape[-1]:
                        dot_product = torch.sum(proj_i * proj_j, dim=-1)  # [batch]
                    else:
                        # Use Frobenius inner product for different dimensions
                        # Project to common dimension or use trace
                        min_dim = min(proj_i.shape[-1], proj_j.shape[-1])
                        dot_product = torch.sum(
                            proj_i[..., :min_dim] * proj_j[..., :min_dim], dim=-1
                        )
                    
                    # Check if orthogonality is violated
                    violation = torch.abs(dot_product) > self.orthogonality_tolerance
                    orthogonality_violations.append(violation)
                    is_irreducible = is_irreducible & (~violation)
        
        # Check rank: rank(⊕_alpha pi_alpha Phi(r)) > 1
        # Stack projections: [num_modules, batch, d_alpha] -> [batch, num_modules, d_alpha]
        # Then check rank of each batch element
        max_proj_dim = max(proj.shape[-1] for proj in projections)
        
        # Pad projections to same dimension
        padded_projections = []
        for proj in projections:
            if proj.shape[-1] < max_proj_dim:
                padding = torch.zeros(
                    proj.shape[0], max_proj_dim - proj.shape[-1],
                    device=proj.device, dtype=proj.dtype
                )
                proj_padded = torch.cat([proj, padding], dim=-1)
            else:
                proj_padded = proj
            padded_projections.append(proj_padded)
        
        # Stack: [num_modules, batch, max_dim] -> [batch, num_modules, max_dim]
        stacked = torch.stack(padded_projections, dim=1)  # [batch, num_modules, max_dim]
        
        # Compute rank for each batch element
        ranks = []
        for b in range(batch_size):
            batch_matrix = stacked[b]  # [num_modules, max_dim]
            rank = torch.linalg.matrix_rank(batch_matrix)
            ranks.append(rank)
            if rank <= self.min_rank:
                is_irreducible[b] = False
        
        ranks_tensor = torch.tensor(ranks, device=residue.device, dtype=torch.long)
        
        diagnostics = {
            'orthogonality_violations': torch.stack(orthogonality_violations) if orthogonality_violations else None,
            'ranks': ranks_tensor,
            'projections': projections
        }
        
        return is_irreducible, diagnostics
    
    def forward(
        self,
        residue: torch.Tensor,
        embedding_fn: callable,
        evidence_modules: List[EvidenceModuleProjection]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: check irreducibility and return diagnostics.
        
        Returns:
            Dictionary with:
            - 'is_irreducible': [batch] boolean
            - 'orthogonality_violations': [num_pairs, batch] boolean
            - 'ranks': [batch] integer ranks
        """
        is_irreducible, diagnostics = self.is_structurally_irreducible(
            residue, embedding_fn, evidence_modules
        )
        
        return {
            'is_irreducible': is_irreducible,
            'orthogonality_violations': diagnostics.get('orthogonality_violations'),
            'ranks': diagnostics.get('ranks'),
            'projections': diagnostics.get('projections')
        }
