"""
Non-Commutativity Curvature Tensor.

Computes the 2-form K = Σ κ_ij e_i∧e_j measuring update order dependence.
When two update operators A, B don't commute (AB ≠ BA), the system has
non-trivial curvature — the order of operations matters structurally.

References:
    - VETO_SUBSPACE_ARCHITECTURE.md
    - ai project report_2-2-2026.txt §"Non-commutativity curvature tensor"
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class NonCommutativityCurvature(nn.Module):
    """
    Computes the non-commutativity curvature 2-form from pairs of update operators.
    
    Given operators A, B ∈ R^{d×d} (weight matrices, Jacobians, or update steps):
        [A, B] = A·B - B·A                    (commutator)
        κ = ½([A, B] - [A, B]^T)              (antisymmetric part = curvature)
        K_norm = ||κ||_F                        (Frobenius norm = total curvature)
    
    High curvature means the system is in a regime where update ordering
    produces divergent trajectories — a structural signal for the veto lattice.
    """
    
    def __init__(self, dim: int, curvature_threshold: float = 0.3,
                 ema_decay: float = 0.95):
        """
        Args:
            dim: Dimensionality of the operator space.
            curvature_threshold: Above this, the operators are "strongly non-commutative."
            ema_decay: Exponential moving average decay for tracking curvature history.
        """
        super().__init__()
        self.dim = dim
        self.curvature_threshold = curvature_threshold
        self.ema_decay = ema_decay
        
        # EMA-smoothed curvature for trend detection
        self.register_buffer('curvature_ema', torch.tensor(0.0))
        self.register_buffer('max_observed', torch.tensor(0.0))
    
    def compute_commutator(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute [A, B] = AB - BA.
        
        Args:
            A: [dim, dim] or [batch, dim, dim] operator
            B: [dim, dim] or [batch, dim, dim] operator
            
        Returns:
            commutator: same shape as inputs
        """
        return torch.matmul(A, B) - torch.matmul(B, A)
    
    def extract_curvature(self, commutator: torch.Tensor) -> torch.Tensor:
        """
        Extract the antisymmetric curvature 2-form from the commutator.
        
        κ_ij = ½([A,B]_ij - [A,B]_ji)
        
        This is the "pure rotation" part — the part that genuinely measures
        non-commutativity rather than symmetric scaling artifacts.
        """
        if commutator.dim() == 2:
            return 0.5 * (commutator - commutator.T)
        else:
            return 0.5 * (commutator - commutator.transpose(-2, -1))
    
    def compute_curvature(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor,
        update_ema: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full curvature computation pipeline.
        
        Args:
            A: [dim, dim] first update operator
            B: [dim, dim] second update operator
            update_ema: Whether to update the running EMA estimate
            
        Returns:
            Dict with:
                'kappa': [dim, dim] antisymmetric curvature tensor
                'curvature_norm': scalar Frobenius norm
                'commutator': [dim, dim] raw commutator
                'is_strongly_noncommutative': bool
                'curvature_ema': scalar running average
                'relative_curvature': norm relative to operator norms
        """
        # Raw commutator
        commutator = self.compute_commutator(A, B)
        
        # Antisymmetric curvature 2-form
        kappa = self.extract_curvature(commutator)
        
        # Curvature magnitude
        curvature_norm = torch.norm(kappa, p='fro')
        
        # Relative curvature: normalize by operator norms to get
        # a scale-invariant measure ∈ [0, 1]
        norm_A = torch.norm(A, p='fro').clamp(min=1e-8)
        norm_B = torch.norm(B, p='fro').clamp(min=1e-8)
        relative_curvature = curvature_norm / (norm_A * norm_B)
        
        # EMA update
        if update_ema:
            self.curvature_ema = (
                self.ema_decay * self.curvature_ema + 
                (1 - self.ema_decay) * curvature_norm.detach()
            )
            self.max_observed = torch.max(self.max_observed, curvature_norm.detach())
        
        # Threshold check
        is_strong = relative_curvature > self.curvature_threshold
        
        return {
            'kappa': kappa,
            'curvature_norm': curvature_norm,
            'commutator': commutator,
            'is_strongly_noncommutative': is_strong,
            'curvature_ema': self.curvature_ema,
            'relative_curvature': relative_curvature,
            'max_observed': self.max_observed
        }
    
    def compute_wedge_components(
        self, 
        kappa: torch.Tensor, 
        basis_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decompose κ into wedge product components κ_ij e_i ∧ e_j.
        
        In the standard basis, these are just the upper-triangular 
        entries of the antisymmetric matrix κ. With a custom basis,
        we project first.
        
        Args:
            kappa: [dim, dim] antisymmetric curvature tensor
            basis_vectors: Optional [dim, dim] custom basis (columns = basis vectors)
            
        Returns:
            components: [dim*(dim-1)/2] wedge product coefficients
        """
        if basis_vectors is not None:
            # Change of basis: κ' = E^T κ E
            kappa = basis_vectors.T @ kappa @ basis_vectors
            
        # Extract upper-triangular (i < j) entries
        indices = torch.triu_indices(kappa.shape[0], kappa.shape[1], offset=1)
        return kappa[indices[0], indices[1]]
    
    def curvature_pressure(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: returns a single scalar pressure from curvature.
        
        Used as a veto signal — high pressure means update order matters
        enough that the system should be cautious about committing to
        either ordering.
        
        Returns:
            pressure: scalar ∈ [0, 1] (sigmoid of relative curvature)
        """
        result = self.compute_curvature(A, B, update_ema=True)
        return torch.sigmoid(result['relative_curvature'] * 3.0 - 1.0)
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Return current state for monitoring."""
        return {
            'curvature_ema': self.curvature_ema.item(),
            'max_observed': self.max_observed.item(),
            'threshold': self.curvature_threshold
        }
