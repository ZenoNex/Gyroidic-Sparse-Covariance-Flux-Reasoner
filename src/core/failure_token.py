"""
Failure Token System: Rupture Functional (Not Error)

Implements the rupture functional R(r) that detects constraint violations
without treating them as errors. Failure tokens are discrete signals that
prevent gradient flow, repair attempts, and memory updates.

Mathematical Foundation:
    R(r) = 1 if exists k: L_k(r, c_k) = ∞
    R(r) = 0 otherwise
    
    On R(r) = 1: r -> ⊥ (no gradient, no repair, no memory update)

Phase 1: Core Constraint Probe Implementation

Author: Implementation Documentation
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from enum import Enum


class FailureTokenType(Enum):
    """Types of failure tokens."""
    BOT = "⊥"  # Rupture - no recovery possible
    REPAIRED = "repaired"  # Successfully repaired
    ALTERNATIVE = "alternative"  # Alternative solution found
    PENDING = "pending"  # Still processing


class FailureToken:
    """
    Failure token representing system state.
    
    Not an error - a discrete signal that prevents certain operations.
    """
    
    def __init__(self, token_type: FailureTokenType, constraint_index: Optional[int] = None, residue: Optional[torch.Tensor] = None):
        """
        Args:
            token_type: Type of failure token
            constraint_index: Optional index of constraint that triggered rupture
            residue: Optional paraconsistent residue (damage state)
        """
        self.type = token_type
        self.constraint_index = constraint_index
        self.residue = residue
        self.gradient_enabled = token_type == FailureTokenType.REPAIRED
        self.repair_enabled = token_type == FailureTokenType.PENDING
        self.memory_update_enabled = token_type in [
            FailureTokenType.REPAIRED, 
            FailureTokenType.ALTERNATIVE
        ]
    
    def __repr__(self):
        return f"FailureToken({self.type.value}, constraint={self.constraint_index})"
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert to tensor representation for status codes.
        
        Returns:
            status: 0=REPAIRED, 1=ALTERNATIVE, 2=BOT, 3=PENDING
        """
        mapping = {
            FailureTokenType.REPAIRED: 0,
            FailureTokenType.ALTERNATIVE: 1,
            FailureTokenType.BOT: 2,
            FailureTokenType.PENDING: 3
        }
        return torch.tensor(mapping[self.type], device=device, dtype=torch.long)
    
    @classmethod
    def from_tensor(cls, status: torch.Tensor, constraint_index: Optional[int] = None):
        """
        Create FailureToken from tensor status code.
        
        Args:
            status: 0=REPAIRED, 1=ALTERNATIVE, 2=BOT, 3=PENDING
            constraint_index: Optional constraint index
        """
        mapping = {
            0: FailureTokenType.REPAIRED,
            1: FailureTokenType.ALTERNATIVE,
            2: FailureTokenType.BOT,
            3: FailureTokenType.PENDING
        }
        status_val = status.item() if isinstance(status, torch.Tensor) else status
        return cls(mapping.get(status_val, FailureTokenType.BOT), constraint_index)


class RuptureFunctional(nn.Module):
    """
    Rupture Functional: R(r)
    
    Detects when constraint losses become infinite or exceed rupture threshold.
    This is NOT an error condition - it's a structural rupture that prevents
    further processing.
    
    R(r) = 1 if exists k: L_k(r, c_k) = ∞ or L_k(r, c_k) > threshold
    R(r) = 0 otherwise
    """
    
    def __init__(self, rupture_threshold: float = 1e6):
        """
        Args:
            rupture_threshold: Maximum loss value before rupture (default: 1e6)
        """
        super().__init__()
        self.rupture_threshold = rupture_threshold
    
    def forward(
        self,
        residue: torch.Tensor,
        constraint_losses: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Check if any constraint loss indicates rupture.
        
        Args:
            residue: [batch, ...] residue tensor
            constraint_losses: Dictionary mapping constraint index -> loss tensor
            
        Returns:
            rupture_flag: 1.0 if rupture detected, 0.0 otherwise
            constraint_index: Index of constraint that caused rupture (or None)
        """
        if not constraint_losses:
            return torch.tensor(0.0, device=residue.device), None
        
        # Check each constraint loss
        for k, loss_k in constraint_losses.items():
            # Check for infinite loss
            if torch.isinf(loss_k).any():
                return torch.tensor(1.0, device=residue.device), k
            
            # Check for loss exceeding threshold
            if isinstance(loss_k, torch.Tensor):
                max_loss = loss_k.max() if loss_k.numel() > 0 else torch.tensor(0.0)
            else:
                max_loss = torch.tensor(float(loss_k), device=residue.device)
            
            if max_loss > self.rupture_threshold:
                return torch.tensor(1.0, device=residue.device), k
        
        return torch.tensor(0.0, device=residue.device), None
    
    def check_rupture(
        self,
        residue: torch.Tensor,
        constraint_losses: Dict[int, torch.Tensor]
    ) -> Optional[FailureToken]:
        """
        Check for rupture and return failure token if detected.
        
        Args:
            residue: [batch, ...] residue tensor
            constraint_losses: Dictionary mapping constraint index -> loss tensor
            
        Returns:
            FailureToken.BOT if rupture detected, None otherwise
        """
        rupture_flag, constraint_idx = self.forward(residue, constraint_losses)
        
        if rupture_flag.item() > 0.5:
            # Capture the current residue as the paraconsistent damage state
            # We take the mean over batch if necessary
            damage_residue = residue.detach().clone()
            if damage_residue.dim() > 1:
                damage_residue = damage_residue.mean(dim=0)
            
            return FailureToken(FailureTokenType.BOT, constraint_index=constraint_idx, residue=damage_residue)
        
        return None
