"""
Dyadic Proficiency Transfer (Dark Matter).

Manages the non-commutative leakage of proficiency between task dyads.
This is the "Dark Matter" that allows skill transfer without merging identities.

Principles:
1. Proficiency Gating: Weak skills do not leak.
2. Non-Commutativity: Transfer A -> B != Transfer B -> A.
3. Dyadic Isolation: Only specific pairs are allowed to resonate.

Author: Implementation Documentation
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class DyadicTransferMap(nn.Module):
    """
    Manages the learned non-commutative transfer map between tasks.
    
    T_{ij}: Task i -> Task j transfer coefficient.
    """
    def __init__(self, num_tasks: int, embedding_dim: int, leakage_threshold: float = 0.7):
        super().__init__()
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.leakage_threshold = leakage_threshold
        
        # Learnable transfer matrix (Directed Graph)
        # We model this as an attention mechanism or direct matrix
        # Let's use a bilinear map for richer interaction: T_ij = u_i^T W u_j
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, embedding_dim))
        self.transfer_bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1)
        
        # Commutator oracle is implicitly this learned structure
        
    def get_transfer_coefficients(self) -> torch.Tensor:
        """
        Compute the full NxN transfer matrix.
        Returns T where T[i,j] is transfer from i TO j.
        """
        # Efficient all-to-all bilinear
        # Expand for broadcasting
        # u_i: [N, 1, D]
        # u_j: [1, N, D]
        u = self.task_embeddings
        u_i = u.unsqueeze(1).expand(-1, self.num_tasks, -1)
        u_j = u.unsqueeze(0).expand(self.num_tasks, -1, -1)
        
        # flatten for batch processing
        u_i_flat = u_i.reshape(-1, self.embedding_dim)
        u_j_flat = u_j.reshape(-1, self.embedding_dim)
        
        transfer_scores = self.transfer_bilinear(u_i_flat, u_j_flat)
        transfer_matrix = transfer_scores.reshape(self.num_tasks, self.num_tasks)
        
        return torch.sigmoid(transfer_matrix)

    def forward(self, task_states: torch.Tensor, proficiency_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply dyadic leakage.
        
        Args:
            task_states: [batch, num_tasks, state_dim]
            proficiency_scores: [batch, num_tasks] (0.0 to 1.0)
            
        Returns:
            leaked_states: [batch, num_tasks, state_dim]
            
        Logic:
           New_State_j = State_j + sum_{i!=j} (Gating_i * T_{ij} * State_i)
           Gating_i = ReLU(Proficiency_i - Threshold)
        """
        batch_size = task_states.shape[0]
        
        # 1. Compute Transfer Matrix T
        T = self.get_transfer_coefficients() # [num_tasks, num_tasks]
        
        # 2. Compute Gating
        # Only high proficiency tasks leak OUT
        gating = torch.relu(proficiency_scores - self.leakage_threshold) # [batch, num_tasks]
        
        # 3. Apply Leakage
        # We want to aggregate inputs TO j FROM all i
        # Contribution_{i->j} = Gating_i * T_{ij} * State_i
        
        # Reshaping for broadcast
        # State_i: [batch, N, 1, D] (Sources)
        # T_{ij}:  [1, N, N, 1]     (Weights)
        # Gating_i: [batch, N, 1, 1]
        
        # But matrix multiplication is easier:
        # Leaked = (Gating * T) @ States ??
        # Let's align dimensions carefully.
        
        # We need for each j: sum_i ( T_{ij} * (Gating_i * State_i) )
        # Let Effective_Source_i = Gating_i * State_i  [batch, num_tasks, dim]
        
        effective_source = task_states * gating.unsqueeze(-1)
        
        # T is [Source, Dest] -> [i, j]
        # We want Dest_j = sum_i Source_i * T_{ij}
        # This is exactly matrix multiplication: Source^T @ T ??
        # Source is [batch, N, D]. T is [N, N].
        # We want output [batch, N, D].
        
        # Einstein summation is safest
        # b: batch, i: source task, j: dest task, d: dimension
        leaked_signal = torch.einsum('bid,ij->bjd', effective_source, T)
        
        # Prevent self-leakage accumulation (optional, but T_ii usually matters)
        # If we want strictly "leakage", we might zero diagonal of T.
        # Let's zero diagonal for "pure leakage".
        # Mask diagonal
        mask = 1.0 - torch.eye(self.num_tasks, device=task_states.device)
        T_masked = T * mask
        
        leaked_signal_pure = torch.einsum('bid,ij->bjd', effective_source, T_masked)
        
        # Add to original state (residual connection)
        return task_states + leaked_signal_pure

class CommutatorOracle(nn.Module):
    """
    Learns and queries the non-commutativity of task pairs.
    Used to optimize the transfer map.
    """
    def __init__(self, transfer_map: DyadicTransferMap):
        super().__init__()
        self.transfer_map = transfer_map
        
    def measure_commutativity_violation(self) -> torch.Tensor:
        """
        Returns matrix V where V_ij = |T_ij - T_ji|.
        High value = Highly non-commutative relationship.
        """
        T = self.transfer_map.get_transfer_coefficients()
        return torch.abs(T - T.t())
        
    def get_leakage_report(self) -> Dict[str, torch.Tensor]:
        T = self.transfer_map.get_transfer_coefficients().detach()
        commutativity = self.measure_commutativity_violation().detach()
        
        return {
            "transfer_matrix": T,
            "commutativity_violation": commutativity,
            "max_non_commutativity": commutativity.max()
        }
