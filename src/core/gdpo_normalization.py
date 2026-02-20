"""
GDPO (Group Pressure-Decoupled Normalization Structural Adaptation) utilities.

Based on arXiv:2601.05242 - implements decoupled per-dimension normalization
to prevent collapse of distinct multi-Pressure (multi-residue) patterns.

Author: William Matthew Bryant
Reference: https://arxiv.org/abs/2601.05242
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np


class SignalSovereignty(nn.Module):
    """
    Signal Sovereignty & Fossilization: Decoupled normalization that preserves structural integrity.
    
    Each functional group maintains its own normalization parameters to prevent 
    signal collapse under Pressure. Implements "Functional Fossilization": 
    successful groups become immutable to prevent structural decay.
    """
    
    def __init__(
        self,
        num_dimensions: int,
        epsilon: float = 1e-8,
        use_batch_norm: bool = True
    ):
        """
        Args:
            num_dimensions: Number of dimensions (K functionals)
            epsilon: Numerical stability constant
            use_batch_norm: Apply final batch normalization after aggregation
        """
        super().__init__()
        
        self.num_dimensions = num_dimensions
        self.epsilon = epsilon
        self.use_batch_norm = use_batch_norm
        
        # Track running statistics for each dimension
        self.register_buffer('running_mean', torch.zeros(num_dimensions))
        self.register_buffer('running_var', torch.ones(num_dimensions))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Signal Sovereignty / Fossilization
        self.register_buffer('is_fossilized', torch.zeros(num_dimensions, dtype=torch.bool))
        self.register_buffer('performance_streak', torch.zeros(num_dimensions, dtype=torch.long))
        self.fossil_threshold = 50 # T generations/batches
        
        self.momentum = 0.1  # For running stats
    
    def group_normalize(
        self,
        values: torch.Tensor,
        group_ids: torch.Tensor,
        dim_idx: int
    ) -> torch.Tensor:
        """
        Normalize values within each group for a specific dimension.
        """
        normalized = torch.zeros_like(values)
        unique_groups = torch.unique(group_ids)
        
        for group in unique_groups:
            mask = (group_ids == group)
            group_values = values[mask]
            
            if len(group_values) > 1:
                mean = group_values.mean()
                std = group_values.std(unbiased=False) + self.epsilon
                normalized[mask] = (group_values - mean) / std
            else:
                # Single sample in group - use running stats
                normalized[mask] = (group_values - self.running_mean[dim_idx]) / \
                                  (torch.sqrt(self.running_var[dim_idx]) + self.epsilon)
        
        return normalized
    
    def forward(
        self,
        multi_dim_pressures: torch.Tensor,
        weights: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply Signal Sovereignty decoupled normalization.
        
        Args:
            multi_dim_pressures: [batch, num_dimensions] multiple pressures/residues
            weights: [num_dimensions] aggregation weights w_k
            group_ids: Optional [batch] group assignments
            
        Returns:
            decoupled: [batch, num_dimensions] Individually normalized pressures
            diagnostics: Dictionary with intermediate values
        """
        batch_size, num_dims = multi_dim_pressures.shape
        assert num_dims == self.num_dimensions
        
        # Default: single group (all samples together)
        if group_ids is None:
            group_ids = torch.zeros(batch_size, dtype=torch.long, device=multi_dim_pressures.device)
        
        # Step 1: Per-dimension group-wise normalization
        decoupled = torch.zeros_like(multi_dim_pressures)
        
        for k in range(num_dims):
            decoupled[:, k] = self.group_normalize(
                multi_dim_pressures[:, k],
                group_ids,
                dim_idx=k
            )
            
            # Update running statistics (ONLY if not fossilized)
            if self.training and not self.is_fossilized[k]:
                with torch.no_grad():
                    mean_k = multi_dim_pressures[:, k].mean()
                    var_k = multi_dim_pressures[:, k].var(unbiased=False)
                    
                    self.running_mean[k] = (1 - self.momentum) * self.running_mean[k] + \
                                           self.momentum * mean_k
                    self.running_var[k] = (1 - self.momentum) * self.running_var[k] + \
                                          self.momentum * var_k
        
        # Check for Fossilization triggers
        if self.training:
            self._update_fossilization_state(multi_dim_pressures)
        
        # Step 2: Weighted aggregation (DEPRECATED - ENFORCING NON-SCALARIZATION)
        # Note: We return decoupled pressures to maintain domain sovereignty.
        # r̂ = Σ w_k · r̃_k (Removed to close the scalarization trap)
        
        # Step 3: Optional batch-level normalization for scale stability
        # if self.use_batch_norm and batch_size > 1:
        #    batch_mean = aggregated.mean()
        #    batch_std = aggregated.std(unbiased=False) + self.epsilon
        #    aggregated = (aggregated - batch_mean) / batch_std
        
        # Diagnostics
        diagnostics = {
            'decoupled': decoupled,  # [batch, num_dims] after per-dim normalization
            'weights_used': weights,
            'capacity_mask': ~self.is_fossilized  # Capacity Removal signal
        }
        
        if self.training:
            self.num_batches_tracked += 1
        
        return decoupled, diagnostics

    def _update_fossilization_state(self, values: torch.Tensor):
        """
        Update fossilization based on stability of signaling.
        """
        with torch.no_grad():
            for k in range(self.num_dimensions):
                if self.is_fossilized[k]:
                    continue
                
                # Metric: Stability of z-score variance
                # If variance of current batch matches running variance tightly, signal is stabilizing
                current_var = values[:, k].var(unbiased=False)
                var_diff = torch.abs(current_var - self.running_var[k]) / (self.running_var[k] + self.epsilon)
                
                if var_diff < 0.05: # High stability
                    self.performance_streak[k] += 1
                else:
                    self.performance_streak[k] = 0
                
                if self.performance_streak[k] >= self.fossil_threshold:
                    self.is_fossilized[k] = True
                    print(f"Signal Sovereignty: functional group {k} has fossilized.")
    
    def compute_separation_pressure(
        self,
        multi_dim_pressures: torch.Tensor,
        use_sovereignty: bool = True
    ) -> float:
        """
        Measure how well distinct pressure patterns remain separated.
        
        Args:
            multi_dim_pressures: [batch, num_dimensions]
            use_sovereignty: If True, use SignalSovereignty normalization
        """
        if use_sovereignty:
            # Use decoupled representation
            group_ids = torch.zeros(multi_dim_pressures.shape[0], dtype=torch.long, 
                                   device=multi_dim_pressures.device)
            weights = torch.ones(self.num_dimensions, device=multi_dim_pressures.device) / self.num_dimensions
            _, diagnostics = self.forward(multi_dim_pressures, weights, group_ids)
            representation = diagnostics['decoupled']
        else:
            # Standard: just sum dimensions
            representation = multi_dim_pressures.sum(dim=1, keepdim=True)
        
        # Compute pairwise distances
        dists = torch.cdist(representation, representation, p=2)
        
        # Mean off-diagonal distance (separation)
        mask = ~torch.eye(len(dists), dtype=torch.bool, device=dists.device)
        separation = dists[mask].mean().item()
        
        return separation


class LearnableWeights(nn.Module):
    """
    Learnable per-dimension weights for SignalSovereignty aggregation.
    
    w_k(θ) determines importance of each functional pressure.
    """
    
    def __init__(
        self,
        num_dimensions: int,
        init_mode: str = 'uniform',
        constraint: str = 'softmax'
    ):
        """
        Args:
            num_dimensions: Number of dimensions (e.g., K functionals)
            init_mode: 'uniform', 'random', or 'inverse_sqrt'
            constraint: 'softmax' (sum to 1), 'positive' (> 0), or 'none'
        """
        super().__init__()
        
        self.num_dimensions = num_dimensions
        self.constraint = constraint
        
        # Initialize raw parameters
        if init_mode == 'uniform':
            raw_weights = torch.ones(num_dimensions)
        elif init_mode == 'random':
            raw_weights = torch.randn(num_dimensions) * 0.1 + 1.0
        elif init_mode == 'inverse_sqrt':
            raw_weights = torch.tensor([1.0 / np.sqrt(k+1) for k in range(num_dimensions)])
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")
        
        self.raw_weights = nn.Parameter(raw_weights)
    
    def forward(self) -> torch.Tensor:
        """
        Get constrained weights.
        
        Returns:
            weights: [num_dimensions] positive or normalized weights
        """
        if self.constraint == 'softmax':
            return torch.softmax(self.raw_weights, dim=0)
        elif self.constraint == 'positive':
            return torch.exp(self.raw_weights)
        elif self.constraint == 'none':
            return self.raw_weights
        else:
            raise ValueError(f"Unknown constraint: {self.constraint}")
    
    def get_weights_dict(self, keys_list: List[int]) -> Dict[int, float]:
        """
        Get weights as dictionary mapping key -> weight.
        
        Args:
            keys_list: List of keys (e.g., functional indices)
            
        Returns:
            Dictionary {key: weight}
        """
        weights = self.forward()
        return {k: w.item() for k, w in zip(keys_list, weights)}


def compare_sovereignty_vs_standard(
    multi_dim_pressures: torch.Tensor,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compare SignalSovereignty vs standard normalization on separation pressure.
    """
    num_dims = multi_dim_pressures.shape[1]
    sovereignty = SignalSovereignty(num_dims)
    
    # Sovereignty separation
    sov_sep = sovereignty.compute_separation_pressure(multi_dim_pressures, use_sovereignty=True)
    
    # Standard separation (sum-based)
    standard_sep = sovereignty.compute_separation_pressure(multi_dim_pressures, use_sovereignty=False)
    
    improvement = (sov_sep / standard_sep - 1.0) * 100 if standard_sep > 0 else 0.0
    
    if verbose:
        print(f"Separation Pressure Comparison:")
        print(f"  Standard (sum-based): {standard_sep:.4f}")
        print(f"  Sovereignty:          {sov_sep:.4f}")
        print(f"  Improvement:          {improvement:+.2f}%")
    
    return {
        'standard': standard_sep,
        'gdpo': gdpo_sep,
        'improvement_pct': improvement
    }
