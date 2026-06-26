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
from src.core.honest_jitter import harvest_honest_jitter


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
        
        Args:
            values: [batch] values for this dimension
            group_ids: [batch] group assignment (e.g., constraint type)
            dim_idx: Which dimension this is (for statistics tracking)
            
        Returns:
            normalized: [batch] group-wise z-scores
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
        
        # Step 1: Vectorized group-wise normalization across all dimensions
        decoupled = torch.zeros_like(multi_dim_pressures)
        unique_groups = torch.unique(group_ids)
        
        for group in unique_groups:
            mask = (group_ids == group)
            group_values = multi_dim_pressures[mask]
            num_samples = group_values.shape[0]
            
            if num_samples > 1:
                mean = group_values.mean(dim=0)
                std = group_values.std(dim=0, unbiased=False) + self.epsilon
                decoupled[mask] = (group_values - mean) / std
            else:
                # Single sample in group - use running stats
                decoupled[mask] = (group_values - self.running_mean) / \
                                  (torch.sqrt(self.running_var) + self.epsilon)
            
        # Update running statistics (ONLY if not fossilized)
        if self.training:
            with torch.no_grad():
                mean_all = multi_dim_pressures.mean(dim=0)
                var_all = multi_dim_pressures.var(dim=0, unbiased=False)
                
                not_fossilized = ~self.is_fossilized
                self.running_mean = torch.where(
                    not_fossilized,
                    (1.0 - self.momentum) * self.running_mean + self.momentum * mean_all,
                    self.running_mean
                )
                self.running_var = torch.where(
                    not_fossilized,
                    (1.0 - self.momentum) * self.running_var + self.momentum * var_all,
                    self.running_var
                )
        
        # Check for Fossilization triggers
        if self.training:
            self._update_fossilization_state(multi_dim_pressures)
        
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
            # Metric: Stability of z-score variance across all dimensions
            current_var = values.var(dim=0, unbiased=False)
            var_diff = torch.abs(current_var - self.running_var) / (self.running_var + self.epsilon)
            
            # Update performance streak
            stable_mask = var_diff < 0.05
            self.performance_streak = torch.where(
                stable_mask,
                self.performance_streak + 1,
                torch.zeros_like(self.performance_streak)
            )
            
            # [ARCHITECTURAL REMEDIATION] Replace naive integer threshold with Mohr-Coulomb 
            # structural yield criteria.
            from src.core.yield_criteria import MohrCoulombProjection
            if not hasattr(self, '_mc_yield'):
                self._mc_yield = MohrCoulombProjection(friction_angle=30.0, cohesion=float(self.fossil_threshold))
            
            # Project the performance streak (pressure) against the cohesion barrier in a vectorized manner
            # pressure shape: [num_dimensions, 1]
            pressure = self.performance_streak.unsqueeze(-1).float()
            load = torch.zeros_like(pressure)
            yielded_pressure = self._mc_yield(pressure, load).squeeze(-1) # [num_dimensions]
            
            # Rupture (Fossilization) only occurs if the MC boundary yields
            newly_fossilized = (~self.is_fossilized) & (yielded_pressure > self.fossil_threshold)
            if newly_fossilized.any():
                for k in torch.where(newly_fossilized)[0].tolist():
                    self.is_fossilized[k] = True
                    print(f"Signal Sovereignty: functional group {k} has fossilized via Mohr-Coulomb yield.")
    
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
    
    w_k() determines importance of each functional pressure.
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
            # SILICON SOVEREIGNTY: Replaced PRNG noise with honest jitter
            raw_weights = harvest_honest_jitter((num_dimensions,), scaled=True) * 0.2 + 1.0
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
        'gdpo': sov_sep,
        'improvement_pct': improvement
    }

class GDPONormalization(nn.Module):
    """
    Drop-in replacement for nn.LayerNorm that utilizes Signal Sovereignty.
    Ensures topological shape preservation by preventing collapse of distinct patterns.
    """
    def __init__(self, num_dimensions: int, epsilon: float = 1e-8):
        super().__init__()
        self.sovereignty = SignalSovereignty(num_dimensions, epsilon=epsilon, use_batch_norm=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is typically [batch, ..., hidden_dim]
        original_shape = x.shape
        # Flatten everything except the last dimension
        x_flat = x.view(-1, original_shape[-1])
        # We need weights, uniformly initialized
        weights = torch.ones(original_shape[-1], device=x.device) / original_shape[-1]
        decoupled, _ = self.sovereignty(x_flat, weights)
        return decoupled.view(original_shape)

