"""
Continuous Co-Primality: Non-Algebraic Independence

Implements entropy-based independence checking for residues.
Uses discrete entropy quantization (not continuous approximations).

Mathematical Foundation:
    E(r_i, r_j) = H(r_i + r_j) - H(r_i) - H(r_j)
    
    Continuous co-primality: E(r_i, r_j) >= 0
    and lim_{t->∞} Cov(r_i^(t), r_j^(t)) = 0
    
    No GCD. Only asymptotic independence under saturation.

Phase 3: Advanced Constraints Implementation

Note: Uses discrete entropy quantization (binary outcomes, bincount, log2)
to match HypergraphOrthogonalityPressure methodology. No continuous approximations.

Author: Implementation Documentation
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import numpy as np


class DiscreteEntropyComputer(nn.Module):
    """
    Computes discrete entropy with proper quantization.
    
    Uses the same quantization method as HypergraphOrthogonalityPressure:
    - Binary outcomes (values > 0)
    - Integer keys via powers of 2
    - Bincount for discrete probabilities
    - log2 for entropy
    """
    
    def __init__(self, num_bins: Optional[int] = None, use_binary: bool = True):
        """
        Args:
            num_bins: Optional number of bins for quantization (auto if None)
            use_binary: If True, use binary quantization (phi > 0), else use bins
        """
        super().__init__()
        self.num_bins = num_bins
        self.use_binary = use_binary
    
    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Quantize values to discrete outcomes.
        
        Args:
            values: [batch, ...] continuous values
            
        Returns:
            quantized: [batch, ...] discrete outcomes
        """
        if self.use_binary:
            # Binary quantization: > 0 -> 1, <= 0 -> 0
            return (values > 0).long()
        else:
            # Multi-bin quantization
            if self.num_bins is None:
                # Auto-determine bins from data
                min_val = values.min().item()
                max_val = values.max().item()
                bins = torch.linspace(min_val, max_val, 10, device=values.device)
            else:
                bins = torch.linspace(values.min(), values.max(), self.num_bins, device=values.device)
            
            # Quantize to bin indices
            quantized = torch.bucketize(values, bins)
            return quantized
    
    def compute_entropy(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute discrete entropy H(values) using proper quantization.
        
        Args:
            values: [batch, ...] continuous values
            
        Returns:
            entropy: Scalar entropy value
        """
        # Quantize to discrete outcomes
        quantized = self.quantize(values)
        
        # Flatten for bincount
        flat_quantized = quantized.flatten()
        
        # Compute discrete probabilities using bincount
        max_val = flat_quantized.max().item()
        min_val = flat_quantized.min().item()
        num_outcomes = max_val - min_val + 1
        
        counts = torch.bincount(flat_quantized - min_val, minlength=num_outcomes).float()
        probs = counts / (flat_quantized.numel() + 1e-8)
        
        # Remove zero probabilities
        probs = probs[probs > 0]
        
        # Compute entropy: H = -Σ p * log2(p)
        entropy = -(probs * torch.log2(probs + 1e-10)).sum()
        
        return entropy
    
    def compute_entropy_batch(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy for each batch element.
        
        Args:
            values: [batch, ...] continuous values
            
        Returns:
            entropies: [batch] entropy values
        """
        batch_size = values.shape[0]
        entropies = []
        
        for b in range(batch_size):
            entropy = self.compute_entropy(values[b])
            entropies.append(entropy)
        
        return torch.tensor(entropies, device=values.device, dtype=values.dtype)


class ContinuousCoprimality(nn.Module):
    """
    Continuous Co-Primality: Entropy-based independence.
    
    E(r_i, r_j) = H(r_i + r_j) - H(r_i) - H(r_j)
    
    Uses discrete entropy quantization (not continuous approximations).
    """
    
    def __init__(
        self,
        use_binary_quantization: bool = True,
        num_bins: Optional[int] = None
    ):
        """
        Args:
            use_binary_quantization: Use binary (>0) quantization (default: True)
            num_bins: Optional number of bins for multi-level quantization
        """
        super().__init__()
        self.entropy_computer = DiscreteEntropyComputer(
            num_bins=num_bins,
            use_binary=use_binary_quantization
        )
    
    def compute_entropy_pressure(
        self,
        residue_i: torch.Tensor,
        residue_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy pressure: E(r_i, r_j) = H(r_i + r_j) - H(r_i) - H(r_j)
        
        Args:
            residue_i: [batch, ...] first residue
            residue_j: [batch, ...] second residue
            
        Returns:
            entropy_pressure: [batch] E values
        """
        # Ensure same shape
        if residue_i.shape != residue_j.shape:
            # Try to align shapes
            if residue_i.numel() == residue_j.numel():
                residue_i = residue_i.reshape(residue_j.shape)
            else:
                # Use minimum common shape
                min_shape = tuple(min(s_i, s_j) for s_i, s_j in zip(residue_i.shape, residue_j.shape))
                residue_i = residue_i[tuple(slice(0, s) for s in min_shape)]
                residue_j = residue_j[tuple(slice(0, s) for s in min_shape)]
        
        # Compute individual entropies
        H_i = self.entropy_computer.compute_entropy_batch(residue_i)
        H_j = self.entropy_computer.compute_entropy_batch(residue_j)
        
        # Compute joint entropy
        residue_sum = residue_i + residue_j
        H_ij = self.entropy_computer.compute_entropy_batch(residue_sum)
        
        # Entropy pressure: E = H_ij - H_i - H_j
        E = H_ij - H_i - H_j
        
        return E
    
    def check_co_primality(
        self,
        residue_i: torch.Tensor,
        residue_j: torch.Tensor,
        threshold: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check continuous co-primality condition.
        
        Co-prime if: E(r_i, r_j) >= 0 (entropy pressure is non-negative)
        
        Args:
            residue_i: [batch, ...] first residue
            residue_j: [batch, ...] second residue
            threshold: Minimum entropy pressure threshold (default: 0.0)
            
        Returns:
            is_co_prime: [batch] boolean tensor
            entropy_pressure: [batch] E values
        """
        E = self.compute_entropy_pressure(residue_i, residue_j)
        is_co_prime = E >= threshold
        
        return is_co_prime, E
    
    def compute_covariance(
        self,
        residue_i: torch.Tensor,
        residue_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute covariance between residues.
        
        Args:
            residue_i: [batch, ...] first residue
            residue_j: [batch, ...] second residue
            
        Returns:
            covariance: [batch] covariance values
        """
        # Flatten for covariance computation
        if residue_i.dim() > 1:
            residue_i_flat = residue_i.flatten(start_dim=1)  # [batch, features]
            residue_j_flat = residue_j.flatten(start_dim=1)  # [batch, features]
        else:
            residue_i_flat = residue_i.unsqueeze(-1)  # [batch, 1]
            residue_j_flat = residue_j.unsqueeze(-1)  # [batch, 1]
        
        # Ensure same feature dimension
        min_features = min(residue_i_flat.shape[-1], residue_j_flat.shape[-1])
        residue_i_flat = residue_i_flat[..., :min_features]
        residue_j_flat = residue_j_flat[..., :min_features]
        
        # Compute covariance: Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
        mean_i = residue_i_flat.mean(dim=-1, keepdim=True)  # [batch, 1]
        mean_j = residue_j_flat.mean(dim=-1, keepdim=True)  # [batch, 1]
        
        centered_i = residue_i_flat - mean_i
        centered_j = residue_j_flat - mean_j
        
        # Covariance: mean of element-wise product
        covariance = (centered_i * centered_j).mean(dim=-1)  # [batch]
        
        return covariance
    
    def check_asymptotic_independence(
        self,
        residue_i: torch.Tensor,
        residue_j: torch.Tensor,
        time_steps: int = 100,
        evolution_fn: Optional[callable] = None,
        tolerance: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check asymptotic independence: lim_{t->∞} Cov(r_i^(t), r_j^(t)) = 0
        
        Args:
            residue_i: [batch, ...] first residue
            residue_j: [batch, ...] second residue
            time_steps: Number of evolution steps to check
            evolution_fn: Optional function to evolve residues: (r, t) -> r_t
            tolerance: Tolerance for zero covariance
            
        Returns:
            is_independent: [batch] boolean tensor
            final_covariance: [batch] final covariance values
        """
        if evolution_fn is None:
            # Default: no evolution (static)
            final_cov = self.compute_covariance(residue_i, residue_j)
            is_independent = torch.abs(final_cov) < tolerance
            return is_independent, final_cov
        
        # Evolve residues and track covariance
        r_i_t = residue_i
        r_j_t = residue_j
        
        covariances = []
        for t in range(time_steps):
            # Evolve
            r_i_t = evolution_fn(r_i_t, t)
            r_j_t = evolution_fn(r_j_t, t)
            
            # Compute covariance
            cov = self.compute_covariance(r_i_t, r_j_t)
            covariances.append(cov)
        
        # Final covariance
        final_cov = covariances[-1]
        
        # Check if covariance decays to zero
        # Use trend: check if final covariance is small and decreasing
        if len(covariances) > 1:
            # Check if trend is decreasing
            cov_tensor = torch.stack(covariances)  # [time_steps, batch]
            trend = cov_tensor[-1] - cov_tensor[-min(10, len(covariances))]  # Recent change
            is_decreasing = trend < 0
            is_small = torch.abs(final_cov) < tolerance
            is_independent = is_small & is_decreasing
        else:
            is_independent = torch.abs(final_cov) < tolerance
        
        return is_independent, final_cov
    
    def forward(
        self,
        residue_i: torch.Tensor,
        residue_j: torch.Tensor,
        check_asymptotic: bool = False,
        time_steps: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: check continuous co-primality.
        
        Returns:
            Dictionary with:
            - 'is_co_prime': [batch] boolean
            - 'entropy_pressure': [batch] E values
            - 'covariance': [batch] covariance (if check_asymptotic)
            - 'is_independent': [batch] boolean (if check_asymptotic)
        """
        is_co_prime, E = self.check_co_primality(residue_i, residue_j)
        
        result = {
            'is_co_prime': is_co_prime,
            'entropy_pressure': E
        }
        
        if check_asymptotic:
            is_independent, final_cov = self.check_asymptotic_independence(
                residue_i, residue_j, time_steps
            )
            result['is_independent'] = is_independent
            result['covariance'] = final_cov
        
        return result
