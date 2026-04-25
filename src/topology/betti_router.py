"""
Betti-Aware Routing: Topological Flow Guidance

Directs information flow toward manifold sectors with high cycle density (Betti-1).
Uses the Cyclotomic TDA Compressor to estimate local Betti numbers.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from src.topology.modular_homology_fft import CyclotomicTDACompressor
from src.core.fgrt_primitives import PrimeResonanceLadder
from src.core.honest_jitter import harvest_honest_jitter

class BettiRouter(nn.Module):
    """
    Directs routing based on localized Betti density.
    """
    def __init__(self, feature_dim: int, num_sectors: int, p: Optional[int] = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_sectors = num_sectors
        
        # Initialize Cyclotomic TDA
        if p is None:
            # Derive p from hardware jitter for Silicon Sovereignty
            # Note: In a real run, this happens at init.
            # Here we provide a fallback for the constructor.
            p = 61 # Default prime
            
        self.tda_compressor = CyclotomicTDACompressor(p=p, ring_size=64)
        
        # Bias projection: Betti density -> sector bias
        self.bias_proj = nn.Linear(1, num_sectors)
        nn.init.orthogonal_(self.bias_proj.weight)
        
    def estimate_sector_betti(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate Betti-1 density for the current state.
        
        Args:
            x: [batch, feature_dim] state tensor
            
        Returns:
            betti_density: [batch, 1]
        """
        # Reshape for ring-FFT: [batch, 1, ring_size]
        ring_size = self.tda_compressor.ring_size
        if x.shape[-1] >= ring_size:
            x_ring = x[..., :ring_size].unsqueeze(1)
        else:
            x_ring = torch.nn.functional.pad(x, (0, ring_size - x.shape[-1])).unsqueeze(1)
            
        # Compute modular persistence
        lifetimes = self.tda_compressor.modular_persistence_approx(x_ring) # [batch, num_cycles]
        
        # Betti-1 is the count of non-trivial cycles (lifetimes > threshold)
        # We use a soft-threshold (sigmoid) for differentiable routing if needed,
        # but the user requested "prioritize," so we can use a hard count or sum of lifetimes.
        betti_1 = (lifetimes > 2.0).sum(dim=-1, keepdim=True).float()
        
        return betti_1

    def compute_routing_bias(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Betti-aware bias for the ZeitgeistRouter.
        
        Returns:
            bias: [batch, num_sectors]
        """
        betti_density = self.estimate_sector_betti(x)
        
        # Project density to sector bias
        # High Betti density -> higher bias for sectors that represent complex manifolds
        bias = self.bias_proj(betti_density)
        
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_routing_bias(x)
