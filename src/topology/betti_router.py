"""
Betti-Aware Routing: Topological Flow Guidance

Directs information flow toward manifold sectors with high cycle density (Betti-1).
Uses the Cyclotomic TDA Compressor to estimate local Betti numbers.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from src.topology.modular_homology_fft import CyclotomicTDACompressor
from src.core.invariants import PrimeResonanceLadder, PHI
from src.core.honest_jitter import harvest_honest_jitter
from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine, PYOPENCL_AVAILABLE
import math

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
        
        # Hardware-accelerated Bouligand Contingent Cone checker
        self.engine = SiliconSovereigntyEngine() if PYOPENCL_AVAILABLE else None
        
    def estimate_sector_betti(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate Betti-1 densities across multiple persistence scales.
        
        Args:
            x: [batch, feature_dim] state tensor
            
        Returns:
            betti_densities: [batch, 8]  Vector of cycle counts at different thresholds
        """
        ring_size = self.tda_compressor.ring_size
        if x.shape[-1] >= ring_size:
            x_ring = x[..., :ring_size].unsqueeze(1)
        else:
            x_ring = torch.nn.functional.pad(x, (0, ring_size - x.shape[-1])).unsqueeze(1)
            
        lifetimes = self.tda_compressor.modular_persistence_approx(x_ring) # [batch, num_cycles]
        
        # Compute Betti-1 at multiple thresholds to capture the 'filtration'
        thresholds = torch.linspace(0.5, 5.0, 8, device=x.device)
        # Result: [batch, 8]
        betti_densities = torch.stack([
            (lifetimes > t).sum(dim=-1).float() for t in thresholds
        ], dim=-1)
        
        return betti_densities

    def compute_routing_bias(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Betti-aware bias for the ZeitgeistRouter.
        """
        betti_densities = self.estimate_sector_betti(x) # [batch, 8]
        
        # Project the topological signature to sector bias
        if not hasattr(self, 'bias_proj_v2'):
             self.bias_proj_v2 = nn.Linear(8, self.num_sectors, device=x.device)
             nn.init.orthogonal_(self.bias_proj_v2.weight)
             
        bias = self.bias_proj_v2(betti_densities)
        
        # If OpenCL Bouligand kernel is available, strictly veto invalid paths (traffic jam physics)
        if self.engine is not None:
            # Check viability using fixed irrationals
            omega_i = math.sqrt(2)
            omega_j = PHI  # Golden ratio from canonical invariants
            t_pseudo = torch.norm(x).item()
            
            # Use betti_densities as the incoming flux vector against state x
            state_i = x.detach().cpu().flatten()
            state_j = betti_densities.detach().cpu().flatten()
            
            # Align dimension sizes
            if state_j.shape[0] > state_i.shape[0]:
                state_j = state_j[:state_i.shape[0]]
            elif state_j.shape[0] < state_i.shape[0]:
                state_j = torch.nn.functional.pad(state_j, (0, state_i.shape[0] - state_j.shape[0]))
                
            opencl_mask = self.engine.evaluate_bouligand_intersection(
                state_i, state_j, omega_i, omega_j, t_pseudo
            )
            
            # Reshape back to bias shape (assume matching num_sectors)
            # Take the first num_sectors elements of the mask
            mask_subset = opencl_mask[:self.num_sectors]
            mask_tensor = torch.from_numpy(mask_subset).to(bias.device)
            
            # Hard-veto rejected sectors by setting them to negative infinity
            bias = torch.where(mask_tensor, bias, torch.full_like(bias, -float('inf')))
            
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_routing_bias(x)
