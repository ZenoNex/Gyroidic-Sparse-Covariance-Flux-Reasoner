"""
Martinova Local Correlation Metric.

Provides a bounded local correlation function that outputs strictly between -1
(maximal dispersion), 0 (spatial randomness/ergodic soup), and 1 (maximal clustering/crystallization).
Used to upgrade heuristic thresholds across the architecture into strict, bounded geometric invariants.
"""

import torch
import torch.nn as nn
from typing import Optional, Union

def compute_bounded_correlation(X: torch.Tensor, r: Optional[float] = None) -> torch.Tensor:
    """
    Computes Martinova's local correlation metric for X.
    
    Args:
        X: Input tensor of shape [..., N, d] representing a spatial point pattern
           of N points in d-dimensional space.
        r: Scale parameter for the local density kernel. If None, dynamically
           uses the mean of pairwise distances.
           
    Returns:
        A tensor of shape [...] containing correlation scores strictly bounded
        in [-1, 1], where:
           -1.0 = Maximal dispersion
            0.0 = Spatial randomness / Ergodic soup
            1.0 = Maximal clustering / Crystallization
    """
    # Keep track of the input shape to restore leading dimensions
    orig_shape = X.shape
    
    if X.dim() == 0:
        X = X.view(1, 1, 1)
    elif X.dim() == 1:
        X = X.view(1, -1, 1)
    elif X.dim() == 2:
        X = X.unsqueeze(0)
        
    B, N, d = X.shape
    device = X.device
    
    if N <= 1:
        # A single point has neutral correlation
        C = torch.zeros(B, device=device)
        if len(orig_shape) < 2:
            return C.squeeze(0)
        return C.view(orig_shape[:-2])
        
    # Pairwise distance matrix [B, N, N]
    # Use squared expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    x_norms = torch.sum(X**2, dim=-1, keepdim=True) # [B, N, 1]
    dist2 = x_norms + x_norms.transpose(-2, -1) - 2 * torch.matmul(X, X.transpose(-2, -1))
    dist = torch.sqrt(torch.clamp(dist2, min=1e-12)) # [B, N, N]
    
    # Determine the local neighborhood radius r dynamically if not provided
    if r is None:
        r_val = dist.mean(dim=[-2, -1], keepdim=True) # [B, 1, 1]
        # Prevent division by zero
        r_val = torch.clamp(r_val, min=1e-6)
    else:
        r_val = torch.tensor(r, device=device).view(1, 1, 1)
        
    # Smooth kernel representation of count within distance r (differentiable Ripley's K)
    # Using Gaussian kernel: K_ij = exp(-d_ij^2 / (2 * r^2))
    kernel_vals = torch.exp(-dist**2 / (2 * r_val**2)) # [B, N, N]
    
    # Zero out self-distances (diagonals)
    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    kernel_vals = kernel_vals * (1.0 - eye)
    
    # Average local density per point
    # Sum over columns (other points) divided by N - 1
    local_density = kernel_vals.sum(dim=-1) / (N - 1) # [B, N]
    
    # Mean local density across all points in the batch
    mu_rho = local_density.mean(dim=-1) # [B]
    
    # Map the average local density from [0, 1] to [-1, 1]
    # mu_rho -> 0 corresponds to maximal dispersion (-1)
    # mu_rho -> 1 corresponds to maximal clustering (1)
    # A random distribution yields intermediate density (normalized here to 0)
    # Under typical Gaussian kernel spacing, the expected value under complete spatial randomness (CSR)
    # is around exp(-0.5) ~ 0.606 if r is the mean distance.
    # To center CSR at 0:
    csr_baseline = 0.5
    C = (mu_rho - csr_baseline) / (1.0 - csr_baseline)
    C = torch.where(mu_rho < csr_baseline, (mu_rho - csr_baseline) / csr_baseline, C)
    
    C = torch.clamp(C, min=-1.0, max=1.0)
    
    # Restore leading batch dimensions
    if len(orig_shape) == 2:
        return C.squeeze(0)
    elif len(orig_shape) < 2:
        return C.squeeze(0)
    else:
        return C.view(orig_shape[:-2])
