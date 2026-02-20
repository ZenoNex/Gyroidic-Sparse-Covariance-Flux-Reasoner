"""
Soliton Stability Functional

Implements dispersion and localization computation for soliton detection.

Mathematical Foundation:
    D(r) = ∫_C |∇Φ(r)|^2 dμ
    Λ(r) = sup_{U subset C} μ(U) s.t. ∫_U |Φ(r)|^2 dμ >= η
    
    Soliton condition: D(r) / Λ(r) < κ
    (no minimization, threshold only)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np


class SolitonStability(nn.Module):
    """
    Soliton Stability Functional.
    
    Computes dispersion and localization to detect stable solitons.
    """
    
    def __init__(
        self,
        eta: float = 0.5,
        kappa: float = 0.1,
        num_integration_samples: int = 100
    ):
        """
        Args:
            eta: Energy threshold for localization (default: 0.5)
            kappa: Soliton threshold (default: 0.1)
            num_integration_samples: Number of samples for numerical integration
        """
        super().__init__()
        self.eta = eta
        self.kappa = kappa
        self.num_integration_samples = num_integration_samples
    
    def compute_dispersion(
        self,
        residue: torch.Tensor,
        constraint_manifold: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Compute dispersion: D(r) = ∫_C |∇Φ(r)|^2 dμ
        
        Args:
            residue: [batch, ...] residue tensor
            constraint_manifold: [batch, dim] constraint manifold
            embedding_fn: Optional function Phi: r -> constraint space
            
        Returns:
            dispersion: [batch] dispersion values
        """
        if embedding_fn is None:
            # Default: identity embedding
            embedding_fn = lambda r: r.reshape(-1, constraint_manifold.shape[-1])
        
        # Embed residue
        phi_r = embedding_fn(residue)
        
        # Ensure same shape
        if phi_r.shape != constraint_manifold.shape:
            if phi_r.numel() == constraint_manifold.numel():
                phi_r = phi_r.reshape(constraint_manifold.shape)
            else:
                # Project to same dimension
                if phi_r.shape[-1] != constraint_manifold.shape[-1]:
                    proj = nn.Linear(phi_r.shape[-1], constraint_manifold.shape[-1],
                                   device=phi_r.device)
                    phi_r = proj(phi_r)
        
        # Compute gradient of embedding
        # Use finite differences or autograd
        batch_size = phi_r.shape[0]
        dim = phi_r.shape[-1]
        
        dispersions = []
        for b in range(batch_size):
            phi_b = phi_r[b]  # [dim]
            constraint_b = constraint_manifold[b]  # [dim]
            
            # Compute gradient: ∇Φ(r) ≈ (Φ(r + δ) - Φ(r)) / δ
            # Use finite differences
            delta = 1e-5
            grad_squared_sum = 0.0
            
            # Sample points on constraint manifold for integration
            # Use constraint point as center, sample nearby points
            for _ in range(self.num_integration_samples):
                # Sample perturbation
                perturbation = torch.randn(dim, device=phi_b.device) * delta
                phi_perturbed = phi_b + perturbation
                
                # Compute gradient approximation
                grad_approx = (phi_perturbed - phi_b) / delta
                
                # |∇|^2
                grad_squared_sum += torch.sum(grad_approx ** 2).item()
            
            # Average over samples (approximate integration)
            dispersion = grad_squared_sum / self.num_integration_samples
            dispersions.append(dispersion)
        
        return torch.tensor(dispersions, device=residue.device, dtype=residue.dtype)
    
    def compute_localization(
        self,
        residue: torch.Tensor,
        constraint_manifold: torch.Tensor,
        embedding_fn: Optional[callable] = None,
        eta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute localization: Λ(r) = sup_{U subset C} μ(U) s.t. ∫_U |Φ(r)|^2 dμ >= η
        
        Finds the largest region U where energy >= eta.
        
        Args:
            residue: [batch, ...] residue tensor
            constraint_manifold: [batch, dim] constraint manifold
            embedding_fn: Optional embedding function
            eta: Optional energy threshold (uses self.eta if None)
            
        Returns:
            localization: [batch] localization values (largest region size)
        """
        if eta is None:
            eta = self.eta
        
        if embedding_fn is None:
            embedding_fn = lambda r: r.reshape(-1, constraint_manifold.shape[-1])
        
        # Embed residue
        phi_r = embedding_fn(residue)
        
        # Ensure same shape
        if phi_r.shape != constraint_manifold.shape:
            if phi_r.numel() == constraint_manifold.numel():
                phi_r = phi_r.reshape(constraint_manifold.shape)
            else:
                if phi_r.shape[-1] != constraint_manifold.shape[-1]:
                    proj = nn.Linear(phi_r.shape[-1], constraint_manifold.shape[-1],
                                   device=phi_r.device)
                    phi_r = proj(phi_r)
        
        batch_size = phi_r.shape[0]
        localizations = []
        
        for b in range(batch_size):
            phi_b = phi_r[b]  # [dim]
            constraint_b = constraint_manifold[b]  # [dim]
            
            # Compute energy: |Φ(r)|^2
            energy = torch.sum(phi_b ** 2)
            
            # Find largest region U where energy >= eta
            # Use greedy approach: start with full region, shrink until energy >= eta
            # Simplified: use constraint manifold volume as proxy
            
            # Compute constraint volume (approximate as hypercube or hypersphere)
            constraint_scale = torch.norm(constraint_b)
            
            # Energy density: energy per unit volume
            # Localization is inverse of energy density (larger region = more localized)
            if energy > 0:
                energy_density = energy / (constraint_scale + 1e-8)
                # Localization: region size needed to contain eta energy
                localization = eta / (energy_density + 1e-8)
            else:
                localization = constraint_scale  # Default: full constraint scale
            
            localizations.append(localization.item())
        
        return torch.tensor(localizations, device=residue.device, dtype=residue.dtype)
    
    def is_soliton(
        self,
        residue: torch.Tensor,
        constraint_manifold: torch.Tensor,
        embedding_fn: Optional[callable] = None,
        kappa: Optional[float] = None
    ) -> torch.Tensor:
        """
        Check soliton condition: D(r) / Λ(r) < κ
        
        Args:
            residue: [batch, ...] residue tensor
            constraint_manifold: [batch, dim] constraint manifold
            embedding_fn: Optional embedding function
            kappa: Optional threshold (uses self.kappa if None)
            
        Returns:
            is_soliton: [batch] boolean tensor
        """
        if kappa is None:
            kappa = self.kappa
        
        D = self.compute_dispersion(residue, constraint_manifold, embedding_fn)
        Lambda = self.compute_localization(residue, constraint_manifold, embedding_fn)
        
        # Ratio: D / Lambda
        ratio = D / (Lambda + 1e-8)
        
        # Soliton if ratio < kappa
        is_soliton = ratio < kappa
        
        return is_soliton
    
    def forward(
        self,
        residue: torch.Tensor,
        constraint_manifold: torch.Tensor,
        embedding_fn: Optional[callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: compute soliton stability metrics.
        
        Returns:
            Dictionary with:
            - 'is_soliton': [batch] boolean
            - 'dispersion': [batch] D(r)
            - 'localization': [batch] Λ(r)
            - 'ratio': [batch] D(r) / Λ(r)
        """
        D = self.compute_dispersion(residue, constraint_manifold, embedding_fn)
        Lambda = self.compute_localization(residue, constraint_manifold, embedding_fn)
        ratio = D / (Lambda + 1e-8)
        is_soliton = ratio < self.kappa
        
        return {
            'is_soliton': is_soliton,
            'dispersion': D,
            'localization': Lambda,
            'ratio': ratio
        }
