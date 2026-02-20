"""
GDPO-enhanced polynomial CRT reconstruction.

Extends GDPO decoupled normalization to polynomial co-prime functionals.

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from src.core.polynomial_crt import PolynomialCRT
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.gdpo_normalization import SignalSovereignty, LearnableWeights


class DecoupledPolynomialCRT(nn.Module):
    """
    Signal Sovereignty-enhanced Polynomial CRT reconstruction.
    
    Applies decoupled normalization to polynomial coefficient distributions
    before reconstruction, preventing collapse of distinct pressure patterns.
    """
    
    def __init__(
        self,
        poly_config: PolynomialCoprimeConfig,
        use_gdpo: bool = True,
        learnable_weights: bool = True,
        weight_init: str = 'uniform'
    ):
        """
        Args:
            poly_config: Polynomial co-prime configuration
            use_gdpo: Enable GDPO decoupling
            learnable_weights: Use learnable per-functional weights
            weight_init: Weight initialization strategy
        """
        super().__init__()
        
        self.config = poly_config
        self.K = poly_config.k
        self.D = poly_config.degree + 1
        self.use_gdpo = use_gdpo
        
        # Base polynomial CRT
        self.poly_crt = PolynomialCRT(poly_config)
        
        # Signal Sovereignty components
        if use_gdpo:
            self.sovereignty = SignalSovereignty(
                num_dimensions=self.K,
                use_batch_norm=True
            )
            
            if learnable_weights:
                self.weight_module = LearnableWeights(
                    num_dimensions=self.K,
                    init_mode=weight_init,
                    constraint='softmax'
                )
            else:
                self.register_buffer(
                    'fixed_weights',
                    torch.ones(self.K) / self.K
                )
        
        self.learnable_weights = learnable_weights
    
    def compute_expected_residues(
        self,
        residue_distributions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute expected polynomial coefficients from distributions.
        
        Args:
            residue_distributions: [batch, K, D] coefficient distributions
            
        Returns:
            expected: [batch, K, D] expected coefficients
        """
        # For polynomial functionals, residue_distributions already represent
        # coefficient distributions, so we just return them
        # (Could add weighted averaging here if needed)
        return residue_distributions
    
    def forward_decoupled(
        self,
        residue_distributions: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        trust_scalars: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        GDPO-enhanced polynomial CRT reconstruction.
        
        Args:
            residue_distributions: [batch, K, D] polynomial coefficient distributions
            group_ids: Optional [batch] group assignments
            trust_scalars: Optional [K] trust weights for fields
            
        Returns:
            reconstruction: [batch, D] reconstructed coefficients
            diagnostics: Dictionary with intermediate values
        """
        batch_size = residue_distributions.shape[0]
        
        # Get expected coefficients per functional
        expected_coeffs = self.compute_expected_residues(residue_distributions)
        # [batch, K, D]
        
        # Apply trust weighting to expected residues if provided
        # (This aligns GDPO with the trust regime)
        # We don't scale the inputs to GDPO directly to keep normalization stable,
        # instead we use trust to bias the GDPO weights or the final reconstruction.
        
        # For GDPO, we normalize across functionals for each coefficient dimension
        # Reshape: [batch, K, D] -> [batch*D, K]
        expected_flat = expected_coeffs.transpose(1, 2).reshape(batch_size * self.D, self.K)
        
        # Get weights
        if self.learnable_weights:
            weights = self.weight_module()
        else:
            weights = self.fixed_weights
        
        # Extend group_ids for all coefficient dimensions
        if group_ids is not None:
            group_ids_extended = group_ids.unsqueeze(1).expand(batch_size, self.D).reshape(-1)
        else:
            group_ids_extended = None
        
        # Apply SignalSovereignty normalization per coefficient dimension
        decoupled_flat, sov_diagnostics = self.sovereignty(
            expected_flat,
            weights,
            group_ids_extended
        )
        # decoupled_flat: [batch*D] aggregated normalized values
        
        # Reshape back and get per-functional decoupled values
        decoupled_per_functional = sov_diagnostics['decoupled']  # [batch*D, K]
        decoupled_per_functional = decoupled_per_functional.reshape(batch_size, self.D, self.K).transpose(1, 2)
        # [batch, K, D]
        
        # Weighted reconstruction using polynomial CRT (Trust-aware)
        reconstruction = self.poly_crt(
            decoupled_per_functional,
            trust_scalars=trust_scalars
        )
        
        diagnostics = {
            'expected_coeffs': expected_coeffs,
            'decoupled_coeffs': decoupled_per_functional,
            'weights': weights,
            'sovereignty_diagnostics': sov_diagnostics
        }
        
        return reconstruction, diagnostics
    
    def forward(
        self,
        residue_distributions: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        trust_scalars: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with automatic GDPO/standard switching.
        
        Args:
            residue_distributions: [batch, K, D]
            group_ids: Optional group assignments
            trust_scalars: Optional [K] trust weights
            return_diagnostics: If True, return diagnostics
            
        Returns:
            reconstruction: [batch, D] or (reconstruction, diagnostics)
        """
        if self.use_gdpo:
            reconstruction, diagnostics = self.forward_decoupled(
                residue_distributions,
                group_ids,
                trust_scalars=trust_scalars
            )
            if return_diagnostics:
                return reconstruction, diagnostics
            return reconstruction
        else:
            reconstruction = self.poly_crt(
                residue_distributions,
                trust_scalars=trust_scalars
            )
            if return_diagnostics:
                return reconstruction, {}
            return reconstruction
    
    def compute_reconstruction_pressure(
        self,
        residue_distributions: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        trust_scalars: Optional[torch.Tensor] = None,
        return_reconstruction: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute reconstruction pressure.
        
        Args:
            residue_distributions: [batch, K, D]
            anchor: Optional [batch, D] ground truth coefficients (for validation only)
            group_ids: Optional group assignments
            trust_scalars: Optional [K] trust weights
            
        Returns:
            pressure: [batch] or (pressure, reconstruction) if return_reconstruction is True
        """
        reconstruction = self.forward(
            residue_distributions, 
            group_ids=group_ids, 
            trust_scalars=trust_scalars
        )
        
        if anchor is not None:
            # Topological Amnesty: Use Angular Pressure to allow magnitude growth
            cos_sim = torch.nn.functional.cosine_similarity(reconstruction, anchor, dim=-1)
            pressure = 1.0 - cos_sim
            if not return_reconstruction:
                 return pressure
            return pressure, reconstruction
        else:
            # Structural tension check
            pressure = torch.std(reconstruction, dim=-1)
        
        return pressure
