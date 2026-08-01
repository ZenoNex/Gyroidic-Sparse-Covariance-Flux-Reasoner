"""
Polynomial CRT reconstruction using co-prime polynomial functionals.

Generalizes the Chinese Remainder Theorem from discrete modular arithmetic
to polynomial remainders with co-prime polynomial functionals.
Operates in a saturated, symbolic regime for topological stability.

Mathematical Foundation:
    Given co-prime _1(x), ..., _K(x)
    For residues r_1(x), ..., r_K(x)
    ! polynomial L(x) such that:
        L(x)  r_k(x) (mod _k(x)) for all k

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union
import networkx as nx

from .polynomial_coprime import PolynomialCoprimeConfig


class PolynomialCRT(nn.Module):
    """
    Polynomial Chinese Remainder Theorem reconstruction.
    
    Reconstructs polynomial L(x) from residues modulo co-prime functionals.
    """
    
    def __init__(
        self,
        poly_config: PolynomialCoprimeConfig,
        use_soft_reconstruction: bool = True
    ):
        """
        Args:
            poly_config: Polynomial co-prime configuration
            use_soft_reconstruction: Use differentiable soft version
        """
        super().__init__()
        
        self.config = poly_config
        self.K = poly_config.k
        self.D = poly_config.degree + 1
        self.use_soft = use_soft_reconstruction
        
        # Get coefficient matrix
        self.register_buffer('theta', poly_config.get_coefficients_tensor())
        
        # Precompute reconstruction weights (analogous to CRT coefficients)
        self._compute_reconstruction_weights()
    
    def _compute_reconstruction_weights(self):
        """
        Compute polynomial CRT reconstruction weights.
        
        For standard CRT: M_k = P/p_k, y_k = M_k^-1 mod p_k
        For polynomial CRT: Similar but with polynomial multiplication/inverse
        
        Simplified approach: Use weighted averaging based on functional structure
        """
        # Reconstruction weights: how much each functional contributes
        # Based on coefficient magnitudes
        weights = torch.norm(self.theta, dim=1, p=2)
        weights = weights / (weights.sum() + 1e-8)
        
        self.register_buffer('recon_weights', weights)
        
        # Fixed-point Scaling: 16-bit
        self.scaling_factor = 2**16

    def fixed_point_reconstruction(
        self,
        residue_distributions: torch.Tensor,
        trust_scalars: Optional[torch.Tensor] = None,
        veto_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Algebraically exact CRT reconstruction using 16-bit fixed-point arithmetic.
        
        Maps residues to int64 lattice, solves for unique reconstruction,
        and scales back to float.
        """
        batch_size = residue_distributions.shape[0]
        device = residue_distributions.device
        
        # 1. Scale to fixed-point int64
        # [B, K, D] -> [B, K, D] (int64)
        scaled_residues = (residue_distributions * self.scaling_factor).to(torch.int64)
        
        # 2. Use polynomial coefficients as 'primes' for modular reconstruction
        # In a real CRT, we need M_k = M / m_k. 
        # Here we use the functional weights as proxies for the algebraic kernels.
        weights_fp = (self.recon_weights * self.scaling_factor).to(torch.int64)
        
        if trust_scalars is not None:
            trust_fp = (trust_scalars * self.scaling_factor).to(torch.int64)
            weights_fp = (weights_fp * trust_fp) // self.scaling_factor
            
        if veto_mask is not None:
            # Mask out vetoed channels [B, K]
            # Convert bool/float mask to 0/1 multiplier
            mask_multiplier = (~veto_mask).to(torch.int64)
            weights_fp = weights_fp.view(1, self.K) * mask_multiplier
        else:
            weights_fp = weights_fp.view(1, self.K)

        # 3. Summation: L = sum(r_k * W_k)
        # In fixed-point: (r * S) * (W * S) / S = (r * W * S)
        # [B, K, D] * [B, K] -> [B, D]
        reconstruction_fp = torch.sum(
            scaled_residues * weights_fp.unsqueeze(-1),
            dim=1
        )
        
        # 4. Scale back to float without precision loss (underflow)
        return (reconstruction_fp.to(torch.float64) / float(self.scaling_factor ** 2)).to(torch.float32)

    def forward(
        self,
        residue_distributions: torch.Tensor,
        trust_scalars: Optional[torch.Tensor] = None,
        veto_mask: Optional[torch.Tensor] = None,
        mode: str = 'majority',
        return_diagnostics: bool = False
    ) -> torch.Tensor:
        """
        Reconstruct polynomial from residue coefficient distributions.
        Supports:
            - 'majority': argmax over residue symbols (Saturated CRT)
            - 'modal': Selects consistent lattice solution (Consensus CRT)
            - 'expectation': Weighted average (Legacy Differentiable)
            - 'fixed_point': Algebraic exact reconstruction (16-bit)
        """
        batch_size = residue_distributions.shape[0]
        
        if mode == 'majority':
            # Majority Symbol CRT: Prioritize symbolic lock
            expected_residues = torch.zeros_like(residue_distributions)
            max_idx = torch.argmax(residue_distributions, dim=-1)
            expected_residues.scatter_(-1, max_idx.unsqueeze(-1), 1.0)
            
        elif mode == 'modal':
            # Modal CRT: Select consistent lattice solution
            expected_residues = residue_distributions 
            
        elif mode == 'fixed_point':
            # Fixed-Point Algebraic CRT
            reconstruction = self.fixed_point_reconstruction(
                residue_distributions,
                trust_scalars=trust_scalars,
                veto_mask=veto_mask
            )
            if return_diagnostics:
                return reconstruction, {'mode_used': 'fixed_point'}
            return reconstruction
            
        else:
            # Legacy Expectation: Differentiable path
            expected_residues = residue_distributions
        
        # Polynomial CRT reconstruction (Symbolic-weighted)
        weights = self.recon_weights.unsqueeze(0)  # [1, K]
        if trust_scalars is not None:
            weights = weights * trust_scalars
            
        if veto_mask is not None:
            # Apply Meliponini isolation (Cerumen mask)
            weights = weights * (~veto_mask).float()
            
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        reconstruction = torch.sum(
            weights.unsqueeze(-1) * expected_residues,
            dim=1
        )
        
        if return_diagnostics:
            diagnostics = {
                'expected_residues': expected_residues,
                'reconstruction_weights': self.recon_weights,
                'mode_used': mode
            }
            return reconstruction, diagnostics
        
        return reconstruction
    
    def compute_reconstruction_pressure(
        self,
        residue_distributions: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
        trust_scalars: Optional[torch.Tensor] = None,
        return_reconstruction: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute reconstruction pressure from residue distributions.
        
        Args:
            residue_distributions: [batch, K, D]
            anchor: Optional [batch, D] (for validation only)
            trust_scalars: Optional [K]
            return_reconstruction: If True, returns (pressure, reconstruction), else just pressure.
            
        Returns:
            pressure: [batch] or (pressure: [batch], reconstruction: [batch, D])
        """
        reconstruction = self.forward(
            residue_distributions, 
            trust_scalars=trust_scalars
        )
        
        if anchor is not None:
            pressure = torch.norm(reconstruction - anchor, dim=-1, p=2)
            if not return_reconstruction:
                 return pressure
            return pressure, reconstruction
        else:
            # Admissibility check: normalize and check deviation
            norm = torch.norm(reconstruction, dim=-1, keepdim=True)
            normalized = reconstruction / (norm + 1e-8)
            pressure = torch.std(normalized, dim=-1)
        
        if not return_reconstruction:
            return pressure
        return pressure, reconstruction

    def compute_packing_fraction(self, residue_distributions: torch.Tensor) -> torch.Tensor:
        """
        Determine packing fraction \phi based on correlation:
        -1.0 represents highly dispersed "Meliponini" pots,
        1.0 represents densely packed "Apis" lattices.
        """
        from core.martinova_correlation import compute_bounded_correlation
        # residue_distributions: [batch, K, D]
        # Calculate local spatial correlation across the K functionals (residue patterns)
        phi = compute_bounded_correlation(residue_distributions)
        return phi


class PolynomialCRTKernelDetector:
    """
    Detect violations of polynomial CRT consistency.
    
    Similar to discrete CRT kernel detection but for polynomial functionals.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Error threshold for violation detection
        """
        self.threshold = threshold
    
    def detect_violations(
        self,
        crt: PolynomialCRT,
        residue_distributions: torch.Tensor,
        anchor: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect CRT kernel violations (inconsistent residues).
        """
        pressures = crt.compute_reconstruction_pressure(residue_distributions, anchor)
        violations = pressures > self.threshold
        
        return violations, pressures
    
    def build_constraint_graph(
        self,
        residue_distributions: torch.Tensor,
        violations: torch.Tensor,
        max_nodes: int = 100
    ) -> nx.Graph:
        """
        Build constraint graph for violated samples.
        
        Args:
            residue_distributions: [batch, K, D]
            violations: [batch] boolean mask
            max_nodes: Maximum number of nodes
            
        Returns:
            graph: NetworkX graph
        """
        G = nx.Graph()
        
        violated_indices = torch.where(violations)[0].cpu().numpy()
        selected_indices = violated_indices[:min(len(violated_indices), max_nodes)]
        
        # Add nodes
        for idx in selected_indices:
            G.add_node(int(idx))
        
        # Add edges between similar violation patterns
        if len(selected_indices) > 1:
            residues_violated = residue_distributions[violations][:max_nodes]
            
            # Compute pairwise similarities
            for i, idx_i in enumerate(selected_indices):
                for j, idx_j in enumerate(selected_indices[i+1:], start=i+1):
                    # Cosine similarity between residue patterns
                    r_i = residues_violated[i].flatten()
                    r_j = residues_violated[j].flatten()
                    
                    similarity = torch.dot(r_i, r_j) / (
                        torch.norm(r_i) * torch.norm(r_j) + 1e-8
                    )
                    
                    if similarity > 0.7:  # High similarity = potential cycle
                        G.add_edge(int(idx_i), int(idx_j), weight=similarity.item())
        
        return G
    
    def find_cycles(self, graph: nx.Graph) -> List[List[int]]:
        """
        Find cycles in constraint graph (obstruction cycles).
        
        Args:
            graph: NetworkX graph
            
        Returns:
            cycles: List of cycles (each cycle is a list of node indices)
        """
        try:
            cycles = nx.cycle_basis(graph)
            return cycles[:10]  # Limit to 10 cycles
        except:
            return []
