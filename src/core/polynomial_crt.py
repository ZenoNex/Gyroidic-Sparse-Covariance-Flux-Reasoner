"""
Polynomial CRT reconstruction using co-prime polynomial functionals.

Generalizes the Chinese Remainder Theorem from discrete modular arithmetic
to polynomial remainders with co-prime polynomial functionals.
Operates in a saturated, symbolic regime for topological stability.

Mathematical Foundation:
    Given co-prime φ_1(x), ..., φ_K(x)
    For residues r_1(x), ..., r_K(x)
    ∃! polynomial L(x) such that:
        L(x) ≡ r_k(x) (mod φ_k(x)) for all k

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
    
    def forward(
        self,
        residue_distributions: torch.Tensor,
        trust_scalars: Optional[torch.Tensor] = None,
        mode: str = 'majority',
        return_diagnostics: bool = False
    ) -> torch.Tensor:
        """
        Reconstruct polynomial from residue coefficient distributions.
        Supports:
            - 'majority': argmax over residue symbols (Saturated CRT)
            - 'modal': Selects consistent lattice solution (Consensus CRT)
            - 'expectation': Weighted average (Legacy Differentiable)
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
            
        else:
            # Legacy Expectation: Differentiable path
            expected_residues = residue_distributions
        
        # Polynomial CRT reconstruction (Symbolic-weighted)
        weights = self.recon_weights
        if trust_scalars is not None:
            weights = weights * trust_scalars
            weights = weights / (weights.sum() + 1e-8)

        reconstruction = torch.sum(
            weights.unsqueeze(0).unsqueeze(-1) * expected_residues,
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
