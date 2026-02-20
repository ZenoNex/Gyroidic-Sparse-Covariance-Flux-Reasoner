"""
Polynomial Coefficient Functional Scaffold.

Allows adaptive shaping of state evolution by modulating polynomial 
coefficients via meta-invariants and resonance contributions.
Uses orthogonal basis functions for stability.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.core.polynomial_coprime import PolynomialBasis


class PolynomialCoefficientFunctional(nn.Module):
    """
    Implements S_i(t) = Σ a_n(t) * p_n(S_i(t)).
    
    Where p_n are orthogonal basis functions.
    Coefficients a_n are modulated by phase-space variance 
    and resonance signals.
    """
    
    def __init__(
        self,
        degree: int = 4,
        state_dim: int = 64,
        basis_type: str = 'chebyshev',
        lambda_coeff: float = 0.05,
        device: str = None
    ):
        """
        Args:
            degree: Maximum polynomial degree.
            state_dim: Dimension of the state.
            basis_type: Type of orthogonal basis.
            lambda_coeff: Rate of coefficient adaptation.
            device: Computing device.
        """
        super().__init__()
        self.degree = degree
        self.state_dim = state_dim
        self.lambda_coeff = lambda_coeff
        self.device = device
        
        # 1. Orthogonal Polynomial Basis
        self.basis = PolynomialBasis(degree=degree, basis_type=basis_type)
        
        # 2. Coefficients a_n: [state_dim, degree + 1]
        # We store them per-dimension to allow anisotropic shaping
        self.a = nn.Parameter(torch.randn(state_dim, degree + 1, device=device) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the polynomial expansion: S' = sum(a_n * p_n(x))
        """
        # x: [batch, state_dim]
        # basis_vals: [batch, state_dim, degree + 1]
        basis_vals = self.basis.evaluate(x)
        
        # Contract over degree: [batch, state_dim, degree+1] * [state_dim, degree+1] -> [batch, state_dim]
        # We use broadcasted multiplication and sum
        out = (basis_vals * self.a.unsqueeze(0)).sum(dim=-1)
        
        return out

    def update_coefficients(
        self, 
        meta_invariant_grad: torch.Tensor, 
        resonance_contribution: torch.Tensor
    ):
        """
        a_n(t+1) = a_n(t) + λ * (∇ I(S) + R(S))
        """
        # Simple gradient-based update within the non-teleological loop
        # update should have shape [state_dim, degree + 1]
        update = meta_invariant_grad + resonance_contribution
        
        if update.shape == self.a.shape:
            self.a.data.add_(self.lambda_coeff * update.detach())
        else:
            # Fallback if update is just state_dim: apply to degree-1 (linear shift)
            if update.dim() == 1:
                self.a.data[:, 1].add_(self.lambda_coeff * update.detach())

    def get_metrics(self) -> Dict[str, float]:
        """Returns metrics for coefficient distribution."""
        return {
            'mean_coeff_magnitude': self.a.abs().mean().item(),
            'max_coeff': self.a.max().item(),
            'coeff_variance': self.a.var().item()
        }

