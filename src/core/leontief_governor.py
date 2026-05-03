"""
Leontief Input-Output Governance for the ADMR Solver.

Implements the Leontief Inverse (I - A)^{-1} as a topological constraint
on resource allocation. Before the system commits VRAM or compute budget
to synthesizing a concept, this governor verifies:

1. Spectral Radius: rho(A) < 1 (productive economy condition).
   If rho(A) >= 1, the system's internal consumption exceeds its output
   and the concept will deflagrate under its own dependency weight.

2. Cascading Cost: (I - A)^{-1} d gives the total production required
   across all sectors to satisfy demand d. This prevents "orphaned"
   concepts -- betting on a Unicorn Soliton without funding its
   coprime polynomial supply chain.

3. Supply Chain Feasibility: If the Neumann series I + A + A^2 + ...
   diverges (rho(A) >= 1), the system falls back to a truncated
   K-term approximation, treating the residual as "structural debt."

The governor does NOT learn. It is an architectural constraint,
like RelationalKappa. Its parameters are derived from the ADMR
solver's transition matrices A[k], not from gradients.

Author: Integrated from Leontief-Kelly research synthesis.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math


class LeontiefGovernor(nn.Module):
    """
    Computes the Leontief Inverse from the ADMR solver's transition matrices
    and provides cascading cost governance for resource allocation.

    The governor aggregates the K facet-wise transition matrices A[k]
    into a single mean consumption matrix A_bar, then computes:
        L = (I - A_bar)^{-1}

    This inverse tells the system: for every unit of external demand,
    how much total cascading production is required across all
    coprime functional channels.
    """

    def __init__(
        self,
        state_dim: int,
        neumann_terms: int = 12,
        spectral_safety_margin: float = 0.95,
        device: str = None
    ):
        """
        Args:
            state_dim: Dimension of the ADMR state space.
            neumann_terms: Number of terms in the truncated Neumann series
                          (fallback when direct inversion is unstable).
            spectral_safety_margin: Maximum allowed spectral radius.
                                   If rho(A) > this, the governor vetoes.
            device: Compute device.
        """
        super().__init__()
        self.state_dim = state_dim
        self.neumann_terms = neumann_terms
        self.spectral_safety_margin = spectral_safety_margin

        # Cache the most recent Leontief inverse for diagnostic access
        self.register_buffer(
            'cached_leontief_inverse',
            torch.eye(state_dim, device=device)
        )
        self.register_buffer(
            'cached_spectral_radius',
            torch.tensor(0.0, device=device)
        )
        self.register_buffer(
            'cached_cascading_cost',
            torch.tensor(1.0, device=device)
        )

    def compute_mean_consumption_matrix(
        self,
        transition_matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregates K facet-wise transition matrices into a single
        consumption matrix A_bar = mean(A[0], A[1], ..., A[K-1]).

        Args:
            transition_matrices: [K, state_dim, state_dim] from ADMR solver.

        Returns:
            A_bar: [state_dim, state_dim] mean consumption matrix.
        """
        return transition_matrices.mean(dim=0)

    def compute_spectral_radius(self, A: torch.Tensor) -> float:
        """
        Computes the spectral radius rho(A) = max(|eigenvalues(A)|).

        For the Leontief model to be productive (convergent Neumann series),
        we need rho(A) < 1 strictly.

        Args:
            A: [state_dim, state_dim] consumption matrix.

        Returns:
            Spectral radius as a float.
        """
        with torch.no_grad():
            try:
                eigenvalues = torch.linalg.eigvals(A)
                rho = eigenvalues.abs().max().item()
            except Exception:
                # Fallback: use Frobenius norm as upper bound
                rho = torch.norm(A, p='fro').item() / math.sqrt(self.state_dim)
        return rho

    def compute_leontief_inverse(
        self,
        transition_matrices: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Computes the Leontief Inverse (I - A_bar)^{-1}.

        If the spectral radius is safe (rho < margin), uses direct inversion.
        If unsafe, falls back to truncated Neumann series.

        Args:
            transition_matrices: [K, state_dim, state_dim] from ADMR solver.

        Returns:
            leontief_inverse: [state_dim, state_dim]
            spectral_radius: float
            is_productive: bool (True if rho < safety margin)
        """
        A_bar = self.compute_mean_consumption_matrix(transition_matrices)
        rho = self.compute_spectral_radius(A_bar)

        self.cached_spectral_radius.fill_(rho)
        is_productive = rho < self.spectral_safety_margin

        I = torch.eye(self.state_dim, device=A_bar.device)

        if is_productive:
            # Direct inversion: (I - A)^{-1}
            try:
                L = torch.linalg.inv(I - A_bar)
            except Exception:
                # Singular or near-singular: fall back to Neumann
                L = self._neumann_series(A_bar, I)
        else:
            # Neumann series truncation (the economy is not productive,
            # but we can still approximate the partial cascade)
            L = self._neumann_series(A_bar, I)

        self.cached_leontief_inverse.copy_(L.detach())
        return L, rho, is_productive

    def _neumann_series(
        self,
        A: torch.Tensor,
        I: torch.Tensor
    ) -> torch.Tensor:
        """
        Truncated Neumann series: L = I + A + A^2 + ... + A^K.

        Each term represents one additional level of cascading dependency.
        The residual A^{K+1} is treated as unresolvable "structural debt."
        """
        L = I.clone()
        A_power = I.clone()

        for k in range(self.neumann_terms):
            A_power = A_power @ A
            L = L + A_power

            # Early termination if powers become negligible
            if A_power.abs().max().item() < 1e-8:
                break

        return L

    def cascading_cost(
        self,
        demand: torch.Tensor,
        transition_matrices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total cascading production required to satisfy
        external demand d, accounting for all internal dependencies.

        x = (I - A)^{-1} d

        This is the Leontief equilibrium: the total output the system
        must generate to sustain both its internal consumption and
        the external demand.

        Args:
            demand: [batch, state_dim] or [state_dim] external demand vector.
            transition_matrices: [K, state_dim, state_dim] from ADMR solver.

        Returns:
            total_production: [batch, state_dim] or [state_dim]
            diagnostics: Dict with spectral_radius, is_productive, cost_ratio
        """
        L, rho, is_productive = self.compute_leontief_inverse(transition_matrices)

        # x = L @ d
        if demand.dim() == 1:
            total_production = L @ demand
        else:
            total_production = demand @ L.T

        # Cost ratio: how much more total production is needed vs raw demand
        demand_norm = demand.norm().item() + 1e-8
        production_norm = total_production.norm().item()
        cost_ratio = production_norm / demand_norm

        self.cached_cascading_cost.fill_(cost_ratio)

        diagnostics = {
            'spectral_radius': rho,
            'is_productive': is_productive,
            'cost_ratio': cost_ratio,
            'neumann_terms_used': self.neumann_terms if not is_productive else 0,
            'cascading_amplification': cost_ratio - 1.0  # How much the cascade adds
        }

        return total_production, diagnostics

    def should_veto_concept(
        self,
        demand: torch.Tensor,
        transition_matrices: torch.Tensor,
        available_budget: float = 1.0
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Governance check: should the system proceed with synthesizing
        this concept given the available compute/memory budget?

        Vetoes if the cascading cost exceeds the available budget,
        or if the economy is non-productive (rho >= margin).

        Args:
            demand: [state_dim] concept demand vector.
            transition_matrices: [K, state_dim, state_dim] from ADMR solver.
            available_budget: Scalar budget (normalized, 1.0 = full capacity).

        Returns:
            should_veto: bool
            diagnostics: Dict with governance details
        """
        total_production, diags = self.cascading_cost(demand, transition_matrices)

        total_cost = total_production.abs().sum().item()
        can_afford = total_cost <= available_budget * self.state_dim

        should_veto = (not diags['is_productive']) or (not can_afford)

        diags['total_cost'] = total_cost
        diags['available_budget'] = available_budget
        diags['can_afford'] = can_afford
        diags['vetoed'] = should_veto

        return should_veto, diags

    def get_metrics(self) -> Dict[str, float]:
        """Diagnostic metrics for the bulletin board."""
        return {
            'leontief_spectral_radius': self.cached_spectral_radius.item(),
            'leontief_cascading_cost': self.cached_cascading_cost.item(),
            'leontief_is_productive': float(
                self.cached_spectral_radius.item() < self.spectral_safety_margin
            )
        }
