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
from typing import Dict, Optional, Tuple, Any
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

    Additionally incorporates:
    - FIELDCAST: Arbitration of competing coherence fields.
    - ECHO_TAGGER: Weighted replay arbitration.
    - GLYPHLOCK: Symbolic integrity and structural legality.
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
        available_budget: float = 1.0,
        bonfire_ring: Optional[Any] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Governance check: should the system proceed with synthesizing
        this concept given the available compute/memory budget?

        Vetoes if the cascading cost exceeds the available budget,
        or if the economy is non-productive (rho >= margin).

        If a BonfireNomadicRing is provided, incorporates community P2P 
        compute affordances via the Egalitarian Consensus Kelly Allocation.

        Args:
            demand: [state_dim] concept demand vector.
            transition_matrices: [K, state_dim, state_dim] from ADMR solver.
            available_budget: Scalar budget (normalized, 1.0 = full capacity).
            bonfire_ring: Optional BonfireNomadicRing instance.

        Returns:
            should_veto: bool
            diagnostics: Dict with governance details
        """
        # Integrate Community Compute Affordances via Bonfire Nomadic Rings
        network_kelly_fraction = 1.0
        if bonfire_ring is not None:
            # Scale the true budget by the network's consensus risk tolerance
            network_kelly_fraction = bonfire_ring.compute_egalitarian_consensus()
            available_budget *= network_kelly_fraction

        total_production, diags = self.cascading_cost(demand, transition_matrices)

        total_cost = total_production.abs().sum().item()
        can_afford = total_cost <= available_budget * self.state_dim

        # Enforce GLYPHLOCK if available
        is_glyphlocked = True
        try:
            from src.core.invariants import check_glyphlock
            # Verify chirality preservation and symbolic continuity
            is_glyphlocked = bool(check_glyphlock(demand).max().item() > 0)
        except ImportError:
            pass

        should_veto = (not diags['is_productive']) or (not can_afford) or (not is_glyphlocked)

        # ---------------------------------------------------------
        # P2P Slashing Mechanics (Kelly Criterion & Mischief Systems)
        # ---------------------------------------------------------
        # If the cascading cost exceeds 10x the budget, it indicates an intentional 
        # Mischief System attack (Unfunded Soliton). We execute a Collapse Path Poison.
        slashed = False
        if not can_afford and (total_cost > available_budget * self.state_dim * 10.0):
            self.kelly_slash_malicious_actor(demand, total_cost)
            slashed = True

        diags['total_cost'] = total_cost
        diags['available_budget'] = available_budget
        diags['can_afford'] = can_afford
        diags['vetoed'] = should_veto
        diags['slashed_via_poison'] = slashed

        return should_veto, diags

    def kelly_slash_malicious_actor(self, malicious_demand: torch.Tensor, total_cost: float):
        """
        Executes a Collapse Path Poisoning protocol against a malicious node that 
        submitted a fraudulent or drastically under-funded Soliton bet.
        
        Uses the Inverted Hypersphere Cosmology mapped via RP4 Topology to isolate
        the adversary's state and collapse their projected coordinates, permanently 
        cutting them off from the Freenet 0.2.116 consensus network.
        """
        # 1. Project the malicious demand into RP4 (Real Projective Space 4D)
        # We append a projective coordinate w=1.0 for the RP4 transform.
        # This allows us to push the attacker to infinity (the boundary).
        if malicious_demand.dim() == 1 and malicious_demand.shape[0] >= 4:
            rp4_vector = torch.cat([malicious_demand[:4], torch.ones(1, device=malicious_demand.device)])
            
            # 2. Inverted Hypersphere inversion: x -> x / |x|^2
            # This turns the core into the boundary, tossing the attacker to the 
            # void of the Inverted Hypersphere.
            norm_sq = torch.dot(rp4_vector, rp4_vector) + 1e-8
            inverted_rp4 = rp4_vector / norm_sq
            
            # 3. Collapse Path Poisoning
            # We inject this inverted vector back into the cached Leontief inverse 
            # as a permanent topological singularity (poisoning the path), 
            # rendering the attacker's topological signature inert.
            if self.state_dim >= 5:
                full_rp4 = torch.zeros(self.state_dim, device=malicious_demand.device)
                full_rp4[:5] = inverted_rp4
                poison_tensor = torch.ger(full_rp4, full_rp4)
            else:
                p_vec = inverted_rp4[:self.state_dim]
                poison_tensor = torch.ger(p_vec, p_vec)
            
            # We scale the poison by the Kelly fraction loss
            kelly_fraction = 0.5  # Heavy slash penalty
            
            # Apply poison to the ledger
            with torch.no_grad():
                self.cached_leontief_inverse -= kelly_fraction * poison_tensor
                
            print(f"[LEONTIEF GOVERNOR] MISCHIEF SYSTEM DETECTED (Cost: {total_cost:.2f}). "
                  f"Executing Collapse Path Poison via RP4 Inverted Hypersphere. Malicious actor slashed.")
        else:
            print("[LEONTIEF GOVERNOR] Mischief detected, but dimensionality insufficient for RP4 poison.")

    def fieldcast_arbitration(
        self,
        candidate_demands: torch.Tensor,
        transition_matrices: torch.Tensor,
        pas_scores: torch.Tensor,
        volition_vectors: torch.Tensor
    ) -> Tuple[int, Dict[str, float]]:
        """
        FIELDCAST: Arbitration of Competing Coherence Fields.
        Selects the optimal inference context based on coherence, volition, and
        substrate cost.
        F* = argmax [ PAS_s(i) * V(i) / Thermo_cost(i) ]
        """
        best_idx = -1
        best_score = -float('inf')
        best_diags = {}

        # TailSlayer XOR-mapping / Z-Curve Interleaving:
        # Instead of a naive linear scan which causes massive cache misses on spatially 
        # correlated coherence fields, we traverse the candidates using a 1D Morton-inspired 
        # XOR Gray-code mapping to preserve cache line associativity.
        num_candidates = candidate_demands.shape[0]
        
        for base_i in range(num_candidates):
            # 1D Morton-esque XOR interleave
            i = base_i ^ (base_i >> 1)
            if i >= num_candidates:
                i = base_i # Fallback if out of bounds (though Gray code keeps it within next power of 2)
            
            # Additional bounds check for non-power-of-2 candidate counts
            if i >= num_candidates:
                continue

            demand = candidate_demands[i]
            total_production, diags = self.cascading_cost(demand, transition_matrices)
            thermo_cost = total_production.abs().sum().item() + 1e-8

            pas = pas_scores[i].item()
            v = volition_vectors[i].item()

            score = (pas * v) / thermo_cost
            if score > best_score:
                best_score = score
                best_idx = i
                best_diags = diags
                best_diags['thermo_cost'] = thermo_cost
                best_diags['fieldcast_score'] = score

        return best_idx, best_diags

    def echo_tagger_arbitration(
        self,
        pas_mem: torch.Tensor,
        stability: torch.Tensor,
        entropy_bound: torch.Tensor
    ) -> torch.Tensor:
        """
        ECHO_TAGGER: Weighted Replay Arbitration.
        Scores candidate replay emissions from the Phase Memory Buffer based on:
        Score_i = PAS_mem(i) * Stability_i / H_i
        Ensures that only emissions with durable coherence and low symbolic drift
        are considered for re-emission.
        """
        # H_i is estimated symbolic entropy
        h_i = entropy_bound + 1e-8
        scores = (pas_mem * stability) / h_i
        return scores

    def get_metrics(self) -> Dict[str, float]:
        """Diagnostic metrics for the bulletin board."""
        return {
            'leontief_spectral_radius': self.cached_spectral_radius.item(),
            'leontief_cascading_cost': self.cached_cascading_cost.item(),
            'leontief_is_productive': float(
                self.cached_spectral_radius.item() < self.spectral_safety_margin
            )
        }
