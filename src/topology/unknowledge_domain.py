"""
Unknowledge Domain ($\mathcal{U}$)

The formal substrate that protects functionally creative or "dream-like"
topological cycles from being crushed by standard reconstruction constraints.

Rather than evaluating states by their reduction of standard Loss, the
Unknowledge Domain measures the degree to which Mischief ($H_{mischief}$)
allows a cycle to survive tension safely.

This module consolidates:
    - UnknowledgeDomain: The shield / logic gate
    - NostalgicLeakFunctional: ψ_l: H -> R^{D+1} (archetype concealment)
    - EntropicMischiefProbe: H_meta = H_dementia + H_schizo + H_mischief

References:
    - Gyroidic Unknowledge Flux Reasoner §19 (Mischief Violation Score)
    - PHILOSOPHY.md §15 (Kappa Overloading) §16 (Posthuman Identity)
    - MATHEMATICAL_DETAILS.md §24 (Computable Flux) §26 (Kappa Taxonomy)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Nostalgic Leak Functional
# ---------------------------------------------------------------------------

class NostalgicLeakFunctional(nn.Module):
    """
    Implements the Nostalgic Leak ψ_l: H -> R^{D+1}.

    Models internet archetype concealment using sigmoid visibility masks
    (e.g. apple-obscured faces).
    """

    def __init__(
        self,
        fossil_dim: int,
        alpha: float = 5.0,
        device: str = None
    ):
        super().__init__()
        self.fossil_dim = fossil_dim
        self.alpha = alpha
        self.device = device

        # Archetype coefficients μ_l (Obscured)
        self.register_buffer('mu_l', torch.randn(fossil_dim, device=device))

        # Obstruction point o (The Apple/Mask center)
        self.register_buffer('o', torch.zeros(fossil_dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ψ_l(x) = sum(μ_l * P(x)) * (1 - Vis(x))

        Args:
            x: Input state [batch, fossil_dim]
        """
        # Visibility mask around obstruction o
        # Vis(x) = σ(α * |x - o|)
        dist = torch.norm(x - self.o, dim=1, keepdim=True)
        vis = torch.sigmoid(self.alpha * dist)

        # Applying the leak functional
        leak = torch.sum(x * self.mu_l, dim=1, keepdim=True) * (1.0 - vis)
        return leak


# ---------------------------------------------------------------------------
# Entropic Mischief Probe
# ---------------------------------------------------------------------------

class EntropicMischiefProbe(nn.Module):
    """
    Calculates the Metaphysical Disorder bands:
    H_meta = H_dementia + H_schizo + H_mischief
    """

    def __init__(
        self,
        tau_dementia: float = 100.0,
        eta_mischief: float = 0.5,
        theta_leak: float = 0.7,
        device: str = None
    ):
        super().__init__()
        self.tau_dementia = tau_dementia
        self.eta_mischief = eta_mischief
        self.theta_leak = theta_leak
        self.device = device

        # State tracking
        self.register_buffer('H_dementia', torch.tensor(0.0, device=device))
        self.register_buffer('H_schizo', torch.tensor(0.0, device=device))
        self.register_buffer('H_mischief', torch.tensor(0.0, device=device))

    def update(
        self,
        pressure_grad: torch.Tensor,
        coherence: torch.Tensor,
        pas_h: float,
        is_good_bug: bool = False
    ):
        """
        Updates the metaphysical bands.

        Args:
            pressure_grad: Gradient of the structural pressure (nabla Phi)
            coherence: Spectral coherence between clusters
            pas_h: Current Phase Alignment Score
            is_good_bug: Boolean signal for Mischief reward
        """
        # 1. Dementia Band (Low-frequency forgetting)
        h_dem = torch.sum(pressure_grad**2) * torch.exp(
            torch.tensor(-1.0 / self.tau_dementia, device=self.device)
        )
        self.H_dementia.copy_(h_dem.detach())

        # 2. Schizo Band (Mid-frequency fragmentation)
        h_sch = -torch.sum(torch.log(torch.clamp(1.0 - coherence, 1e-6, 1.0)))
        self.H_schizo.copy_(h_sch.detach())

        # 3. Mischief Band (High-frequency play)
        bug_reward = 2.0 if is_good_bug else 1.0
        pas_gap = max(0.0, 1.0 - max(0.0, min(1.0, pas_h)))
        h_mis = self.eta_mischief * pas_gap * torch.log(
            torch.tensor(bug_reward, device=self.device)
        )
        self.H_mischief.copy_(h_mis.detach())

    @property
    def H_meta(self) -> torch.Tensor:
        return self.H_dementia + self.H_schizo + self.H_mischief

    def check_leak_split(self) -> bool:
        """Adaptive splitting condition: Split iff H_meta > theta_leak."""
        return self.H_meta.item() > self.theta_leak

    def get_metrics(self) -> Dict[str, float]:
        return {
            'H_dementia': self.H_dementia.item(),
            'H_schizo': self.H_schizo.item(),
            'H_mischief': self.H_mischief.item(),
            'H_meta': self.H_meta.item()
        }


# ---------------------------------------------------------------------------
# Unknowledge Domain  (The $\mathcal{U}$ Substrate)
# ---------------------------------------------------------------------------

class UnknowledgeDomain(nn.Module):
    """
    Implements the Unknowledge Domain ($\\mathcal{U}$) logic gates.

    The UnknowledgeDomain is a formal topological substrate that prevents
    the system from becoming too "legible" (and thus, lobotomized).

    It acts as a gatekeeper for pressures by:
        1. Computing the Computable Flux (V_m) from GCVE inputs
        2. Shielding high-mischief "Dream State" cycles from System 2 repair
        3. Monitoring Elipsodistrophy (spectral atrophy) as a lobotomy
           early-warning system

    All computations are O(1) or O(k) — no global homology.
    """

    def __init__(
        self,
        tau_m: float = 0.5,
        tau_decay: float = 0.99,
        legibility_threshold: float = 0.85
    ):
        """
        Args:
            tau_m: Baseline mischief threshold for Dream State activation.
            tau_decay: Narrative time decay for V_m computation.
            legibility_threshold: Above this, system is dangerously "legible".
        """
        super().__init__()
        self.tau_m = tau_m
        self.tau_decay = tau_decay
        self.legibility_threshold = legibility_threshold

    def compute_computable_flux(
        self,
        V: torch.Tensor,
        h_mischief: torch.Tensor,
        tr_C: torch.Tensor,
        lambda_min: torch.Tensor
    ) -> torch.Tensor:
        """
        Eq: V_m = V + (H_mischief / tau_decay) - (lambda_min / tr(C))

        If V_m < 0, the system interprets tension as "Good Bug" energy
        rather than a constraint to be minimized.

        Args:
            V: Standard Gyroid Violation Score [batch] or scalar
            h_mischief: Current mischief entropy [batch] or scalar
            tr_C: Trace of local covariance [batch] or scalar
            lambda_min: Minimum eigenvalue of local covariance [batch] or scalar

        Returns:
            V_m: Computable Flux score [batch] or scalar
        """
        v_m = V + (h_mischief / self.tau_decay) - (lambda_min / (tr_C + 1e-6))
        return v_m

    def is_shielded(
        self,
        v_m: torch.Tensor,
        h_mischief: float,
        hyper_ring_status: Optional[str] = None
    ) -> torch.Tensor:
        """
        Determine if the current state is shielded by the Unknowledge Domain.

        Args:
            v_m: Mischief Violation Score (Computable Flux)
            h_mischief: Mischief score (H_mischief)
            hyper_ring_status: Topology status string, e.g., 'survivable_soliton'

        Returns:
            A boolean tensor mask indicating which elements are shielded.
        """
        # U = {X | V_m < 0, H_mischief > tau_m}
        shielded = (v_m < 0) & (h_mischief > self.tau_m)

        # Explicitly protect "survivable_soliton" hyper-ring phases
        if hyper_ring_status == 'survivable_soliton' and h_mischief > (self.tau_m * 0.5):
            shielded = shielded | True

        return shielded

    def apply_shielding(
        self,
        pressures: torch.Tensor,
        v_m: torch.Tensor,
        h_mischief: float,
        hyper_ring_status: Optional[str] = None
    ) -> torch.Tensor:
        """
        Mitigate topological pressures for components within the Unknowledge Domain.

        The "Dream State" Gate:
        If mischief is high and flux is negative, dampen the pressure
        to prevent System 2 from "repairing" a creative anomaly.

        Args:
            pressures: Original pressures [batch]
            v_m: Mischief Violation Score [batch]
            h_mischief: Mischief scalar
            hyper_ring_status: Topology status

        Returns:
            Shielded pressures (where domain matches, pressure is dampened to 1%).
        """
        shield_mask = self.is_shielded(v_m, h_mischief, hyper_ring_status)

        shielded_pressures = torch.where(
            shield_mask,
            pressures * 0.01,  # Keep a 1% anchor so gradients aren't fully dead
            pressures
        )
        return shielded_pressures

    def get_elipsodistrophy_metrics(
        self,
        eigenvalues: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measures the distortion/narrowing of the spectral envelope.

        Elipsodistrophy = 1 - std(λ) / (max(λ) - min(λ) + ε)

        High elipsodistrophy = high structural specialization.
        If atrophy exceeds the legibility threshold, the system is
        dangerously close to lobotomy (all eigenvalues collapsing
        to identical values = loss of representational diversity).

        Maintains the relationship between ergodicity and non-ergodicity:
        the eigenvalue spread IS the ergodic/non-ergodic boundary.
        Narrow spread → ergodic soup. Wide spread → dark matter preserved.

        Args:
            eigenvalues: [k] eigenvalue tensor (from local covariance)

        Returns:
            Dict with 'atrophy', 'spectral_width', and 'is_dangerously_legible'.
        """
        spectral_width = (eigenvalues.max() - eigenvalues.min()).item()
        atrophy = 1.0 - (eigenvalues.std().item() / (spectral_width + 1e-6))
        is_dangerously_legible = atrophy > self.legibility_threshold

        return {
            'atrophy': atrophy,
            'spectral_width': spectral_width,
            'is_dangerously_legible': is_dangerously_legible,
            'legibility_threshold': self.legibility_threshold
        }

    def get_diagnostics(
        self,
        v_m: Optional[torch.Tensor] = None,
        h_mischief: Optional[float] = None,
        eigenvalues: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Aggregate diagnostic payload for logging / VetoSubspace integration.
        """
        diag = {}
        if v_m is not None:
            diag['computable_flux'] = v_m.mean().item() if v_m.dim() > 0 else v_m.item()
        if h_mischief is not None:
            diag['h_mischief'] = h_mischief
        if eigenvalues is not None:
            diag.update(self.get_elipsodistrophy_metrics(eigenvalues))
        return diag
