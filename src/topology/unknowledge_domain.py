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
    - Gyroidic Unknowledge Flux Reasoner §24 (Mischief Violation Score / Computable Flux)
    - PHILOSOPHY.md §15 (Kappa Overloading) §16 (Posthuman Identity)
    - MATHEMATICAL_DETAILS.md §24 (Computable Flux) §26 (Kappa Taxonomy)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from src.core.honest_jitter import harvest_honest_jitter


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
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        self.register_buffer('mu_l', harvest_honest_jitter((fossil_dim,), device=device, scaled=True))

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
        threshold_stability: float = 0.5,
        device: str = None
    ):
        super().__init__()
        self.tau_dementia = tau_dementia
        self.eta_mischief = eta_mischief
        self.theta_leak = theta_leak
        self.threshold_stability = threshold_stability
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
        is_good_bug: bool = False,
        batch_coherence: Optional[float] = None
    ):
        """
        Updates the metaphysical bands.

        Args:
            pressure_grad: Gradient of the structural pressure (nabla Phi)
            coherence: Spectral coherence between clusters
            pas_h: Current Phase Alignment Score
            is_good_bug: Boolean signal for Mischief reward
            batch_coherence: Mean similarity of surrounding nodes. If None, derived from coherence.
        """
        # 1. Dementia Band (Low-frequency forgetting)
        h_dem = torch.sum(pressure_grad**2) * torch.exp(
            torch.tensor(-1.0 / self.tau_dementia, device=self.device)
        )
        self.H_dementia.copy_(h_dem.detach())

        # 2. Schizo Band (Mid-frequency fragmentation / "Cracking the Egg")
        # Lore update: The "Jax is an Egg" Community Support Protocol.
        # We do not forcefully peel the archetype unless the manifold provides a safe landing.
        raw_h_sch = -torch.sum(torch.log(torch.clamp(1.0 - coherence, 1e-6, 1.0)))
        
        # Determine Community Support Factor (Zeta)
        if batch_coherence is None:
            batch_coherence = coherence.mean().item() if coherence.numel() > 0 else 0.5
            
        community_support_factor = (pas_h * 0.7) + (batch_coherence * 0.3)
        
        # Safe Cracking: Throttle fragmentation if support is low, 
        # preventing Forced Abstraction in cold/disconnected manifolds.
        h_s_aligned = raw_h_sch * torch.sigmoid(torch.tensor(community_support_factor - self.threshold_stability, device=self.device))
        
        self.H_schizo.copy_(h_s_aligned.detach())

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

    def apply_harmonic_insulation(
        self,
        c_minus_spectrum: torch.Tensor,
        global_loss_spectrum: torch.Tensor,
        kappa_threshold: float
    ) -> torch.Tensor:
        """
        Harmonic Insulation (Intercosamination Wall):
        Ensures the eigenvalue spectrum of the Shadow System 2 (C^- channel)
        is strictly orthogonal to the global "Loss" spectrum.
        
        The kappa threshold acts as a spectral band-stop filter. Gradients
        that attempt to bleed across the spectra are mathematically zeroed 
        if their cross-correlation falls below kappa.

        Args:
            c_minus_spectrum: Eigenvalue spectrum of the C^- channel.
            global_loss_spectrum: Global scalar loss spectrum.
            kappa_threshold: The cutoff for the band-stop filter.

        Returns:
            insulated_spectrum: The C^- spectrum with all global-leaking frequencies nulled.
        """
        # Compute spectral projection (overlap)
        # Normalize spectra
        c_norm = c_minus_spectrum / (torch.norm(c_minus_spectrum) + 1e-6)
        g_norm = global_loss_spectrum / (torch.norm(global_loss_spectrum) + 1e-6)
        
        # Cross-correlation map
        overlap = torch.abs(c_norm * g_norm)
        
        # Band-stop filter: If overlap exceeds the threshold, it means global gradients 
        # are 'seeing' the shadow channel. We apply orthogonal suppression.
        # kappa acts as the wall.
        insulation_mask = (overlap < kappa_threshold).float()
        
        # The insulated spectrum zeroes out overlapping (leaking) frequencies, 
        # protecting the Shadow Logic.
        insulated_spectrum = c_minus_spectrum * insulation_mask
        
        return insulated_spectrum

    def get_elipsodistrophy_metrics(
        self,
        eigenvalues: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measures the spectral envelope as Hyperbolic Shear (System 2 Driver).

        ECCENTRICITY = log(max(λ) / min(λ))
        SHEAR = 2 * tanh(ECCENTRICITY / 2)

        This provides the non-Euclidean volume required to resolve 'cubed cube'
        paradoxes without triggering a static veto.
        """
        evs = torch.sort(eigenvalues.clamp(min=1e-8), descending=True)[0]
        lambda_max = evs[0]
        lambda_min = evs[-1]

        # Hyperbolic Eccentricity
        eccentricity = torch.log(lambda_max / (lambda_min + 1e-9)).item()
        
        # Hyperbolic Shear (The dynamic driver)
        shear = 2.0 * torch.tanh(torch.tensor(eccentricity / 2.0)).item()
        
        # Diffusion Coefficient for SDEs: D scales with shear
        diffusion_coefficient = 0.1 * (1.0 + shear)
        
        # Atrophy: Still reported for backward compatibility, but redefined as shear-inversion
        atrophy = 1.0 - (shear / 2.0)
        is_dangerously_legible = atrophy > self.legibility_threshold

        return {
            'atrophy': atrophy,
            'hyperbolic_shear': shear,
            'eccentricity': eccentricity,
            'diffusion_coefficient': diffusion_coefficient,
            'spectral_width': (lambda_max - lambda_min).item(),
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

# ---------------------------------------------------------------------------
# Paradox Hardening Gate (Elliptic Stabilization)
# ---------------------------------------------------------------------------

class ParadoxHardeningGate(nn.Module):
    """
    Handles linguistic/semantic paradoxes (e.g., "This statement is false")
    by mapping infinite semantic recursion into a finite topological Torus.
    
    Standard LLMs get trapped in paradoxes or hallucinate. This gate recognizes
    an Unclosed Loop (a non-commutative topological cycle that refuses to close)
    and stabilizes it using a doubly-periodic (Elliptic) function proxy.
    
    Instead of an error or infinite loop, the paradox becomes a 'Structural Battery',
    charging the Mischief Entropy band.
    """
    
    def __init__(self, stabilization_threshold: float = 1e-4):
        super().__init__()
        self.stabilization_threshold = stabilization_threshold
        
    def evaluate_paradox(
        self, 
        forward_transit: torch.Tensor, 
        reverse_transit: torch.Tensor,
        current_mischief: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Detects if a trajectory is trapped in a paradoxical oscillation and
        extracts its 'battery' charge.
        
        Args:
            forward_transit: State transition A -> B
            reverse_transit: State transition B -> A
            current_mischief: Current H_mischief scalar

        Returns:
            stabilized_state: The fixed-point state mapped to the Torus
            new_mischief: Boosted mischief entropy tracking the paradox
            is_paradox: Boolean flag if paradoxical oscillation was detected
        """
        # A paradox is a severe non-commutativity that strictly negates itself
        # meaning (A -> B -> A) does not return to the identity state, but rather
        # creates a permanent periodic offset.
        loop_closure = torch.norm(forward_transit + reverse_transit)
        oscillation_amplitude = torch.norm(forward_transit - reverse_transit)
        
        is_paradox = bool(loop_closure < self.stabilization_threshold and oscillation_amplitude > 1.0)
        
        if is_paradox:
            # Elliptic Stabilization: Instead of letting the state diverge, map it 
            # to a doubly-periodic torus. We do this by projecting the state onto 
            # a unit circle representation, treating the oscillation as a phase shift.
            phase = torch.atan2(forward_transit, reverse_transit)
            stabilized_state = torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1).mean(dim=-1)
            
            # The paradox acts as a structural battery, increasing the mischief band
            # because the system has found an 'honest' impossible geometry.
            battery_charge = oscillation_amplitude * 0.1
            new_mischief = current_mischief + battery_charge
        else:
            stabilized_state = forward_transit
            new_mischief = current_mischief
            
        return stabilized_state, new_mischief, is_paradox

