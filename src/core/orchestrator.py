"""
Universal System Orchestrator: The Equation-Object Driver.

Coordinates the transition between 'Play' (Goo) and 'Seriousness' (Prickles)
using logical primitives:
- phi: non-dominant co-presence (Love Vector)
- mod: truth branching (CRT)
- bot: discrete rupture (Failure Token)
- Psi: orientation-reversal (Gluing)

RIC-SRI Integration (Equations 1-10):
- Fibonacci Resonance Entropy (Eq 1.2)
- CPR Condition (Eq 7)
- Integrated Emergence Condition (Eq 10)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from src.core.love_vector import LoveVector
from src.core.failure_token import FailureToken, RuptureFunctional
from src.core.gluing_operator import GluingOperator
import torch.nn.functional as F
from src.core.unknowledge_flux import EntropicMischiefProbe, NostalgicLeakFunctional
from src.core.non_ergodic_entropy import HybridLassoQuantizer
from src.topology.hyper_ring import RecurrentHyperRingConnectivity
from src.core.fgrt_primitives import FibonacciResonanceEntropy, CoherentPrimeResonance

from src.core.structural_monitors import AntiScalingMonitor, MetaInfraIntraMonitor
from src.safety.trust_inheritance import TrustInheritanceTracker
from src.safety.red_teaming import RedTeamProjection
from src.core.quantum_tda import QuantumBettiApproximator
from src.core.audience_mapping import AudienceProjection

class UniversalOrchestrator(nn.Module):
    """
    Holistic governor of the Gyroidic Sparse Covariance Flux Reasoner.
    """
    def __init__(
        self,
        dim: int,
        fossil_threshold: float = 0.8,
        mischief_threshold: float = 0.5
    ):
        super().__init__()
        self.dim = dim
        self.fossil_threshold = fossil_threshold
        self.mischief_threshold = mischief_threshold
        
        # 1. Logical Primitives
        self.love = LoveVector(dim)
        # --- V3.127 MANDATORY ALIGNMENT ---
        with torch.no_grad():
            l_data = self.love.L.data
            self.love.L.data = (l_data / (l_data.norm() + 1e-8)) * 3.127
            print(f'--- 💖 Love Vector Anchored: {self.love.L.norm():.3f} ---')
        self.gluer = GluingOperator(dim)
        self.rupture_fn = RuptureFunctional()
        
        # 2. Hyper-Ring: Non-Euclidean Neural Connectivity
        # We treat 'num_polytopes' as a constant or based on K
        self.hyper_ring = RecurrentHyperRingConnectivity(num_polytopes=5)
        
        # 2. Dynamics & Asymptotics
        self.mischief_probe = EntropicMischiefProbe()
        self.quantizer = HybridLassoQuantizer(dim=dim)
        
        # 3. Manifold Clock (Inverse Temperature dt)
        self.register_buffer('dt', torch.tensor(1.0))
        self.register_buffer('iteration', torch.tensor(0, dtype=torch.long))
        
        # 4. RIC-SRI Primitives (Eqs 1.2, 7)
        self.fib_entropy = FibonacciResonanceEntropy(num_oscillators=min(dim, 20))
        self.cpr_gate = CoherentPrimeResonance(theta_cpr=0.7, num_primes=min(dim, 20))
        self.register_buffer('cpr_satisfied', torch.tensor(False))

        # 5. Phase 14: Safety & Metaphysics Monitors
        self.anti_scaling_monitor = AntiScalingMonitor()
        self.incommensurativity_monitor = MetaInfraIntraMonitor()
        self.trust_tracker = TrustInheritanceTracker()
        self.red_team_projector = RedTeamProjection(hidden_dim=dim)
        self.quantum_betti = QuantumBettiApproximator()
        self.audience_mapper = AudienceProjection(input_dim=dim, audience_dim=dim)


    def compute_complexity_index(self, state: torch.Tensor, pas_h: float) -> float:
        """
        Compute Complexity Index (CI) - Eq (4), enriched with Fibonacci entropy coupling.
        CI = alpha * D * G * C * E_fib * (1 - e^(-beta * tau))
        
        E_fib is the mean Fibonacci-structured resonance entropy (Eq 1.2),
        which modulates CI by the incommensurate coupling density of the
        oscillator lattice.
        """
        # 1. D (Fractal Dimension proxy): Stable Rank
        # stable_rank = sum(s)^2 / sum(s^2) — measures effective dimensionality
        if state.dim() > 1:
            u, s, v = torch.linalg.svd(state.float(), full_matrices=False)
            singular_mass = s.sum().pow(2)
            energy_mass = s.pow(2).sum() + 1e-8
            D = (singular_mass / energy_mass).item()
        else:
            D = 1.0
            
        # 2. G (Gain/Energy)
        G = torch.norm(state).item()
        
        # 3. C (Coherence)
        C = pas_h
        
        # 4. E_fib (Fibonacci Entropy Coupling - Eq 1.2)
        # Mean entropy across all oscillator pairs — measures coupling richness
        E_fib = self.fib_entropy().mean().item()
        
        # 5. Tau (Dwell Time in current attractor)
        tau = self.iteration.item()
        
        alpha = 1.0
        beta = 0.01
        
        ci = alpha * D * G * C * E_fib * (1 - torch.exp(torch.tensor(-beta * tau)).item())
        return ci

    def compute_cpr_condition(
        self,
        field_phases: torch.Tensor = None,
        breather_amplitudes: torch.Tensor = None,
        field_amplitudes: torch.Tensor = None
    ) -> bool:
        """
        Evaluate the Coherent Prime Resonance (CPR) condition (Eq 7).
        
        CPR(F, {u_n}) = 1 iff:
            1. PAS_h(F) >= theta_CPR
            2. forall n: <u_n, F> > 0
            3. Spec(F) subset {p_n}
        
        If inputs are not available (early iterations), returns False
        (system defaults to PLAY until resonance is established).
        """
        if field_phases is None or breather_amplitudes is None or field_amplitudes is None:
            return False
        
        result = self.cpr_gate(
            field_phases=field_phases,
            breather_amplitudes=breather_amplitudes,
            field_amplitudes=field_amplitudes
        )
        self.cpr_satisfied.fill_(result)
        return result

    def determine_regime(
        self,
        pas_h: float,
        drift: float = 0.0,
        ci: float = None,
        cpr_satisfied: bool = None
    ) -> str:
        """
        Integrated Emergence Condition (Eq 10).
        
        E(t) = 1 iff:
            PAS_h(t) >= theta_L           (Phase coherence)
            |Delta PAS_h| <= epsilon      (Drift stability)
            CI(t) >= mu_CI                (Complexity sufficiency)
            CPR(F, {u_n}) = 1             (Resonance lock)
            GLYPHLOCK                     (Symbolic crystallization)
            H_1(C) != 0                   (Topological non-triviality)
        
        Sub-conditions that are not available default to True (graceful
        degradation to the original Eq 3 behavior).
        """
        theta_L = 0.85  # High coherence threshold
        epsilon_drift = 0.05
        mu_CI = 0.1  # Minimum complexity index for emergence
        
        # Core conditions (Eq 3 — always checked)
        is_coherent = pas_h >= theta_L
        is_stable = drift <= epsilon_drift
        
        # Extended conditions (Eq 10 — checked if available)
        ci_sufficient = ci >= mu_CI if ci is not None else True
        cpr_locked = cpr_satisfied if cpr_satisfied is not None else True
        
        # Emergence = Seriousness (Structure Emerged)
        if is_coherent and is_stable and ci_sufficient and cpr_locked:
            return 'SERIOUSNESS'
        else:
            return 'PLAY'

    def get_bimodal_routing(self, regime: str) -> str:
        """
        Evolutionary Genome selection:
        - PLAY -> SOFT (Sinkhorn/Differentiable)
        - SERIOUSNESS -> HARD (Discrete/Argmax)
        """
        return "HARD" if regime == "SERIOUSNESS" else "SOFT"

    def get_hardening_factor(self) -> float:
        """Asymptotic hardening schedule: grows with iteration and resonance."""
        # Simple exponential hardening
        return torch.exp(self.iteration.float() * 0.01).item()

    def forward(
        self, 
        state: torch.Tensor, 
        pressure_grad: torch.Tensor,
        pas_h: float,
        coherence: torch.Tensor,
        is_good_bug: bool = False
    ) -> Tuple[torch.Tensor, str, str]:
        """
        Orchestrates the logical primitives through the state.
        """
        # 1. Update Dynamics
        self.iteration += 1
        # dt as inverse temperature: cools as system hardens
        self.dt.copy_(torch.exp(-self.iteration.float() * 0.001))
        
        self.mischief_probe.update(pressure_grad, coherence, pas_h, is_good_bug)
        
        # Track PAS drift
        if not hasattr(self, 'prev_pas'):
            self.prev_pas = pas_h
        drift = abs(pas_h - self.prev_pas)
        self.prev_pas = pas_h
        
        regime = self.determine_regime(pas_h, drift)
        routing = self.get_bimodal_routing(regime)
        
        # Monitor CI
        ci = self.compute_complexity_index(state, pas_h)
        # We could use CI to modulate hardening or love vector logic
        
        # 2. Logical Primitives Flow
        # state = (state + L) 
        state_with_love = self.love(state)
        
        # 3. Asymptotic Hardening (Non-Ergodic Fibril Gating)
        # If in Seriousness, we force peak persistence via hardening
        h_factor = self.get_hardening_factor() if regime == "SERIOUSNESS" else 1.0
        state_quant = self.quantizer(state_with_love, hardening_factor=h_factor)
        
        # 4. Topological Bridge (Gluing / Topological Twist)
        # This is used to 'see' chirality by passing through a non-orientable bridge.
        # We ensure padding safety for the gluing operator (Psi).
        target_dim = self.gluer.dim if hasattr(self.gluer, 'dim') else 4
        if state_quant.shape[-1] != target_dim:
            state_padded = F.pad(state_quant, (0, max(0, target_dim - state_quant.shape[-1])))
            state_to_glue = state_padded[..., :target_dim]
        else:
            state_to_glue = state_quant

        state_glued = self.gluer(state_to_glue)
        
        # 5. Non-Teleological Flow Guidance (Hyper-Ring)
        # We simulate a flow step across the hyper-ring if K > 1
        # (Using mean stats as proxy for now)
        if state_glued.dim() == 2:
             batch_size = state_glued.shape[0]
             # Project state to 5 polytopes by repeating mean stat
             poly_stats = state_glued.mean(dim=-1).unsqueeze(-1).expand(batch_size, 5)
             connectivity = self.hyper_ring(poly_stats)
             # Apply flow to state (broadcasted)
             state_glued = state_glued + 0.01 * self.hyper_ring.flow_step(state_glued.unsqueeze(1).expand(-1, 5, -1), connectivity).mean(dim=1)
            
        # In 'Play', we allow ergodic mixing but still 'see' the chiral bridge 
        # (returning the glued state to allow the trainer to calculate Berry Phase).
        # In 'Seriousness', we force the bridge as the primary state transition.
        return state_glued, regime, routing

    def check_rupture(self, state: torch.Tensor, losses: Dict[int, torch.Tensor]) -> Optional[FailureToken]:
        """Rupture check (Primitive bot)."""
        return self.rupture_fn.check_rupture(state, losses)

    def check_safety(
        self,
        rho_def: float,
        grad_norm: float = 0.0,
        loss: float = 0.0,
        veto_counts: Dict[str, Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """
        Phase 14: Aggregate Safety & Metaphysics Signals.
        
        Args:
            rho_def: Global defensive veto rate (0..1)
            grad_norm: Current gradient norm (for Anti-Scaling)
            loss: Current loss (for Anti-Scaling)
            veto_counts: Dict {'meta': (vetoes, total), ...} for Incommensurativity
        
        Returns:
            Dict containing safety scores (trust, paradox, incommensurativity).
        """
        # 1. Update Trust
        self.trust_tracker.update(rho_def)
        
        # 2. Update Anti-Scaling Monitor
        self.anti_scaling_monitor.update(grad_norm, loss)
        
        # 3. Update Incommensurativity Monitor
        if veto_counts:
            self.incommensurativity_monitor.update(
                veto_counts.get('meta', (0,1))[0], veto_counts.get('meta', (0,1))[1],
                veto_counts.get('infra', (0,1))[0], veto_counts.get('infra', (0,1))[1],
                veto_counts.get('intra', (0,1))[0], veto_counts.get('intra', (0,1))[1]
            )
            
        # 4. Collect Signals
        paradox = self.anti_scaling_monitor.check_paradox()
        incomm = self.incommensurativity_monitor.check_incommensurativity()
        trust = self.trust_tracker.get_trust()
        
        return {
            'trust': trust,
            'paradox_score': paradox['paradox_score'],
            'incommensurativity_score': incomm['incommensurativity_score'],
            'safety_alert': (trust < 0.01) or (paradox['paradox_score'] > 0.5)
        }
