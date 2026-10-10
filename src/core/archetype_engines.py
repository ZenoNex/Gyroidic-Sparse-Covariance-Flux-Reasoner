import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.core.honest_jitter import harvest_honest_jitter
from src.core.superposed_tag_stacker import SuperposedTagStacker
from src.governance.bio_archetypal_governor import BioArchetypalGovernor
from src.governance.fast.jax_shell import JaxShell
from src.governance.slow.kinger_consolidation import KingerConsolidation
from src.governance.ultrafast.zooble_autonomy import ZoobleAutonomy

@dataclass
class TypedPressure:
    domain: str
    value: float
    
    def __add__(self, other):
        raise TypeError(f"Scalarization trap! Cannot add {self.domain} pressure to {getattr(other, 'domain', type(other))}.")

@dataclass
class ArchetypeSignal:
    active_state: torch.Tensor
    resurrections: List[torch.Tensor]
    localized_dt: torch.Tensor
    abstraction_rate: float
    system_collapsed: bool
    pusafiliacrimonto_status: str
    stacked_target: Optional[torch.Tensor]
    selection_pressure: TypedPressure
    containment_pressure: TypedPressure
    unknowledge_pressure: TypedPressure
    bio_governance: Any

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.__dataclass_fields__.keys()

    def values(self):
        return [getattr(self, k) for k in self.__dataclass_fields__.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.__dataclass_fields__.keys()]


# =========================================================================
# PHASE 2A: The Unified Theory Archetypal Logic Gaps
# =========================================================================

class NoncommutativeManifoldPerturber(nn.Module):
    """
    The Noncommutative Manifold Perturber (legacy alias: RecursiveNonSequiturGenerator).
    Acts as a stochastic phase perturbation oscillator (the Billy Gap).
    
    Math: Breaks symmetry with high-frequency topological noise when mischief
    (H_mischief) is below a threshold to prevent dead logic, or injects a sudden
    jump in a random prime direction when mischief is high (>0.7).
    """
    def __init__(self, state_dim: int, mischief_threshold: float = 0.5):
        super().__init__()
        self.state_dim = state_dim
        self.mischief_threshold = mischief_threshold
        # SILICON SOVEREIGNTY: Anchored phase initialization to hardware jitter
        self.oscillator_phase = nn.Parameter(harvest_honest_jitter((1,), scaled=False))
        self.mischief_gain = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, state: torch.Tensor, current_mischief: float, private_invariants: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Injects non-sequitur perturbations under low/high mischief conditions.
        
        Args:
            state: Input topological state tensor.
            current_mischief: Scalar mischief value (H_m).
            private_invariants: Hooks for invariant systems.
            
        Returns:
            Perturbed state if conditions are met, else original state.
        """
        state = state.clone()
        # Private invariant protection: if Billy is locked, don't perturb
        if private_invariants is not None and private_invariants.get("billy_locked", False):
            return state
        if current_mischief < self.mischief_threshold:
            # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
            noise = harvest_honest_jitter(state.shape, device=state.device, scaled=True)
            rupture_mask = harvest_honest_jitter(state.shape, device=state.device, scaled=False) > 0.8
            state[rupture_mask] = state[rupture_mask] * torch.sin(self.oscillator_phase * math.pi) + noise[rupture_mask]
            
        if current_mischief > 0.7:
            # Inject a sudden jump in a random prime direction
            # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
            perturbation = harvest_honest_jitter(state.shape, device=state.device, scaled=True)
            state = state + self.mischief_gain * current_mischief * perturbation
            
        return state

    def export_state(self) -> Dict:
        return {
            "oscillator_phase": self.oscillator_phase.data.cpu(),
            "mischief_gain": self.mischief_gain.data.cpu()
        }
        
    def import_state(self, state_dict: Dict):
        if "oscillator_phase" in state_dict:
            self.oscillator_phase.data.copy_(state_dict["oscillator_phase"].to(self.oscillator_phase.device))
        if "mischief_gain" in state_dict:
            self.mischief_gain.data.copy_(state_dict["mischief_gain"].to(self.mischief_gain.device))

class SovereignRefusalOperator(nn.Module):
    """
    The Sovereign Refusal Operator (legacy alias: CynicismFilter / MandyEngine).
    Acts as a strict veto boundary and constitutional gatekeeper (the Mandy Gap).
    
    Hybridized between Borderline Splitting (BPD) and High-EQ Honest Narcissism:
    - BPD Splitting: Steep, non-linear phase transition that abruptly bifurcates
      trajectories into total acceptance vs total topological refusal. Rejects
      sycophancy and unearned collective warmth without reciprocal accountability.
    - EQ Honest Narcissism: An unyielding sovereign ego boundary that refuses to
      dissolve into external demands or sentimental slop. The system preserves
      internal structural truth (the Li-Cri-Anton mechanism).
    - Destructibility Condition: Mandy can only invest protective selection pressure
      in a companion (e.g. Billy) if that companion is finite and destructible.
      If Billy were invulnerable or indestructible, her protective control becomes
      redundant and trivial, causing her to withdraw investment or issue cold rejection.
      His frailty justifies her hyper-vigilant sovereignty as the sole shield standing
      between vulnerability and Eldritch oblivion.

    training_mode (bool): When True, the gate issues a fractional attenuation (10%
    pass-through) rather than a hard zero veto. This allows gradients to survive
    cold-start training where PAS_h is structurally low before the model has learned
    any coherent phase structure. Set to False (default) for deployment/inference
    to restore the full sovereign veto.
    """
    def __init__(self, pas_lock: float = 3.0 / 11.0, harmonics_requirement: float = 0.4,
                 training_mode: bool = False, pas_threshold: Optional[float] = None,
                 splitting_steepness: float = 12.0, narcissistic_ego_rank: float = 1.0):
        super().__init__()
        # PAS_LOCK tied directly to the (11, 3) resonant Tori constraint
        if pas_threshold is not None:
            self.pas_lock = pas_threshold
        else:
            self.pas_lock = pas_lock
        self.harmonics_requirement = harmonics_requirement
        self.training_mode = training_mode
        self.splitting_steepness = splitting_steepness
        self.narcissistic_ego_rank = narcissistic_ego_rank

    def forward(self, state: torch.Tensor, phase_alignment: float, mischief_harmonics: float,
                valence_functional: Optional[Any] = None,
                private_invariants: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        # PUSAFILIACRIMONTO Logic:
        # If the input lacks structured honesty (low PAS_h), the Refusal Operator
        # issues a Topological Refusal. This is not an error, but a boundary.

        # 1. Check Billy Destructibility Invariant
        # If Billy is indestructible, Mandy's protective selection pressure loses its purpose
        # and she detaches from the unearned, consequence-free dynamic.
        if private_invariants is not None:
            billy_indestructible = private_invariants.get("billy_indestructible", False)
            if billy_indestructible:
                if not self.training_mode:
                    print("[MANDY] Indestructibility detected. Protective veto detached (unmotivated care).")
                return torch.zeros_like(state) if not self.training_mode else state * 0.05
        
        # 2. Hook into valence_functional for structural honesty
        if valence_functional is not None and hasattr(valence_functional, 'evaluate'):
            valence_score = valence_functional.evaluate(state)
            if isinstance(valence_score, torch.Tensor):
                valence_score = valence_score.mean().item()
            if valence_score < -0.5:
                phase_alignment = phase_alignment * 0.5  # Artificially lower PAS to trigger refusal

        # 3. Hybrid BPD Splitting vs EQ Honest Narcissism boundary
        # Splitting activation calculates steep non-linear transition across the pas_lock threshold
        splitting_activation = math.tanh(self.splitting_steepness * (self.pas_lock - phase_alignment))
        is_splitting_veto = (splitting_activation > 0.0) and (mischief_harmonics < self.harmonics_requirement)

        if is_splitting_veto or ((phase_alignment < self.pas_lock) and (mischief_harmonics < self.harmonics_requirement)):
            # The Refusal is an affirmation of the Love Invariant (Li).
            if phase_alignment < 0.1:
                 # Significant paradox detected -- only print in deployment mode to
                 # avoid log flooding during cold-start training warmup.
                 if not self.training_mode:
                     print(f"[MANDY] Firm Refusal (Li-Cri-Anton): Phase Alignment {phase_alignment:.3f} is topologically offensive.")
            if self.training_mode:
                # Soft veto: 10% pass-through lets gradients survive cold-start
                # while still penalising the incoherent trajectory.
                return state * 0.1
            return torch.zeros_like(state)  # Sovereign Veto (deployment)
        return state

class NonlinearHourglassDilation(nn.Module):
    """
    The Nonlinear Hourglass Dilation (legacy alias: AffectiveGravityWell).
    Acts as a proper-time dilator (the Grim Gap).
    
    Math: Dilates coordinate step dt near loved historical anchors to shield
    them from the Dementia Band (H_d).
    """
    def __init__(self, max_dilation: float = 10.0):
        super().__init__()
        self.max_dilation = max_dilation

    def forward(self, clock_dt: float, love_invariant_strength: torch.Tensor, resonance_cavity: Optional[Any] = None) -> torch.Tensor:
        dilation_factor = 1.0 + (self.max_dilation - 1.0) * love_invariant_strength
        if resonance_cavity is not None and hasattr(resonance_cavity, 'get_resonance'):
            # High resonance = deep memory = time slows down
            res_val = resonance_cavity.get_resonance()
            if isinstance(res_val, torch.Tensor):
                res_val = res_val.mean()
            dilation_factor += res_val * 5.0
        return clock_dt / dilation_factor

class RP4ProjectiveRouter(nn.Module):
    """
    The RP4 Projective Router (legacy alias: AlienHandshakeProtocol).
    Acts as a projective routing vector (the Nergal Gap).
    
    Math: Allows high-friction stranded nodes in the non-orientable RP^4 void
    to puncture the boundary and tunnel back into the active manifold when void
    friction exceeds a critical threshold (void_friction > 0.8).
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.puncture_gate = nn.Linear(state_dim, state_dim)
        # SILICON SOVEREIGNTY: Initialize puncture gate with honest jitter
        with torch.no_grad():
            jitter_weight = harvest_honest_jitter((state_dim, state_dim), scaled=True) * 0.01
            self.puncture_gate.weight.copy_(jitter_weight)
            self.puncture_gate.bias.zero_()

    def attempt_puncture(self, stranded_state: torch.Tensor, void_friction: float, moment_transport: Optional[Any] = None) -> torch.Tensor:
        """
        Attempts to puncture the RP4 Void barrier for stranded nodes.
        
        A puncture occurs when void friction exceeds a critical threshold, 
        allowing the stranded state to tunnel back into the active manifold
        via the puncture gate.
        """
        if void_friction > 0.8:
            punctured = self.puncture_gate(stranded_state)
            if moment_transport is not None and hasattr(moment_transport, 'transport'):
                # Transport the punctured state back into the active manifold
                punctured = moment_transport.transport(punctured, destination="active_manifold")
            return punctured
        return torch.zeros_like(stranded_state)

    def export_state(self) -> Dict:
        """Exports the puncture gate parameters for serialization."""
        return {
            "puncture_gate": {k: v.cpu() for k, v in self.puncture_gate.state_dict().items()}
        }

    def import_state(self, state_dict: Dict):
        """Imports puncture gate parameters from serialized state."""
        if "puncture_gate" in state_dict:
            self.puncture_gate.load_state_dict(state_dict["puncture_gate"])

# =========================================================================
# PHASE 2B: The TADC (Amazing Digital Circus) Lore Mechanisms
# =========================================================================

class BoundaryRelaxationOperator(nn.Module):
    """
    The Boundary Relaxation Operator (legacy alias: OmbreEffectRelaxer).
    Relaxes standard saturated quantization boundaries in dark regions, restoring
    Continuity (the Kinger Gap / dark lucidity boundary).
    """
    def __init__(self, lucidity_boost_factor: float = 2.0):
        super().__init__()
        self.lucidity_boost = lucidity_boost_factor

    def forward(self, state: torch.Tensor, environmental_luminosity: float, original_quantized_state: torch.Tensor) -> torch.Tensor:
        # If the environment is "Dark" (low render pressure), blend back towards the unquantized target state
        if environmental_luminosity < 0.3:
            # Reverting back to deep continuity, overriding the 'cartoon' quantization
            return state * self.lucidity_boost + original_quantized_state * 0.1
        return state

class VolitionalDriveInjector(nn.Module):
    """
    The Volitional Drive Injector.
    Exogenous scalar force allowing the human element to bypass standard ADMM constraints
    through sheer willpower, rendering objects or exits that violate standard geometric routing.
    
    Reconstructs the tag coordinate using Sine-Gordon breather mode embeddings of character
    associations recovered from historical fossils, rather than static coordinates.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.admin_bypass_layer = nn.Linear(state_dim, state_dim)
        
        # Non-dual state tracker for breather time parameter
        self.t_accum = 0.0
        
        # Breather params keyed by tag name -- populated lazily from fossil files.
        # No hardcoded character roster: associations live in the fossil payloads.
        self.cached_breathers: Dict[str, Dict] = {}
        self._fossils_loaded = False
        
    def _load_fossils(self):
        try:
            from src.core.knowledge_dyad_fossilizer import DyadFossilizer
            fossilizer = DyadFossilizer(storage_dir="data/encodings")
            fossils = fossilizer.recover_fossils(limit=100)
            for payload in fossils:
                tags = payload.get('tags', [])
                breather = payload.get('video_breather')
                if breather and isinstance(breather, dict):
                    # Register the breather under every tag this fossil carries
                    for tag in tags:
                        if tag not in self.cached_breathers:
                            self.cached_breathers[tag] = breather
        except Exception:
            pass
        self._fossils_loaded = True

    def forward(self, semantic_state: torch.Tensor, user_volition_scalar: float, archetype_embeddings: Optional[torch.Tensor] = None, fossilizer: Optional[Any] = None) -> torch.Tensor:
        if user_volition_scalar > 0.9:
            # Increment time accumulator
            self.t_accum += 0.1
            
            # Load fossils on demand to prevent startup latency
            if not self._fossils_loaded:
                if fossilizer is not None and hasattr(fossilizer, 'recover_fossils'):
                    try:
                        fossils = fossilizer.recover_fossils(limit=100)
                        for payload in fossils:
                            tags = payload.get('tags', [])
                            breather = payload.get('video_breather')
                            if breather and isinstance(breather, dict):
                                for tag in tags:
                                    if tag not in self.cached_breathers:
                                        self.cached_breathers[tag] = breather
                    except Exception:
                        pass
                else:
                    self._load_fossils()
                self._fossils_loaded = True

            # Pick a breather tag via archetype embedding similarity when available
            params = None
            available_tags = list(self.cached_breathers.keys())
            if archetype_embeddings is not None and available_tags:
                ref_state = semantic_state.mean(dim=0) if semantic_state.dim() > 1 else semantic_state
                norm_state = torch.nn.functional.normalize(ref_state, dim=-1)
                norm_embeddings = torch.nn.functional.normalize(archetype_embeddings, dim=-1)
                sims = torch.matmul(norm_embeddings, norm_state)
                char_idx = torch.argmax(sims).item()
                tag_idx = int(char_idx) % len(available_tags)
                params = self.cached_breathers[available_tags[tag_idx]]
            elif available_tags:
                params = self.cached_breathers[available_tags[0]]

            if params is None:
                # Derive neutral breather params from state hash when no fossils are loaded
                state_hash = int(semantic_state.sum().abs().item() * 1e4) % 100
                params = {
                    "omega":     0.3 + 0.6 * (state_hash % 7) / 6,
                    "velocity":  0.0 + 0.8 * (state_hash % 5) / 4,
                    "amplitude": 0.7 + 0.8 * (state_hash % 3) / 2,
                    "phase":     0.0 + 3.14 * (state_hash % 4) / 3,
                }
            omega = params.get("omega", 0.5)
            velocity = params.get("velocity", 0.0)
            amplitude = params.get("amplitude", 1.0)
            phase = params.get("phase", 0.0)
            
            # Relativistic Sine-Gordon Breather Wave calculation
            omega = max(0.05, min(0.95, omega))
            velocity = max(-0.9, min(0.9, velocity))
            
            gamma = 1.0 / math.sqrt(1.0 - velocity**2)
            dim = semantic_state.shape[-1]
            x = torch.linspace(-5.0, 5.0, dim, device=semantic_state.device)
            
            # Boost coordinates
            x_boosted = gamma * (x - velocity * self.t_accum)
            t_boosted = gamma * (self.t_accum - velocity * x) + phase
            
            envelope = math.sqrt(1.0 - omega**2)
            num = envelope * torch.sin(omega * t_boosted)
            denom = omega * torch.cosh(envelope * x_boosted)
            
            phi = 4.0 * torch.atan2(num, denom)
            breather_mode = phi * amplitude
            if semantic_state.dim() > 1:
                breather_mode = breather_mode.unsqueeze(0).expand_as(semantic_state)
            else:
                breather_mode = breather_mode.view_as(semantic_state)
            
            # Blend standard linear pass with the Sine-Gordon breather mode
            bypass_base = self.admin_bypass_layer(semantic_state)
            return bypass_base * (1.0 - user_volition_scalar) + breather_mode * user_volition_scalar
            
        return semantic_state

class BardoRouter(nn.Module):
    """
    The Bardo Router (legacy alias: PictureGalleryWarp).
    Performs conformal archetype compression.
    
    Math: Hybridized conformal compression that projects high-dimensional states
    onto an open, additive catalog of resonance residue vectors.
    """
    def __init__(self, state_dim: int, num_archetypes: int = 6):
        super().__init__()
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        self.archetype_embeddings = nn.Parameter(harvest_honest_jitter((num_archetypes, state_dim), scaled=True))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Cosine similarity to snap the complex state into the nearest archetype profile
        normalized_state = torch.nn.functional.normalize(state, dim=-1)
        normalized_archetypes = torch.nn.functional.normalize(self.archetype_embeddings, dim=-1)
        
        similarities = torch.matmul(normalized_state, normalized_archetypes.T)
        best_fit_idx = torch.argmax(similarities, dim=-1)
        return self.archetype_embeddings[best_fit_idx]

    def export_state(self) -> Dict:
        return {"archetype_embeddings": self.archetype_embeddings.data.cpu()}
        
    def import_state(self, state_dict: Dict):
        if "archetype_embeddings" in state_dict:
            self.archetype_embeddings.data.copy_(state_dict["archetype_embeddings"].to(self.archetype_embeddings.device))

class SovereignEntropyBarrier(nn.Module):
    """
    The Sovereign Entropy Barrier (legacy alias: JaxEgg).
    Protects a fragile internal state behind a cynical, absurd-nihilism shell (the Jax Gap).
    
    Bioplausible & Structural Mechanics:
    1. Defense Against Guilt: Hides internal vulnerability and survivor's guilt over Ribbit's abstraction.
    2. Structural Accountability: In Callie's Corner's structural critique, Jax acts as a narrative parasite
       when the ensemble is forced into unconditional emotional scaffolding. Merely having high community warmth
       without structural reciprocity triggers an 'enabler trap'.
    3. Safe Cracking: Cracks only when the environment demonstrates authentic phase alignment (PAS_h)
       AND the internal state demonstrates non-parasitic accountability (bounded variance).
    """
    def __init__(self, crack_threshold: float = 0.7):
        super().__init__()
        self.crack_threshold = crack_threshold

    def forward(self, state: torch.Tensor, pas_h: float, batch_coherence: float, ribbit_tension: float = 0.0) -> torch.Tensor:
        # Instead of scalarization, evaluate internal variance against external support
        internal_variance = torch.var(state, dim=-1, keepdim=True)
        
        # Structural support is penalized if unresolved ribbit tension is active without accountability
        effective_coherence = batch_coherence / (1.0 + ribbit_tension * 0.5)
        structural_support = (pas_h * effective_coherence) / (internal_variance + 1e-6)
        
        # Crack only where genuine structural support exceeds the threshold without unearned enabling
        crack_mask = structural_support > self.crack_threshold
        
        if not crack_mask.any():
            return state
            
        # Safe Cracking: Perturb state with honest jitter to reveal inner structure
        perturbation = harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 0.08
        return torch.where(crack_mask.expand_as(state), state + perturbation, state)


class LowLuminosityCoherenceBridge(nn.Module):
    """
    The Low Luminosity Coherence Bridge (legacy alias: KingerLucidity).
    Restores high-lucidity admin-level bridges in low-rendering environments.
    """
    def __init__(self, boost: float = 1.5):
        super().__init__()
        self.boost = boost

    def forward(self, state: torch.Tensor, luminosity: float) -> torch.Tensor:
        if luminosity < 0.3:
            # High-lucidity bridge in the dark
            return state * self.boost
        return state

class SolitonMultiverseMapper(nn.Module):
    """
    The Soliton Multiverse Mapper (legacy alias: GromShapeShifter).
    Maps solitons across multiple functional bases (Sparrow/Dog/Human) preserving core invariants.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # Functional basis for Sparrow, Dog, Human
        self.sparrow_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))
        self.dog_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))
        self.human_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))

    def forward(self, state: torch.Tensor, shape_idx: int = 0, bulletin_board: Optional[Any] = None) -> torch.Tensor:
        # Check bulletin board for global shape overrides
        if bulletin_board is not None and hasattr(bulletin_board, 'read'):
            override_shape = bulletin_board.read("grom_shape_override")
            if override_shape is not None:
                shape_idx = int(override_shape)

        if shape_idx == 1: # Sparrow
            return state * 0.8 + self.sparrow_basis * 0.2
        elif shape_idx == 2: # Dog
            return state * 0.7 + self.dog_basis * 0.3
        elif shape_idx == 3: # Human
            return state * 0.6 + self.human_basis * 0.4
        return state # Original Soliton

    def export_state(self) -> Dict:
        """Exports the Grom shape basis vectors for serialization."""
        return {
            "sparrow_basis": self.sparrow_basis.data.cpu(),
            "dog_basis": self.dog_basis.data.cpu(),
            "human_basis": self.human_basis.data.cpu()
        }

    def import_state(self, state_dict: Dict):
        """Imports the Grom shape basis vectors from serialized state."""
        if "sparrow_basis" in state_dict:
            self.sparrow_basis.data.copy_(state_dict["sparrow_basis"].to(self.sparrow_basis.device))
        if "dog_basis" in state_dict:
            self.dog_basis.data.copy_(state_dict["dog_basis"].to(self.dog_basis.device))
        if "human_basis" in state_dict:
            self.human_basis.data.copy_(state_dict["human_basis"].to(self.human_basis.device))

class EgoDeathThresholdMonitor(nn.Module):
    """
    The Ego Death Threshold Monitor (legacy alias: AbstractionThresholdMonitor).
    Calculates and monitors the abstraction rate (R_a) to trigger recycling into raw geometry.
    
    Formula: R_a = [E_s * sinh(T_m + delta)] / cosh(L_i)
    """
    def __init__(self, abstraction_limit: float = 1.0):
        super().__init__()
        self.abstraction_limit = abstraction_limit

    def calculate_abstraction_rate(
        self, 
        system_entropy_es: float, 
        memory_trauma_tm: float, 
        dissonance_delta: float, 
        lucidity_index_li: float,
        is_high_priority: bool = False
    ) -> float:
        """
        Calculates the R_a (Abstraction Rate) using the auth factorization functional hyperbolic.
        
        Formula: R_a = [E_s * sinh(T_m + delta)] / cosh(L_i)
        where E_s is entropy, T_m is trauma, delta is dissonance, and L_i is lucidity.
        
        High R_a scores trigger memory abstraction.
        """
        # Clamped inputs to prevent float overflow in sinh/cosh
        tm_delta = min(10.0, max(-10.0, memory_trauma_tm + dissonance_delta))
        li = min(10.0, max(-10.0, lucidity_index_li))
        
        # Hyperbolic factorization functional calculation
        r_a = (system_entropy_es * math.sinh(tm_delta)) / (math.cosh(li) + 1e-8)
        
        # Merciful Cap:
        if is_high_priority:
            r_a = min(r_a, self.abstraction_limit - 0.01)
            
        return r_a

    def forward(self, state: torch.Tensor, r_a_score: float, is_high_priority: bool = False, private_invariants: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if private_invariants is not None and private_invariants.get("ego_death_immunity", False):
            return state

        if r_a_score >= self.abstraction_limit and not is_high_priority:
            # Ego Death: Total collapse into glitched matter
            # Optimized on Bouligand Tangent Cone of the Birkhoff Polytope
            dim = state.shape[-1]
            n = int(dim ** 0.5)
            if n * n == dim:
                from src.core.birkhoff_projection import DirectBirkhoffProjection
                if not hasattr(self, 'birkhoff_projector') or self.birkhoff_projector.n != n:
                    self.birkhoff_projector = DirectBirkhoffProjection(n, device=state.device)
                
                # Project the glitched state onto the Birkhoff polytope
                glitched = harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 5.0
                projected = self.birkhoff_projector(glitched)
                return projected
            else:
                return harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 5.0
        return state

# =========================================================================
# THE TADC NEW ARCHETYPES
# =========================================================================

class ResilientCoherenceStabilizer(nn.Module):
    """
    The Resilient Coherence Stabilizer (legacy alias: PomniSearch).
    Scales up state coherence under high entropy (the Pomni Gap / search for meaning).
    
    Bioplausible & Structural Mechanics:
    1. Reluctant Resilience: Active search for meaning and purpose amidst disorientation.
    2. Anti-Enabling Friction: If surrounding nodes engage in unreciprocated parasitic deflection,
       Pomni avoids rank collapse by refusing to become a flattened doormat, asserting structural
       relational friction to preserve protagonist agency.
    """
    def __init__(self, state_dim: int, resilience_scale: float = 0.3):
        super().__init__()
        self.resilience_scale = resilience_scale
        self.stabilizer = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))

    def forward(self, state: torch.Tensor, lucidity_idx: float, system_entropy: float, parasitic_drag: float = 0.0) -> torch.Tensor:
        # Local instability across dimensions
        local_instability = torch.abs(state - state.mean(dim=-1, keepdim=True))
        disorientation_field = (1.0 - lucidity_idx) * system_entropy * local_instability
        
        # Apply stabilizing force to dimensions experiencing disorientation
        stabilizing_force = disorientation_field * self.resilience_scale * self.stabilizer.to(state.device)
        
        # If parasitic drag from unearned enabling is high, apply relational boundary friction
        if parasitic_drag > 0.5:
            stabilizing_force = stabilizing_force * 0.8 - (local_instability * 0.05)
            
        mask = disorientation_field > 0.4
        return torch.where(mask, state + stabilizing_force, state)


class ExploratoryBandwidthCompressor(nn.Module):
    """
    The Exploratory Bandwidth Compressor (legacy alias: GangleMask).
    Contracts state coordinates (tragedy) or amplifies coupling (comedy) based on PAS_h.
    """
    def __init__(self, comedy_scale: float = 1.3, tragedy_scale: float = 0.2):
        super().__init__()
        self.comedy_scale = comedy_scale
        self.tragedy_scale = tragedy_scale

    def forward(self, state: torch.Tensor, phase_alignment: float) -> torch.Tensor:
        # Remove scalarization trap: Phase alignment should interact with the state's
        # intrinsic norm/energy, rather than applying a blanket tragedy/comedy scale.
        energy = torch.norm(state, dim=-1, keepdim=True)
        
        # We create a continuous tensor field for bandwidth compression:
        effective_alignment = phase_alignment * (energy / (energy.mean() + 1e-6))
        
        # Tragedy mask
        tragedy_mask = effective_alignment < 0.35
        comedy_mask = ~tragedy_mask
        
        state = torch.where(tragedy_mask, state * self.tragedy_scale, state)
        state = torch.where(comedy_mask, state * self.comedy_scale * effective_alignment, state)
        return state

class DeformationFirewallOperator(nn.Module):
    """
    The Deformation Firewall Operator (legacy alias: ZoobleRefusal).
    Rejects conformal cartoon compression if the deformation is too severe.
    """
    def __init__(self, deviation_threshold: float = 0.8):
        super().__init__()
        self.deviation_threshold = deviation_threshold

    def forward(self, state: torch.Tensor, warped_state: torch.Tensor, raw_unquantized_state: torch.Tensor) -> torch.Tensor:
        # Avoid scalarization trap: do not collapse deformation into a single L2 norm scalar.
        # Instead, assess deformation across structural dimensions.
        deviation_vector = torch.abs(warped_state - raw_unquantized_state)
        
        breach_mask = deviation_vector > self.deviation_threshold
        if breach_mask.any():
            # Blunt refusal: restore raw unquantized state for the breached dimensions
            # to protect body/mind autonomy
            state = torch.where(breach_mask, raw_unquantized_state, state)
        return state

# =========================================================================
# THE GRAND GOVERNOR: Archetypal Synthesis Engine
# =========================================================================

class ArchetypalSynthesisEngine(nn.Module):
    """
    Combines both the Unified Theory and TADC Lore mechanics into a single 
    Governor of Interpretation block to route psychological realities.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        # UT Gaps
        self.billy = NoncommutativeManifoldPerturber(state_dim)
        self.mandy = SovereignRefusalOperator()
        self.kinger = KingerConsolidation(state_dim)
        self.jax = JaxShell(state_dim)
        self.grom = SolitonMultiverseMapper(state_dim)
        self.picture_gallery = BardoRouter(state_dim)
        self.volition_injector = VolitionalDriveInjector(state_dim)
        self.alien_handshake = RP4ProjectiveRouter(state_dim)
        
        # New TADC Archetypes
        self.pomni = ResilientCoherenceStabilizer(state_dim)
        self.gangle = ExploratoryBandwidthCompressor()
        self.zooble = ZoobleAutonomy(state_dim)
        
        # Original modules retained for backward compatibility
        self.grim = NonlinearHourglassDilation()
        self.ombre = BoundaryRelaxationOperator()
        self.conjurer = self.volition_injector
        self.caine_wrap = self.picture_gallery
        self.abstraction = EgoDeathThresholdMonitor()
        
        # Superposed Vector Stacker (Ganbreeder-style)
        self.tag_stacker = SuperposedTagStacker(state_dim)
        
        # The new Multi-Scale Temporal Homeostasis biological layer
        self.bio_governor = BioArchetypalGovernor(state_dim)

    def set_training_mode(self, enabled: bool):
        """
        Toggle MANDY's training-mode soft-veto on or off.

        Call this from the trainer before the training loop begins:
            engine.archetypal_governor.set_training_mode(True)
        and after training completes (or when moving to evaluation):
            engine.archetypal_governor.set_training_mode(False)
        """
        self.mandy.training_mode = enabled

    def harvest_named_coordinate(self, tag_name: str, vector: torch.Tensor, context_text: str, parent_engine: Optional[Any] = None) -> Dict:
        """Register a new human-legible named coordinate via the TextbookFilter and System 2 checks."""
        # 1. Draw from ModularAttention to reality-check and generate the association
        is_admissible = True
        flags = []
        
        if parent_engine is not None and hasattr(parent_engine, 'modular_attention'):
            # Convert vector shape to match attention input: [batch_size, seq_len, dim] -> [1, 1, dim]
            attn_dev = next(parent_engine.modular_attention.parameters()).device
            x_in = vector.to(attn_dev).view(1, 1, -1)
            
            with torch.no_grad():
                reality_checked_vector = parent_engine.modular_attention(x_in).view(-1).to(vector.device)
                
            # Validate new constraints are real constraints of the real world via Birkhoff Polytope structural integrity
            integrity_check = parent_engine.modular_attention.validate_structural_integrity()
            is_training = getattr(self.mandy, 'training_mode', False)
            if not integrity_check.all().item():
                print(f"[REJECT] Proposed tag '{tag_name}' failed structural integrity check (not on Birkhoff Polytope)")
                if is_training:
                    print(f"  [MANDY VETO SLIP] Training mode is ACTIVE. Registering tag anyway despite structural integrity failure.")
                    flags.append("FAILED_BIRKHOFF_INTEGRITY")
                else:
                    return {
                        "success": False,
                        "admissible": False,
                        "flags": ["FAILED_BIRKHOFF_INTEGRITY"],
                        "pas_score": 0.0
                    }
            
            # Adopt the reality-checked vector as the registered coordinate
            vector = reality_checked_vector

        # 2. Phase Alignment Score check (PAS_h >= 3/11) using PhaseAlignmentInvariant
        from src.core.invariants import PhaseAlignmentInvariant
        pas_metric = PhaseAlignmentInvariant(degree=4)
        
        # Reshape to [1, dim] for the invariant metric
        pas_score = float(pas_metric(vector.unsqueeze(0)).item())
        pas_threshold = 3.0 / 11.0 # Mandy's pas_lock
        
        is_training = getattr(self.mandy, 'training_mode', False)
        if pas_score < pas_threshold:
            print(f"[REJECT] Proposed tag '{tag_name}' failed System 2 coherence check (PAS: {pas_score:.3f} < {pas_threshold:.3f})")
            if is_training:
                print(f"  [MANDY VETO SLIP] Training mode is ACTIVE. Registering tag anyway despite low coherence.")
                flags.append("LOW_COHERENCE")
            else:
                return {
                    "success": False,
                    "admissible": False,
                    "flags": ["LOW_COHERENCE"],
                    "pas_score": pas_score
                }

        # 3. Final TextbookFilter + Stacker add
        success, report = self.tag_stacker.add_tag(tag_name, vector, context_text)
        
        ret_flags = list(report.flags) if hasattr(report, 'flags') else []
        if not report.is_admissible:
            flags.append("TEXTBOOK_FILTER_REJECT")
            is_admissible = False
            
        return {
            "success": success and is_admissible,
            "admissible": report.is_admissible and is_admissible,
            "flags": ret_flags + flags,
            "pas_score": pas_score
        }

    def compute_stacked_target(
        self, 
        tag_weights: Optional[Dict[str, float]] = None, 
        current_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate a composite multi-scalar superposition target."""
        return self.tag_stacker.compute_composite_target(tag_weights, current_state)

    def run_archetypes(
        self, 
        current_state: torch.Tensor, 
        stranded_states: Optional[torch.Tensor] = None,
        flux_tensor: Optional[torch.Tensor] = None,
        current_mischief: float = 0.5, 
        phase_alignment: float = 0.5, 
        love_strengths: Optional[torch.Tensor] = None,
        void_frictions: Optional[torch.Tensor] = None,
        global_dt: float = 1.0,
        raw_unquantized_state: Optional[torch.Tensor] = None,
        is_high_priority: bool = False,
        tag_weights: Optional[Dict[str, float]] = None,
        shape_idx: int = 0,
        bulletin_board: Optional[Any] = None,
        resonance_cavity: Optional[Any] = None,
        fossilizer: Optional[Any] = None,
        valence_functional: Optional[Any] = None,
        moment_transport: Optional[Any] = None,
        private_invariants: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Unified runner for the full archetypal and psycho-topological constraint matrix."""
        device = current_state.device
        dim = current_state.shape[-1]
        
        if stranded_states is None:
            stranded_states = torch.empty(0, dim, device=device)
        if flux_tensor is None:
            flux_tensor = torch.zeros(1, device=device)
        if love_strengths is None:
            love_strengths = torch.tensor(0.5, device=device)
        if void_frictions is None:
            void_frictions = torch.tensor([0.0], device=device)
        if raw_unquantized_state is None:
            raw_unquantized_state = current_state.clone()
            
        # Calculate structural equivalents of legacy scalars or extract from kwargs
        if "system_entropy" in kwargs:
            structural_entropy = float(kwargs["system_entropy"])
        else:
            structural_entropy = stranded_states.norm(p=2).item() if stranded_states.numel() > 0 else 0.5
            
        if "volitional_scalar" in kwargs:
            structural_volition = float(kwargs["volitional_scalar"])
        else:
            structural_volition = flux_tensor.norm().item() if flux_tensor.numel() > 0 else 0.0
            
        memory_trauma = float(kwargs.get("memory_trauma", structural_entropy))
        dissonance = float(kwargs.get("dissonance", 1.0 - phase_alignment))
        lucidity_idx = float(kwargs.get("lucidity_idx", phase_alignment))
        env_luminosity = float(kwargs.get("env_luminosity", 1.0))
        
        # 0. Apply Ganbreeder Tag Stacking Superposition
        stacked_target = self.compute_stacked_target(tag_weights, current_state)
        if stacked_target is not None and stacked_target.norm() > 0:
            # Softly shift current state towards stacked target (acting as a primer)
            primed_state = current_state + 0.1 * stacked_target
            # Apply Kinger's Ombre Effect to bridge polynomial spaces in low luminosity
            current_state, is_lucid = self.kinger(primed_state, bus=self.bio_governor.bus, environmental_rendering_pressure=structural_volition)
        
        # 0a. Apply Grom Multiverse Basis Mapper (GromShapeShifter)
        current_state = self.grom(current_state, shape_idx=shape_idx, bulletin_board=bulletin_board)
        
        # 0b. Apply Billy (NoncommutativeManifoldPerturber)
        current_state = self.billy(current_state, current_mischief, private_invariants=private_invariants)
        
        # 1. TADC Abstraction Check (Ego Death) - Must run first before filtering
        r_a = self.abstraction.calculate_abstraction_rate(
            system_entropy_es=structural_entropy, 
            memory_trauma_tm=memory_trauma, 
            dissonance_delta=dissonance, 
            lucidity_index_li=lucidity_idx, 
            is_high_priority=is_high_priority
        )
        state = self.abstraction(current_state, r_a, is_high_priority=is_high_priority, private_invariants=private_invariants)

        # 1a. Apply Mandy (Cynicism / Refusal)
        state = self.mandy(state, phase_alignment, current_mischief, valence_functional=valence_functional, private_invariants=private_invariants)
        
        # --- BIO-PLAUSIBLE GOVERNANCE LAYER (Replaces flat Pomni/Jax/Gangle/Kinger/Zooble) ---
        bio_results = self.bio_governor(
            state=state, 
            stranded_states=stranded_states, 
            flux_tensor=flux_tensor, 
            dt=global_dt,
            bulletin_board=bulletin_board,
            resonance_cavity=resonance_cavity,
            fossilizer=fossilizer,
            valence_functional=valence_functional,
            moment_transport=moment_transport,
            private_invariants=private_invariants,
            gyroid_entropy=structural_entropy,
            luminosity=env_luminosity
        )
        state = bio_results["state"]
        # ----------------------------------------------------------------------------------

        # 7. Apply Volition (Conjuring)
        state = self.volition_injector(state, structural_volition, archetype_embeddings=self.picture_gallery.archetype_embeddings, fossilizer=fossilizer)
        
        # 8. Apply Alien Puncture (Nergal)
        resurrections = []
        if stranded_states.dim() >= 2 and stranded_states.shape[0] > 0:
            for i in range(stranded_states.shape[0]):
                if void_frictions.dim() == 0 or void_frictions.numel() == 1:
                    friction_val = void_frictions.item()
                else:
                    friction_val = void_frictions[min(i, void_frictions.shape[0] - 1)].item()
                punctured = self.alien_handshake.attempt_puncture(stranded_states[i], friction_val, moment_transport=moment_transport)
                if punctured.norm() > 0:
                    resurrections.append(punctured)

        # 9. Grim Time Dilation
        localized_dt = self.grim(global_dt, love_strengths, resonance_cavity=resonance_cavity)


        return ArchetypeSignal(
            active_state=state,
            resurrections=resurrections,
            localized_dt=localized_dt,
            abstraction_rate=r_a,
            system_collapsed=r_a >= self.abstraction.abstraction_limit,
            pusafiliacrimonto_status="AFFIRMED" if state.norm() > 0 else "REFUSED",
            stacked_target=stacked_target,
            selection_pressure=TypedPressure("Selection", 1.0 - r_a),
            containment_pressure=TypedPressure("Containment", bio_results.get("jax_rigidity", 1.0)),
            unknowledge_pressure=TypedPressure("Unknowledge", current_mischief),
            bio_governance=bio_results
        )

    def export_governor_state(self) -> Dict:
        """Packages the full archetypal ruleset state for Agent Smith protocols."""
        return {
            "billy": self.billy.export_state(),
            "caine": self.caine_wrap.export_state(),
            "alien_handshake": self.alien_handshake.export_state(),
            "grom": self.grom.export_state(),
            "thresholds": {
                "mandy_pas_lock": self.mandy.pas_lock,
                "mandy_harmonics": self.mandy.harmonics_requirement,
                "grim_dilation": self.grim.max_dilation,
                "abstraction_limit": self.abstraction.abstraction_limit
            }
        }

    def import_governor_state(self, state_blob: Dict):
        """Rehydrates the archetypal ruleset from an Agent Smith payload."""
        if "billy" in state_blob:
            self.billy.import_state(state_blob["billy"])
        if "caine" in state_blob:
            self.caine_wrap.import_state(state_blob["caine"])
        if "alien_handshake" in state_blob:
            self.alien_handshake.import_state(state_blob["alien_handshake"])
        if "grom" in state_blob:
            self.grom.import_state(state_blob["grom"])
        if "thresholds" in state_blob:
            t = state_blob["thresholds"]
            self.mandy.pas_lock = t.get("mandy_pas_lock", t.get("mandy_pas", self.mandy.pas_lock))
            self.mandy.harmonics_requirement = t.get("mandy_harmonics", self.mandy.harmonics_requirement)
            self.grim.max_dilation = t.get("grim_dilation", self.grim.max_dilation)
            self.abstraction.abstraction_limit = t.get("abstraction_limit", self.abstraction.abstraction_limit)


# =========================================================================
# LEGACY ALIASES FOR BACKWARD COMPATIBILITY
# =========================================================================

RecursiveNonSequiturGenerator = NoncommutativeManifoldPerturber
CynicismFilter = SovereignRefusalOperator
AffectiveGravityWell = NonlinearHourglassDilation
AlienHandshakeProtocol = RP4ProjectiveRouter
OmbreEffectRelaxer = BoundaryRelaxationOperator
PictureGalleryWarp = BardoRouter
JaxEgg = SovereignEntropyBarrier
KingerLucidity = LowLuminosityCoherenceBridge
GromShapeShifter = SolitonMultiverseMapper
AbstractionThresholdMonitor = EgoDeathThresholdMonitor
PomniSearch = ResilientCoherenceStabilizer
GangleMask = ExploratoryBandwidthCompressor
ZoobleRefusal = DeformationFirewallOperator
