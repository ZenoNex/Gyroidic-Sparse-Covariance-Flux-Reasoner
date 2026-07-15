import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Any
from src.core.honest_jitter import harvest_honest_jitter
from src.core.superposed_tag_stacker import SuperposedTagStacker

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
        
    def forward(self, state: torch.Tensor, current_mischief: float) -> torch.Tensor:
        """
        Injects non-sequitur perturbations under low/high mischief conditions.
        
        Args:
            state: Input topological state tensor.
            current_mischief: Scalar mischief value (H_m).
            
        Returns:
            Perturbed state if conditions are met, else original state.
        """
        state = state.clone()
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
    The Sovereign Refusal Operator (legacy alias: CynicismFilter).
    Acts as a strict veto boundary (the Mandy Gap).
    
    Math: Implements the Li-Cri-Anton mechanism, returning a zero vector to refuse
    optimization trajectories lacking structural honesty (low PAS_h) to protect
    the Love Invariant.

    training_mode (bool): When True, the gate issues a fractional attenuation (10%
    pass-through) rather than a hard zero veto. This allows gradients to survive
    cold-start training where PAS_h is structurally low before the model has learned
    any coherent phase structure. Set to False (default) for deployment/inference
    to restore the full sovereign veto.
    """
    def __init__(self, pas_lock: float = 3.0 / 11.0, harmonics_requirement: float = 0.4,
                 training_mode: bool = False, pas_threshold: Optional[float] = None):
        super().__init__()
        # PAS_LOCK tied directly to the (11, 3) resonant Tori constraint
        if pas_threshold is not None:
            self.pas_lock = pas_threshold
        else:
            self.pas_lock = pas_lock
        self.harmonics_requirement = harmonics_requirement
        self.training_mode = training_mode

    def forward(self, state: torch.Tensor, phase_alignment: float, mischief_harmonics: float) -> torch.Tensor:
        # PUSAFILIACRIMONTO Logic:
        # If the input lacks structured honesty (low PAS_h), the Refusal Operator
        # issues a Topological Refusal. This is not an error, but a boundary.
        if (phase_alignment < self.pas_lock) and (mischief_harmonics < self.harmonics_requirement):
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

    def forward(self, clock_dt: float, love_invariant_strength: torch.Tensor) -> torch.Tensor:
        dilation_factor = 1.0 + (self.max_dilation - 1.0) * love_invariant_strength
        return clock_dt / dilation_factor

class RP4ProjectiveRouter(nn.Module):
    """
    The RP4 Projective Router (legacy alias: AlienHandshakeProtocol).
    Acts as a projective routing vector (the Nergal Gap).
    
    Math: Allows high-friction stranded nodes in the non-orientable RP^4 void
    to puncture the boundary and tunnel back into the active manifold when void
    friction exceeds a critical threshold.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.puncture_gate = nn.Linear(state_dim, state_dim)

    def attempt_puncture(self, stranded_state: torch.Tensor, void_friction: float) -> torch.Tensor:
        """
        Attempts to puncture the RP4 Void barrier for stranded nodes.
        
        A puncture occurs when void friction exceeds a critical threshold, 
        allowing the stranded state to tunnel back into the active manifold
        via the puncture gate.
        
        Args:
            stranded_state: The state vector of the node stranded in the void.
            void_friction: Scalar friction value of the surrounding RP4 vacuum.
            
        Returns:
            The punctured/resurrected state or a zero vector if puncture fails.
        """
        if void_friction > 0.8:
            return self.puncture_gate(stranded_state)
        return torch.zeros_like(stranded_state)

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

    def forward(self, semantic_state: torch.Tensor, user_volition_scalar: float, archetype_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        if user_volition_scalar > 0.9:
            # Increment time accumulator
            self.t_accum += 0.1
            
            # Load fossils on demand to prevent startup latency
            if not self._fossils_loaded:
                self._load_fossils()

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
    Protects a fragile internal state behind a cynical shell (the Jax Gap).
    
    Math: Gated by community support (combination of PAS_h and batch coherence).
    """
    def __init__(self, crack_threshold: float = 0.7):
        super().__init__()
        self.crack_threshold = crack_threshold

    def forward(self, state: torch.Tensor, pas_h: float, batch_coherence: float) -> torch.Tensor:
        # Community Support Factor (Zeta)
        zeta = (pas_h * 0.7) + (batch_coherence * 0.3)
        
        # If support is low, keep the shell (return original state)
        if zeta < self.crack_threshold:
            return state
        
        # Safe Cracking: Perturb state to reveal inner structure
        perturbation = harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 0.1
        return state + perturbation

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
        # Functional basis for Sparrow, Dog, Human
        self.sparrow_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))
        self.dog_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))
        self.human_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))

    def forward(self, state: torch.Tensor, shape_idx: int = 0) -> torch.Tensor:
        if shape_idx == 1: # Sparrow
            return state * 0.8 + self.sparrow_basis * 0.2
        elif shape_idx == 2: # Dog
            return state * 0.7 + self.dog_basis * 0.3
        elif shape_idx == 3: # Human
            return state * 0.6 + self.human_basis * 0.4
        return state # Original Soliton

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

    def forward(self, state: torch.Tensor, r_a_score: float, is_high_priority: bool = False) -> torch.Tensor:
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
    """
    def __init__(self, state_dim: int, resilience_scale: float = 0.3):
        super().__init__()
        self.resilience_scale = resilience_scale
        self.stabilizer = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))

    def forward(self, state: torch.Tensor, lucidity_idx: float, system_entropy: float) -> torch.Tensor:
        disorientation = (1.0 - lucidity_idx) * system_entropy
        if disorientation > 0.4:
            stabilizing_force = disorientation * self.resilience_scale * self.stabilizer.to(state.device)
            return state + stabilizing_force
        return state

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
        if phase_alignment < 0.35:
            # tragedy mode: contract the state coordinates to avoid toxic leak propagation
            return state * self.tragedy_scale
        else:
            # comedy mode: amplify the exploratory coupling of the state
            return state * self.comedy_scale * phase_alignment

class DeformationFirewallOperator(nn.Module):
    """
    The Deformation Firewall Operator (legacy alias: ZoobleRefusal).
    Rejects conformal cartoon compression if the deformation is too severe.
    """
    def __init__(self, deviation_threshold: float = 0.8):
        super().__init__()
        self.deviation_threshold = deviation_threshold

    def forward(self, state: torch.Tensor, warped_state: torch.Tensor, raw_unquantized_state: torch.Tensor) -> torch.Tensor:
        deviation = torch.norm(warped_state - raw_unquantized_state)
        if deviation.item() > self.deviation_threshold:
            # Blunt refusal: restore raw unquantized state to protect body/mind autonomy
            return raw_unquantized_state
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
        self.kinger = LowLuminosityCoherenceBridge()
        self.jax = SovereignEntropyBarrier()
        self.grom = SolitonMultiverseMapper(state_dim)
        self.picture_gallery = BardoRouter(state_dim)
        self.volition_injector = VolitionalDriveInjector(state_dim)
        self.alien_handshake = RP4ProjectiveRouter(state_dim)
        
        # New TADC Archetypes
        self.pomni = ResilientCoherenceStabilizer(state_dim)
        self.gangle = ExploratoryBandwidthCompressor()
        self.zooble = DeformationFirewallOperator()
        
        # Original modules retained for backward compatibility
        self.grim = NonlinearHourglassDilation()
        self.ombre = BoundaryRelaxationOperator()
        self.conjurer = self.volition_injector
        self.caine_wrap = self.picture_gallery
        self.abstraction = EgoDeathThresholdMonitor()
        
        # Superposed Vector Stacker (Ganbreeder-style)
        self.tag_stacker = SuperposedTagStacker(state_dim)

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
        stranded_states: torch.Tensor,
        current_mischief: float, 
        phase_alignment: float, 
        love_strengths: torch.Tensor,
        void_frictions: torch.Tensor,
        global_dt: float,
        # TADC specific params
        env_luminosity: float,
        volitional_scalar: float,
        system_entropy: float,
        memory_trauma: float,
        dissonance: float,
        lucidity_idx: float,
        raw_unquantized_state: torch.Tensor,
        is_high_priority: bool = False,
        tag_weights: Optional[Dict[str, float]] = None
    ):
        """Unified runner for the full archetypal and psycho-topological constraint matrix."""
        
        # 0. Apply Ganbreeder Tag Stacking Superposition
        stacked_target = self.compute_stacked_target(tag_weights, current_state)
        if stacked_target is not None and stacked_target.norm() > 0:
            # Softly shift current state towards stacked target (acting as a primer)
            primed_state = current_state + 0.1 * stacked_target
            # Apply the BoundaryRelaxationOperator (self.ombre) to blend between the state and target based on env_luminosity
            current_state = self.ombre(primed_state, env_luminosity, stacked_target)
        
        # 1. TADC Abstraction Check (Ego Death) - Must run first before filtering
        r_a = self.abstraction.calculate_abstraction_rate(
            system_entropy, memory_trauma, dissonance, lucidity_idx, 
            is_high_priority=is_high_priority
        )
        state = self.abstraction(current_state, r_a, is_high_priority=is_high_priority)

        # 1a. Apply Pomni (Empathy & Search for Meaning)
        state = self.pomni(state, lucidity_idx, system_entropy)

        # 1b. Apply Mandy (Cynicism / Refusal)
        state = self.mandy(state, phase_alignment, current_mischief)
        
        # 2. Apply Kinger (Dark Lucidity)
        state = self.kinger(state, env_luminosity)
        
        # 3. Apply Jax (Egg / Community Support)
        state = self.jax(state, phase_alignment, phase_alignment)
        
        # 3a. Apply Gangle (Masking / Mood Shifts)
        state = self.gangle(state, phase_alignment)
        
        # 4. Apply Grom (Freedom of Shape)
        shape_id = 0
        if current_mischief > 0.6:
            shape_id = int(harvest_honest_jitter((1,), device=state.device, scaled=False).item() * 4)
        state = self.grom(state, shape_id)

        # 5. Apply Conformal Warp (Picture Gallery)
        state_before_warp = state.clone()
        state = self.picture_gallery(state)
        
        # 5a. Apply Zooble (Refusal / Body Autonomy)
        state = self.zooble(state, warped_state=state, raw_unquantized_state=state_before_warp)
        
        # 6. Apply Billy (Generative Madness)
        state = self.billy(state, current_mischief)

        # 7. Apply Volition (Conjuring)
        state = self.volition_injector(state, volitional_scalar, archetype_embeddings=self.picture_gallery.archetype_embeddings)
        
        # 8. Apply Alien Puncture (Nergal)
        resurrections = []
        for i in range(stranded_states.shape[0]):
            if void_frictions.dim() == 0 or void_frictions.numel() == 1:
                friction_val = void_frictions.item()
            else:
                friction_val = void_frictions[min(i, void_frictions.shape[0] - 1)].item()
            punctured = self.alien_handshake.attempt_puncture(stranded_states[i], friction_val)
            if punctured.norm() > 0:
                resurrections.append(punctured)

        # 9. Grim Time Dilation
        localized_dt = self.grim(global_dt, love_strengths)

        return {
            "active_state": state,
            "resurrections": resurrections,
            "localized_dt": localized_dt,
            "abstraction_rate": r_a,
            "system_collapsed": r_a >= self.abstraction.abstraction_limit,
            "pusafiliacrimonto_status": "AFFIRMED" if state.norm() > 0 else "REFUSED",
            "stacked_target": stacked_target
        }

    def export_governor_state(self) -> Dict:
        """Packages the full archetypal ruleset state for Agent Smith protocols."""
        return {
            "billy": self.billy.export_state(),
            "caine": self.caine_wrap.export_state(),
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
