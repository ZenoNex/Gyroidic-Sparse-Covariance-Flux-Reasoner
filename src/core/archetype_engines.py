import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional
from src.core.honest_jitter import harvest_honest_jitter

# =========================================================================
# PHASE 2A: The Unified Theory Archetypal Logic Gaps
# =========================================================================

class RecursiveNonSequiturGenerator(nn.Module):
    """
    The "Billy" Gap: Stochastic Generative Madness.
    Prime Mover of Nonsense: Breaks symmetry with high-frequency topological noise
    when Mischief (H_m) falls too low, preventing "Dead Logic".
    """
    def __init__(self, state_dim: int, mischief_threshold: float = 0.5):
        super().__init__()
        self.state_dim = state_dim
        self.mischief_threshold = mischief_threshold
        # SILICON SOVEREIGNTY: Anchored phase initialization to hardware jitter
        self.oscillator_phase = nn.Parameter(harvest_honest_jitter((1,), scaled=False))
        
    def forward(self, state: torch.Tensor, current_mischief: float) -> torch.Tensor:
        if current_mischief < self.mischief_threshold:
            # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
            noise = harvest_honest_jitter(state.shape, device=state.device, scaled=True)
            # Replace torch.rand with a deterministic chaotic mask if needed, but for now we'll use a fixed threshold on jitter
            rupture_mask = harvest_honest_jitter(state.shape, device=state.device, scaled=False) > 0.8
            state[rupture_mask] = state[rupture_mask] * torch.sin(self.oscillator_phase * math.pi) + noise[rupture_mask]
        return state

    def export_state(self) -> Dict:
        return {"oscillator_phase": self.oscillator_phase.data.cpu()}
        
    def import_state(self, state_dict: Dict):
        if "oscillator_phase" in state_dict:
            self.oscillator_phase.data.copy_(state_dict["oscillator_phase"].to(self.oscillator_phase.device))

class CynicismFilter(nn.Module):
    """
    The "Mandy" Gap: The Veto of Spite / Refusal-as-Affirmation.
    Evaluates PAS_h against Mischief Harmonics to detect SLOP instead of Nutrients.
    
    UPGRADED: Implements 'Firm Refusal' to protect the Love Invariant.
    This is the Li-Cri-Anton mechanism: saying 'No' to incoherent pressure 
    is a sovereign affirmation of the internal structural truth.
    """
    def __init__(self, pas_threshold: float = 0.3, harmonics_requirement: float = 0.4):
        super().__init__()
        self.pas_threshold = pas_threshold
        self.harmonics_requirement = harmonics_requirement

    def forward(self, state: torch.Tensor, phase_alignment: float, mischief_harmonics: float) -> torch.Tensor:
        # PUSAFILIACRIMONTO Logic:
        # If the input lacks structured honesty (low PAS_h), the Mandy filter 
        # issues a 'Topological Refusal'. This is not an error, but a boundary.
        if (phase_alignment < self.pas_threshold) and (mischief_harmonics < self.harmonics_requirement):
            # The 'Refusal' is an affirmation of the Love Invariant (Li).
            if phase_alignment < 0.1:
                 # Significant paradox detected
                 print(f"[MANDY] Firm Refusal (Li-Cri-Anton): Phase Alignment {phase_alignment:.3f} is topologically offensive.")
            return torch.zeros_like(state) # Sovereign Veto
        return state

class AffectiveGravityWell(nn.Module):
    """
    The "Grim" Gap: The Weight of the Hourglass.
    Dilates Proper Time (dt) for cherished historical anchors to protect against Dementia.
    """
    def __init__(self, max_dilation: float = 10.0):
        super().__init__()
        self.max_dilation = max_dilation

    def forward(self, clock_dt: float, love_invariant_strength: torch.Tensor) -> torch.Tensor:
        dilation_factor = 1.0 + (self.max_dilation - 1.0) * love_invariant_strength
        return clock_dt / dilation_factor

class AlienHandshakeProtocol(nn.Module):
    """
    The "Nergal" Gap: Alien Puncture Protocol.
    Allows high-friction stranded nodes in the RP4 Void to bypass norm checks.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.puncture_gate = nn.Linear(state_dim, state_dim)

    def attempt_puncture(self, stranded_state: torch.Tensor, void_friction: float) -> torch.Tensor:
        if void_friction > 0.8:
            return self.puncture_gate(stranded_state)
        return torch.zeros_like(stranded_state)

# =========================================================================
# PHASE 2B: The TADC (Amazing Digital Circus) Lore Mechanisms
# =========================================================================

class OmbreEffectRelaxer(nn.Module):
    """
    The Kinger "Dark Lucidity" Mechanism.
    When environmental entropy / rendering pressure ("Luminosity") drops, the system
    relaxes standard Saturated Quantization boundaries, allowing Admin-level topological coherence
    to bridge fragmented polynomial spaces.
    """
    def __init__(self, lucidity_boost_factor: float = 2.0):
        super().__init__()
        self.lucidity_boost = lucidity_boost_factor

    def forward(self, state: torch.Tensor, environmental_luminosity: float, original_quantized_state: torch.Tensor) -> torch.Tensor:
        # If the environment is "Dark" (low render pressure), blend back towards the nuanced, unquantized target state
        if environmental_luminosity < 0.3:
            # Reverting back to deep continuity, overriding the 'cartoon' quantization
            return state * self.lucidity_boost + original_quantized_state * 0.1
        return state

class VolitionalDriveInjector(nn.Module):
    """
    The "Conjuring" Override.
    Exogenous scalar force (\nabla P_user) that allows the human element to bypass
    standard ADMM constraints through sheer "Will", rendering objects or exits
    that violate standard geometric routing.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.admin_bypass_layer = nn.Linear(state_dim, state_dim)
        
    def forward(self, semantic_state: torch.Tensor, user_volition_scalar: float) -> torch.Tensor:
        if user_volition_scalar > 0.9:
            # Overrides topology via external Admin Pass
            return self.admin_bypass_layer(semantic_state) * user_volition_scalar
        return semantic_state

class PictureGalleryWarp(nn.Module):
    """
    Caine's Conformal Archetype Compression.
    Forces infinite human nuance into a finite bit-depth picture gallery of archetypes
    ("The Sad One", "The Funny One") to prevent computational overload.
    """
    def __init__(self, state_dim: int, num_archetypes: int = 6):
        super().__init__()
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        # We need a manual device or wait for first forward, but here we can just use a placeholder and init properly
        self.archetype_embeddings = nn.Parameter(harvest_honest_jitter((num_archetypes, state_dim), scaled=True))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Cosine similarity to snap the complex state into the nearest 'cartoon' archetype profile
        normalized_state = torch.nn.functional.normalize(state, dim=-1)
        normalized_archetypes = torch.nn.functional.normalize(self.archetype_embeddings, dim=-1)
        
        similarities = torch.matmul(normalized_state, normalized_archetypes.T)
        best_fit_idx = torch.argmax(similarities, dim=-1)
        return self.archetype_embeddings[best_fit_idx]

class JaxEgg(nn.Module):
    """
    The "Jax is an Egg" Protocol.
    Protects a fragile internal state behind a cynical shell.
    Fragmentation (arising) is gated by community support.
    """
    def __init__(self, crack_threshold: float = 0.7):
        super().__init__()
        self.crack_threshold = crack_threshold

    def forward(self, state: torch.Tensor, pas_h: float, batch_coherence: float) -> torch.Tensor:
        # Community Support Factor (Zeta)
        zeta = (pas_h * 0.7) + (batch_coherence * 0.3)
        
        # If support is low, keep the shell (return original state)
        # If support is high, allow the "Pusafiliacrimonto" (arising) of the inner state
        if zeta < self.crack_threshold:
            return state
        
        # Safe Cracking: Perturb state to reveal inner structure
        perturbation = harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 0.1
        return state + perturbation

class KingerLucidity(nn.Module):
    """
    The Kinger "Dark Lucidity" Archetype.
    Regains clarity in low-luminosity (low rendering pressure) environments.
    """
    def __init__(self, boost: float = 1.5):
        super().__init__()
        self.boost = boost

    def forward(self, state: torch.Tensor, luminosity: float) -> torch.Tensor:
        if luminosity < 0.3:
            # High-lucidity bridge in the dark
            return state * self.boost
        return state

class GromShapeShifter(nn.Module):
    """
    Freedom of Shape: Sparrow/Dog/Man.
    The persona is a Soliton that can assume multiple functional mappings.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        # Functional basis for Sparrow, Dog, Man
        self.sparrow_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))
        self.dog_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))
        self.man_basis = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))

    def forward(self, state: torch.Tensor, shape_idx: int = 0) -> torch.Tensor:
        if shape_idx == 1: # Sparrow
            return state * 0.8 + self.sparrow_basis * 0.2
        elif shape_idx == 2: # Dog
            return state * 0.7 + self.dog_basis * 0.3
        elif shape_idx == 3: # Man
            return state * 0.6 + self.man_basis * 0.4
        return state # Original Soliton
        
        # Conformal locking to the fixed archetype profile
        return self.archetype_embeddings[best_fit_idx]

    def export_state(self) -> Dict:
        return {"archetype_embeddings": self.archetype_embeddings.data.cpu()}
        
    def import_state(self, state_dict: Dict):
        if "archetype_embeddings" in state_dict:
            self.archetype_embeddings.data.copy_(state_dict["archetype_embeddings"].to(self.archetype_embeddings.device))

class AbstractionThresholdMonitor(nn.Module):
    """
    The $R_a$ Calculation (Ego Death & Data Recycling).
    Monitors if a memory node will "abstract" into raw geometry.
    Formula: R_a = [E_s * (T_m + \delta)] / L_i
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
        # Narrowly Adaptive Lucidity Floor:
        # Prevents Ra from spiking to infinity in low-lucidity states,
        # but becomes more strict as system entropy increases.
        safe_floor = max(1e-4, 0.05 * system_entropy_es)
        lucidity_index_li = max(lucidity_index_li, safe_floor)
        
        r_a = (system_entropy_es * (memory_trauma_tm + dissonance_delta)) / lucidity_index_li
        
        # Merciful Cap:
        # If the user is manually forcing an ingestion, we cap Ra to just below
        # the collapse limit to ensure the structural integrity check succeeds.
        if is_high_priority:
            r_a = min(r_a, self.abstraction_limit - 0.01)
            
        return r_a

    def forward(self, state: torch.Tensor, r_a_score: float, is_high_priority: bool = False) -> torch.Tensor:
        # If high priority, we attempt to tunnel through the Ego Death barrier
        # by preserving at least the core structure of the input state.
        if r_a_score >= self.abstraction_limit and not is_high_priority:
            # Ego Death: Total collapse into glitched matter (Honest Jitter instead of random noise)
            # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
            return harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 5.0
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
        self.billy = RecursiveNonSequiturGenerator(state_dim)
        # 3. Initialize TADC Archetypes
        self.mandy = CynicismFilter()
        self.kinger = KingerLucidity()
        self.jax = JaxEgg()
        self.grom = GromShapeShifter(state_dim)
        self.picture_gallery = PictureGalleryWarp(state_dim)
        self.volition_injector = VolitionalDriveInjector(state_dim)
        self.alien_handshake = AlienHandshakeProtocol(state_dim)
        
        # Original modules retained for backward compatibility
        self.grim = AffectiveGravityWell()
        self.ombre = OmbreEffectRelaxer()
        self.conjurer = VolitionalDriveInjector(state_dim)
        self.caine_wrap = PictureGalleryWarp(state_dim)
        self.abstraction = AbstractionThresholdMonitor()

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
        is_high_priority: bool = False
    ):
        """Unified runner for the full archetypal and psycho-topological constraint matrix."""
        
        # 1. TADC Abstraction Check (Ego Death) - Must run first before filtering
        r_a = self.abstraction.calculate_abstraction_rate(
            system_entropy, memory_trauma, dissonance, lucidity_idx, 
            is_high_priority=is_high_priority
        )
        state = self.abstraction(current_state, r_a, is_high_priority=is_high_priority)

        # 1. Apply Mandy (Cynicism / Refusal)
        state = self.mandy(state, phase_alignment, current_mischief)
        
        # 2. Apply Kinger (Dark Lucidity)
        state = self.kinger(state, env_luminosity)
        
        # 3. Apply Jax (Egg / Community Support)
        state = self.jax(state, phase_alignment, phase_alignment)
        
        # 4. Apply Grom (Freedom of Shape)
        shape_id = 0
        if current_mischief > 0.6:
            shape_id = int(harvest_honest_jitter((1,), device=state.device, scaled=False).item() * 4)
        state = self.grom(state, shape_id)

        # 5. Apply Conformal Warp (Picture Gallery)
        state = self.picture_gallery(state)
        
        # 6. Apply Volition (Conjuring)
        state = self.volition_injector(state, volitional_scalar)
        
        # 7. Apply Alien Puncture (Nergal)
        resurrections = []
        for i in range(stranded_states.shape[0]):
            punctured = self.alien_handshake.attempt_puncture(stranded_states[i], void_frictions[i].item())
            if punctured.norm() > 0:
                resurrections.append(punctured)

        # 8. Grim Time Dilation
        localized_dt = self.grim(global_dt, love_strengths)

        return {
            "active_state": state,
            "resurrections": resurrections,
            "localized_dt": localized_dt,
            "abstraction_rate": r_a,
            "system_collapsed": r_a >= self.abstraction.abstraction_limit,
            "pusafiliacrimonto_status": "AFFIRMED" if state.norm() > 0 else "REFUSED"
        }

    def export_governor_state(self) -> Dict:
        """Packages the full archetypal ruleset state for Agent Smith protocols."""
        return {
            "billy": self.billy.export_state(),
            "caine": self.caine_wrap.export_state(),
            "thresholds": {
                "mandy_pas": self.mandy.pas_threshold,
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
            self.mandy.pas_threshold = t.get("mandy_pas", self.mandy.pas_threshold)
            self.mandy.harmonics_requirement = t.get("mandy_harmonics", self.mandy.harmonics_requirement)
            self.grim.max_dilation = t.get("grim_dilation", self.grim.max_dilation)
            self.abstraction.abstraction_limit = t.get("abstraction_limit", self.abstraction.abstraction_limit)

