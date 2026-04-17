import torch
import torch.nn as nn
import math

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
        self.oscillator_phase = nn.Parameter(torch.rand(1))
        
    def forward(self, state: torch.Tensor, current_mischief: float) -> torch.Tensor:
        if current_mischief < self.mischief_threshold:
            noise = torch.randn_like(state)
            rupture_mask = torch.rand_like(state) > 0.8
            state[rupture_mask] = state[rupture_mask] * torch.sin(self.oscillator_phase * math.pi) + noise[rupture_mask]
        return state

class CynicismFilter(nn.Module):
    """
    The "Mandy" Gap: The Veto of Spite.
    Evaluates PAS_h against Mischief Harmonics to detect SLOP instead of Nutrients.
    """
    def __init__(self, pas_threshold: float = 0.3, harmonics_requirement: float = 0.4):
        super().__init__()
        self.pas_threshold = pas_threshold
        self.harmonics_requirement = harmonics_requirement

    def forward(self, state: torch.Tensor, phase_alignment: float, mischief_harmonics: float) -> torch.Tensor:
        if (phase_alignment < self.pas_threshold) and (mischief_harmonics < self.harmonics_requirement):
            return torch.zeros_like(state) # Veto
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
        self.archetype_embeddings = nn.Parameter(torch.randn(num_archetypes, state_dim))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Cosine similarity to snap the complex state into the nearest 'cartoon' archetype profile
        normalized_state = torch.nn.functional.normalize(state, dim=-1)
        normalized_archetypes = torch.nn.functional.normalize(self.archetype_embeddings, dim=-1)
        
        similarities = torch.matmul(normalized_state, normalized_archetypes.T)
        best_fit_idx = torch.argmax(similarities, dim=-1)
        
        # Conformal locking to the fixed archetype profile
        snapped_state = self.archetype_embeddings[best_fit_idx]
        return snapped_state

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
            # Ego Death: Total collapse into glitched matter (random unstructured noise)
            return torch.randn_like(state) * 5.0
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
        self.mandy = CynicismFilter()
        self.grim = AffectiveGravityWell()
        self.nergal = AlienHandshakeProtocol(state_dim)
        
        # TADC Mechanics
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
        raw_unquantized_state: torch.Tensor
    ):
        """Unified runner for the full archetypal and psycho-topological constraint matrix."""
        
        # 1. TADC Abstraction Check (Ego Death) - Must run first before filtering
        r_a = self.abstraction.calculate_abstraction_rate(system_entropy, memory_trauma, dissonance, lucidity_idx)
        state_alive = self.abstraction(current_state, r_a)

        # If abstracted (random noise), further psychology checks are mostly moot, but let's flow it.
        # 2. TADC Volition / Conjuring Check
        state_volitional = self.conjurer(state_alive, volitional_scalar)

        # 3. TADC Ombre Effect / Dark Lucidity
        state_lucid = self.ombre(state_volitional, env_luminosity, raw_unquantized_state)

        # 4. TADC Picture Gallery Warp (Only applies if environment is highly rendered, else lucidity overrides)
        if env_luminosity >= 0.3:
            state_lucid = self.caine_wrap(state_lucid)

        # 5. Mandy Veto filtering
        state_filtered = self.mandy(state_lucid, phase_alignment, current_mischief)
        
        # 6. Billy Generative Madness check
        state_final = self.billy(state_filtered, current_mischief)

        # 7. Nergal Void Puncture
        resurrections = []
        for i in range(stranded_states.shape[0]):
            punctured = self.nergal.attempt_puncture(stranded_states[i], void_frictions[i].item())
            if punctured.norm() > 0:
                resurrections.append(punctured)

        # 8. Grim Time Dilation
        localized_dt = self.grim(global_dt, love_strengths)

        return {
            "active_state": state_final,
            "resurrections": resurrections,
            "localized_dt": localized_dt,
            "abstraction_rate": r_a,
            "system_collapsed": r_a >= self.abstraction.abstraction_limit
        }
