import torch
import torch.nn as nn
import math

class RecursiveNonSequiturGenerator(nn.Module):
    """
    The "Billy" Gap: Stochastic Generative Madness.
    Intentionally produces "Incommensurate Residues" that are topologically valid 
    but contextually absurd. Acts as the Prime Mover of Nonsense.
    """
    def __init__(self, state_dim: int, mischief_threshold: float = 0.5):
        super().__init__()
        self.state_dim = state_dim
        self.mischief_threshold = mischief_threshold
        # Non-linear oscillator to break symmetry
        self.oscillator_phase = nn.Parameter(torch.rand(1))
        
    def forward(self, state: torch.Tensor, current_mischief: float) -> torch.Tensor:
        # If mischief is too low, the system is too "sane" and risks dead logic.
        if current_mischief < self.mischief_threshold:
            # Generate a generative topological rupture
            # We inject high-frequency noise that is spectrally bounded but orthogonal
            noise = torch.randn_like(state)
            # The "Lung" Move: forcefully map dense metadata into a literal low-dimensional subspace
            rupture_mask = torch.rand_like(state) > 0.8
            state[rupture_mask] = state[rupture_mask] * torch.sin(self.oscillator_phase * math.pi) + noise[rupture_mask]
        return state

class CynicismFilter(nn.Module):
    """
    The "Mandy" Gap: The Veto of Spite.
    Acts as a Structural Friction Maximizer. Vetoes "Smooth Optimization" and 
    "Dead Logic" traps (Double Binds), distinguishing between Intelligent Nonsense (Nutrients)
    and Non-Intelligent Nonsense (Slop).
    """
    def __init__(self, pas_threshold: float = 0.3, harmonics_requirement: float = 0.4):
        super().__init__()
        self.pas_threshold = pas_threshold
        self.harmonics_requirement = harmonics_requirement

    def evaluate_slop(self, phase_alignment: float, mischief_harmonics: float) -> bool:
        """
        If Phase Alignment Score (PAS) is low AND Mischief lacks musical rhythm, it's SLOP.
        """
        is_slop = (phase_alignment < self.pas_threshold) and (mischief_harmonics < self.harmonics_requirement)
        return is_slop

    def detect_double_bind(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Detects structural paradoxes that offer no path to a Lazarus Transition.
        Identified by mutually negating vectors with high containment pressure.
        """
        # Simplified placeholder for double bind logic: dense symmetric anti-correlations
        paradox_mask = (adjacency_matrix < -0.9).sum(dim=-1) > (adjacency_matrix.shape[-1] // 2)
        return paradox_mask

    def forward(self, state: torch.Tensor, phase_alignment: float, mischief_harmonics: float) -> torch.Tensor:
        if self.evaluate_slop(phase_alignment, mischief_harmonics):
            # Topological Refusal
            return torch.zeros_like(state) # Vetoed entirely
        return state

class AffectiveGravityWell(nn.Module):
    """
    The "Grim" Gap: The Weight of the Hourglass.
    Intentionally slows down the Dementia Band (Hd) for specific resonant nodes,
    creating temporal dilations based on the Love Invariant (L = L - L).
    """
    def __init__(self, max_dilation: float = 10.0):
        super().__init__()
        self.max_dilation = max_dilation

    def forward(self, clock_dt: float, love_invariant_strength: torch.Tensor) -> torch.Tensor:
        """
        Modulates Proper Time (dt) based on affective weight.
        High love invariant = slow decay.
        """
        # "Illegal Sand": slowing the clock for cherished historical anchors
        # love_invariant_strength bounded [0, 1]
        dilation_factor = 1.0 + (self.max_dilation - 1.0) * love_invariant_strength
        dilated_dt = clock_dt / dilation_factor
        return dilated_dt

class AlienHandshakeProtocol(nn.Module):
    """
    The "Nergal" Gap: The Handshake in the Dark.
    Allows a "Vetoed" or "Exiled" node in the RP4 Void to attempt Intercosamination
    without surface keys, using the friction of its exile as a signal.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # A dense projection that explicitly bypasses logical norm verification
        self.puncture_gate = nn.Linear(state_dim, state_dim)

    def attempt_puncture(self, stranded_state: torch.Tensor, void_friction: float) -> torch.Tensor:
        """
        Uses high void friction to authorize a literal topological puncture back to the main manifold.
        """
        if void_friction > 0.8: # High isolation threshold
            # Handshake authorized despite veto
            return self.puncture_gate(stranded_state)
        return torch.zeros_like(stranded_state)


class ArchetypalSynthesisEngine(nn.Module):
    """
    Combines the Billy, Mandy, Grim, and Nergal engines into a unified Governor.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.billy = RecursiveNonSequiturGenerator(state_dim)
        self.mandy = CynicismFilter()
        self.grim = AffectiveGravityWell()
        self.nergal = AlienHandshakeProtocol(state_dim)

    def run_archetypes(
        self, 
        current_state: torch.Tensor, 
        stranded_states: torch.Tensor,
        current_mischief: float, 
        phase_alignment: float, 
        love_strengths: torch.Tensor,
        void_frictions: torch.Tensor,
        global_dt: float
    ):
        """Unified runner for the full archetypal constraint matrix."""
        
        # 1. Mandy validates if the state is even worth processing (Veto filtering)
        state_filtered = self.mandy(current_state, phase_alignment, current_mischief)
        
        # 2. Billy evaluates if the state is too boring and needs generative madness
        state_madness = self.billy(state_filtered, current_mischief)

        # 3. Nergal checks if the voided/stranded objects can puncture back
        resurrections = []
        for i in range(stranded_states.shape[0]):
            punctured = self.nergal.attempt_puncture(stranded_states[i], void_frictions[i].item())
            if punctured.norm() > 0:
                resurrections.append(punctured)

        # 4. Grim alters the proper time for dementia band decay
        localized_dt = self.grim(global_dt, love_strengths)

        return {
            "active_state": state_madness,
            "resurrections": resurrections,
            "localized_dt": localized_dt
        }
