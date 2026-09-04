import torch
import torch.nn as nn
from typing import Dict, Any

from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.environment.caine_precision import CainePrecisionGenerator
from src.governance.interoceptive.pomni_uncertainty import PomniUncertaintyPredictor
from src.governance.ultrafast.zooble_autonomy import ZoobleBodySchema
from src.governance.fast.jax_shell import JaxShell
from src.governance.fast.ragatha_bonding import RagathaBonding
from src.governance.medium.gangle_oscillator import GangleOscillator
from src.governance.slow.kinger_consolidation import KingerConsolidation

class BioArchetypalGovernor(nn.Module):
    """
    Bio-Archetypal Governor (Multi-Scale Temporal Homeostasis).
    
    Replaces the flat ArchetypalSynthesisEngine with a biologically grounded
    cascade of temporal neighborhoods.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        
        # Core Bus & Environment
        self.bus = NeuromodulatoryBus()
        self.caine_precision = CainePrecisionGenerator(state_dim)
        
        # Interoceptive (Surprise/Noradrenaline)
        self.pomni = PomniUncertaintyPredictor(state_dim)
        
        # Ultrafast (GABA/Body-Schema)
        self.zooble = ZoobleBodySchema(state_dim)
        
        # Fast (Serotonin/Oxytocin/Approach-Avoidance)
        self.jax = JaxShell(state_dim)
        self.ragatha = RagathaBonding(state_dim)
        
        # Medium (Dopamine/Limit-Cycle)
        self.gangle = GangleOscillator(state_dim)
        
        # Slow (Acetylcholine/Consolidation)
        self.kinger = KingerConsolidation(state_dim)

    def forward(
        self, 
        state: torch.Tensor, 
        gyroid_entropy: float = 0.5, 
        luminosity: float = 1.0, 
        dt: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes the biological cascade.
        
        Returns:
            Dict containing:
                - state: The mutated gyroidal state
                - neuro_bus: Snapshot of neuromodulator concentrations
                - precision_matrix: Caine's gaslighting precision
                - panic: Jax's panic flag
                - consolidating: Kinger's sleep flag
                - step_factor: Gangle's mood-driven learning rate modifier
        """
        # 1. Environment: Caine determines base precision based on entropy
        precision_matrix = self.caine_precision(gyroid_entropy)
        
        # 2. Interoceptive: Pomni calculates surprise and broadcasts Noradrenaline
        state = self.pomni(state, gyroid_entropy, self.bus)
        
        # 3. Ultrafast: Zooble asserts body schema, potentially gating the signal (GABA)
        state = self.zooble(state, self.bus)
        
        # 4. Fast: Jax evaluates approach/avoidance, triggering panic if unsafe
        state, panic = self.jax(state, self.bus)
        
        # 5. Fast: Ragatha responds to Pomni's distress (Oxytocin)
        state = self.ragatha(state, self.bus)
        
        # 6. Medium: Gangle cycles mood, dictating the step factor (Dopamine)
        state, step_factor = self.gangle(state, self.bus, dt=dt)
        
        # 7. Slow: Kinger consolidates memory if it is dark (Acetylcholine)
        state, consolidating = self.kinger(state, self.bus, luminosity)
        
        return {
            "state": state,
            "neuro_bus": self.bus.snapshot(),
            "precision_matrix": precision_matrix,
            "panic": panic,
            "consolidating": consolidating,
            "step_factor": step_factor
        }
