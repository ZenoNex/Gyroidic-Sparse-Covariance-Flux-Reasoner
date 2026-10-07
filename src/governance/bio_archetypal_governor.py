import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.environment.caine_precision import CainePrecisionGenerator
from src.governance.interoceptive.pomni_uncertainty import PomniUncertaintyPredictor
from src.governance.ultrafast.zooble_autonomy import ZoobleAutonomy
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
        self.zooble = ZoobleAutonomy(state_dim)
        
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
        stranded_states: torch.Tensor, 
        flux_tensor: torch.Tensor, 
        dt: float = 1.0,
        bulletin_board: Optional[Any] = None,
        resonance_cavity: Optional[Any] = None,
        fossilizer: Optional[Any] = None,
        valence_functional: Optional[Any] = None,
        moment_transport: Optional[Any] = None,
        private_invariants: Optional[Dict[str, Any]] = None
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
        # 1. Environment: Caine determines base precision based on structural trauma (stranded states)
        structural_entropy = stranded_states.norm(p=2).item() if stranded_states.numel() > 0 else 0.5
        precision_matrix = self.caine_precision(structural_entropy)
        
        # 2. Interoceptive: Pomni calculates surprise and broadcasts Noradrenaline
        state = self.pomni(state, structural_entropy, self.bus)
        
        # 3. Ultrafast: Zooble asserts body schema, potentially gating the signal (GABA)
        if bulletin_board is not None:
            raw_state = bulletin_board.read_residue()
            if raw_state.dim() == 1 and state.dim() > 1:
                raw_state = raw_state.unsqueeze(0).expand_as(state)
        else:
            raw_state = state
            
        state, zooble_signal = self.zooble(raw_state, state, self.bus)
        if hasattr(zooble_signal, 'is_refused') and zooble_signal.is_refused:
            print(f"[ZOOBLE] {zooble_signal.reason}")
        
        # 4. Fast: Jax evaluates approach/avoidance, triggering panic if unsafe
        if bulletin_board is not None:
            from src.core.invariants import PhaseAlignmentInvariant
            batch_tensors = bulletin_board.residue_history
            pas_metric = PhaseAlignmentInvariant(degree=4).to(state.device)
            surrounding_pas_h = pas_metric(batch_tensors)
            if surrounding_pas_h.dim() == 0:
                surrounding_pas_h = surrounding_pas_h.unsqueeze(0)
            
            admm_force = bulletin_board.read_force()
            external_pressure = admm_force.norm().item()
        else:
            surrounding_pas_h = torch.tensor([0.5], device=state.device)
            batch_tensors = state.unsqueeze(0) if state.dim() == 1 else state
            external_pressure = flux_tensor.norm().item() if flux_tensor.numel() > 0 else 0.5
            
        state, jax_rigidity = self.jax(
            state, 
            surrounding_pas_h=surrounding_pas_h,
            batch_tensors=batch_tensors,
            internal_entropy=structural_entropy,
            external_pressure=external_pressure,
            bus=self.bus
        )
        panic = jax_rigidity > 1.2
        
        # 5. Fast: Ragatha responds to Pomni's distress (Oxytocin)
        state = self.ragatha(state, self.bus)
        
        # 6. Medium: Gangle cycles mood, dictating the step factor (Dopamine)
        state, step_factor = self.gangle(state, self.bus, dt=dt)
        
        # 7. Slow: Kinger consolidates memory based on flux pressure (Acetylcholine)
        luminosity_scalar = flux_tensor.norm().item() if flux_tensor.numel() > 0 else 0.5
        state, consolidating = self.kinger(state, self.bus, luminosity_scalar)
        
        return {
            "state": state,
            "neuro_bus": self.bus.snapshot(),
            "precision_matrix": precision_matrix,
            "panic": panic,
            "consolidating": consolidating,
            "step_factor": step_factor,
            "jax_rigidity": jax_rigidity
        }
