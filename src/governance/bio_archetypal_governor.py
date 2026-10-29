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
from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType, TypedComputeAffordance
from src.core.valence_drive import ValenceFunctional

class BioArchetypalGovernor(nn.Module):
    """
    Bio-Archetypal Governor (Multi-Scale Temporal Homeostasis).
    
    Replaces the flat ArchetypalSynthesisEngine with a biologically grounded
    cascade of temporal neighborhoods, intimately coupled to ValenceFunctional
    hunger dynamics and TypedComputeAffordance physical constraints.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        
        # Core Bus, Environment & Valence Drive
        self.bus = NeuromodulatoryBus()
        self.caine_precision = CainePrecisionGenerator(state_dim)
        self.valence_drive = ValenceFunctional(decay=0.99, hunger_scale=1.0)
        
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
        self.last_active_reflex: str = "Tathata_Mirror"
        self.reflex_streak: int = 0

    def evaluate_threat_recalibration_and_fixed_reflexes(
        self,
        structural_entropy: float,
        external_pressure: float,
        surrounding_pas_h: float,
        valence_hunger: float,
        jax_rigidity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Branch 5 Ontological Governance:
        1. Threat Recalibration (TheraminTrees):
           Separates structural danger (torsional collapse where PAS_h < 0.25 and pressure > 1.5)
           from high-entropy inconvenience / playful mischief (high entropy with intact PAS_h).
        2. 8 Fixed Survival Reflexes & Anti-Fixation Hysteresis:
           Audits the 8 childhood/systemic survival reflexes:
           Fight, Flounce, Fester, Flight, Fantasy, Fix, Flood, Fool.
           Applies anti-fixation hysteresis so the governor never locks into a single reflex,
           returning to the Alan Watts Tathata mirror (non-grasping witness) and auditing
           Kastrup's 3 Decombination Mechanisms (Resonance/Attention, Posturing/Masking, Dissociation/Forgetting).
        """
        is_structural_danger = (surrounding_pas_h < 0.25) and (external_pressure > 1.2)
        is_high_entropy_inconvenience = (not is_structural_danger) and (structural_entropy > 0.8)

        # Score the 8 Fixed Survival Reflexes
        reflex_scores = {
            "Fight": float(max(0.0, external_pressure * jax_rigidity - 1.0)),
            "Flounce": float(max(0.0, (jax_rigidity - 1.1) * structural_entropy)),
            "Fester": float(max(0.0, valence_hunger * (1.0 - surrounding_pas_h))),
            "Flight": float(max(0.0, external_pressure - surrounding_pas_h)),
            "Fantasy": float(max(0.0, structural_entropy * (1.0 - external_pressure))),
            "Fix": float(max(0.0, (1.0 - structural_entropy) * jax_rigidity)),
            "Flood": float(max(0.0, structural_entropy + valence_hunger - 1.2)),
            "Fool": float(max(0.0, structural_entropy * surrounding_pas_h))
        }
        dominant_reflex = max(reflex_scores, key=reflex_scores.get)
        dominant_score = reflex_scores[dominant_reflex]

        if dominant_score < 0.35 or is_high_entropy_inconvenience:
            active_mode = "Tathata_Mirror"
            self.reflex_streak = 0
        else:
            if dominant_reflex == self.last_active_reflex:
                self.reflex_streak += 1
            else:
                self.reflex_streak = 1
            active_mode = dominant_reflex

        # Anti-Fixation Hysteresis: if locked into any fixed survival reflex for >= 3 steps, release grasp
        hysteresis_damping = 1.0
        if self.reflex_streak >= 3 and active_mode != "Tathata_Mirror":
            hysteresis_damping = 0.5 / float(self.reflex_streak)
            active_mode = f"{dominant_reflex}->Tathata_Release"
            self.reflex_streak = 0

        self.last_active_reflex = dominant_reflex

        # Kastrup 3 Decombination Mechanisms audit
        decombination_mechanisms = {
            "resonance_attention": float(surrounding_pas_h),
            "posturing_masking": float(min(1.0, max(0.0, jax_rigidity - 0.8))),
            "dissociation_forgetting": float(min(1.0, max(0.0, 1.0 - surrounding_pas_h)))
        }

        return {
            "is_structural_danger": bool(is_structural_danger),
            "is_high_entropy_inconvenience": bool(is_high_entropy_inconvenience),
            "threat_recalibration_verdict": (
                "STRUCTURAL_DANGER" if is_structural_danger
                else ("PLAYFUL_INCONVENIENCE" if is_high_entropy_inconvenience else "HOMEORHETIC_EQUILIBRIUM")
            ),
            "active_reflex_mode": active_mode,
            "dominant_reflex": dominant_reflex,
            "reflex_scores": reflex_scores,
            "hysteresis_damping": round(hysteresis_damping, 4),
            "decombination_mechanisms": decombination_mechanisms
        }

    def forward(
        self, 
        state: torch.Tensor, 
        stranded_states: Optional[torch.Tensor] = None, 
        flux_tensor: Optional[torch.Tensor] = None, 
        dt: float = 1.0,
        bulletin_board: Optional[Any] = None,
        resonance_cavity: Optional[Any] = None,
        fossilizer: Optional[Any] = None,
        valence_functional: Optional[Any] = None,
        moment_transport: Optional[Any] = None,
        private_invariants: Optional[Dict[str, Any]] = None,
        gyroid_entropy: Optional[float] = None,
        luminosity: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes the biological cascade coupled with typed compute affordances and valence drive.
        
        Returns:
            Dict containing:
                - state: The mutated gyroidal state
                - neuro_bus: Snapshot of neuromodulator concentrations
                - precision_matrix: Caine's gaslighting precision
                - panic: Jax's panic flag
                - consolidating: Kinger's sleep flag
                - step_factor: Gangle's mood-driven learning rate modifier
                - jax_rigidity: Cynical boundary strength
                - valence_hunger: Live hunger scalar
                - valence_metrics: Satisfaction and starvation metrics
                - affordances: Dict of typed compute affordances
        """
        # 0. Hardware Affordance Monitoring across all typed regimes
        hw = HardwareMonitor()
        silicon_aff = hw.get_typed_affordance(ComputeAffordanceType.SILICON)
        thermal_aff = hw.get_typed_affordance(ComputeAffordanceType.THERMAL)
        algo_aff = hw.get_typed_affordance(ComputeAffordanceType.ALGORITHMIC)
        batch_aff = hw.get_typed_affordance(ComputeAffordanceType.COMPUTE_BATCH)
        topological_aff = hw.get_typed_affordance(ComputeAffordanceType.TOPOLOGICAL)
        
        # 1. Structural Entropy & Base Environment
        if gyroid_entropy is not None:
            structural_entropy = float(gyroid_entropy)
        elif stranded_states is not None and stranded_states.numel() > 0:
            structural_entropy = stranded_states.norm(p=2).item()
        else:
            structural_entropy = 0.5
            
        precision_matrix = self.caine_precision(structural_entropy)
        
        # 2. Valence Drive: Compute Hunger from structural pressure, mischief, and entropy
        v_func = valence_functional if valence_functional is not None else self.valence_drive
        
        if flux_tensor is not None and flux_tensor.numel() > 0:
            pressure_tensor = flux_tensor
        elif state.numel() > 0:
            pressure_tensor = state.norm(dim=-1, keepdim=True) if state.dim() > 1 else state.norm().unsqueeze(0)
        else:
            pressure_tensor = torch.tensor([0.5], device=state.device)
            
        mischief_val = kwargs.get('mischief', kwargs.get('current_mischief', None))
        mischief_tensor = torch.tensor([float(mischief_val)], device=state.device) if mischief_val is not None else None
        
        entropy_tensor = torch.tensor([float(structural_entropy)], device=state.device)
        categories_tensor = kwargs.get('categories', None)
        
        try:
            valence_hunger_tensor = v_func(
                current_pressure=pressure_tensor,
                mischief=mischief_tensor,
                entropy=entropy_tensor,
                categories=categories_tensor
            )
            valence_hunger_scalar = float(valence_hunger_tensor.mean().item())
        except Exception:
            valence_hunger_scalar = 0.5
            
        valence_metrics = v_func.get_metrics() if hasattr(v_func, 'get_metrics') else {}
        
        # 3. Interoceptive: Pomni calculates surprise and broadcasts Noradrenaline
        # Heightened by valence hunger and topological affordance deficits
        state = self.pomni(
            state=state, 
            gyroid_entropy=structural_entropy, 
            bus=self.bus,
            topological_affordance=topological_aff,
            valence_hunger=valence_hunger_scalar
        )
        
        # 4. Ultrafast: Zooble asserts body schema and Silicon compute affordance firewall (GABA)
        if bulletin_board is not None:
            raw_state = bulletin_board.read_residue()
            if raw_state.dim() == 1 and state.dim() > 1:
                raw_state = raw_state.unsqueeze(0).expand_as(state)
        else:
            raw_state = state
            
        state, zooble_signal = self.zooble(
            raw_unquantized_state=raw_state, 
            warped_state=state, 
            bus=self.bus,
            silicon_affordance=silicon_aff,
            valence_hunger=valence_hunger_scalar
        )
        if hasattr(zooble_signal, 'is_refused') and zooble_signal.is_refused:
            print(f"[ZOOBLE] {zooble_signal.reason}")
        
        # 5. Fast: Jax evaluates approach/avoidance, incorporating hunger and algorithmic headroom
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
            if flux_tensor is not None and flux_tensor.numel() > 0:
                external_pressure = flux_tensor.norm().item()
            else:
                external_pressure = 0.5
            
        state, jax_rigidity = self.jax(
            state=state, 
            surrounding_pas_h=surrounding_pas_h,
            batch_tensors=batch_tensors,
            internal_entropy=structural_entropy,
            external_pressure=external_pressure,
            bus=self.bus,
            algorithmic_affordance=algo_aff,
            valence_hunger=valence_hunger_scalar
        )
        panic = jax_rigidity > 1.2
        pas_scalar = float(surrounding_pas_h.mean().item()) if isinstance(surrounding_pas_h, torch.Tensor) else float(surrounding_pas_h)
        threat_recalibration = self.evaluate_threat_recalibration_and_fixed_reflexes(
            structural_entropy=structural_entropy,
            external_pressure=external_pressure,
            surrounding_pas_h=pas_scalar,
            valence_hunger=valence_hunger_scalar,
            jax_rigidity=float(jax_rigidity)
        )
        if threat_recalibration["is_high_entropy_inconvenience"]:
            panic = False
            jax_rigidity = float(jax_rigidity) * threat_recalibration["hysteresis_damping"]
        
        # 6. Fast: Ragatha responds to distress, soothing hunger and grief (Oxytocin)
        boundary_friction = max(0.0, jax_rigidity - 1.0)
        state = self.ragatha(
            state=state, 
            bus=self.bus, 
            boundary_friction=boundary_friction,
            valence_hunger=valence_hunger_scalar
        )
        
        # 7. Medium: Gangle cycles mood, dictating step factor (Dopamine & Batch Affordance)
        state, step_factor = self.gangle(
            state=state, 
            bus=self.bus, 
            dt=dt,
            batch_affordance=batch_aff,
            valence_hunger=valence_hunger_scalar
        )
        
        # 8. Slow: Kinger consolidates memory, acting as a Thermodynamic Heat Sink (Acetylcholine)
        if luminosity is not None:
            luminosity_scalar = float(luminosity)
        elif flux_tensor is not None and flux_tensor.numel() > 0:
            luminosity_scalar = flux_tensor.norm().item()
        else:
            luminosity_scalar = 0.5
            
        state, consolidating = self.kinger(
            state=state, 
            bus=self.bus, 
            environmental_rendering_pressure=luminosity_scalar,
            thermal_affordance=thermal_aff,
            valence_hunger=valence_hunger_scalar
        )
        
        return {
            "state": state,
            "neuro_bus": self.bus.snapshot(),
            "precision_matrix": precision_matrix,
            "panic": panic,
            "consolidating": consolidating,
            "step_factor": step_factor,
            "jax_rigidity": jax_rigidity,
            "valence_hunger": valence_hunger_scalar,
            "valence_metrics": valence_metrics,
            "threat_recalibration": threat_recalibration,
            "affordances": {
                "silicon": silicon_aff,
                "thermal": thermal_aff,
                "algorithmic": algo_aff,
                "batch": batch_aff,
                "topological": topological_aff
            }
        }

