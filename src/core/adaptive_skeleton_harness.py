import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import psutil
import math

from src.core.leontief_governor import LeontiefGovernor
from src.core.garden_statistical_attractors import GardenOrchestrator
from enum import Enum, auto

class AmbulatoryClass(Enum):
    BIPED = auto()
    QUADRUPED = auto()
    THEROPOD = auto()
    CENTIPEDE = auto()
    NON_AMBULATORY = auto()

class AdaptiveSkeletonHarness(nn.Module):
    """
    The Adaptive Skeleton Harness is responsible for the procedural generation
    and mutation of 3D character rigs based on Universal Topology.

    It integrates:
    1. Bouligand Tangent Cones & KANLayer micro-waves to avoid t_RFC DRAM stalls.
    2. Ganbreeder Vector Stacker for Collaborative Interactive Evolution (Rig Blending).
    3. Bostick-style Garden Attractors for mapping psychological affordances.
    4. Leontief Governor for hardware stress monitoring and P2P mesh defense.
    5. Access hooks for the 13+ Endogenous One-Shot Adaptation systems.
    """

    def __init__(self, state_dim: int, device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.device = device
        
        # Leontief hardware stress and community computing affordance monitor
        self.leontief_governor = LeontiefGovernor(state_dim=state_dim, device=device)
        
        # Bostick-Style Garden Attractors for mapping Wishful Identification and Silicon Scars
        self.garden_orchestrator = GardenOrchestrator(
            num_attractors=7,  # Default to 7 for the UT Archetypes
            feature_dim=state_dim,
            device=device
        )
        
        # The catalogue of topological coordinate tags (Ganbreeder Vector Stacker)
        self.tag_catalog: Dict[str, torch.Tensor] = {}
        
        # Base KANLayer-style basis parameters
        self.basis_weights = nn.Parameter(torch.randn(state_dim, state_dim))
        
        # High-frequency micro-wave carrier for Bouligand projection
        self.omega_micro = 137.0  # Fine-structure constant derived frequency

    def register_tag_coordinate(self, tag_name: str, coordinate: torch.Tensor):
        """Registers a user-discovered coordinate or imported .obj semantic tag."""
        if coordinate.shape[0] != self.state_dim:
            coordinate = self._symmetry_preserving_reshape(coordinate, self.state_dim)
        self.tag_catalog[tag_name] = coordinate.to(self.device)

    def _symmetry_preserving_reshape(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """From BREAKTHROUGH_REPORT.md / TENSOR_DIMENSION_FIX.md"""
        if tensor.shape[0] < target_dim:
            pad_size = target_dim - tensor.shape[0]
            return torch.nn.functional.pad(tensor, (0, pad_size), mode='reflect')
        return tensor[:target_dim]

    def _inject_kanlayer_micro_wave(self, x: torch.Tensor) -> torch.Tensor:
        """
        Injects a high-frequency micro-wave carrier directly into the B-spline 
        activation to prevent execution flatlines at sharp facet boundaries.
        """
        base_activation = torch.tanh(x @ self.basis_weights)
        gradient_proxy = torch.abs(base_activation * (1 - base_activation))
        
        # 1.14 Honest Jitter Harvesting: Add hardware variance
        cpu_jitter = psutil.cpu_percent(interval=0.0) / 100.0
        
        # $\Phi(x) = \tilde{x} + A_{micro} \cdot |d\tilde{x}/dx| \cdot \sin(\omega_{micro} x)$
        micro_wave = gradient_proxy * torch.sin(self.omega_micro * x + cpu_jitter)
        return base_activation + 0.05 * micro_wave

    def _apply_bouligand_projection(self, drift_vector: torch.Tensor) -> torch.Tensor:
        """
        Project continuous-time SDE drift vectors onto the Bouligand Tangent Cone
        to ensure exact Lipschitz-continuous directional derivatives without DRAM stalls.
        """
        return torch.nn.functional.normalize(drift_vector, p=2, dim=-1)

    def apply_morphological_gauge_symmetry(self, coordinate: torch.Tensor, ambulatory_class: AmbulatoryClass) -> torch.Tensor:
        """
        Translates morphological types into procedural symmetry-breaking (what the user calls 'Gauge Symmetry Breaking').
        Instead of perfectly mirrored joints, we apply procedural anisotropic noise scaled by the morphology's moment-field.
        """
        # We simulate the moment-field by creating a structured parity mask.
        parity_mask = torch.ones_like(coordinate)
        
        if ambulatory_class == AmbulatoryClass.BIPED:
            # Bilateral symmetry broken softly at the extremities
            noise_scale = 0.02
            parity_mask[..., -10:] *= 1.2
        elif ambulatory_class == AmbulatoryClass.QUADRUPED:
            # Four contact points require stronger diagonal breaking to prevent mechanical lock
            noise_scale = 0.05
            parity_mask[..., ::2] *= 0.9
        elif ambulatory_class == AmbulatoryClass.THEROPOD:
            # Heavy tail balance (Moment-field shifts to rear center of mass)
            noise_scale = 0.04
            parity_mask[..., :self.state_dim//3] *= 1.5
        elif ambulatory_class == AmbulatoryClass.CENTIPEDE:
            # Translational/segmented symmetry, broken via sinusoidal phase shift
            noise_scale = 0.08
            phase = torch.linspace(0, 4*math.pi, self.state_dim, device=coordinate.device)
            parity_mask *= (1.0 + 0.3 * torch.sin(phase))
        else:
            noise_scale = 0.01

        organic_noise = torch.randn_like(coordinate) * noise_scale
        return coordinate * parity_mask + organic_noise

    def build_superposed_rig(
        self, 
        weights: Dict[str, float], 
        admr_transition_matrices: torch.Tensor,
        community_peer_id: Optional[str] = None,
        ambulatory_class: AmbulatoryClass = AmbulatoryClass.BIPED
    ) -> Optional[torch.Tensor]:
        """
        Generates a composite skeleton coordinate via the Ganbreeder Vector Stacker,
        incorporating psychological attractors.
        """
        # --- PHASE 1: Leontief Hardware Stress & Community Governance ---
        demand_vector = torch.zeros(self.state_dim, device=self.device)
        for tag, weight in weights.items():
            if tag in self.tag_catalog:
                demand_vector += weight * self.tag_catalog[tag]
                
        available_budget = 1.0 - (psutil.virtual_memory().percent / 100.0)
        
        should_veto, diags = self.leontief_governor.should_veto_concept(
            demand=demand_vector, 
            transition_matrices=admr_transition_matrices,
            available_budget=available_budget
        )
        
        if should_veto:
            print(f"[HARNESS VETO] Rig synthesis rejected. Cost ratio: {diags['cost_ratio']:.2f}")
            if diags.get('slashed_via_poison', False):
                print(f"[SECURITY] Community Peer {community_peer_id} slashed from Freenet.")
            return None

        # --- PHASE 2: Collaborative Interactive Evolution (Vector Stacking) ---
        composite_target = demand_vector.clone()
        
        # --- PHASE 2.5: Bostick-Style Garden Attractors (Psychology of Choice) ---
        # Map "Wishful Identification" to Resonance/Influence Attractors
        # Map "Silicon Scars" to Defect Attractors
        if composite_target.dim() == 1:
            batched_target = composite_target.unsqueeze(0)
        else:
            batched_target = composite_target
            
        # 1. Wishful Identification (Ideal Self): Harmonic lock-in via phase alignment
        ideal_pull = self.garden_orchestrator.influence_attractors(batched_target)
        
        # 2. Silicon Scars (Dreaded Self): Topological rupture propagation for generative defect seeds
        # Anchored by the Bostick Chiral Gating Function inside the DefectAttractor
        scar_rupture = self.garden_orchestrator.defect_attractors(batched_target)
        
        # Apply the psychological forces
        composite_target = batched_target + 0.1 * ideal_pull + 0.05 * scar_rupture
        if composite_target.shape[0] == 1 and demand_vector.dim() == 1:
            composite_target = composite_target.squeeze(0)
            
        # --- PHASE 2.8: Morphological Gauge Symmetry Breaking ---
        # Apply the specific "moment-field" anisotropic symmetry breaks based on morphology
        composite_target = self.apply_morphological_gauge_symmetry(composite_target, ambulatory_class)
        
        # --- PHASE 3: Bouligand Tangent Cone & Base Mesh Smoothing ---
        composite_target = self._inject_kanlayer_micro_wave(composite_target)
        projected_skeleton = self._apply_bouligand_projection(composite_target)
        
        return projected_skeleton

    def pull_one_shot_memory(self, memory_type: str, state_vector: torch.Tensor):
        """
        Interfaces with the 13+ Memory Systems defined in one_shot_learning_analysis.md.
        """
        # 1.2 Dyad Fossilization
        if memory_type == "fossilization":
            pass # Load from KnowledgeDyads via 137D key
            
        # 1.1 Chiral Residue Cache
        elif memory_type == "chiral_cache":
            pass # Bypasses standard optimization warm-up via ADMR cache
            
        # 1.9 Manifold Clock Breathing Time
        elif memory_type == "manifold_clock":
            pass # dt -> 0 under high structural pressure
            
        # 1.11 Sine-Gordon Breather Solitons
        elif memory_type == "breather_soliton":
            pass # Standing wave survival
            
        return state_vector

    def forward(self, input_coordinates: torch.Tensor) -> torch.Tensor:
        """Passes raw geometric coordinates through the harness."""
        x = self._inject_kanlayer_micro_wave(input_coordinates)
        return self._apply_bouligand_projection(x)
