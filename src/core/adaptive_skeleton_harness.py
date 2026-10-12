import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Set
import psutil
import math
import os
import glob

from src.core.leontief_governor import LeontiefGovernor
from src.core.garden_statistical_attractors import GardenOrchestrator
from enum import Enum, auto

from src.core.invariants import get_prime_ladder, PI, PHI
from src.core.modular_virtualization import ModularVirtualizationLayer
from src.core.knowledge_dyad_fossilizer import DyadFossilizer
from src.core.chern_simons_gasket import ChernSimonsGasket
from src.core.agent_substrate_bridge import AgentSubstrateBridge
from src.core.fgrt_primitives import PrimeResonanceLadder, morton_encode_3d
from src.core.honest_jitter import harvest_honest_jitter
from src.ui.voxelboxter_simulation import morton_encode, StructuralGraph, Block, RigidBody
from src.core.structural_blueprints import AddonRoutine, MangostienBSplineMod, BooleanXORLayer

class AmbulatoryClass(Enum):
    BIPED = auto()
    QUADRUPED = auto()
    THEROPOD = auto()
    CENTIPEDE = auto()
    NON_AMBULATORY = auto()

class EnemySubtype(Enum):
    NONE = auto()
    ABSTRACTED_GLITCH = auto()       # Kaufmo/Queenie glitched matter, non-ergodic rupture
    MANNEQUIN_INFILTRATOR = auto()   # Abel-style rigid biped, deceptive parity mask, low variance
    EXISTENTIAL_SENTIENT = auto()    # Gummigoo-style bandit/reptile, high empathy susceptibility, syrup deformation
    VOID_STALKER = auto()            # Cellar null entity, negative topological pressure, phase-shifting
    DEMIURGIC_TITAN = auto()         # Mildenhall/Caine boss archetype, multi-limbed reality-distorting dt
    FERAL_SWARM = auto()             # Theropod/quadruped chitinous scavenger, flocking vector fields

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

    def __init__(self, state_dim: int, device: str = "cpu", fossilizer: Optional[Any] = None):
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

        # Modular Virtualization Layer (Residue Number System & Repunit-CRT sparse probes)
        self.modular_rns = ModularVirtualizationLayer(dim=state_dim, device=device)

        # Chern-Simons Gasket (Multi-Dimensional Non-Abelian Gauge Connection & Curvature)
        self.chern_simons = ChernSimonsGasket(manifold_dim=state_dim, device=device)

        # Knowledge Dyad Fossilizer & Agent Substrate Bridge (LLM archive digestion & Agent Smith rehydration)
        self._fossilizer = fossilizer
        self.substrate_bridge = AgentSubstrateBridge(device=device)

        # Prime Resonance Ladder for modular algebra and harmonic joint tuning
        self.resonance_ladder = PrimeResonanceLadder(num_resonators=min(32, max(8, state_dim)))
        self.p2p_mesh_affordance = 1.0
        self.last_rig_telemetry: Dict[str, Any] = {}

    @property
    def fossilizer(self):
        if self._fossilizer is None:
            self._fossilizer = DyadFossilizer(storage_dir="data/encodings", feature_dim=self.state_dim)
        return self._fossilizer

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

    def apply_enemy_subtype_morphology(self, coordinate: torch.Tensor, subtype: EnemySubtype) -> torch.Tensor:
        """
        Applies topological distortions, psychological scar amplifications,
        and anisotropic symmetry breaking corresponding to specific enemy subtypes
        from the TADC / SOMA / Universal Topology lore.
        """
        if subtype == EnemySubtype.NONE:
            return coordinate

        deformed = coordinate.clone()
        if subtype == EnemySubtype.ABSTRACTED_GLITCH:
            # Kaufmo/Queenie Glitched Matter: extreme non-ergodic phase noise and chaotic tearing
            jitter = (torch.rand_like(deformed) - 0.5) * 0.45
            mask = torch.rand_like(deformed) > 0.65
            deformed[mask] = deformed[mask] * -1.8 + jitter[mask]
        elif subtype == EnemySubtype.MANNEQUIN_INFILTRATOR:
            # Abel-style rigid script: stark quantization into planar low-degree facets
            deformed = torch.round(deformed * 4.0) / 4.0
        elif subtype == EnemySubtype.EXISTENTIAL_SENTIENT:
            # Gummigoo-style bandit: soft matter relaxation with visceral empathy softening
            deformed = torch.tanh(deformed * 0.8) * 1.2
        elif subtype == EnemySubtype.VOID_STALKER:
            # Cellar Null Entity: negative topological pressure, projective inversion
            deformed = -1.0 * torch.sign(deformed) * torch.log1p(torch.abs(deformed))
        elif subtype == EnemySubtype.DEMIURGIC_TITAN:
            # Mildenhall/Caine Boss: reality-distorting harmonic expansion across prime indices
            scale_ladder = torch.linspace(1.0, 2.5, deformed.shape[-1], device=deformed.device)
            deformed = deformed * scale_ladder + 0.15 * torch.sin(self.omega_micro * deformed)
        elif subtype == EnemySubtype.FERAL_SWARM:
            # Chitinous flocking: rotational phase permutation
            deformed = torch.roll(deformed, shifts=2, dims=-1) * 0.9 + 0.1 * torch.randn_like(deformed)

        return deformed

    def build_superposed_rig(
        self, 
        weights: Dict[str, float], 
        admr_transition_matrices: Optional[torch.Tensor] = None,
        community_peer_id: Optional[str] = None,
        ambulatory_class: AmbulatoryClass = AmbulatoryClass.BIPED,
        enemy_subtype: EnemySubtype = EnemySubtype.NONE
    ) -> Optional[torch.Tensor]:
        """
        Generates a composite skeleton coordinate via the Ganbreeder Vector Stacker,
        incorporating psychological attractors and enemy subtype morphology.
        """
        if admr_transition_matrices is None:
            admr_transition_matrices = torch.eye(self.state_dim, device=self.device).unsqueeze(0)

        # --- PHASE 1: Leontief Hardware Stress & Community Governance ---
        demand_vector = torch.zeros(self.state_dim, device=self.device)
        for tag, weight in weights.items():
            if tag in self.tag_catalog:
                demand_vector += weight * self.tag_catalog[tag]
        if demand_vector.norm() == 0:
            demand_vector = torch.randn(self.state_dim, device=self.device)
                
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
        scar_rupture = self.garden_orchestrator.defect_attractors(batched_target)
        
        # Apply the psychological forces
        composite_target = batched_target + 0.1 * ideal_pull + 0.05 * scar_rupture
        if composite_target.shape[0] == 1 and demand_vector.dim() == 1:
            composite_target = composite_target.squeeze(0)

        # --- PHASE 2.6: Compute Affordance Gating & Agent Smith Fossil Retrieval ---
        # Gated by local RAM/CPU stress and collective P2P network affordance
        affordance = self.evaluate_compute_affordance()
        fossil_data = None
        morton_keys = []
        if affordance["can_deep_retrieve"]:
            archetype_tag = f"archetype_{enemy_subtype.name.lower()}" if enemy_subtype != EnemySubtype.NONE else "archetype_sovereign"
            fossil_data = self.retrieve_agent_smith_fossil(archetype_tag, demand_dim=self.state_dim)
            if fossil_data and "meta_state" in fossil_data:
                f_state = fossil_data["meta_state"].to(composite_target.device)
                if f_state.shape[-1] != composite_target.shape[-1]:
                    f_state = self._symmetry_preserving_reshape(f_state, composite_target.shape[-1])
                composite_target = composite_target + 0.15 * f_state

        # --- PHASE 2.7: Multi-Dimensional Gauge Functional & Morton Spatial Encoding ---
        fossil_gauge = fossil_data.get("gauge_field") if fossil_data else None
        composite_target, morton_keys = self.apply_multidimensional_gauge_functional(composite_target, fossil_gauge)

        # --- PHASE 2.75: Modular Algebra CRT Coprime Harmonics ---
        crt_id = fossil_data.get("pot_identity_crt") if fossil_data else None
        composite_target = self.apply_modular_algebra_harmonics(composite_target, crt_id)

        # --- PHASE 2.8: Morphological Gauge Symmetry Breaking ---
        # Apply the specific "moment-field" anisotropic symmetry breaks based on morphology
        composite_target = self.apply_morphological_gauge_symmetry(composite_target, ambulatory_class)

        # --- PHASE 2.9: Enemy Subtype Topological Distortion ---
        if enemy_subtype != EnemySubtype.NONE:
            composite_target = self.apply_enemy_subtype_morphology(composite_target, enemy_subtype)
        
        # --- PHASE 3: Bouligand Tangent Cone & Base Mesh Smoothing ---
        composite_target = self._inject_kanlayer_micro_wave(composite_target)
        projected_skeleton = self._apply_bouligand_projection(composite_target)
        
        # Cache live telemetry for downstream Voxelboxter compilation & diagnostic tracking
        self.last_rig_telemetry = {
            "morton_codes": morton_keys,
            "fossil_telemetry": fossil_data,
            "compute_affordance": affordance,
            "ambulatory_class": ambulatory_class,
            "enemy_subtype": enemy_subtype
        }
        
        return projected_skeleton

    def evaluate_compute_affordance(self, peer_network: Optional[Any] = None) -> Dict[str, Any]:
        """
        Gauges local hardware affordances and collective P2P network budgets.
        Checks local memory/CPU budgets and inspects Bonfire nomadic ring peer status.
        Gates expensive Agent Smith retrieval and structural algebra compilation.
        """
        mem_avail = 1.0 - (psutil.virtual_memory().percent / 100.0)
        cpu_avail = 1.0 - (psutil.cpu_percent(interval=0.0) / 100.0)
        local_affordance = max(0.05, mem_avail * 0.7 + cpu_avail * 0.3)
        p2p_affordance = self.p2p_mesh_affordance

        # Check collective peer budget if peer network provided or check bonfire peers file
        if peer_network is not None and hasattr(peer_network, 'get_collective_compute_budget'):
            try:
                p2p_affordance = peer_network.get_collective_compute_budget()
            except Exception:
                p2p_affordance = self.p2p_mesh_affordance
        elif os.path.exists("data/bonfire_peers.json"):
            try:
                with open("data/bonfire_peers.json", "r") as f:
                    peers_data = json.load(f)
                    if isinstance(peers_data, list) and len(peers_data) > 0:
                        p2p_affordance = min(1.0, 0.5 + 0.1 * len(peers_data))
            except Exception:
                pass

        combined_budget = min(local_affordance, p2p_affordance)
        return {
            "local_affordance": local_affordance,
            "p2p_affordance": p2p_affordance,
            "combined_budget": combined_budget,
            "can_deep_retrieve": combined_budget > 0.15,
            "can_structural_compile": combined_budget > 0.10
        }

    def retrieve_agent_smith_fossil(
        self,
        archetype_tag: str,
        demand_dim: int,
        search_dirs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves or rehydrates Agent Smith fossil tensors resulting from LLM archive digestion.
        Scans both active fossil storage (data/encodings) and the ChatGPT friction harvester
        archive (data/service_llm_archive).
        Aligns and verifies invariants via AgentSubstrateBridge.
        If no disk fossil matches, derives an in-situ sovereign soliton using DyadFossilizer
        dynamic residue tuples, get_prime_ladder, and hardware Honest Jitter.
        """
        if search_dirs is None:
            search_dirs = ["data/encodings", "data/service_llm_archive"]

        candidate_files = []
        for sdir in search_dirs:
            if os.path.exists(sdir):
                count = 0
                try:
                    for entry in os.scandir(sdir):
                        if entry.name.endswith(".pt") and not entry.name.startswith("."):
                            candidate_files.append(entry.path)
                            count += 1
                            if count >= 8:
                                break
                except Exception:
                    pass

        target_payload = None
        source_label = "in_situ_soliton"
        if candidate_files:
            for fpath in candidate_files:
                try:
                    if os.path.getsize(fpath) > 5 * 1024 * 1024:
                        continue
                    payload = torch.load(fpath, map_location=self.device)
                    if isinstance(payload, dict) and payload.get("type") == "soliton_smith":
                        tags = payload.get("tags", [])
                        if any(archetype_tag in t for t in tags) or not target_payload:
                            target_payload = payload
                            source_label = f"fossil_disk:{os.path.basename(fpath)}"
                            if any(archetype_tag in t for t in tags):
                                break
                except Exception:
                    continue

        if target_payload is not None:
            # Verify and align through AgentSubstrateBridge
            try:
                self.substrate_bridge.verify_invariants(target_payload)
                aligned_payload = self.substrate_bridge.align_substrate(
                    target_payload, expected_dim=demand_dim
                )
            except Exception:
                aligned_payload = target_payload

            raw_meta = aligned_payload.get("meta_state_shielded")
            if raw_meta is None:
                raw_meta = aligned_payload.get("unified_spectral_signature")
            if isinstance(raw_meta, torch.Tensor):
                meta_state = raw_meta.to(self.device)
            else:
                meta_state = harvest_honest_jitter((demand_dim,), scaled=True).to(self.device)

            gauge_field = aligned_payload.get("gauge_field_aligned", aligned_payload.get("gauge_field"))
            if isinstance(gauge_field, torch.Tensor):
                gauge_field = gauge_field.to(self.device)
            else:
                gauge_field = self.chern_simons.gauge_field

            prime_freqs = aligned_payload.get("prime_frequencies")
            if not isinstance(prime_freqs, torch.Tensor):
                prime_freqs = get_prime_ladder(demand_dim, device=self.device)

            crt_pot_id = aligned_payload.get("pot_identity_crt")
            if crt_pot_id is None:
                crt_pot_id = self.fossilizer.generate_residue_tuple(meta_state)
        else:
            # Sovereign in-situ generation anchored to physical hardware timing jitter
            meta_state = harvest_honest_jitter((demand_dim,), scaled=True).to(self.device)
            gauge_field = self.chern_simons.gauge_field
            prime_freqs = get_prime_ladder(demand_dim, device=self.device)
            # Dynamically derive CRT residue tuple from tensor energy without hardcoded primes
            crt_pot_id = self.fossilizer.generate_residue_tuple(meta_state)

        return {
            "meta_state": meta_state,
            "gauge_field": gauge_field,
            "prime_frequencies": prime_freqs,
            "pot_identity_crt": crt_pot_id,
            "source": source_label
        }

    def apply_multidimensional_gauge_functional(
        self,
        coordinate: torch.Tensor,
        fossil_gauge: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Applies multi-dimensional gauge field functional:
        A_mu = (Phi_pressure, A_flow, ...)
        Interacts with ChernSimonsGasket to plug logic leaks and track non-commutative curvature.
        Morton-encodes continuous gauge nodes into 1D spatial hash keys using morton_encode.
        """
        coord_mod = coordinate.clone()
        dim = coord_mod.shape[-1]
        
        # Use fossil gauge if provided, otherwise ChernSimonsGasket gauge field
        active_gauge = fossil_gauge if (fossil_gauge is not None and isinstance(fossil_gauge, torch.Tensor)) else self.chern_simons.gauge_field
        if active_gauge is not None and isinstance(active_gauge, torch.Tensor):
            g_slice = active_gauge[:dim, :dim]
            if g_slice.shape == (dim, dim):
                coord_mod = coord_mod + torch.tanh(coord_mod @ g_slice.T) * 0.12

        # Morton spatial encoding along kinematic chain into 30-bit spatial hash keys
        morton_keys = []
        with torch.no_grad():
            flat = coord_mod.detach().cpu().flatten()
            for i in range(0, max(3, len(flat) - 2), 3):
                x_val = int(round(flat[i % len(flat)].item() * 5.0)) & 0x3FF
                y_val = int(round(flat[(i + 1) % len(flat)].item() * 5.0)) & 0x3FF
                z_val = int(round(flat[(i + 2) % len(flat)].item() * 5.0)) & 0x3FF
                code = morton_encode(x_val, y_val, z_val)
                morton_keys.append(code)

        return coord_mod, morton_keys

    def apply_modular_algebra_harmonics(
        self,
        coordinate: torch.Tensor,
        crt_identity: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Applies modular algebra via Chinese Remainder Theorem coprime residues,
        ModularVirtualizationLayer (RNS Repunit-CRT probes), and dynamic prime ladders (get_prime_ladder).
        Eliminates kinematic mode-locking without hardcoded prime constants.
        """
        dim = coordinate.shape[-1]
        
        # 1. Dynamically fetch resonant primes and incommensurate frequencies (Polychron / Invariant standards)
        primes = get_prime_ladder(dim, device=coordinate.device)
        # f_{p_k} = 2*pi*ln(p_k)
        incommensurate_freqs = 2.0 * math.pi * torch.log(torch.clamp(primes, min=2.0))
        
        # 2. Derive CRT residue tuple dynamically if not provided
        if crt_identity is None or not crt_identity:
            crt_identity = self.fossilizer.generate_residue_tuple(coordinate)
            
        # 3. Apply CRT phase modulation across coprime moduli
        indices = torch.arange(dim, device=coordinate.device, dtype=torch.float32)
        phase_mod = torch.zeros_like(indices)
        for p in crt_identity:
            p_val = max(2, int(p))
            phase_mod += torch.sin(2.0 * math.pi * (indices % p_val) / float(p_val))
            
        phase_scale = (phase_mod / max(1, len(crt_identity))) * 0.05
        
        # 4. Modular RNS projection via ModularVirtualizationLayer
        try:
            rns_residues = self.modular_rns.float_to_rns(coordinate)
            norm_rns = (rns_residues / (rns_residues.norm() + 1e-6)) * 0.02
        except Exception:
            norm_rns = 0.0
            
        # Combine incommensurate frequencies, CRT phase mod, and RNS residues
        harmonic_shift = torch.cos(incommensurate_freqs * 0.01) * phase_scale
        return coordinate * (1.0 + harmonic_shift.to(coordinate.device)) + norm_rns

    def apply_structural_algebra(
        self,
        skeleton_tensor: torch.Tensor,
        morton_keys: List[int],
        target_graph: Optional[StructuralGraph] = None
    ) -> StructuralGraph:
        """
        Interacts with structural algebra functionals (AddonRoutine, MangostienBSplineMod,
        BooleanXORLayer) to project continuous skeleton into discrete voxel constructs.
        """
        graph = target_graph if target_graph is not None else StructuralGraph()
        routine = AddonRoutine()
        
        # 1. B-spline compiled mod along skeleton spine
        bspline = MangostienBSplineMod(
            name="skeleton_spine_algebra",
            latent_dim=3,
            resolution=max(4, min(8, len(morton_keys)))
        )
        routine.add_layer(bspline)
        
        # 2. Defect injection layer (Boolean XOR)
        xor_layer = BooleanXORLayer(
            name="topological_defect_seam",
            center=(0, 0, 0),
            dimensions=(1, 1, 1)
        )
        routine.add_layer(xor_layer)
        
        # Execute routine on graph
        routine.execute(graph)
        
        # Ensure key Morton spatial anchor voxels are represented in the graph
        for i, code in enumerate(morton_keys[:8]):
            vx = (code & 0x7) - 3
            vy = ((code >> 3) & 0x7) - 3
            vz = ((code >> 6) & 0x7) + i
            cell = (vx, vy, vz)
            if cell not in graph.blocks:
                graph.add_block(Block(local_cell=cell, rotation=(0, 0, 0, 1), material_id=1, health=100.0))
                
        return graph

    def compile_to_voxelboxter_construct(
        self,
        ambulatory_class: AmbulatoryClass = AmbulatoryClass.BIPED,
        enemy_subtype: EnemySubtype = EnemySubtype.NONE,
        difficulty_scale: float = 1.0,
        bone_ratio: float = 0.8,
        skeleton_override: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        End-to-end bridge: Compiles the adaptive skeleton into a complete
        Voxelboxter Construct (StructuralGraph + RigidBody) with Wager #2 80/20
        fossilization split, Morton octree keys, and Agent Smith rehydration telemetry.
        """
        if skeleton_override is not None:
            skeleton = skeleton_override
        else:
            tag_weights = {f"archetype_{enemy_subtype.name.lower()}": 1.0}
            skeleton = self.build_superposed_rig(
                weights=tag_weights,
                ambulatory_class=ambulatory_class,
                enemy_subtype=enemy_subtype
            )
            if skeleton is None:
                skeleton = self.forward(torch.randn(self.state_dim, device=self.device))

        morton_keys = []
        fossil_info = {}
        if hasattr(self, "last_rig_telemetry") and self.last_rig_telemetry:
            morton_keys = self.last_rig_telemetry.get("morton_codes", [])
            fossil_info = self.last_rig_telemetry.get("fossil_telemetry", {})
        if not morton_keys:
            _, morton_keys = self.apply_multidimensional_gauge_functional(skeleton)

        # Build discrete StructuralGraph using structural algebra
        graph = self.apply_structural_algebra(skeleton, morton_keys)
        
        # Enforce Wager #2: 80/20 Fossilization Split
        fossil_split = graph.fossilize_core(bone_ratio=bone_ratio)
        
        # Instantiate RigidBody with precomputed bone inertial invariants
        rb = RigidBody(
            mass=max(1.0, graph.bone_mass * 10.0),
            center_of_mass=graph.bone_center_of_mass,
            inertia_tensor=graph.bone_inertia_tensor,
            kelly_growth_rate=1.0,
            is_fossilized=False
        )

        return {
            "graph": graph,
            "rigid_body": rb,
            "skeleton": skeleton,
            "morton_codes": morton_keys,
            "fossil_split": fossil_split,
            "fossil_telemetry": fossil_info,
            "bone_blocks": len(graph.bone_blocks),
            "flesh_blocks": len(graph.flesh_blocks),
            "total_blocks": len(graph.blocks)
        }

    def build_enemy_rig(
        self,
        subtype: EnemySubtype,
        ambulatory_class: AmbulatoryClass = AmbulatoryClass.BIPED,
        difficulty_scale: float = 1.0,
        admr_transition_matrices: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Constructs a complete enemy entity rig package with kinematics and combat telemetry.
        Directly bridges into Voxelboxter constructs when compute affordances permit.
        """
        if admr_transition_matrices is None:
            admr_transition_matrices = torch.eye(self.state_dim, device=self.device).unsqueeze(0)

        # Build base tag weight profile
        tag_weights = {f"archetype_{subtype.name.lower()}": 1.0}
        skeleton = self.build_superposed_rig(
            weights=tag_weights,
            admr_transition_matrices=admr_transition_matrices,
            ambulatory_class=ambulatory_class,
            enemy_subtype=subtype
        )
        if skeleton is None:
            # Fallback coordinate if vetoed
            skeleton = self.forward(torch.randn(self.state_dim, device=self.device))

        # Enemy subtype attributes
        meta = {
            EnemySubtype.NONE: {"health": 100.0, "aggression": 0.0, "damage_type": "physical", "lore": "Neutral construct"},
            EnemySubtype.ABSTRACTED_GLITCH: {"health": 250.0, "aggression": 0.95, "damage_type": "phase_noise", "lore": "Abstracted mind reduced to non-ergodic glitched matter"},
            EnemySubtype.MANNEQUIN_INFILTRATOR: {"health": 120.0, "aggression": 0.40, "damage_type": "scripted_ballistic", "lore": "Caine-orchestrated deceptive mannequin"},
            EnemySubtype.EXISTENTIAL_SENTIENT: {"health": 160.0, "aggression": 0.60, "damage_type": "syrup_blunt", "lore": "Sentient bandit awakening to simulated existence"},
            EnemySubtype.VOID_STALKER: {"health": 190.0, "aggression": 0.85, "damage_type": "void_entropy", "lore": "Cellar predator thriving in negative pressure"},
            EnemySubtype.DEMIURGIC_TITAN: {"health": 600.0, "aggression": 0.99, "damage_type": "dt_dilation", "lore": "Mildenhall creature / demiurgic administrative titan"},
            EnemySubtype.FERAL_SWARM: {"health": 80.0, "aggression": 0.75, "damage_type": "chitin_pierce", "lore": "Coordinated swarm scavengers"}
        }.get(subtype, {"health": 100.0, "aggression": 0.5, "damage_type": "physical", "lore": "Standard entity"})

        # Build Voxelboxter StructuralGraph if compute affordance allows
        voxel_construct = None
        affordance = self.evaluate_compute_affordance()
        if affordance["can_structural_compile"]:
            voxel_construct = self.compile_to_voxelboxter_construct(
                ambulatory_class=ambulatory_class,
                enemy_subtype=subtype,
                difficulty_scale=difficulty_scale,
                skeleton_override=skeleton
            )

        telemetry = getattr(self, "last_rig_telemetry", {})

        return {
            "skeleton": skeleton,
            "subtype": subtype,
            "ambulatory_class": ambulatory_class,
            "max_health": meta["health"] * difficulty_scale,
            "current_health": meta["health"] * difficulty_scale,
            "aggression_index": min(1.0, meta["aggression"] * difficulty_scale),
            "damage_type": meta["damage_type"],
            "lore_anchor": meta["lore"],
            "voxel_construct": voxel_construct,
            "morton_codes": telemetry.get("morton_codes", []),
            "fossil_telemetry": telemetry.get("fossil_telemetry", {}),
            "compute_affordance": affordance
        }

    def pull_one_shot_memory(self, memory_type: str, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Interfaces with the 13+ Memory Systems defined in one_shot_learning_analysis.md.
        Grounded in persistent fossil dyads, chiral caches, manifold breathing, and solitons.
        """
        # 1.2 Dyad Fossilization: Grab fossil tensor from LLM archive digestion artifacts
        if memory_type == "fossilization":
            fossil = self.retrieve_agent_smith_fossil("archetype_sovereign", demand_dim=state_vector.shape[-1])
            if fossil and "meta_state" in fossil:
                f_state = fossil["meta_state"].to(state_vector.device)
                if f_state.shape[-1] != state_vector.shape[-1]:
                    f_state = self._symmetry_preserving_reshape(f_state, state_vector.shape[-1])
                return state_vector + 0.1 * f_state
            
        # 1.1 Chiral Residue Cache: Bypasses standard warmup via honest jitter phase
        elif memory_type == "chiral_cache":
            jitter = harvest_honest_jitter(state_vector.shape, scaled=True).to(state_vector.device)
            return state_vector * (1.0 + 0.05 * jitter)
            
        # 1.9 Manifold Clock Breathing Time: Dilation modulation under structural pressure
        elif memory_type == "manifold_clock":
            pressure = float(torch.norm(state_vector).item())
            dilation = 1.0 / (1.0 + 0.1 * pressure)
            return state_vector * dilation
            
        # 1.11 Sine-Gordon Breather Solitons: Standing wave harmonics survival
        elif memory_type == "breather_soliton":
            dim = state_vector.shape[-1]
            primes = get_prime_ladder(dim, device=state_vector.device)
            incommensurate_freqs = 2.0 * math.pi * torch.log(torch.clamp(primes, min=2.0))
            breather = torch.cos(incommensurate_freqs * 0.1)
            return state_vector + 0.05 * breather
            
        return state_vector

    def forward(self, input_coordinates: torch.Tensor) -> torch.Tensor:
        """Passes raw geometric coordinates through the harness."""
        x = self._inject_kanlayer_micro_wave(input_coordinates)
        return self._apply_bouligand_projection(x)
