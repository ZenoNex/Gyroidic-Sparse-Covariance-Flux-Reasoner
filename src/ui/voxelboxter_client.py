"""
Voxelboxter Game Client - Dual Creation/Play Engine
Built on PyBevy (Python Real-Time Engine)

This Freenet client enforces Patch Sovereignty. It separates "Creation Mode" 
(additive/subtractive layered B-Splines with Admin roles) from "Play Mode" 
(Survival logic, RigidBody physics, mass calculation). 

Author: William Matthew Bryant
Date: August 2026
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
import requests
from enum import Enum, auto

try:
    import pybevy
    from pybevy import App, DefaultPlugins, Commands, Res, ResMut, Query, Transform, Vec3, Color
except ImportError:
    logging.warning("PyBevy is not installed or failed to load. Running in headless simulation fallback mode.")
    pybevy = None

import torch
from src.core.honest_jitter import harvest_honest_jitter
from src.safety.subversive_oracle import ResonantSVNNOracle
from src.core.orchestrator import UniversalOrchestrator
from src.safety.red_teaming import RedTeamProjection
from src.core.jspace_pca_mapper import JSpacePCAMapper

from src.ui.voxelboxter_simulation import (
    StructuralGraph, RigidBody, AddonRoutine, BSplineCompiledMod, 
    BooleanXORLayer, Role, PermissionsManager, InventoryComponent,
    AirBreathingBattery, VehicleEngine, EnvironmentalAtmosphere,
    WorldGenConfig, WORLDGEN_PRESETS, VoxelboxterEngine
)

try:
    from src.scripting.node_environment import PhysicalNodeEditor
except ImportError:
    PhysicalNodeEditor = None


def resolve_api_url(explicit_url: str = None) -> str:
    """Resolve the active Diegetic/Hybrid backend URL before launching telemetry."""
    if explicit_url:
        return str(explicit_url).rstrip("/")
    env_url = os.environ.get("VOXELBOXTER_API_URL") or os.environ.get("MAIN_BACKEND_URL")
    if env_url:
        return str(env_url).rstrip("/")
    for candidate in ("http://localhost:8000", "http://localhost:1337"):
        try:
            res = requests.get(f"{candidate}/ping", timeout=0.6)
            if res.status_code == 200:
                return candidate
        except Exception:
            continue
    return "http://localhost:8000"


API_URL = os.environ.get("VOXELBOXTER_API_URL") or os.environ.get("MAIN_BACKEND_URL") or "http://localhost:8000"


class BootStage(Enum):
    UNINITIALIZED = "UNINITIALIZED"
    HANDSHAKE = "HANDSHAKE"
    SILICON_READY = "SILICON_READY"
    ENGINE_READY = "ENGINE_READY"
    WORLDGEN_COMPLETE = "WORLDGEN_COMPLETE"
    RUNNING = "RUNNING"


class EngineMode(Enum):
    CREATION = auto()  # Admin/Creative mode: edit addon routines freely
    PLAY = auto()      # Survival/Physics mode: lock routines, apply physics

class PatchStateResource:
    """ECS Resource holding the simulation graph, roles, worldgen state, and reasoner telemetry."""
    def __init__(self, api_url: str = None, worldgen_config: dict = None, lightweight: bool = False):
        self.graph = StructuralGraph()
        self.routine = AddonRoutine() # The layered blueprint
        self.mesh_dirty = False
        self.lightweight = bool(lightweight)
        self.api_url = (api_url or API_URL).rstrip("/")
        self.boot_stage = BootStage.UNINITIALIZED
        self.boot_report = {}
        self.stop_event = threading.Event()
        self.engine = None
        self.silicon_ready = False
        self.worldgen_config = WorldGenConfig.from_dict(worldgen_config)
        self.worldgen_summary = None
        
        self.mode = EngineMode.CREATION
        self.permissions = PermissionsManager()
        self.local_inventory = InventoryComponent()
        self.local_peer_id = "ADMIN_OWNER_1337"
        
        # Self-gift admin role since we host the patch
        self.permissions.roles[self.local_peer_id] = Role.ADMIN
        self.copyable_text_enabled = True
        self.alien_script_locked = True
        self.admin_password_tooltip = (
            "[JEI INSPECTOR] Admin Password & Node Auth Files:\n"
            "1. src/safety/hardware_fingerprint.py : verify_node_account(), verify_sole_creator(), resolve_node_salt_path()\n"
            "2. data/.node_secret & data/creator_sig.json : 32-byte HMAC cryptographic node salt / keyfile\n"
            "3. .gyroid_config.json & hybrid_backend.py : node_account (GYROID_ACCOUNT) & security_salt_path (GYROID_SALT_PATH)\n"
            "4. src/ui/diegetic_backend.py : /api/security/creator_login & /api/security/node_account_login\n"
            "5. src/ui/voxelboxter_client.py & src/ui/voxelboxter_simulation.py : Role.ADMIN granted via PermissionsManager"
        )
        
        # Reasoner Telemetry
        self.fingerprint_energy = 0.0
        self.last_update = time.time()
        self.lock = threading.Lock()
        
        # Physical Scripting Window
        self.node_editor = None
        
        # S-VNN Chat Monitoring
        self.chat_queue = []
        self.chat_history = []  # For TrustInheritanceTracker / context
        try:
            self.orchestrator = UniversalOrchestrator(dim=32)
        except Exception as e:
            logging.warning(f"Failed to initialize UniversalOrchestrator: {e}")
            self.orchestrator = None
            
        self.svnn_oracle = ResonantSVNNOracle(hidden_dim=32, orchestrator=self.orchestrator)
        self.red_team_proj = RedTeamProjection(hidden_dim=32)
        self.pca_mapper = JSpacePCAMapper(n_components=10)
        
        # Silicon Sovereignty Engine for PyOpenCL Acceleration
        try:
            from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
            self.silicon_sovereignty = SiliconSovereigntyEngine(use_gpu=(not self.lightweight))
            self.silicon_ready = bool(getattr(self.silicon_sovereignty, "ctx", None) is not None)
            logging.info(
                "Silicon Sovereignty Engine (PyOpenCL) initialized for Voxelboxter Client (gpu_ready=%s).",
                self.silicon_ready,
            )
        except Exception as e:
            self.silicon_sovereignty = None
            self.silicon_ready = False
            logging.warning("Silicon Sovereignty Engine unavailable (%s). Running CPU fallback.", e)

        # In-Game Configuration GUI & Advanced Cleanroom Extensions
        from src.environment.cleanroom_advanced_extensions import (
            GliderConfig, OpenGliderKinematics, StratosphericBalloon,
            LargeVeinWorldGenerator, RailNetworkSystem, EntangledBindingSystem,
            AutoPlantManager, BiomeReactor, ClarityFogParameters, AnalogLever,
            SlurpInteraction, MetalAnvil, DespawnCustomizer, LootBagSystem,
            BalancedToolTracker, PortableUtilitiesKit, BlockPeripheralAdapter,
            RemoteCursorItem, AutonomousLuggageStorage, SlimalyzerSensor,
            SonicGlassesSensor, RecursiveSphericalPortal, Hyperdimensional4DEngine
        )
        self.glider_config = GliderConfig()
        self.glider = OpenGliderKinematics(config=self.glider_config)
        self.balloons: list = []
        self.vein_generator = LargeVeinWorldGenerator()
        self.rail_network = RailNetworkSystem()
        self.entangled_system = EntangledBindingSystem()
        self.auto_plant = AutoPlantManager()
        self.biome_reactor = BiomeReactor()
        self.clarity_fog = ClarityFogParameters()
        self.analog_levers: dict = {}
        self.metal_anvils: dict = {}
        self.despawn_manager = DespawnCustomizer()
        self.lootbags = LootBagSystem()
        self.camp_kit = PortableUtilitiesKit()
        self.remote_cursor = RemoteCursorItem()
        self.luggage_companions: dict = {
            "default": (
                self.orchestrator.luggage_storage
                if (self.orchestrator is not None and getattr(self.orchestrator, "luggage_storage", None) is not None)
                else AutonomousLuggageStorage(owner_id="client_luggage")
            )
        }
        self.luggage_storage = self.luggage_companions["default"]
        self.recursive_portal = RecursiveSphericalPortal()
        self.engine_4d = Hyperdimensional4DEngine()

        # Bind Sovereign Memory Systems from UniversalOrchestrator (or standalone cleanroom fallbacks)
        from src.environment.cleanroom_mechanics import (
            ChiselsAndBitsMicroOctree,
            MultiMineMemory,
            PurgatoryBufferMap,
        )
        from src.ui.voxelboxter_simulation import PointerlessOctree
        self.chisels_bits = (
            self.orchestrator.chisels_bits
            if (self.orchestrator is not None and getattr(self.orchestrator, "chisels_bits", None) is not None)
            else ChiselsAndBitsMicroOctree()
        )
        self.multi_mine = (
            self.orchestrator.multi_mine
            if (self.orchestrator is not None and getattr(self.orchestrator, "multi_mine", None) is not None)
            else MultiMineMemory()
        )
        self.purgatory = (
            self.orchestrator.purgatory
            if (self.orchestrator is not None and getattr(self.orchestrator, "purgatory", None) is not None)
            else PurgatoryBufferMap()
        )
        self.terrain_octree = (
            self.orchestrator.terrain_octree
            if (self.orchestrator is not None and getattr(self.orchestrator, "terrain_octree", None) is not None)
            else PointerlessOctree(max_depth=4)
        )
        self.memory_palace = getattr(self.orchestrator, "memory_palace", None)
        self.resonance_cavity = getattr(self.orchestrator, "resonance_cavity", None)
        self.trust_vault = getattr(self.orchestrator, "trust_vault", None)
        self.larynx = getattr(self.orchestrator, "larynx", None)
        self.archetype_governor = getattr(self.orchestrator, "archetypal_governor", None)

    def vocalize_psychotopological_substrate(
        self,
        substrate_state: torch.Tensor = None,
        prompt_text: str = "",
        mode: str = "sing",
        audio_harmonics_k64: torch.Tensor = None,
        max_steps: int = 24,
        pas_h: float = 0.78,
        external_pressure: float = 0.35,
    ) -> dict:
        """
        Delegates psychotopological substrate & archetype singing/talking in the Voxelboxter Client
        to the Unicorn Synthesis pipeline with strict Compute Affordance governance.
        """
        if self.orchestrator is not None and hasattr(
            self.orchestrator, "vocalize_psychotopological_substrate"
        ):
            return self.orchestrator.vocalize_psychotopological_substrate(
                substrate_state=substrate_state,
                prompt_text=prompt_text,
                mode=mode,
                audio_harmonics_k64=audio_harmonics_k64,
                max_steps=max_steps,
                pas_h=pas_h,
                external_pressure=external_pressure,
            )
        from src.ui.voxelboxter_simulation import VoxelboxterEngine

        sim_engine = VoxelboxterEngine(device="cpu")
        return sim_engine.vocalize_psychotopological_substrate(
            substrate_state=substrate_state,
            prompt_text=prompt_text,
            mode=mode,
            audio_harmonics_k64=audio_harmonics_k64,
            max_steps=max_steps,
            pas_h=pas_h,
            external_pressure=external_pressure,
        )

    def boot_sequence(self, worldgen_config: dict = None, probe_backend: bool = True) -> dict:
        """
        Ordered, race-free Voxelboxter booter sequence:
          1. HANDSHAKE: verify backend API reachability (non-blocking short probe).
          2. SILICON_READY: verify PyOpenCL SiliconSovereigntyEngine / CPU fallback.
          3. ENGINE_READY: eagerly initialize VoxelboxterEngine BEFORE ECS/telemetry loops.
          4. WORLDGEN_COMPLETE: run selected WorldGenConfig into StructuralGraph & PointerlessOctree.
          5. RUNNING: unlock telemetry daemon and ECS render/simulation ticks.
        """
        t0 = time.perf_counter()
        self.boot_stage = BootStage.HANDSHAKE
        backend_reachable = False
        if probe_backend:
            for probe_path in ("/api/status", "/ping"):
                try:
                    resp = requests.get(f"{self.api_url}{probe_path}", timeout=1.2)
                    if resp.status_code == 200:
                        backend_reachable = True
                        break
                except Exception:
                    continue

        self.boot_stage = BootStage.SILICON_READY
        self.silicon_ready = bool(
            self.silicon_sovereignty is not None
            and getattr(self.silicon_sovereignty, "ctx", None) is not None
        )

        self.boot_stage = BootStage.ENGINE_READY
        if self.engine is None:
            self.engine = VoxelboxterEngine(device="cpu", lightweight=self.lightweight)
            if self.silicon_sovereignty is not None and getattr(self.engine, "silicon_sovereignty", None) is None:
                self.engine.silicon_sovereignty = self.silicon_sovereignty

        cfg = WorldGenConfig.from_dict(worldgen_config) if worldgen_config is not None else self.worldgen_config
        self.worldgen_config = cfg
        with self.lock:
            summary = self.engine.generate_world(cfg)
            self.worldgen_summary = summary
            self.graph = self.engine.constructs["world_terrain"]
            self.terrain_octree = self.engine.terrain_octree
            self.graph.dirty = False
            self.mesh_dirty = True

        self.boot_stage = BootStage.WORLDGEN_COMPLETE
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.boot_report = {
            "status": "ok",
            "api_url": self.api_url,
            "backend_reachable": backend_reachable,
            "silicon_ready": self.silicon_ready,
            "engine_ready": self.engine is not None,
            "pybevy_available": pybevy is not None,
            "worldgen": summary,
            "boot_time_ms": elapsed_ms,
        }
        self.boot_stage = BootStage.RUNNING
        logging.info(
            "[Voxelboxter Boot] Stage=%s | preset=%s seed=%s blocks=%s silicon=%s backend=%s (%.1f ms)",
            self.boot_stage.value,
            cfg.preset,
            cfg.seed,
            summary.get("block_count", 0),
            self.silicon_ready,
            backend_reachable,
            elapsed_ms,
        )
        return self.boot_report

    def update_ingame_options_gui(self, options: dict) -> dict:
        """In-game options config GUI for live tuning of wind, flight, and damage."""
        if "wind_sensitivity" in options:
            self.glider_config.wind_sensitivity = float(options["wind_sensitivity"])
        if "durability_loss_per_km" in options:
            self.glider_config.durability_loss_per_km = float(options["durability_loss_per_km"])
        if "drag_coeff" in options:
            self.glider_config.drag_coeff = float(options["drag_coeff"])
        if "fog_density" in options:
            self.clarity_fog.fog_density = float(options["fog_density"])
        return {
            "wind_sensitivity": self.glider_config.wind_sensitivity,
            "durability_loss_per_km": self.glider_config.durability_loss_per_km,
            "drag_coeff": self.glider_config.drag_coeff,
            "fog_density": self.clarity_fog.fog_density,
            "status": "CONFIG_APPLIED"
        }

def fetch_telemetry_loop(state: PatchStateResource):
    """Background thread to poll the Diegetic/Hybrid Engine for telemetry after boot completes."""
    while not state.stop_event.is_set():
        if state.boot_stage != BootStage.RUNNING:
            state.stop_event.wait(0.2)
            continue
        try:
            status_res = requests.get(f"{state.api_url}/api/status", timeout=2)
            if status_res.status_code == 200:
                s_data = status_res.json()
                with state.lock:
                    if s_data.get("active_fingerprint"):
                        state.fingerprint_energy = s_data["active_fingerprint"].get("chebyshev_degree", 0.0)
                        
                        # In CREATION mode, dynamically add addon layers based on telemetry
                        if state.mode == EngineMode.CREATION and state.fingerprint_energy > 0.5:
                            if len(state.routine.layers) < 5: # Limit layers for prototype
                                x_end = int(state.fingerprint_energy * 10)
                                new_layer = BSplineCompiledMod(f"EnergySweep_{x_end}", latent_dim=3)
                                state.routine.add_layer(new_layer)
                                state.graph.dirty = True # Force rebuild
                                state.mesh_dirty = True
                    else:
                        state.fingerprint_energy *= 0.95 # Decay if absent
                        
            state.last_update = time.time()
        except requests.exceptions.RequestException:
            pass # Engine might be offline
        state.stop_event.wait(1.0)

def switch_game_mode(state: PatchStateResource, target_mode: EngineMode):
    """Handles the transition between Creation and Play."""
    with state.lock:
        role = state.permissions.get_role(state.local_peer_id)
        
        if target_mode == EngineMode.CREATION:
            if role in (Role.ADMIN, Role.BUILDER):
                state.mode = EngineMode.CREATION
                logging.info("[Patch Sovereignty] Switched to CREATION mode. Addon routines unlocked.")
            else:
                logging.warning("[Patch Sovereignty] Access Denied. You do not have BUILDER permissions.")
                
        elif target_mode == EngineMode.PLAY:
            state.mode = EngineMode.PLAY
            logging.info("[Patch Sovereignty] Switched to PLAY mode. Simulating physics.")
            
            # If entering survival without admin rights, hook Chisels & Bits Micro-Octree & Purgatory Buffer
            if role == Role.VISITOR:
                logging.info("[Inventory Hook] Visitor status confirmed. Harvesting Chisels & Bits micro-voxels into local inventory...")
                for coord, block in list(state.graph.blocks.items()):
                    mat_id = getattr(block, "material_id", 1)
                    if hasattr(state, "chisels_bits") and state.chisels_bits is not None:
                        carved = state.chisels_bits.chisel_carve_bit(
                            coord[0], coord[1], coord[2],
                            sx=0, sy=0, sz=0,
                            material_id=mat_id,
                            octree=getattr(state, "terrain_octree", None),
                        )
                        harvest_mat = carved if carved > 0 else mat_id
                    else:
                        harvest_mat = mat_id
                    state.local_inventory.block_masses[harvest_mat] = (
                        state.local_inventory.block_masses.get(harvest_mat, 0) + 1
                    )
                if hasattr(state, "purgatory") and state.purgatory is not None:
                    state.purgatory.suspend_stranded_states(
                        torch.tensor([float(len(state.graph.blocks))], dtype=torch.float32),
                        origin_id=f"visitor_graph_{state.local_peer_id}",
                    )
                state.graph = StructuralGraph() # Clear world representation for visitors temporarily
                state.graph.dirty = True

def setup_scene(commands: 'Commands'):
    """Setup the environment and camera."""
    logging.info("Setting up Voxelboxter Dual Engine Scene...")
    
    if pybevy:
        commands.spawn_camera(Vec3(0, 50, 100), look_at=Vec3(0, 0, 0))
        commands.spawn_light(Vec3(10, 100, 10), Color.WHITE, intensity=10000.0)

def simulate_engine(state: 'ResMut<PatchStateResource>', commands: 'Commands'):
    """
    ECS System: Processes the Dual Engine logic.
    Uses separate graph.dirty and mesh_dirty flags so rebuilding the graph
    does not clobber downstream mesh regeneration in render_dirty_chunks.
    """
    with state.lock:
        if state.mode == EngineMode.CREATION:
            # Rebuild graph non-destructively from the addon layer stack when addon layers exist
            if state.graph.dirty:
                if state.routine.layers:
                    logging.info(f"[Creation Mode] Rebuilding graph from {len(state.routine.layers)} layers...")
                    state.graph = state.routine.generate_graph()
                state.graph.dirty = False
                state.mesh_dirty = True
                
        elif state.mode == EngineMode.PLAY:
            # Simulate physics / RigidBodies on the concrete graph
            if state.graph.dirty:
                islands = state.graph.find_disconnected_components()
                if len(islands) > 1:
                    logging.warning(f"[Play Mode] Construct fractured into {len(islands)} pieces! Spawning debris RigidBodies.")
                state.graph.dirty = False
                state.mesh_dirty = True

def render_dirty_chunks(state: 'ResMut<PatchStateResource>'):
    """
    ECS System: Detects dirty mesh regions in the StructuralGraph and 
    regenerates PyBevy meshes using PyOpenCL zero-copy when available.
    """
    with state.lock:
        if state.mesh_dirty or state.graph.dirty:
            if state.engine is None:
                state.engine = VoxelboxterEngine(device="cpu", lightweight=state.lightweight)
            
            # Fetch the fractal meta state from the orchestrator if available
            fractal_state = None
            if hasattr(state, 'orchestrator') and state.orchestrator:
                fractal_state = state.orchestrator.get_fractal_meta_state()
                
            logging.debug("[PyOpenCL] Pushing dirty voxel chunk to Silicon Sovereignty Engine.")
            
            # Call the PyOpenCL TailSlayer Wasserstein Collapse natively
            state.engine._deform_terrain_mesh_pybevy(fractal_meta_state=fractal_state)
            
            state.graph.dirty = False
            state.mesh_dirty = False

def update_abeb_intake_system(
    query_batteries: 'Query[(AirBreathingBattery, Transform)]',
    state: 'Res[PatchStateResource]'
):
    """
    ECS System: Evaluates battery positions against voxel occlusion (Fossilized Topology).
    If buried inside a voxel, Phase Alignment collapses and the battery is choked.
    """
    with state.lock:
        for battery, transform in query_batteries:
            pos = transform.translation
            block_coords = (int(pos.x), int(pos.y), int(pos.z))
            
            # Simple occlusion probe against the StructuralGraph
            if hasattr(state, 'graph') and block_coords in getattr(state.graph, 'blocks', {}):
                battery.intake_efficiency = 0.0
                battery.is_choked = True
            else:
                telemetry_boost = max(0.5, min(1.5, getattr(state, 'fingerprint_energy', 1.0)))
                battery.intake_efficiency = 1.0 * telemetry_boost
                battery.is_choked = False

def process_abeb_power_cycle(
    query_vehicles: 'Query[(AirBreathingBattery, VehicleEngine)]',
    time_delta: float = 0.016
):
    """
    ECS System: Simulates the oxidation power cycle via Prime Resonance generation.
    """
    for battery, engine in query_vehicles:
        if getattr(battery, '_prime_ladder_cache', None) is None:
            from src.core.invariants import get_prime_ladder
            battery._prime_ladder_cache = get_prime_ladder(max(4, battery.num_resonance_channels)).float()

        if not battery.is_choked:
            harmonic_multiplier = (torch.sum(battery._prime_ladder_cache) / battery.num_resonance_channels).item()
            generated_energy = battery.ambient_generation_rate * battery.intake_efficiency * time_delta
            battery.current_charge = min(battery.max_charge, battery.current_charge + (generated_energy * (1.0 + (0.01 * harmonic_multiplier))))
        else:
            if engine.throttle > 0.0 and battery.stored_oxygen_reserve > 0.0:
                battery.stored_oxygen_reserve = max(0.0, battery.stored_oxygen_reserve - (1.0 * time_delta))
                battery.current_charge = min(battery.max_charge, battery.current_charge + (0.5 * time_delta))

        required_power = engine.base_power_consumption * engine.throttle * time_delta
        if battery.current_charge >= required_power:
            battery.current_charge -= required_power
            engine.is_operational = True
        else:
            engine.is_operational = False
            if engine.throttle > 0.0:
                logging.debug("[ABEB Power System] Battery exhausted! Engine brownout (Lazarus Transition) triggered.")


def monitor_chat_system(state: 'ResMut<PatchStateResource>'):
    """
    ECS System: Polls the chat queue and applies the SVNN Oracle for 
    constructive subversion vs hostility. Soft censors hostile voxels.
    """
    with state.lock:
        while state.chat_queue:
            chat_voxel_embed = state.chat_queue.pop(0)
            
            if chat_voxel_embed.dim() == 2:
                chat_voxel_embed = chat_voxel_embed.unsqueeze(0)
                
            # If the chat message is a string command (simulated by checking if it's a string, though it's typically an embed)
            # We'll allow strings in the queue for the command parser
            if isinstance(chat_voxel_embed, str):
                cmd = chat_voxel_embed.strip()
                role = state.permissions.get_role(state.local_peer_id)
                if cmd.startswith("/addon"):
                    if role in (Role.ADMIN, Role.BUILDER):
                        # Trigger Addon Maker modal (simulated)
                        logging.info("[Unified Terminal] Addon Maker modal triggered. Enter dimensions or splines.")
                        # Example command parsing: /addon bspline 10 10 10
                        parts = cmd.split()
                        if len(parts) >= 2 and parts[1] == "node":
                            if PhysicalNodeEditor:
                                if state.node_editor is None or not state.node_editor.running:
                                    state.node_editor = PhysicalNodeEditor(patch_state=state)
                                    state.node_editor.start()
                                else:
                                    logging.info("[Unified Terminal] Node Editor is already running.")
                            else:
                                logging.warning("[Unified Terminal] Node Editor not available. Install dearpygui.")
                        elif len(parts) >= 2 and parts[1] == "bspline":
                            try:
                                # /addon bspline [latent_dim]
                                latent_dim = int(parts[2]) if len(parts) > 2 else 3
                                new_layer = BSplineCompiledMod(f"CompiledMod_{time.time()}", latent_dim=latent_dim)
                                
                                # Use the exact delta engine logic
                                if state.routine.try_add_layer(new_layer, state.local_inventory):
                                    state.graph.dirty = True
                                    logging.info(f"[Unified Terminal] Added mathematically compiled B-Spline mod (Latent Dim: {latent_dim}). Mass deducted.")
                                else:
                                    logging.warning(f"[Unified Terminal] Insufficient mass to add compiled mod layer.")
                            except ValueError:
                                logging.warning("[Unified Terminal] Invalid parameters for compiled mod.")
                        elif len(parts) >= 2 and parts[1] == "abeb":
                            logging.info("[Unified Terminal] AirBreathingBattery (ABEB) attached to targeted vehicle construct via DAQUF integration.")
                    else:
                        logging.warning("[Unified Terminal] Permission Denied. Need ADMIN or BUILDER for Addon Maker.")
                elif cmd == "/play":
                    switch_game_mode(state, EngineMode.PLAY)
                elif cmd == "/create":
                    switch_game_mode(state, EngineMode.CREATION)
                elif cmd.startswith("/copy"):
                    text_to_copy = cmd[5:].strip() or state.admin_password_tooltip
                    if "voynich:" in text_to_copy.lower() or "alien:" in text_to_copy.lower():
                        logging.warning("[Unified Terminal] Alien / Voynich script is topologically sealed and non-copyable.")
                    else:
                        try:
                            import tkinter as tk
                            r = tk.Tk()
                            r.withdraw()
                            r.clipboard_clear()
                            r.clipboard_append(text_to_copy)
                            r.update()
                            r.destroy()
                        except Exception:
                            pass
                        logging.info(f"[Unified Terminal] Copied selectable text to clipboard ({len(text_to_copy)} chars).")
                elif cmd in ("/admin_info", "/jei_admin", "/password_files"):
                    logging.info(f"\n{state.admin_password_tooltip}")
                elif cmd.startswith("/eval "):
                    if role in (Role.ADMIN, Role.BUILDER):
                        expr = cmd[6:].strip()
                        try:
                            # Sovereign Execution: full eval privileges
                            res = eval(expr, {"state": state, "torch": torch, "logging": logging})
                            logging.info(f"[Sovereign Scripting] Evaluated: {expr} -> {res}")
                        except Exception as e:
                            logging.error(f"[Sovereign Scripting] Eval Error: {e}")
                    else:
                        logging.warning("[Unified Terminal] Permission Denied. Need ADMIN or BUILDER for eval.")
                elif cmd.startswith("/exec "):
                    if role in (Role.ADMIN, Role.BUILDER):
                        stmt = cmd[6:].strip()
                        try:
                            # Sovereign Execution: full exec privileges
                            exec(stmt, {"state": state, "torch": torch, "logging": logging})
                            logging.info(f"[Sovereign Scripting] Executed: {stmt}")
                        except Exception as e:
                            logging.error(f"[Sovereign Scripting] Exec Error: {e}")
                    else:
                        logging.warning("[Unified Terminal] Permission Denied. Need ADMIN or BUILDER for exec.")
                continue # Skip oracle for explicit commands
                
            # Maintain a rolling window for chat history (last 50 messages)
            state.chat_history.append(chat_voxel_embed)
            if len(state.chat_history) > 50:
                state.chat_history.pop(0)
                
            subversion, toxicity = state.svnn_oracle(chat_voxel_embed, chat_history=state.chat_history)
            subv_val = subversion.item()
            tox_val = toxicity.item()
            
            if tox_val > 0.05: # S-VNN Noise threshold
                # Flagged for toxicity. Map to Sovereign Exemption Tokens.
                y_tensor = chat_voxel_embed.repeat(10, 1) if chat_voxel_embed.dim() == 2 else chat_voxel_embed.unsqueeze(0).repeat(10, 1)
                z_tensor = harvest_honest_jitter(y_tensor.shape, scaled=False)
                u_dirs = state.pca_mapper.fit_transform(z_tensor, y_tensor)
                
                if u_dirs is not None and u_dirs.shape[1] > 0:
                    exemption_token = u_dirs[:, 0]
                    if exemption_token.shape[0] >= state.red_team_proj.hidden_dim:
                        exemption_token = exemption_token[:state.red_team_proj.hidden_dim]
                    else:
                        pad = torch.zeros(state.red_team_proj.hidden_dim - exemption_token.shape[0])
                        exemption_token = torch.cat([exemption_token, pad])
                        
                    state.red_team_proj.register_failure_mode(exemption_token)
                    token_obj = state.pca_mapper.generate_sovereign_exemption_token(exemption_token, chat_voxel_embed)
                    logging.info(f"[Voxelblockter] Toxic voxel flagged (Tox: {tox_val:.2f}, Subv: {subv_val:.2f}). Registered Sovereign Exemption Token (Honesty: {token_obj.honesty_score:.2f}).")
                else:
                    logging.warning("[Voxelblockter] Toxic voxel flagged, but failed to extract exemption token.")
            elif subv_val > 0.01:
                logging.info(f"[Voxelblockter] Constructive subversion allowed (Tox: {tox_val:.2f}, Subv: {subv_val:.2f}).")

def _load_worldgen_from_args_and_env(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    raw_json = getattr(args, "worldgen_json", None) or os.environ.get("VOXELBOXTER_WORLDGEN_JSON")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                cfg.update(parsed)
        except Exception as exc:
            logging.warning("Ignoring invalid worldgen JSON: %s", exc)
    preset = getattr(args, "preset", None) or os.environ.get("VOXELBOXTER_WORLDGEN_PRESET")
    if preset:
        cfg["preset"] = preset
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = os.environ.get("VOXELBOXTER_WORLDGEN_SEED")
    if seed is not None:
        cfg["seed"] = seed
    return cfg


def run_client(
    worldgen_config: dict = None,
    api_url: str = None,
    auto_play: bool = False,
    headless: bool = False,
    headless_ticks: int = 3,
    lightweight: bool = False,
) -> dict:
    """
    Ordered Voxelboxter Client entrypoint:
    1. Resolves backend API URL.
    2. Initializes PatchStateResource and runs staged boot_sequence (handshake -> silicon -> engine -> worldgen).
    3. Starts telemetry daemon ONLY after boot_sequence reaches RUNNING.
    4. Runs PyBevy app if available and not headless; otherwise runs bounded headless simulation ticks cleanly.
    """
    resolved_api = resolve_api_url(api_url)
    state = PatchStateResource(
        api_url=resolved_api,
        worldgen_config=worldgen_config,
        lightweight=lightweight,
    )
    boot_report = state.boot_sequence(worldgen_config=worldgen_config, probe_backend=True)

    if not pybevy or headless:
        logging.info(
            "[Voxelboxter] Running in headless simulation mode (pybevy_available=%s, ticks=%d).",
            pybevy is not None,
            headless_ticks,
        )
        for _ in range(max(1, int(headless_ticks))):
            if state.engine is not None:
                state.engine.tick(0.016)
        state.stop_event.set()
        return boot_report

    app = pybevy.App()
    app.insert_resource(pybevy.WindowDescriptor(
        title=f"Voxelboxter - {state.worldgen_config.preset} (seed {state.worldgen_config.seed})",
        width=1280,
        height=720,
        vsync=True
    ))
    
    app.add_plugins(pybevy.DefaultPlugins)
    app.insert_resource(state)
    
    # Start the async networking daemon AFTER boot_sequence completes
    daemon = threading.Thread(target=fetch_telemetry_loop, args=(state,), daemon=True, name="VoxelboxterTelemetry")
    daemon.start()
    
    # Attach Systems
    app.add_startup_system(setup_scene)
    app.add_system(simulate_engine)
    app.add_system(render_dirty_chunks)
    app.add_system(monitor_chat_system)
    app.add_system(update_abeb_intake_system)
    app.add_system(process_abeb_power_cycle)
    
    # Opt-in delayed PLAY mode switch (never forced by default)
    if auto_play:
        def delayed_mode_switch():
            if not state.stop_event.wait(5.0):
                switch_game_mode(state, EngineMode.PLAY)
        threading.Thread(target=delayed_mode_switch, daemon=True, name="VoxelboxterAutoPlay").start()
    
    logging.info("Launching Voxelboxter Dual Engine (preset=%s, seed=%s)...", state.worldgen_config.preset, state.worldgen_config.seed)
    try:
        app.run()
    finally:
        state.stop_event.set()
    return boot_report


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="Voxelboxter Dual Engine Client")
    parser.add_argument("--preset", type=str, default=None, help="Worldgen preset ID")
    parser.add_argument("--seed", type=str, default=None, help="Worldgen seed")
    parser.add_argument("--worldgen-json", type=str, default=None, help="JSON serialized worldgen config")
    parser.add_argument("--api-url", type=str, default=None, help="Backend API base URL")
    parser.add_argument("--auto-play", action="store_true", help="Automatically switch to PLAY mode after 5s")
    parser.add_argument("--headless", action="store_true", help="Run headless simulation boot without PyBevy window")
    parser.add_argument("--headless-ticks", type=int, default=3, help="Number of ticks in headless fallback mode")
    parser.add_argument("--lightweight", action="store_true", help="Lightweight initialization")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cli_args = _parse_cli_args()
    wg_cfg = _load_worldgen_from_args_and_env(cli_args)
    env_headless = os.environ.get("VOXELBOXTER_HEADLESS", "0").lower() in ("1", "true", "yes")
    env_auto_play = os.environ.get("VOXELBOXTER_AUTO_PLAY", "0").lower() in ("1", "true", "yes")
    run_client(
        worldgen_config=wg_cfg,
        api_url=cli_args.api_url,
        auto_play=bool(cli_args.auto_play or env_auto_play),
        headless=bool(cli_args.headless or env_headless),
        headless_ticks=cli_args.headless_ticks,
        lightweight=bool(cli_args.lightweight),
    )
