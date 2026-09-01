"""
Voxelboxter Game Client - Dual Creation/Play Engine
Built on PyBevy (Python Real-Time Engine)

This Freenet client enforces Patch Sovereignty. It separates "Creation Mode" 
(additive/subtractive layered B-Splines with Admin roles) from "Play Mode" 
(Survival logic, RigidBody physics, mass calculation). 

Author: William Matthew Bryant
Date: August 2026
"""

import sys
import time
import logging
import threading
import requests
from enum import Enum, auto

try:
    import pybevy
    from pybevy import App, DefaultPlugins, Commands, Res, ResMut, Query, Transform, Vec3, Color
except ImportError:
    logging.warning("PyBevy is not installed or failed to load. The client requires Python 3.12+.")
    pybevy = None

import torch
from src.safety.subversive_oracle import ResonantSVNNOracle
from src.core.orchestrator import UniversalOrchestrator
from src.safety.red_teaming import RedTeamProjection
from src.core.jspace_pca_mapper import JSpacePCAMapper

from src.ui.voxelboxter_simulation import (
    StructuralGraph, RigidBody, AddonRoutine, BSplineSweepLayer, 
    BooleanXORLayer, Role, PermissionsManager, InventoryComponent
)

API_URL = "http://localhost:1337"

class EngineMode(Enum):
    CREATION = auto()  # Admin/Creative mode: edit addon routines freely
    PLAY = auto()      # Survival/Physics mode: lock routines, apply physics

class PatchStateResource:
    """ECS Resource holding the simulation graph, roles, and reasoner telemetry."""
    def __init__(self):
        self.graph = StructuralGraph()
        self.routine = AddonRoutine() # The layered blueprint
        
        self.mode = EngineMode.CREATION
        self.permissions = PermissionsManager()
        self.local_inventory = InventoryComponent()
        self.local_peer_id = "ADMIN_OWNER_1337"
        
        # Self-gift admin role since we host the patch
        self.permissions.roles[self.local_peer_id] = Role.ADMIN
        
        # Reasoner Telemetry
        self.fingerprint_energy = 0.0
        self.last_update = time.time()
        self.lock = threading.Lock()
        
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

def fetch_telemetry_loop(state: PatchStateResource):
    """Background thread to poll the Diegetic Engine for Freenet telemetry."""
    while True:
        try:
            status_res = requests.get(f"{API_URL}/api/status", timeout=2)
            if status_res.status_code == 200:
                s_data = status_res.json()
                with state.lock:
                    if s_data.get("active_fingerprint"):
                        state.fingerprint_energy = s_data["active_fingerprint"].get("chebyshev_degree", 0.0)
                        
                        # In CREATION mode, dynamically add addon layers based on telemetry
                        if state.mode == EngineMode.CREATION and state.fingerprint_energy > 0.5:
                            if len(state.routine.layers) < 5: # Limit layers for prototype
                                x_end = int(state.fingerprint_energy * 10)
                                new_layer = BSplineSweepLayer(f"EnergySweep_{x_end}", (0,0,0), (x_end, 5, x_end))
                                state.routine.add_layer(new_layer)
                                state.graph.dirty = True # Force rebuild
                    else:
                        state.fingerprint_energy *= 0.95 # Decay if absent
                        
            state.last_update = time.time()
        except requests.exceptions.RequestException:
            pass # Engine might be offline
        time.sleep(1.0)

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
            
            # If entering survival without admin rights, hook Chisels & Bits
            if role == Role.VISITOR:
                logging.info("[Inventory Hook] Visitor status confirmed. Storing cut mass into local inventory...")
                # Pseudo-logic: calculate mass from routine and store to inventory
                cut_mass = len(state.graph.blocks)
                state.local_inventory.block_masses[1] = cut_mass
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
    """
    with state.lock:
        if state.mode == EngineMode.CREATION:
            # Rebuild graph non-destructively from the addon layer stack
            if state.graph.dirty:
                logging.info(f"[Creation Mode] Rebuilding graph from {len(state.routine.layers)} layers...")
                state.graph = state.routine.generate_graph()
                state.graph.dirty = False
                
        elif state.mode == EngineMode.PLAY:
            # Simulate physics / RigidBodies on the concrete graph
            # Damage simulation (fracturing)
            if state.graph.dirty:
                islands = state.graph.find_disconnected_components()
                if len(islands) > 1:
                    logging.warning(f"[Play Mode] Construct fractured into {len(islands)} pieces! Spawning debris RigidBodies.")
                state.graph.dirty = False

def render_dirty_chunks():
    """
    ECS System: Detects dirty regions in the StructuralGraph and 
    regenerates PyBevy meshes only for modified regions.
    """
    pass

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
                
            # Maintain a rolling window for chat history (last 50 messages)
            state.chat_history.append(chat_voxel_embed)
            if len(state.chat_history) > 50:
                state.chat_history.pop(0)
                
            subversion, toxicity = state.svnn_oracle(chat_voxel_embed, chat_history=state.chat_history)
            subv_val = subversion.item()
            tox_val = toxicity.item()
            
            if tox_val > 0.05: # S-VNN Noise threshold
                # Flagged for toxicity. Map to Sovereign Exemption Tokens.
                y_tensor = torch.randn(10, 32)
                z_tensor = torch.randn(10, 32)
                u_dirs = state.pca_mapper.fit_transform(z_tensor, y_tensor)
                
                if u_dirs is not None and u_dirs.shape[1] > 0:
                    exemption_token = u_dirs[:, 0]
                    if exemption_token.shape[0] >= state.red_team_proj.hidden_dim:
                        exemption_token = exemption_token[:state.red_team_proj.hidden_dim]
                    else:
                        pad = torch.zeros(state.red_team_proj.hidden_dim - exemption_token.shape[0])
                        exemption_token = torch.cat([exemption_token, pad])
                        
                    state.red_team_proj.register_failure_mode(exemption_token)
                    logging.info(f"[Voxelblockter] Toxic voxel flagged (Tox: {tox_val:.2f}, Subv: {subv_val:.2f}). Registered Sovereign Exemption Token.")
                else:
                    logging.warning("[Voxelblockter] Toxic voxel flagged, but failed to extract exemption token.")
            elif subv_val > 0.01:
                logging.info(f"[Voxelblockter] Constructive subversion allowed (Tox: {tox_val:.2f}, Subv: {subv_val:.2f}).")

def run_client():
    if not pybevy:
        return
        
    app = pybevy.App()
    app.insert_resource(pybevy.WindowDescriptor(
        title="Voxelboxter - Dual Engine Sandbox",
        width=1280,
        height=720,
        vsync=True
    ))
    
    app.add_plugins(pybevy.DefaultPlugins)
    
    # Initialize our ECS Resource
    state = PatchStateResource()
    app.insert_resource(state)
    
    # Start the async networking daemon to sync with DiegeticPhysicsEngine
    daemon = threading.Thread(target=fetch_telemetry_loop, args=(state,), daemon=True)
    daemon.start()
    
    # Attach Systems
    app.add_startup_system(setup_scene)
    app.add_system(simulate_engine)
    app.add_system(render_dirty_chunks)
    app.add_system(monitor_chat_system)
    
    # Simulate a user mode switch a few seconds in
    def delayed_mode_switch():
        time.sleep(5)
        switch_game_mode(state, EngineMode.PLAY)
    
    threading.Thread(target=delayed_mode_switch, daemon=True).start()
    
    logging.info("Launching Voxelboxter Dual Engine...")
    app.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_client()
