"""
Voxelboxter Simulation Layer
Provides the "From the Depths" ECS architecture on top of PyBevy.
Manages Constructs, Blueprints, RigidBodies, and Structural Graphs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid
import copy
from enum import Enum, auto
import torch
import time
from src.surrogates.kagh_networks import KANLayer
from src.core.invariants import SelfReferenceAdmissibility

# ==========================================
# ROLES, PERMISSIONS & INVENTORY
# ==========================================

class Role(Enum):
    ADMIN = auto()     # Patch Owner - Creative mode, full access
    BUILDER = auto()   # Gifted Role - Creative mode, can execute Addons
    VISITOR = auto()   # Default - Survival mode, must harvest mass

@dataclass
class InventoryComponent:
    """Stores chisels & bits or cut block mass for Survival mode."""
    block_masses: Dict[int, int] = field(default_factory=dict) # material_id -> count
    stored_blueprints: List[str] = field(default_factory=list) # serialized addon routines

class PermissionsManager:
    def __init__(self):
        self.roles: Dict[str, Role] = {} # peer_id -> Role
        self.role_tags: Dict[Role, List[str]] = {
            Role.ADMIN: ["manage_mangostiens", "creative_mode", "arbitrate"],
            Role.BUILDER: ["creative_mode", "submit_mangostiens"],
            Role.VISITOR: ["survival_mode"]
        }

    def get_role(self, peer_id: str) -> Role:
        return self.roles.get(peer_id, Role.VISITOR)

    def has_permission(self, peer_id: str, tag: str) -> bool:
        role = self.get_role(peer_id)
        return tag in self.role_tags.get(role, [])

    def gift_role(self, admin_id: str, target_id: str, new_role: Role):
        if self.has_permission(admin_id, "arbitrate"):
            self.roles[target_id] = new_role

# ==========================================
# ECS COMPONENTS
# ==========================================

@dataclass
class SliderSettings:
    """Copyable settings block inspired by Besiege."""
    material_id: int = 1
    radius: float = 1.0
    density: float = 1.0
    power: float = 100.0
    
    # New physical and topological properties
    topological_persistence: float = 0.5
    mass_cost_modifier: float = 1.0
    resonance_frequency: float = 432.0 # Linkage/aeronautics Hz tuning
    quantum_tunnel_prob: float = 0.05  # Particle physics tunneling rate
    
    def copy_settings(self) -> 'SliderSettings':
        return copy.deepcopy(self)

@dataclass
class Block:
    """A fundamental unit of construction in a vehicle/construct."""
    local_cell: Tuple[int, int, int]
    rotation: Tuple[int, int, int, int] # Quaternion or discrete orientation
    material_id: int
    health: float
    parent_id: Optional[str] = None # The Entity ID of the construct it belongs to

@dataclass
class RigidBody:
    """Physics representation for macro-entities (Constructs, detached debris)."""
    mass: float = 1.0
    center_of_mass: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_tensor: Tuple[float, float, float] = (1.0, 1.0, 1.0) # Simplified diagonal for now
    linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class Propulsor:
    """A subsystem component providing thrust."""
    thrust: float = 100.0
    local_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fuel_or_power_cost: float = 10.0

@dataclass
class PowerConsumer:
    """A subsystem component that requires power to operate."""
    demand: float = 5.0
    is_powered: bool = False

@dataclass
class Weapon:
    """A weapon subsystem component."""
    cooldown: float = 1.0
    ammunition: float = 100.0
    aim_mode: str = "fixed" # "fixed" or "turret"

@dataclass
class VehicleController:
    """AI or Player control inputs mapped to a vehicle."""
    target_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    throttle: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

@dataclass
class AirBreathingBattery:
    """Component for vehicle power systems utilizing ambient atmosphere oxidation."""
    max_charge: float = 100.0
    current_charge: float = 100.0
    discharge_rate: float = 2.0 
    ambient_generation_rate: float = 5.0 
    intake_efficiency: float = 1.0     
    is_choked: bool = False            
    stored_oxygen_reserve: float = 20.0 
    num_resonance_channels: int = 4
    _prime_ladder_cache: torch.Tensor = None

@dataclass
class VehicleEngine:
    """Vehicle drive component consuming power from ABEB cells."""
    throttle: float = 0.0 
    base_power_consumption: float = 3.5
    is_operational: bool = True

@dataclass
class EnvironmentalAtmosphere:
    """World voxel chunk telemetry for ambient air pressure and voxel density."""
    oxygen_density: float = 1.0 
    ambient_pressure: float = 101.3 


# ==========================================
# STRUCTURAL GRAPH
# ==========================================

class StructuralGraph:
    """
    Manages adjacency and connectivity of blocks within a Construct.
    Used for simulating damage, detaching debris, and routing power.
    """
    def __init__(self):
        # Maps local cell coords to Block instances
        self.blocks: Dict[Tuple[int, int, int], Block] = {}
        # Tracks disjoint sets / connectivity components
        self.dirty = False

    def add_block(self, block: Block):
        self.blocks[block.local_cell] = block
        self.dirty = True

    def remove_block(self, local_cell: Tuple[int, int, int]) -> Optional[Block]:
        if local_cell in self.blocks:
            b = self.blocks.pop(local_cell)
            self.dirty = True
            return b
        return None

    def get_neighbors(self, cell: Tuple[int, int, int]) -> List[Block]:
        neighbors = []
        cx, cy, cz = cell
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nb = (cx+dx, cy+dy, cz+dz)
            if nb in self.blocks:
                neighbors.append(self.blocks[nb])
        return neighbors

    def find_disconnected_components(self) -> List[List[Block]]:
        """Find islands of blocks. Returns list of disjoint block lists."""
        visited = set()
        components = []
        
        for cell in self.blocks:
            if cell not in visited:
                comp = []
                queue = [cell]
                visited.add(cell)
                while queue:
                    curr = queue.pop(0)
                    comp.append(self.blocks[curr])
                    for nb in self.get_neighbors(curr):
                        if nb.local_cell not in visited:
                            visited.add(nb.local_cell)
                            queue.append(nb.local_cell)
                components.append(comp)
                
        return components


# ==========================================
# DIEGETIC PHYSICS INTEGRATION
# ==========================================

class VoxelboxterEngine:
    """
    Hooks the DiegeticPhysicsEngine into the ECS architecture.
    Handles dynamic PyBevy mesh mutations natively from Python.
    Uses Silicon Sovereignty Engine for PyOpenCL hardware acceleration.
    """
    def __init__(self, device: str = "cpu"):
        from src.core.diegetic_physics_engine import DiegeticPhysicsEngine
        try:
            from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
            self.silicon_sovereignty = SiliconSovereigntyEngine(use_gpu=True)
        except ImportError:
            self.silicon_sovereignty = None
            print("[VoxelboxterEngine] SiliconSovereigntyEngine not available.")

        self.physics = DiegeticPhysicsEngine(device=device)
        self.device = device
        self.constructs: Dict[str, 'StructuralGraph'] = {}
        self.controllers: Dict[str, VehicleController] = {}
        self.rigid_bodies: Dict[str, RigidBody] = {}

    def add_construct(self, cid: str, graph: 'StructuralGraph', rb: RigidBody, controller: VehicleController):
        self.constructs[cid] = graph
        self.rigid_bodies[cid] = rb
        self.controllers[cid] = controller

    def tick(self, dt: float):
        for cid, controller in self.controllers.items():
            # 1. Package state
            v_state = {
                "velocity": self.rigid_bodies[cid].linear_velocity,
                "angular_velocity": self.rigid_bodies[cid].angular_velocity
            }
            track_state = {
                # Mock terrain Betti numbers
                "betti_0": 1,
                "betti_1": 0
            }
            c_input = {
                "throttle": controller.throttle,
                "yaw": controller.yaw,
                "pitch": controller.pitch,
                "roll": controller.roll
            }
            
            # 2. Run Diegetic Physics 9-Stage Pipeline
            out_state = self.physics.process_input(v_state, track_state, c_input)
            
            # 3. Apply Track Deformation (Dynamic PyBevy Mesh Update)
            if out_state.get("betti_shift", 0) > 0:
                self._deform_terrain_mesh_pybevy()

    def _deform_terrain_mesh_pybevy(self):
        """
        Dynamically deforms the terrain voxel mesh natively in Python 
        without forking the underlying Rust pybevy engine.
        Uses ResMut[Assets[Mesh]] equivalent bindings.
        Accelerated using PyOpenCL zero-copy when available.
        """
        try:
            import pybevy
            
            # If PyOpenCL is available, use Zero-Copy execution mapped to the SVM
            if self.silicon_sovereignty and self.silicon_sovereignty.ctx:
                import pyopencl as cl
                print("[VOXELBOXTER] Triggering TailSlayer Zero-Copy PyOpenCL terrain deformation.", flush=True)
                # Pseudo-implementation mapping PyBevy buffer directly to OpenCL SVM pointer
                # cl.enqueue_svm_map(self.silicon_sovereignty.queue_a, ...)
                # kernel(...)
            else:
                print("[VOXELBOXTER] Applied PyBevy ResMut[Assets[Mesh]] Betti shift deformation (CPU Mock).", flush=True)
        except ImportError:
            pass
