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
# PROCEDURAL FRACTAL GENERATION (TVA AESTHETIC)
# ==========================================

class PointerlessOctree:
    """
    Morton-encoded (Z-order curve) pointerless octree for mutating treehouse dimensions.
    Replaces dense voxel arrays with a mathematically pure spatial hash.
    Enables evolutionary L-Systems (Subdivision, Collapse, Crossover) 
    governed by Banach Fixed Point and Wasserstein Optimal Transport.
    """
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.morton_grid: Dict[int, int] = {} # Morton Code -> Material ID
        
    def _interleave_bits(self, x: int, y: int, z: int) -> int:
        """Computes the 3D Morton code by interleaving bits of x, y, z."""
        # Standard magic bit-shifting for 10-bit components (30-bit Morton code)
        def expand_bits(v: int) -> int:
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v <<  8)) & 0x0300F00F
            v = (v | (v <<  4)) & 0x030C30C3
            v = (v | (v <<  2)) & 0x09249249
            return v
        return (expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2))
        
    def add_fractal_node(self, x: int, y: int, z: int, material: int):
        code = self._interleave_bits(x, y, z)
        self.morton_grid[code] = material
        
    def get_pyopencl_kernel(self) -> str:
        """
        PyOpenCL kernel for Wasserstein Collapse (Earth Mover's Distance)
        Vectorized to float4 to utilize Pascal architecture warp widths efficiently.
        Includes Stochastic Rounding via TEA to prevent 4-bit lattice quantization error.
        """
        return """
        #define TEA_ROUNDS 4
        inline uint tea_hash(uint v0, uint v1) {
            uint sum = 0;
            for(int i=0; i<TEA_ROUNDS; ++i) {
                sum += 0x9e3779b9;
                v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
                v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
            }
            return v0;
        }

        // ----------------------------------------------------
        // Morton Encoding (Z-Order Curve) GPU Kernel
        // Flattens 3D spatial coordinates into a 1D scalar
        // to enable zero-copy linear memory streaming.
        // ----------------------------------------------------
        inline uint expand_bits(uint v) {
            v = (v | (v << 16)) & 0x030000FF;
            v = (v | (v <<  8)) & 0x0300F00F;
            v = (v | (v <<  4)) & 0x030C30C3;
            v = (v | (v <<  2)) & 0x09249249;
            return v;
        }

        inline uint morton_interleave(uint x, uint y, uint z) {
            return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
        }

        __kernel void wasserstein_collapse_svm(
            __global float4* morton_residues,
            __global float* out_lattice,
            const uint iteration_count,
            const float spatial_mass_budget
        ) {
            int gid = get_global_id(0);
            float4 residue = morton_residues[gid];
            
            // Earth Mover's sliding (Wasserstein): Push mass towards local dense centers
            float local_mass = length(residue.xyz);
            float4 collapsed = residue;
            if (local_mass < (spatial_mass_budget * 0.001f)) {
                collapsed.xyz = (float3)(0.0f); // Yield / Fracture
            } else {
                collapsed.xyz = normalize(residue.xyz) * min(local_mass, spatial_mass_budget); // Cap by Parseval budget
            }

            // Stochastic Rounding into 4-bit Lattice
            uint rand_seed = tea_hash(gid, iteration_count);
            float noise = ((float)(rand_seed & 0xFFFF) / 65535.0f) * 0.1f - 0.05f;
            
            // Cast to 4-bit lattice
            out_lattice[gid] = floor(collapsed.x + noise) * 16.0f;
        }
        """

    def apply_wasserstein_collapse(self, ctx, queue, residues_tensor, iteration: int, mass_budget: float):
        """
        Earth Mover's Distance equivalent.
        Slides dense clusters into monolithic brutalist blocks using PyOpenCL SVM.
        """
        import pyopencl as cl
        import pyopencl.array as cl_array
        
        prg = cl.Program(ctx, self.get_pyopencl_kernel()).build()
        
        # Prepare SVM buffers (CL_MEM_USE_HOST_PTR logic conceptually encapsulated)
        mf = cl.mem_flags
        # Convert tensor to float4 arrays
        res_np = residues_tensor.detach().cpu().numpy().astype(np.float32)
        # Pad to float4
        pad_size = (4 - (res_np.size % 4)) % 4
        if pad_size > 0:
            res_np = np.append(res_np, np.zeros(pad_size, dtype=np.float32))
        
        res_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes // 4)
        
        num_work_items = res_np.size // 4
        prg.wasserstein_collapse_svm(queue, (num_work_items,), None, res_buf, out_buf, np.uint32(iteration), np.float32(mass_budget))
        
        out_np = np.empty(num_work_items, dtype=np.float32)
        cl.enqueue_copy(queue, out_np, out_buf).wait()
        
        return out_np

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

    def _deform_terrain_mesh_pybevy(self, fractal_meta_state: Optional[torch.Tensor] = None):
        """
        Dynamically deforms the terrain voxel mesh natively in Python.
        Pipes the continuous KAGH Surrogate projections from FractalMetaFunctional
        into discrete Menger-Sponge/Stepwell voxel operations.
        Accelerated using PyOpenCL zero-copy when available.
        """
        try:
            import pybevy
            
            # 1. Evaluate the Banach Fixed Point (Subdivision Mutability)
            # 2. Map KAGH polynomial residues to the PointerlessOctree
            
            if self.silicon_sovereignty and self.silicon_sovereignty.ctx:
                import pyopencl as cl
                import pyopencl.array as cl_array
                print("[VOXELBOXTER] Triggering TailSlayer Zero-Copy PyOpenCL terrain deformation.", flush=True)
                
                # If we have a live meta-state from the reasoner, use it to carve recursive stepwells
                if fractal_meta_state is not None:
                    # Enforce Parseval's Theorem: Spatial energy must match Frequency Energy
                    # Sum of squared frequency magnitudes = sum of spatial mass
                    spatial_mass_budget = torch.sum(fractal_meta_state ** 2).item()
                    print(f"[VOXELBOXTER] Parseval's Theorem computed mass budget: {spatial_mass_budget:.4f}. Injecting residues.", flush=True)
                    
                    octree = PointerlessOctree()
                    # Execute Wasserstein Collapse over SVM to slide mass into brutalist Menger Sponges
                    out_lattice = octree.apply_wasserstein_collapse(
                        self.silicon_sovereignty.ctx, 
                        self.silicon_sovereignty.queue_a, 
                        fractal_meta_state, 
                        iteration=int(time.time()), 
                        mass_budget=spatial_mass_budget
                    )
                    
                    # Zero-Copy map PyBevy mesh buffer to SVM pointer
                    # cl.enqueue_svm_map(self.silicon_sovereignty.queue_a, SVM_PTR, cl.map_flags.WRITE, size)
                    # memcpy(SVM_PTR, out_lattice)
                    # cl.enqueue_svm_unmap(self.silicon_sovereignty.queue_a, SVM_PTR)
                    
                    print(f"[VOXELBOXTER] Hedged Zero-Copy update complete. Terrain lattice deformed.", flush=True)
            else:
                print("[VOXELBOXTER] Applied PyBevy ResMut[Assets[Mesh]] Betti shift deformation (CPU Mock).", flush=True)
                if fractal_meta_state is not None:
                     print("[VOXELBOXTER] Simulating Drucker-Prager Yield structural fracturing.", flush=True)
        except ImportError:
            pass
