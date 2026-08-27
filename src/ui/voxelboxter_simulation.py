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

    def get_role(self, peer_id: str) -> Role:
        return self.roles.get(peer_id, Role.VISITOR)

    def gift_role(self, admin_id: str, target_id: str, new_role: Role):
        if self.get_role(admin_id) == Role.ADMIN:
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
# B-SPLINE ADDON / BLUEPRINT LOGIC (DUAL ENGINE)
# ==========================================

class AddonLayer:
    """Base class for Paint.net style additive/subtractive blueprint layers."""
    def __init__(self, name: str):
        self.name = name
        self.settings = SliderSettings()
        self.enabled = True

    def execute(self, graph: StructuralGraph):
        pass

class BSplineSweepLayer(AddonLayer):
    """Generates blocks along a spline (additive)."""
    def __init__(self, name: str, start_pos: Tuple[int, int, int], end_pos: Tuple[int, int, int]):
        super().__init__(name)
        self.start = start_pos
        self.end = end_pos

    def execute(self, graph: StructuralGraph):
        if not self.enabled: return
        
        steps = max(abs(self.end[0]-self.start[0]), abs(self.end[1]-self.start[1]), abs(self.end[2]-self.start[2]))
        if steps == 0: steps = 1
        
        r = int(self.settings.radius)
        for i in range(steps + 1):
            t = i / steps
            cx = int(self.start[0] + (self.end[0] - self.start[0]) * t)
            cy = int(self.start[1] + (self.end[1] - self.start[1]) * t)
            cz = int(self.start[2] + (self.end[2] - self.start[2]) * t)
            
            # Apply radius thickness
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    for dz in range(-r, r+1):
                        b = Block(local_cell=(cx+dx, cy+dy, cz+dz), rotation=(0,0,0,1), 
                                  material_id=self.settings.material_id, health=100.0)
                        graph.add_block(b)

class BooleanXORLayer(AddonLayer):
    """Subtractive XOR cut (carving out engine bays, etc)."""
    def __init__(self, name: str, center: Tuple[int, int, int], dimensions: Tuple[int, int, int]):
        super().__init__(name)
        self.center = center
        self.dims = dimensions

    def execute(self, graph: StructuralGraph):
        if not self.enabled: return
        cx, cy, cz = self.center
        hx, hy, hz = self.dims[0]//2, self.dims[1]//2, self.dims[2]//2
        
        cells_to_remove = []
        for x in range(cx - hx, cx + hx + 1):
            for y in range(cy - hy, cy + hy + 1):
                for z in range(cz - hz, cz + hz + 1):
                    if (x, y, z) in graph.blocks:
                        cells_to_remove.append((x, y, z))
        
        for cell in cells_to_remove:
            graph.remove_block(cell)

class MirrorSymmetryLayer(AddonLayer):
    """Duplicates current graph across an axis."""
    def __init__(self, name: str, axis: str = 'x'):
        super().__init__(name)
        self.axis = axis

    def execute(self, graph: StructuralGraph):
        if not self.enabled: return
        
        new_blocks = []
        for cell, block in graph.blocks.items():
            nx, ny, nz = cell
            if self.axis == 'x': nx = -nx
            elif self.axis == 'y': ny = -ny
            elif self.axis == 'z': nz = -nz
            
            if (nx, ny, nz) not in graph.blocks:
                new_b = Block(local_cell=(nx, ny, nz), rotation=block.rotation, 
                              material_id=block.material_id, health=block.health)
                new_blocks.append(new_b)
                
        for b in new_blocks:
            graph.add_block(b)

class AddonRoutine:
    """Manages the stack of layers (Blueprint)."""
    def __init__(self):
        self.layers: List[AddonLayer] = []
        
    def add_layer(self, layer: AddonLayer):
        self.layers.append(layer)
        
    def generate_graph(self) -> StructuralGraph:
        """Executes the entire layer stack non-destructively."""
        graph = StructuralGraph()
        for layer in self.layers:
            layer.execute(graph)
        return graph
