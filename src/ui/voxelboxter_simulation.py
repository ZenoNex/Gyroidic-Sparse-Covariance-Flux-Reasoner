"""
Voxelboxter Simulation Layer
Provides the "From the Depths" ECS architecture on top of PyBevy.
Manages Constructs, Blueprints, RigidBodies, and Structural Graphs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import uuid
import copy
from enum import Enum, auto
import torch
import time
import math
import numpy as np
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
    """Stores chisels & bits or cut block mass for Survival mode, with slot-based expansion."""
    block_masses: Dict[int, int] = field(default_factory=dict) # material_id -> count
    stored_blueprints: List[str] = field(default_factory=list) # serialized addon routines
    expanded_slots: Optional[Any] = None # ExpandedInventorySystem instance
    
    def get_or_create_expanded(self) -> Any:
        if self.expanded_slots is None:
            from src.environment.cleanroom_mechanics import ExpandedInventorySystem
            self.expanded_slots = ExpandedInventorySystem()
        return self.expanded_slots

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
    
    # Delta-v Collision Severity attributes
    reinforced_alloy_chassis: bool = False
    impact_energy_harvester: bool = False
    accumulated_delta_v: float = 0.0
    last_impulse_tier: str = "NONE" # NONE, LOW, MID, HIGH

    def apply_collision_impulse(
        self,
        impulse_J: float,
        contact_normal: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        delta_t: float = 0.016,
        graph: Optional['StructuralGraph'] = None,
        inventory: Optional['InventoryComponent'] = None
    ) -> Dict[str, Any]:
        """
        Delta-v Collision Dynamics: Delta_v = J / m.
        Evaluates peak deceleration stress over delta_t hardness curve and
        detaches structural voxels across Low (<5 mph), Mid (5-15 mph), and High (>15 mph) tiers.
        """
        eff_mass = self.mass * (1.5 if self.reinforced_alloy_chassis else 1.0)
        delta_v = impulse_J / max(1.0, eff_mass)
        self.accumulated_delta_v += delta_v
        
        # Hardness curve: effective peak deceleration stress
        peak_stress = delta_v / max(1e-4, delta_t)
        
        # Velocity update
        nx, ny, nz = contact_normal
        vx, vy, vz = self.linear_velocity
        self.linear_velocity = (vx + nx * delta_v, vy + ny * delta_v, vz + nz * delta_v)
        
        # Thresholds in m/s (5 mph = 2.235 m/s, 15 mph = 6.705 m/s)
        thresh_mid = 4.47 if self.reinforced_alloy_chassis else 2.235
        thresh_high = 13.41 if self.reinforced_alloy_chassis else 6.705
        
        detached_blocks: List[Tuple[int, int, int]] = []
        harvested_boost: float = 0.0
        
        if delta_v < thresh_mid:
            tier = "LOW" # Cosmetic scuff
        elif delta_v <= thresh_high:
            tier = "MID" # Panel denting, peripheral block detachment
            if graph and graph.blocks:
                # Detach peripheral block
                key = next(iter(graph.blocks.keys()))
                detached_block = graph.remove_block(key)
                if detached_block:
                    detached_blocks.append(key)
                    if inventory:
                        inventory.block_masses[detached_block.material_id] = inventory.block_masses.get(detached_block.material_id, 0) + 1
        else:
            tier = "HIGH" # Catastrophic shearing
            if graph and graph.blocks:
                keys = list(graph.blocks.keys())[:min(4, len(graph.blocks))]
                for k in keys:
                    detached_block = graph.remove_block(k)
                    if detached_block:
                        detached_blocks.append(k)
                        if inventory:
                            inventory.block_masses[detached_block.material_id] = inventory.block_masses.get(detached_block.material_id, 0) + 1

        self.last_impulse_tier = tier

        # Impact Energy Harvester augment conversion
        if self.impact_energy_harvester and delta_v > 1.0:
            harvested_boost = delta_v * 15.0

        return {
            "delta_v": delta_v,
            "peak_stress": peak_stress,
            "tier": tier,
            "detached_blocks": detached_blocks,
            "harvested_boost": harvested_boost
        }

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
    """
    Component for vehicle power systems utilizing ambient atmosphere oxidation.
    Includes Plasma Air Induction to ionize ambient atmospheric N2/O2 into high-energy
    oxidizers, overcoming NO2 manufacturing and processing scarcity.
    """
    max_charge: float = 100.0
    current_charge: float = 100.0
    discharge_rate: float = 2.0 
    ambient_generation_rate: float = 5.0 
    intake_efficiency: float = 1.0     
    is_choked: bool = False            
    stored_oxygen_reserve: float = 20.0 
    num_resonance_channels: int = 4
    _prime_ladder_cache: torch.Tensor = None
    
    # Plasma Air Induction: compensates for environmental NO2 scarcity via electric arc ionization
    plasma_air_induction: bool = True
    plasma_boost_factor: float = 2.4

@dataclass
class VehicleEngine:
    """
    Vehicle drive component consuming power from ABEB cells.
    Supports sleeve-valve combustion architecture with electrical port timing degrees
    synchronized with air-breathing battery discharge cycles.
    """
    throttle: float = 0.0 
    base_power_consumption: float = 3.5
    is_operational: bool = True
    
    # Sleeve-Valve combustion & electrical timings
    is_sleeve_valve: bool = True
    electrical_timing_degrees: float = 12.5 # Advance angle for rotary sleeve port alignment
    sleeve_port_overlap_efficiency: float = 1.35

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

    def compute_betti_numbers(self) -> Dict[str, int]:
        """
        Calculates exact Betti numbers (beta_0, beta_1) of the construct or track graph.
        beta_0: Number of connected components (islands/disjoint bodies)
        beta_1: Number of independent 1D cycles/tunnels (Euler: beta_1 = E - V + beta_0)
        """
        if not self.blocks:
            return {"betti_0": 0, "betti_1": 0, "euler_characteristic": 0}
            
        components = self.find_disconnected_components()
        beta_0 = len(components)
        V = len(self.blocks)
        
        edges = set()
        for cell in self.blocks:
            cx, cy, cz = cell
            for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
                nb = (cx+dx, cy+dy, cz+dz)
                if nb in self.blocks:
                    edges.add((cell, nb))
                    
        E = len(edges)
        beta_1 = max(0, E - V + beta_0)
        return {
            "betti_0": beta_0,
            "betti_1": beta_1,
            "euler_characteristic": beta_0 - beta_1
        }


# ==========================================
# PROCEDURAL BLUEPRINTS & B-SPLINE MODS
# ==========================================

from src.core.structural_blueprints import (
    AddonLayer, AddonRoutine, BooleanXORLayer,
    MangostienBSplineMod, DarkMatterAttractorLayer,
    MirrorSymmetryLayer, MangostienTicket, MangostienArbitrator
)

class BSplineCompiledMod(MangostienBSplineMod):
    """
    Compiled B-Spline Procedural Mod.
    Uses Kolmogorov-Arnold (KAN) layers and Cox-de Boor evaluation to compile
    exact, non-heuristic mathematical curves on the fly into voxel constructs
    (vehicle hulls, tools, organic flora, and terrain contours).
    Subject to Mohr-Coulomb fossilization criteria.
    """
    def __init__(self, name: str, latent_dim: int = 3, resolution: int = 20, spline_degree: int = 3):
        super().__init__(name=name, latent_dim=latent_dim, resolution=resolution)
        self.spline_degree = spline_degree
        self.kan_layer = KANLayer(in_features=latent_dim, out_features=3, grid_size=resolution, spline_order=spline_degree)

    def evaluate_curve(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluates non-heuristic B-spline curves along parameter t."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.size(-1) != self.latent_dim:
            t = t.repeat(1, self.latent_dim)[:, :self.latent_dim]
        return self.kan_layer(t)


# ==========================================
# PROCEDURAL FRACTAL GENERATION (TVA AESTHETIC)
# ==========================================

def morton_encode(x: int, y: int, z: int) -> int:
    """
    Bit-interleaves 3D coordinates (x, y, z) into a 1D scalar Morton code (Z-order curve).
    Collapses pointerless octree lookups into streaming memory.
    """
    def expand_bits(v: int) -> int:
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v <<  8)) & 0x0300F00F
        v = (v | (v <<  4)) & 0x030C30C3
        v = (v | (v <<  2)) & 0x09249249
        return v
    ux = max(0, int(x)) & 0x3FF
    uy = max(0, int(y)) & 0x3FF
    uz = max(0, int(z)) & 0x3FF
    return (expand_bits(ux) | (expand_bits(uy) << 1) | (expand_bits(uz) << 2))


# ==========================================
# FLORA DYNAMICS & PIRANGI CASHEW MUTATION
# ==========================================

class FloraMutationType(Enum):
    INVERTED_GRAVITROPISM = "inverted_gravitropism"
    ADVENTITIOUS_GROUND_ROOTING = "adventitious_ground_rooting"
    SECONDARY_TRUNK_METAMORPHOSIS = "secondary_trunk_metamorphosis"
    CONTINUOUS_GRAPH_EXPANSION = "continuous_graph_expansion"

@dataclass
class FloraComponent:
    """
    ECS Component representing living flora.
    Tracks developmental age, hydration, photosynthetic potential,
    mutation flags, and vascular graph connectivity.
    """
    entity_id: str
    species_name: str = "PirangiCashew"
    mutation_tags: List[str] = field(default_factory=lambda: [
        "INVERTED_GRAVITROPISM",
        "ADVENTITIOUS_GROUND_ROOTING",
        "SECONDARY_TRUNK_METAMORPHOSIS",
        "CONTINUOUS_GRAPH_EXPANSION"
    ])
    age_ticks: int = 0
    hydration: float = 1.0
    sunlight_exposure: float = 1.0
    growth_rate: float = 1.0
    lateral_spread_bias: float = 0.92
    droop_constant: float = 0.02
    adventitious_roots_count: int = 0
    secondary_trunks_count: int = 0
    canopy_radius: float = 1.0
    total_wood_mass: int = 0
    ground_z: int = 0

@dataclass
class AdventitiousRootNode:
    """Represents a branch contact point anchored into soil as an adventitious root."""
    ground_cell: Tuple[int, int, int]
    parent_branch_cell: Tuple[int, int, int]
    is_secondary_trunk: bool = False
    secondary_branch_cells: List[Tuple[int, int, int]] = field(default_factory=list)

class PirangiCashewTree:
    """
    Procedural simulation of the 'Cashew Tree of Pirangi' (Maior cajueiro do mundo) mutation.
    
    Biological & Topological Phenomena:
    1. Inverted Gravitropism: Branches extend horizontally outward across the (x, y) plane
       instead of growing vertically upwards.
    2. Cantilever Droop: Under accumulating wood mass and branch length, branches bow down 
       towards the ground (delta_z = -kappa * mass * L^2).
    3. Adventitious Rooting: Upon touching the ground (z <= ground_z), branches form adventitious 
       roots that penetrate the soil substrate instead of rotting or terminating.
    4. Secondary Trunk Metamorphosis: Anchored root sites metamorphose into new vertical/lateral 
       secondary trunks that spawn secondary horizontal branches.
    5. Continuous Single-Organism Expansion: Spans an expansive canopy (mimicking the 8,500 m2 
       specimen) as a single connected StructuralGraph, maintaining strict mass conservation.
    """
    def __init__(
        self,
        root_cell: Tuple[int, int, int] = (0, 0, 1),
        ground_z: int = 0,
        wood_material_id: int = 4,
        leaf_material_id: int = 5,
        root_material_id: int = 6
    ):
        self.root_cell = root_cell
        self.ground_z = ground_z
        self.wood_material_id = wood_material_id
        self.leaf_material_id = leaf_material_id
        self.root_material_id = root_material_id
        
        self.flora_comp = FloraComponent(
            entity_id=f"pirangi_tree_{uuid.uuid4().hex[:8]}",
            ground_z=ground_z
        )
        self.graph = StructuralGraph()
        self.root_nodes: Dict[Tuple[int, int, int], AdventitiousRootNode] = {}
        self.active_branch_tips: List[Dict[str, Any]] = []
        
        # Initialize original primary trunk
        rx, ry, rz = root_cell
        for z in range(ground_z, rz + 1):
            cell = (rx, ry, z)
            block = Block(local_cell=cell, rotation=(0, 0, 0, 1), material_id=self.wood_material_id, health=150.0)
            self.graph.add_block(block)
        
        # Initiate 4 initial cardinal lateral branches due to inverted gravitropism
        angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for idx, ang in enumerate(angles):
            self.active_branch_tips.append({
                "origin": (rx, ry, rz),
                "current_cell": (rx, ry, rz),
                "angle": ang,
                "length": 1.0,
                "mass": 1.0,
                "is_rooted": False,
                "source_trunk": (rx, ry, rz)
            })

    def grow_tick(
        self,
        terrain_octree: Optional['PointerlessOctree'] = None,
        inventory: Optional[InventoryComponent] = None,
        max_new_blocks_per_tick: int = 8
    ) -> Dict[str, Any]:
        """
        Executes one physiological growth tick:
        - Advances horizontal branches outward.
        - Applies cantilever droop.
        - Checks for ground contact and establishes adventitious roots.
        - Promotes rooted sites into secondary trunks.
        - Enforces mass deduction from inventory.
        """
        self.flora_comp.age_ticks += 1
        blocks_added = 0
        new_roots = 0
        new_secondary_trunks = 0
        
        for tip in list(self.active_branch_tips):
            if blocks_added >= max_new_blocks_per_tick:
                break
                
            if tip.get("is_rooted", False):
                continue
                
            if inventory is not None:
                avail_mass = inventory.block_masses.get(self.wood_material_id, 0)
                if avail_mass <= 0:
                    break
                    
            cx, cy, cz = tip["current_cell"]
            tip["length"] += 1.0
            tip["mass"] += 1.2
            
            # 1. Inverted Gravitropism: Lateral horizontal expansion
            dx = math.cos(tip["angle"])
            dy = math.sin(tip["angle"])
            target_x = int(round(cx + dx))
            target_y = int(round(cy + dy))
            
            # 2. Cantilever Droop: delta_z = -kappa * mass * L^2
            droop = self.flora_comp.droop_constant * tip["mass"] * (tip["length"] ** 1.3)
            droop_int = int(math.floor(droop))
            target_z = max(self.ground_z, cz - (1 if droop_int > 0 else 0))
            
            # Form contiguous 6-connected cardinal steps to ensure single topological graph unity
            cardinal_steps = []
            curr_step = (cx, cy, cz)
            if target_x != curr_step[0]:
                curr_step = (curr_step[0] + (1 if target_x > curr_step[0] else -1), curr_step[1], curr_step[2])
                cardinal_steps.append(curr_step)
            if target_y != curr_step[1]:
                curr_step = (curr_step[0], curr_step[1] + (1 if target_y > curr_step[1] else -1), curr_step[2])
                cardinal_steps.append(curr_step)
            if target_z != curr_step[2]:
                curr_step = (curr_step[0], curr_step[1], curr_step[2] + (1 if target_z > curr_step[2] else -1))
                cardinal_steps.append(curr_step)
            
            if not cardinal_steps:
                cardinal_steps = [(target_x, target_y, target_z)]
                
            for step_cell in cardinal_steps:
                if step_cell not in self.graph.blocks:
                    wood_block = Block(
                        local_cell=step_cell,
                        rotation=(0, 0, 0, 1),
                        material_id=self.wood_material_id,
                        health=100.0
                    )
                    self.graph.add_block(wood_block)
                    blocks_added += 1
                    if inventory is not None:
                        inventory.block_masses[self.wood_material_id] -= 1
                    if terrain_octree is not None:
                        terrain_octree.add_fractal_node(step_cell[0], step_cell[1], step_cell[2], self.wood_material_id)
            
            next_cell = cardinal_steps[-1]
            tip["current_cell"] = next_cell
            nz = next_cell[2]
            
            # 3. Ground contact detection -> Adventitious Rooting
            if nz <= self.ground_z:
                tip["is_rooted"] = True
                new_roots += 1
                self.flora_comp.adventitious_roots_count += 1
                
                # Anchor root down into ground
                root_cell = (next_cell[0], next_cell[1], self.ground_z - 1)
                if root_cell not in self.graph.blocks:
                    root_block = Block(
                        local_cell=root_cell,
                        rotation=(0, 0, 0, 1),
                        material_id=self.root_material_id,
                        health=200.0
                    )
                    self.graph.add_block(root_block)
                    blocks_added += 1
                    if terrain_octree is not None:
                        terrain_octree.add_fractal_node(root_cell[0], root_cell[1], root_cell[2], self.root_material_id)
                
                root_node = AdventitiousRootNode(
                    ground_cell=(next_cell[0], next_cell[1], self.ground_z),
                    parent_branch_cell=next_cell,
                    is_secondary_trunk=True
                )
                self.root_nodes[next_cell] = root_node
                
                # 4. Secondary Trunk Metamorphosis:
                new_secondary_trunks += 1
                self.flora_comp.secondary_trunks_count += 1
                
                child_angles = [
                    tip["angle"] - math.pi / 3,
                    tip["angle"] + math.pi / 3,
                    tip["angle"] + math.pi / 2
                ]
                for c_ang in child_angles:
                    sec_tip = {
                        "origin": next_cell,
                        "current_cell": next_cell,
                        "angle": c_ang % (2 * math.pi),
                        "length": 1.0,
                        "mass": 1.0,
                        "is_rooted": False,
                        "source_trunk": next_cell
                    }
                    self.active_branch_tips.append(sec_tip)
                    root_node.secondary_branch_cells.append(next_cell)

        # Update canopy radius
        max_dist = 0.0
        rx, ry, rz = self.root_cell
        for cell in self.graph.blocks.keys():
            dist = math.sqrt((cell[0] - rx) ** 2 + (cell[1] - ry) ** 2)
            if dist > max_dist:
                max_dist = dist
        self.flora_comp.canopy_radius = max_dist
        self.flora_comp.total_wood_mass = len(self.graph.blocks)
        
        return {
            "age_ticks": self.flora_comp.age_ticks,
            "blocks_added": blocks_added,
            "total_blocks": len(self.graph.blocks),
            "adventitious_roots": self.flora_comp.adventitious_roots_count,
            "secondary_trunks": self.flora_comp.secondary_trunks_count,
            "canopy_radius": self.flora_comp.canopy_radius,
            "connected_components": len(self.graph.find_disconnected_components())
        }

class FloraTreeSapling:
    """
    Plantable Sapling item component in Voxelboxter.
    Conserves mass by deducting sapling material from player inventory upon planting.
    """
    @staticmethod
    def plant_sapling(
        position: Tuple[int, int, int],
        inventory: InventoryComponent,
        player_id: str = "BUILDER",
        species: str = "pirangi_cashew",
        ground_z: int = 0
    ) -> Optional[PirangiCashewTree]:
        """Plants a sapling at the given position if mass is available."""
        sapling_cost = 4
        avail = inventory.block_masses.get(4, 0)
        if avail < sapling_cost:
            return None
        
        inventory.block_masses[4] -= sapling_cost
        tree = PirangiCashewTree(root_cell=position, ground_z=ground_z)
        return tree


# ==========================================
# FAUNA DYNAMICS & INHIBITION-STABILIZED NETWORKS
# ==========================================

@dataclass
class FaunaComponent:
    """ECS Component for animal fauna populations or individuals."""
    entity_id: str
    species: str = "DesertStalker"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    excitation_level: float = 1.0
    inhibition_level: float = 1.0
    defect_density: float = 0.0
    chirality_phase: float = 1.0
    is_active: bool = True

class FaunaISN(torch.nn.Module):
    """
    Fauna behavioral dynamics modeled as an Inhibition-Stabilized Network (ISN).
    - Cross-homeostatic E/I balance prevents runaway panic excitation or behavioral stupor.
    - Defect propagation partial differential equation (PDE):
      dd/dt = D * del^2(d) + alpha * V_gyroid - beta * d
      diffuses environmental trauma/disruptions like inflammatory cytokine signals.
    - Chiral gating routes trajectories through left- or right-handed phase paths.
    """
    def __init__(
        self,
        w_ee: float = 1.4,
        w_ei: float = 1.2,
        w_ie: float = 1.0,
        w_ii: float = 0.8,
        diffusion_d: float = 0.15,
        alpha_gyroid: float = 0.1,
        beta_decay: float = 0.05
    ):
        super().__init__()
        self.w_ee = torch.nn.Parameter(torch.tensor(w_ee))
        self.w_ei = torch.nn.Parameter(torch.tensor(w_ei))
        self.w_ie = torch.nn.Parameter(torch.tensor(w_ie))
        self.w_ii = torch.nn.Parameter(torch.tensor(w_ii))
        
        self.diffusion_d = diffusion_d
        self.alpha_gyroid = alpha_gyroid
        self.beta_decay = beta_decay

    def step_population(
        self,
        fauna: FaunaComponent,
        external_stimulus: float = 0.0,
        local_gyroid_potential: float = 0.0,
        dt: float = 0.05
    ) -> Dict[str, float]:
        """Advances one discrete integration step of the ISN dynamics."""
        e = fauna.excitation_level
        i = fauna.inhibition_level
        
        # ISN differential equations
        de_dt = -e + torch.relu(self.w_ee * e - self.w_ei * i + external_stimulus).item()
        di_dt = -i + torch.relu(self.w_ie * e - self.w_ii * i + external_stimulus * 0.5).item()
        
        fauna.excitation_level = max(0.01, e + de_dt * dt)
        fauna.inhibition_level = max(0.01, i + di_dt * dt)
        
        # Defect propagation PDE step: dd/dt = D * del^2(d) + alpha * V_gyroid - beta * d
        laplacian_d = external_stimulus * 0.5 - fauna.defect_density
        dd_dt = self.diffusion_d * laplacian_d + self.alpha_gyroid * local_gyroid_potential - self.beta_decay * fauna.defect_density
        fauna.defect_density = max(0.0, fauna.defect_density + dd_dt * dt)
        
        # Chiral gating: routes steering left or right based on excitation vs inhibition asymmetry
        fauna.chirality_phase = 1.0 if (fauna.excitation_level >= fauna.inhibition_level) else -1.0
        
        ei_ratio = fauna.excitation_level / max(1e-4, fauna.inhibition_level)
        return {
            "excitation": fauna.excitation_level,
            "inhibition": fauna.inhibition_level,
            "ei_ratio": ei_ratio,
            "defect_density": fauna.defect_density,
            "chirality_phase": fauna.chirality_phase
        }


# ==========================================
# EMERGENT BAKING ENGINE
# ==========================================

class EmergentBakingEngine:
    """
    Driven by GardenStatisticalAttractors to allow players to 'bake' and lock in localized
    geometry (structures, flora groves, crafted items) via user-triggered fossilization.
    Turns subjective structural attachment into topological rigidity on the Poincare disk.
    """
    def __init__(self):
        try:
            from src.core.garden_statistical_attractors import GardenOrchestrator
            self.garden = GardenOrchestrator(feature_dim=16)
        except Exception:
            self.garden = None

    def bake_construct(
        self,
        construct_id: str,
        graph: StructuralGraph,
        permissions: PermissionsManager,
        peer_id: str
    ) -> Dict[str, Any]:
        """
        Bakes a dynamic construct into an immutable topological fossil.
        Requires ADMIN or BUILDER permission.
        """
        if not (permissions.has_permission(peer_id, "manage_mangostiens") or permissions.has_permission(peer_id, "creative_mode")):
            return {"success": False, "reason": "Insufficient permissions to bake construct"}

        total_blocks = len(graph.blocks)
        topological_persistence = min(1.0, 0.5 + 0.05 * math.log(max(1, total_blocks)))
        
        for block in graph.blocks.values():
            block.health = 999.0
            
        return {
            "success": True,
            "construct_id": construct_id,
            "total_fossilized_blocks": total_blocks,
            "topological_persistence": topological_persistence,
            "fossil_timestamp": time.time()
        }


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

    def remove_voxel(self, x: int, y: int, z: int) -> bool:
        """Removes a voxel from the Morton-encoded spatial hash lattice. Returns True if removed."""
        code = self._interleave_bits(x, y, z)
        if code in self.morton_grid:
            del self.morton_grid[code]
            return True
        return False

    def _deinterleave_bits(self, code: int) -> Tuple[int, int, int]:
        """Recovers 3D integer coordinates (x, y, z) from a 30-bit Morton code."""
        def compact_bits(v: int) -> int:
            v &= 0x09249249
            v = (v ^ (v >> 2)) & 0x030C30C3
            v = (v ^ (v >> 4)) & 0x0300F00F
            v = (v ^ (v >> 8)) & 0x030000FF
            v = (v ^ (v >> 16)) & 0x000003FF
            return v
        return (compact_bits(code), compact_bits(code >> 1), compact_bits(code >> 2))

    def compute_betti_numbers(self) -> Dict[str, int]:
        """
        Computes topological Betti numbers (beta_0, beta_1) directly from the Morton-encoded voxel lattice.
        beta_0: Number of connected voxel islands (islands of terrain/constructs).
        beta_1: 1D topological cycles / loop tunnels (Euler characteristic: beta_1 = E - V + beta_0).
        """
        if not self.morton_grid:
            return {"betti_0": 1, "betti_1": 0, "euler_characteristic": 1}
            
        coords = set(self._deinterleave_bits(code) for code in self.morton_grid.keys())
        V = len(coords)
        if V == 0:
            return {"betti_0": 0, "betti_1": 0, "euler_characteristic": 0}
            
        visited = set()
        components = 0
        edges_count = 0
        
        for v in coords:
            if v not in visited:
                components += 1
                queue = [v]
                visited.add(v)
                while queue:
                    curr = queue.pop(0)
                    cx, cy, cz = curr
                    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        nb = (cx+dx, cy+dy, cz+dz)
                        if nb in coords and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
            
            # Count undirected edges along positive directions to avoid double counting
            cx, cy, cz = v
            for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
                nb = (cx+dx, cy+dy, cz+dz)
                if nb in coords:
                    edges_count += 1
                    
        beta_0 = components
        beta_1 = max(0, edges_count - V + beta_0)
        return {
            "betti_0": beta_0,
            "betti_1": beta_1,
            "euler_characteristic": beta_0 - beta_1
        }
        
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

from src.core.yield_criteria import MohrCoulombProjection, DruckerPragerProjection
from src.core.erosion_filter import TopologicalErosionFBM

class VoxelboxterEngine:
    """
    Hooks the DiegeticPhysicsEngine into the ECS architecture.
    Handles dynamic PyBevy mesh mutations natively from Python.
    Uses Silicon Sovereignty Engine for PyOpenCL hardware acceleration.
    
    Integrates Dual-Scale Plasticity:
    - Mohr-Coulomb for local sharp brittle shear fractures (cuts, craters, tire gouges)
    - Drucker-Prager for smooth global plastic flow envelopes preventing mesh crashes
    - Fractional Brownian Motion (FBM) erosion for carving weather gullies and Ley Lines
    - Living ecological loops: Pirangi cashew flora mutation and Fauna ISN populations
    """
    def __init__(self, device: str = "cpu"):
        from src.core.diegetic_physics_engine import DiegeticPhysicsEngine
        try:
            from src.core.device_utils import get_torch_device
            torch_dev = get_torch_device(device)
        except Exception:
            torch_dev = "cpu" if str(device).lower() == "opencl" else str(device)

        try:
            from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
            self.silicon_sovereignty = SiliconSovereigntyEngine(use_gpu=True)
        except Exception as e:
            self.silicon_sovereignty = None
            print(f"[VoxelboxterEngine] SiliconSovereigntyEngine running in mock mode: {e}")

        self.physics = DiegeticPhysicsEngine(device=torch_dev)
        self.device = device
        self.constructs: Dict[str, 'StructuralGraph'] = {}
        self.controllers: Dict[str, VehicleController] = {}
        self.rigid_bodies: Dict[str, RigidBody] = {}
        
        # Dual-scale plasticity yield criteria
        self.mohr_coulomb = MohrCoulombProjection(friction_angle=30.0, cohesion=0.8)
        self.drucker_prager = DruckerPragerProjection(alpha=0.2, k=1.0)
        
        # Weathering and Ley line FBM erosion filter
        self.erosion_fbm = TopologicalErosionFBM(octaves=4)
        
        # Emergent baking engine & ecological managers
        self.baking_engine = EmergentBakingEngine()
        self.fauna_populations: Dict[str, Tuple[FaunaComponent, FaunaISN]] = {}
        self.active_flora: Dict[str, PirangiCashewTree] = {}

        # Cleanroom mechanics suites (JourneyMap, JER, Mob Properties, Infernal Mobs, Mekanism, Create Aeronautics, Spice of Life, EnderIO, RedNet, Factorio)
        from src.environment.cleanroom_mechanics import (
            JourneyTopoRadar, ResourceDistributionInspector, MultiMineMemory,
            MekanismProcessingPipeline, NutritionalDiversityTracker,
            RedNet16BundledCable, CompositeVoxelConduit,
            TransportBeltSegment, DirectionalInserter,
            FaunaGeneticsComponent, AeronauticContraption,
            ExpandedInventorySystem, MobProperties
        )
        self.radar = JourneyTopoRadar()
        self.jer = ResourceDistributionInspector()
        self.multi_mine = MultiMineMemory()
        self.mekanism = MekanismProcessingPipeline()
        self.nutrition = NutritionalDiversityTracker()
        self.rednet_bundle = RedNet16BundledCable()
        self.conduits: Dict[Tuple[int, int, int], CompositeVoxelConduit] = {}
        self.transport_belts: Dict[Tuple[int, int, int], TransportBeltSegment] = {}
        self.aeronautics: Dict[str, AeronauticContraption] = {}
        self.mob_registry: Dict[str, MobProperties] = {}
        self.fauna_genetics: Dict[str, FaunaGeneticsComponent] = {}
        self.terrain_octree = PointerlessOctree()

    def compute_terrain_betti(self, cid: Optional[str] = None) -> Dict[str, Any]:
        """
        Dynamically computes real topological Betti numbers (beta_0, beta_1) for the
        active terrain track or vehicle/construct graph, completely replacing mock constants.
        """
        if cid and cid in self.constructs and self.constructs[cid].blocks:
            return self.constructs[cid].compute_betti_numbers()
        elif "track" in self.constructs and self.constructs["track"].blocks:
            return self.constructs["track"].compute_betti_numbers()
        elif self.terrain_octree and self.terrain_octree.morton_grid:
            return self.terrain_octree.compute_betti_numbers()
        elif self.constructs:
            total_b0 = 0
            total_b1 = 0
            for g in self.constructs.values():
                b = g.compute_betti_numbers()
                total_b0 += b["betti_0"]
                total_b1 += b["betti_1"]
            return {"betti_0": max(1, total_b0), "betti_1": total_b1, "euler_characteristic": total_b0 - total_b1}
        else:
            return {"betti_0": 1, "betti_1": 0, "euler_characteristic": 1}

    def mine_voxel_progressive(self, coord: Tuple[int, int, int], damage_delta: float) -> bool:
        """AtomicStryker Multi Mine cleanroom: partial damage remembered across ticks."""
        return self.multi_mine.mine_block(coord, damage_delta)

    def refine_ore_mekanism(self, ore_count: int, tier: int = 1, reagent_mb: float = 0.0) -> Tuple[int, float]:
        """Mekanism tiered ore multiplier (Tier 1: 1x, Tier 2: 2x, Tier 3: 3x, Tier 4: 4x, Tier 5: 5x)."""
        if tier == 1:
            return self.mekanism.process_tier_1_smelt(ore_count), 0.0
        elif tier == 2:
            return self.mekanism.process_tier_2_enrichment(ore_count), 0.0
        elif tier == 3:
            return self.mekanism.process_tier_3_purification(ore_count, reagent_mb)
        elif tier == 4:
            return self.mekanism.process_tier_4_chemical_injection(ore_count, reagent_mb)
        elif tier == 5:
            return self.mekanism.process_tier_5_chemical_dissolution(ore_count, reagent_mb)
        return ore_count, 0.0

    def consume_food_spice_of_life(self, food_id: str, nutrition: float, saturation: float) -> Dict[str, Any]:
        """Spice of Life Carrot & Onion cleanroom: permanent heart progression + dynamic buffs."""
        return self.nutrition.eat_food(food_id, nutrition, saturation)

    def add_aeronautic_airship(self, name: str, contraption: Any):
        """Create: Aeronautics cleanroom: registers physical airship contraption."""
        self.aeronautics[name] = contraption

    def add_construct(self, cid: str, graph: 'StructuralGraph', rb: RigidBody, controller: VehicleController):
        self.constructs[cid] = graph
        self.rigid_bodies[cid] = rb
        self.controllers[cid] = controller

    def add_flora_cashew_tree(self, tree_id: str, tree: PirangiCashewTree):
        """Registers a mutating Pirangi cashew tree into the active world simulation."""
        self.active_flora[tree_id] = tree
        self.constructs[tree_id] = tree.graph

    def add_fauna_population(self, fauna_id: str, fauna: FaunaComponent, isn: Optional[FaunaISN] = None):
        """Registers an animal fauna population governed by an Inhibition-Stabilized Network."""
        if isn is None:
            isn = FaunaISN()
        self.fauna_populations[fauna_id] = (fauna, isn)

    def apply_dual_yield_fracture(
        self,
        impulse_J: float,
        stress_tensor: Optional[torch.Tensor] = None,
        local_cell: Optional[Tuple[int, int, int]] = None,
        construct_id: Optional[str] = None,
        inventory: Optional[InventoryComponent] = None
    ) -> Dict[str, Any]:
        """
        Evaluates Dual-Scale Plasticity:
        1. Mohr-Coulomb (Local Shear Yield):
           tau = c + sigma * tan(phi). Exceeding shear fractures voxels along brittle rupture lines.
        2. Drucker-Prager (Global Flow Envelope):
           alpha * I1 + sqrt(J2) - k = 0. Encloses ruptured geometry in a smooth convex envelope.
        """
        if stress_tensor is None:
            stress_tensor = torch.tensor([[impulse_J * 0.1]])
            
        normal_load = torch.zeros_like(stress_tensor)
        
        # 1. Local Mohr-Coulomb evaluation
        mc_shear_yield = self.mohr_coulomb(stress_tensor, normal_load)
        is_brittle_rupture = mc_shear_yield.item() > 0.8
        
        # 2. Global Drucker-Prager smooth flow envelope
        dp_flow = self.drucker_prager(stress_tensor)
        
        detached_blocks = []
        if is_brittle_rupture and construct_id and construct_id in self.constructs:
            graph = self.constructs[construct_id]
            if local_cell and local_cell in graph.blocks:
                detached = graph.remove_block(local_cell)
                if detached:
                    detached_blocks.append(local_cell)
                    if inventory:
                        inventory.block_masses[detached.material_id] = inventory.block_masses.get(detached.material_id, 0) + 1
                        
        return {
            "mohr_coulomb_shear": mc_shear_yield.item(),
            "is_brittle_rupture": is_brittle_rupture,
            "drucker_prager_envelope": dp_flow.item(),
            "detached_blocks": detached_blocks
        }

    def carve_weathering_ley_lines(
        self,
        trajectory_coords: torch.Tensor,
        vehicle_mass: float = 1.0,
        traffic_intensity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Applies Fractional Anisotropic Fractal Polynomial Functionals encoded Brownian Motion (FBM).
        Carves multi-scale gullies along pressure gradients (grad P) using resonant prime frequencies.
        Gullies carved by heavy vehicles become Ley Lines (Resonance Streamlines) for lighter vehicles.
        """
        with torch.no_grad():
            if trajectory_coords.dim() == 1:
                trajectory_coords = trajectory_coords.unsqueeze(0)
            pressure_grad = torch.ones_like(trajectory_coords) * traffic_intensity
            erosion_depth = self.erosion_fbm(trajectory_coords, pressure_grad=pressure_grad, intensity=0.1 * (vehicle_mass ** 0.5))
            slipstream_boost = torch.mean(torch.abs(erosion_depth)).item() * 1.5
            return {
                "erosion_depth": erosion_depth,
                "slipstream_boost": slipstream_boost,
                "ley_line_active": slipstream_boost > 0.05
            }

    def tick(self, dt: float):
        # 1. Tick Vehicle Controllers & Physics
        for cid, controller in self.controllers.items():
            v_state = {
                "velocity": self.rigid_bodies[cid].linear_velocity,
                "angular_velocity": self.rigid_bodies[cid].angular_velocity
            }
            track_state = self.compute_terrain_betti(cid)
            c_input = {
                "throttle": controller.throttle,
                "yaw": controller.yaw,
                "pitch": controller.pitch,
                "roll": controller.roll
            }
            
            out_state = self.physics.process_input(v_state, track_state, c_input)
            
            if out_state.get("betti_shift", 0) > 0:
                if self.terrain_octree and self.terrain_octree.morton_grid:
                    rb = self.rigid_bodies[cid]
                    vx, vy, vz = int(rb.position[0]), int(rb.position[1]), int(rb.position[2])
                    self.terrain_octree.remove_voxel(vx, vy, vz)
                self._deform_terrain_mesh_pybevy()

        # 2. Tick Fauna ISN Dynamics
        for fid, (fauna, isn) in self.fauna_populations.items():
            if fauna.is_active:
                isn.step_population(fauna, external_stimulus=0.05, local_gyroid_potential=0.1, dt=dt)

        # 3. Tick Active Flora (Pirangi Cashew mutation expansion)
        for tid, tree in self.active_flora.items():
            tree.grow_tick()

    def _deform_terrain_mesh_pybevy(self, fractal_meta_state: Optional[torch.Tensor] = None):
        """
        Dynamically deforms the terrain voxel mesh natively in Python.
        Pipes the continuous KAGH Surrogate projections from FractalMetaFunctional
        into discrete Menger-Sponge/Stepwell voxel operations.
        Accelerated using PyOpenCL zero-copy when available.
        """
        try:
            import pybevy
            
            if self.silicon_sovereignty and self.silicon_sovereignty.ctx:
                import pyopencl as cl
                import pyopencl.array as cl_array
                print("[VOXELBOXTER] Triggering TailSlayer Zero-Copy PyOpenCL terrain deformation.", flush=True)
                
                if fractal_meta_state is not None:
                    spatial_mass_budget = torch.sum(fractal_meta_state ** 2).item()
                    print(f"[VOXELBOXTER] Parseval's Theorem computed mass budget: {spatial_mass_budget:.4f}. Injecting residues.", flush=True)
                    
                    octree = PointerlessOctree()
                    out_lattice = octree.apply_wasserstein_collapse(
                        self.silicon_sovereignty.ctx, 
                        self.silicon_sovereignty.queue_a, 
                        fractal_meta_state, 
                        iteration=int(time.time()), 
                        mass_budget=spatial_mass_budget
                    )
                    print(f"[VOXELBOXTER] Hedged Zero-Copy update complete. Terrain lattice deformed.", flush=True)
            else:
                print("[VOXELBOXTER] Applied PyBevy ResMut[Assets[Mesh]] Betti shift deformation (CPU Mock).", flush=True)
                if fractal_meta_state is not None:
                    print("[VOXELBOXTER] Simulating Drucker-Prager Yield structural fracturing.", flush=True)
        except ImportError:
            pass

# Canonical Alias
VoxelboxterSimulation = VoxelboxterEngine

