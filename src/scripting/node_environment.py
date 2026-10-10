"""
Dedicated DearPyGui Node Environment for Physical Scripting.
Implements Blender-style "Plug & Play" Dataflow Node Graph Architecture:
- Evaluates data dependency from right to left (pull) while streaming data left to right (push).
- Typed sockets: Vector (Blue), Float (Gray), Int/Bool (Green/Pink), Color (Yellow).
- Master Node Groups (reusable functions for bounding box, center of gravity, named attributes).
- Direct integration with core canonical subsystems:
  * PointerlessOctree & BooleanXORLayer for Chisels & Bits micro-voxel carving.
  * AddonRoutine & InventoryComponent for construct crafting and mass accounting.
  * VoxelboxterEngine, AirBreathingBattery (with Plasma Air Induction), and sleeve-valve combustion timings.
  * Delta-v (Delta_v = J / m) frame-by-frame impulse dynamics and collision severity tiers.
  * ValenceFunctional & EgoDeathThresholdMonitor for life and manifold hunger.
  * AdaptiveSkeletonHarness & EnemySubtype for procedural character rigs and enemy morphologies.
  * CarnotMobiusLedger & LeontiefGovernor for thermodynamic economy and admin shop clearance.
"""

import threading
import logging
import time
import math
from typing import Dict, Any, Callable, List, Tuple, Optional, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import dearpygui.dearpygui as dpg

import torch

# Canonical engine and simulation imports
from src.ui.voxelboxter_simulation import (
    StructuralGraph, Block, SliderSettings, InventoryComponent,
    Role, PermissionsManager, RigidBody, Propulsor, PowerConsumer,
    Weapon, VehicleController, AirBreathingBattery, VehicleEngine,
    EnvironmentalAtmosphere, PointerlessOctree, VoxelboxterEngine
)
from src.core.structural_blueprints import (
    AddonLayer, AddonRoutine, BooleanXORLayer,
    MangostienBSplineMod, DarkMatterAttractorLayer
)
from src.core.adaptive_skeleton_harness import (
    AdaptiveSkeletonHarness, AmbulatoryClass, EnemySubtype
)
from src.core.yield_criteria import DruckerPragerProjection, MohrCoulombProjection
from src.core.valence_drive import ValenceFunctional
from src.core.archetype_engines import EgoDeathThresholdMonitor
from src.core.carnot_mobius_ledger import CarnotMobiusLedger
from src.core.leontief_governor import LeontiefGovernor
from src.core.honest_jitter import harvest_honest_jitter
from src.core.hardware_monitor import has_headroom
from src.p2p.bonfire_consensus import BonfireNomadicRing
from src.p2p.freenet_ws_client import FreenetClient
from src.p2p.zk_aggregator import ZKAggregator
from src.surrogates.calm_predictor import CALM

logger = logging.getLogger(__name__)

# =========================================================================
# 1. BLENDER-STYLE DATAFLOW SOCKETS & WIRE ARCHITECTURE
# =========================================================================

class SocketType(Enum):
    VECTOR = auto()    # Blue: 3D Directional/Position data [X, Y, Z]
    FLOAT = auto()     # Gray: Single numerical scalar (Friction, Hardness, Delta-v, Mass)
    INT = auto()       # Green: Discrete integer counting number (Material ID, Iterations)
    BOOLEAN = auto()   # Pink: True/False toggle switch
    COLOR = auto()     # Yellow: Color / RGBA 4-element vector


@dataclass
class NodeSocket:
    name: str
    socket_type: SocketType
    is_output: bool
    value: Any = None
    connected_to: List['NodeSocket'] = field(default_factory=list)


class DataflowNode:
    """
    Base dataflow node executing right-to-left dependency pull and left-to-right push.
    """
    def __init__(self, node_id: str, label: str):
        self.node_id = node_id
        self.label = label
        self.inputs: Dict[str, NodeSocket] = {}
        self.outputs: Dict[str, NodeSocket] = {}

    def add_input(self, name: str, socket_type: SocketType, default_value: Any = None) -> NodeSocket:
        sock = NodeSocket(name, socket_type, is_output=False, value=default_value)
        self.inputs[name] = sock
        return sock

    def add_output(self, name: str, socket_type: SocketType, default_value: Any = None) -> NodeSocket:
        sock = NodeSocket(name, socket_type, is_output=True, value=default_value)
        self.outputs[name] = sock
        return sock

    def evaluate(self):
        """Calculates equations at this node and pushes results to output sockets."""
        pass


class MasterNodeGroup(DataflowNode):
    """
    Reusable Master Node Group (macro block).
    Calculates Bounding Box midpoints for Center of Gravity offset,
    and injects custom game engine attributes ('phys_hardness', 'phys_friction', 'phys_delta_v').
    """
    def __init__(self, group_id: str = "group_vehicle_attr_injector"):
        super().__init__(group_id, "Vehicle Attribute Injector (Master Group)")
        self.add_input("in_min_bounds", SocketType.VECTOR, (-1.0, -1.0, -1.0))
        self.add_input("in_max_bounds", SocketType.VECTOR, (1.0, 1.0, 1.0))
        self.add_input("in_hardness", SocketType.FLOAT, 10.0)
        self.add_input("in_friction", SocketType.FLOAT, 0.85)
        self.add_input("in_hp", SocketType.FLOAT, 100.0)
        
        self.add_output("out_cog_offset", SocketType.VECTOR, (0.0, 0.0, 0.0))
        self.add_output("out_attributes", SocketType.INT, {}) # Dictionary of named attributes

    def evaluate(self):
        # Center of Gravity calculation: size * -0.5 midpoint offset
        min_b = self.inputs["in_min_bounds"].value or (-1.0, -1.0, -1.0)
        max_b = self.inputs["in_max_bounds"].value or (1.0, 1.0, 1.0)
        mid_x = (min_b[0] + max_b[0]) * 0.5
        mid_y = (min_b[1] + max_b[1]) * 0.5
        mid_z = (min_b[2] + max_b[2]) * 0.5
        cog_offset = (-mid_x, -mid_y, -mid_z)
        self.outputs["out_cog_offset"].value = cog_offset

        # Named attributes baked into vertex/mesh export layer
        attrs = {
            "phys_hardness": float(self.inputs["in_hardness"].value or 10.0),
            "phys_friction": float(self.inputs["in_friction"].value or 0.85),
            "phys_hp": float(self.inputs["in_hp"].value or 100.0),
            "cog_offset": cog_offset
        }
        self.outputs["out_attributes"].value = attrs


# =========================================================================
# 2. ORE DICTIONARY & CANONICAL UNIFICATION
# =========================================================================

class OreDictionary:
    """
    Unifies disparate block, ore, scrap, and ingot IDs under canonical tags.
    """
    def __init__(self):
        self._tag_to_materials: Dict[str, Set[int]] = {}
        self._material_to_tags: Dict[int, Set[str]] = {}
        self._setup_defaults()

    def _setup_defaults(self):
        defaults = {
            "oreIron": {10, 11, 12},
            "ingotIron": {20, 21},
            "gemDiamond": {30},
            "dustRedstone": {40},
            "woodLog": {50, 51},
            "chiselBitStone": {100},
            "chiselBitIron": {101},
            "carbideVoxel": {150},
            "darkMatterFlux": {200},
            "cellOxygen": {300},
            "bioRation": {400},
            "thrusterBlock": {501},
            "batteryBlock": {502}
        }
        for tag, mats in defaults.items():
            for m in mats:
                self.register_ore(tag, m)

    def register_ore(self, tag: str, material_id: int):
        self._tag_to_materials.setdefault(tag, set()).add(material_id)
        self._material_to_tags.setdefault(material_id, set()).add(tag)

    def get_materials(self, tag: str) -> List[int]:
        return list(self._tag_to_materials.get(tag, set()))

    def get_tags(self, material_id: int) -> List[str]:
        return list(self._material_to_tags.get(material_id, set()))

    def matches_tag(self, material_id: int, tag: str) -> bool:
        return tag in self._material_to_tags.get(material_id, set())


# =========================================================================
# 3. PHYSICAL NODE EDITOR (UNIFIED INTEGRATION HARNESS)
# =========================================================================

class PhysicalNodeEditor:
    """
    Dedicated DearPyGui Node Environment for Physical Scripting.
    Implements Blender-style dataflow graph and connects directly into canonical engines:
    - PointerlessOctree & BooleanXORLayer for micro-voxel Chisels & Bits.
    - AddonRoutine & InventoryComponent for blueprint mass crafting.
    - VoxelboxterEngine, AirBreathingBattery (with Plasma Air Induction), and sleeve-valve timings.
    - Delta-v (Delta_v = J / m) impulse dynamics and collision severity tiers.
    - AdaptiveSkeletonHarness with EnemySubtype rig generation.
    - CarnotMobiusLedger & LeontiefGovernor for economy and admin shop clearance.
    - ValenceFunctional & EgoDeathThresholdMonitor for life and hunger.
    """
    def __init__(self, patch_state=None):
        self.patch_state = patch_state
        self.running = False
        self.thread = None
        self.eval_thread = None
        
        # Canonical engine components (no duplicate wrappers)
        self.oredict = OreDictionary()
        self.octree = PointerlessOctree(max_depth=8)
        self.routine = AddonRoutine()
        self.inventory = InventoryComponent()
        self.permissions = PermissionsManager()
        self.harness = AdaptiveSkeletonHarness(state_dim=8)
        
        # Vehicle & Kinetics
        self.rigid_body = RigidBody(mass=100.0)
        self.battery = AirBreathingBattery(max_charge=500.0, current_charge=500.0, plasma_air_induction=True)
        self.engine = VehicleEngine(throttle=0.0, is_sleeve_valve=True, electrical_timing_degrees=12.5)
        self.atmosphere = EnvironmentalAtmosphere(oxygen_density=1.0, ambient_pressure=101.3)
        self.propulsors: List[Propulsor] = [Propulsor(thrust=250.0)]
        
        # Life, Hunger & Ego Death
        self.valence = ValenceFunctional(decay=0.98, hunger_scale=1.0)
        self.ego_death_monitor = EgoDeathThresholdMonitor(abstraction_limit=0.85)
        self.health = 100.0
        self.hunger = 100.0
        self.saturation = 20.0
        self.metabolic_burn_rate = 1.0
        
        # Economy & Governance Ledgers
        self.carnot_ledger = CarnotMobiusLedger(xi_shear=0.1, p_cool_max=1000.0, p_nuclear=2000.0)
        self.leontief_governor = LeontiefGovernor(state_dim=2)
        self.admin_margin = 0.15
        self.wallet_balance = 1000.0
        self.transaction_history: List[float] = []

        # Master Node Group (Blender-style attribute injector)
        self.master_group = MasterNodeGroup()

        # Cleanroom Mechanics Modules (JourneyMap, JER, Mekanism, Create Aeronautics, Spice of Life, EnderIO, RedNet, Factorio, Animal Husbandry)
        from src.environment.cleanroom_mechanics import (
            JourneyTopoRadar, ResourceDistributionInspector, MultiMineMemory,
            MekanismProcessingPipeline, NutritionalDiversityTracker,
            RedNet16BundledCable, CompositeVoxelConduit,
            TransportBeltSegment, DirectionalInserter,
            AeronauticContraption, FaunaGeneticsComponent,
            ExpandedInventorySystem, MobProperties,
            SEMElectrochemicalExtractor, ItemStack,
            FluidStack, GasStack, InfernalAffix,
            RotationalKineticNetwork, RotationalNode,
            ModularTool, ToolMaterial, ToolPartType, MaterialTrait, SmelterySystem,
            RusticDelightManager, FermentationBarrel,
            JadeRaycastInspector, FastMapBlockState,
            WaystonesNetwork, WaystoneNode, LootrContainerManager,
            EasyAnvilsSystem, AllTheHeadsRegistry,
            EnchantmentIndustryPipeline, ClimateRiverSegment, RiverBiomeType,
            CombatNouveauProfile, WeaponCategory, HotbarSwapper,
            ProjectEEMCSolver, DiegeticCauldron,
            DistinctPotionProfile, ActiveEffectInsight,
            MultipartVoxelCell, SubgridRotationalWireHarness,
            BagOfHolding, BagTier, IC2CropGenome, GiantCropCluster, FoodQualityTier
        )
        self.radar = JourneyTopoRadar()
        self.jer = ResourceDistributionInspector()
        self.multi_mine = MultiMineMemory()
        self.mekanism = MekanismProcessingPipeline()
        self.sem_extractor = SEMElectrochemicalExtractor()
        self.nutrition = NutritionalDiversityTracker()
        self.rednet = RedNet16BundledCable()
        self.conduit = CompositeVoxelConduit()
        self.transport_belt = TransportBeltSegment()
        self.inserter = DirectionalInserter(pickup_coord=(0, 0, 0), dropoff_coord=(1, 0, 0))
        self.expanded_inventory = ExpandedInventorySystem()
        self.aeronautics = AeronauticContraption(name="SovereignAirship")
        self.mob_properties = MobProperties(mob_id="sample_mob")
        self.fauna_genetics = FaunaGeneticsComponent()

        # Extended Cleanroom Systems
        self.rotational_net = RotationalKineticNetwork()
        self.smeltery = SmelterySystem()
        self.modular_tool = ModularTool(tool_type="pickaxe")
        self.rustic_delight = RusticDelightManager()
        self.fermentation_barrel = FermentationBarrel()
        self.jade = JadeRaycastInspector()
        self.fast_map = FastMapBlockState()
        self.waystones = WaystonesNetwork()
        self.waystones.register_waystone(WaystoneNode(waystone_id="spawn", name="Spawn_Sanctuary", dimension="overworld", coordinates=(0, 64, 0), is_global=True))
        self.lootr = LootrContainerManager()
        self.enchantment_industry = EnchantmentIndustryPipeline()
        self.combat_nouveau = CombatNouveauProfile.create(WeaponCategory.SWORD)
        self.hotbar_swapper = HotbarSwapper()
        self.project_e = ProjectEEMCSolver()
        self.cauldron = DiegeticCauldron()
        self.wire_harness = SubgridRotationalWireHarness()
        self.bag_of_holding = BagOfHolding(tier=BagTier.IRON)
        self.crop_genome = IC2CropGenome()
        self.giant_crop = GiantCropCluster()

        # Virtual Links and Sidechain parameters
        self.virtual_links: List[Tuple[str, str]] = []
        self.last_state_tuple = None

    # ---------------------------------------------------------------------
    # Canonical Chisels & Bits Operations via PointerlessOctree & BooleanXOR
    # ---------------------------------------------------------------------
    def chisel_carve_bit(self, x: int, y: int, z: int, material_id: int) -> int:
        """Adds or carves a micro-voxel bit into the Morton-encoded octree."""
        code = self.octree._interleave_bits(x, y, z)
        prev = self.octree.morton_grid.pop(code, None)
        if prev is not None:
            # Store scrap directly into local inventory
            self.inventory.block_masses[prev] = self.inventory.block_masses.get(prev, 0) + 1
            return prev
        return 0

    def chisel_place_bit(self, x: int, y: int, z: int, material_id: int):
        """Places a micro-bit into the PointerlessOctree."""
        self.octree.add_fractal_node(x, y, z, material_id)

    # ---------------------------------------------------------------------
    # Canonical Crafting via AddonRoutine & InventoryComponent
    # ---------------------------------------------------------------------
    def craft_addon_layer(self, layer: AddonLayer) -> bool:
        """Executes construct fabrication using canonical AddonRoutine and mass deduction."""
        return self.routine.try_add_layer(layer, self.inventory)

    # ---------------------------------------------------------------------
    # Delta-v Collision Physics & Vehicle Kinetics
    # ---------------------------------------------------------------------
    def simulate_collision_impulse(
        self,
        impulse_J: float,
        contact_normal: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        delta_t: float = 0.016,
        graph: Optional[StructuralGraph] = None
    ) -> Dict[str, Any]:
        """Calculates Delta-v collision severity frame-by-frame and detaches voxels."""
        res = self.rigid_body.apply_collision_impulse(
            impulse_J=impulse_J,
            contact_normal=contact_normal,
            delta_t=delta_t,
            graph=graph,
            inventory=self.inventory
        )
        # If impact energy harvester is installed, recharge battery with harvested boost
        if res.get("harvested_boost", 0.0) > 0.0:
            self.battery.current_charge = min(self.battery.max_charge, self.battery.current_charge + res["harvested_boost"])
        return res

    def tick_vehicle_kinetics(self, dt: float) -> Dict[str, float]:
        """
        Simulates vehicle aerodynamics with Plasma Air Induction and sleeve-valve combustion.
        Plasma air induction overcomes NO2 synthesis bottlenecks by ionizing atmospheric N2/O2.
        """
        if not self.battery.is_choked:
            plasma_mult = self.battery.plasma_boost_factor if self.battery.plasma_air_induction else 1.0
            gen = self.battery.ambient_generation_rate * self.atmosphere.oxygen_density * plasma_mult * dt
            self.battery.current_charge = min(self.battery.max_charge, self.battery.current_charge + gen)

        # Sleeve valve electrical timings and port overlap efficiency
        timing_eff = math.cos(math.radians(self.engine.electrical_timing_degrees)) * self.engine.sleeve_port_overlap_efficiency
        power_demand = self.engine.base_power_consumption * self.engine.throttle * (1.0 / max(0.1, timing_eff))
        for p in self.propulsors:
            power_demand += p.fuel_or_power_cost * self.engine.throttle

        power_satisfied = 1.0
        step_power = power_demand * dt
        if self.battery.current_charge >= step_power:
            self.battery.current_charge -= step_power
        else:
            power_satisfied = self.battery.current_charge / max(1e-6, step_power)
            self.battery.current_charge = 0.0

        net_thrust = sum(p.thrust for p in self.propulsors) * self.engine.throttle * power_satisfied * timing_eff
        accel = net_thrust / max(1.0, self.rigid_body.mass)
        vx, vy, vz = self.rigid_body.linear_velocity
        vz += accel * dt
        # Atmospheric drag
        drag = 0.05 * self.atmosphere.ambient_pressure / 101.3
        vz *= max(0.0, 1.0 - drag * dt)
        self.rigid_body.linear_velocity = (vx, vy, vz)

        return {
            "battery_charge": self.battery.current_charge,
            "net_thrust": net_thrust,
            "speed": abs(vz),
            "accumulated_delta_v": self.rigid_body.accumulated_delta_v,
            "last_tier": self.rigid_body.last_impulse_tier
        }

    # ---------------------------------------------------------------------
    # Life, Hunger & Valence Dynamics
    # ---------------------------------------------------------------------
    def tick_life_and_hunger(self, dt: float, current_pressure: float = 0.5, current_mischief: float = 0.2) -> Dict[str, Any]:
        burn = self.metabolic_burn_rate * dt * (1.0 + current_mischief * 0.5)
        if self.saturation > burn:
            self.saturation -= burn
        else:
            rem = burn - self.saturation
            self.saturation = 0.0
            self.hunger = max(0.0, self.hunger - rem)

        # Starvation health loss
        starving = False
        if self.hunger <= 0.0:
            self.health = max(0.0, self.health - 2.5 * dt)
            starving = True

        p_tensor = torch.tensor([[current_pressure]])
        m_tensor = torch.tensor([[current_mischief]])
        valence_tensor = self.valence(p_tensor, mischief=m_tensor)
        valence_hunger = valence_tensor.mean().item()

        trauma = (100.0 - self.health) / 100.0
        lucidity = max(0.1, self.health / 100.0)
        r_a = self.ego_death_monitor.calculate_abstraction_rate(
            system_entropy_es=current_mischief,
            memory_trauma_tm=trauma,
            dissonance_delta=valence_hunger,
            lucidity_index_li=lucidity
        )
        ego_death = r_a >= self.ego_death_monitor.abstraction_limit

        return {
            "health": self.health,
            "hunger": self.hunger,
            "saturation": self.saturation,
            "starving": starving,
            "valence_hunger": valence_hunger,
            "abstraction_rate": r_a,
            "ego_death": ego_death
        }

    # ---------------------------------------------------------------------
    # Character Model & Enemy Rig Generation via AdaptiveSkeletonHarness
    # ---------------------------------------------------------------------
    def generate_enemy_rig(
        self,
        subtype: EnemySubtype,
        ambulatory_class: AmbulatoryClass = AmbulatoryClass.BIPED,
        difficulty_scale: float = 1.0
    ) -> Dict[str, Any]:
        return self.harness.build_enemy_rig(
            subtype=subtype,
            ambulatory_class=ambulatory_class,
            difficulty_scale=difficulty_scale
        )

    # ---------------------------------------------------------------------
    # Economy & Admin Shop Clearing via CarnotMobiusLedger & Leontief
    # ---------------------------------------------------------------------
    def execute_admin_trade(
        self,
        material_id: int,
        quantity: int,
        is_buy: bool,
        role: Role = Role.VISITOR
    ) -> Tuple[bool, float, str]:
        base_val = 10.0 # Standard metallurgical price
        unit_price = 0.0 if role == Role.ADMIN else (base_val * (1.0 + self.admin_margin) if is_buy else base_val * (1.0 - self.admin_margin))
        total = unit_price * quantity
        
        if is_buy:
            if role != Role.ADMIN and self.wallet_balance < total:
                return False, self.wallet_balance, f"Insufficient funds ({self.wallet_balance:.2f} < {total:.2f})"
            self.inventory.block_masses[material_id] = self.inventory.block_masses.get(material_id, 0) + quantity
            if role != Role.ADMIN:
                self.wallet_balance -= total
            self.transaction_history.append(total)
            return True, self.wallet_balance, f"Acquired {quantity}x of material {material_id}"
        else:
            have = self.inventory.block_masses.get(material_id, 0)
            if have < quantity:
                return False, self.wallet_balance, f"Insufficient inventory ({have} < {quantity})"
            self.inventory.block_masses[material_id] -= quantity
            self.wallet_balance += total
            self.transaction_history.append(total)
            return True, self.wallet_balance, f"Disposed {quantity}x of material {material_id}"

    def evaluate_economy_thermodynamics(self) -> Dict[str, Any]:
        vol = sum(self.transaction_history[-10:]) if self.transaction_history else 10.0
        admr_res = torch.tensor([min(2.0, vol / 500.0)])
        metrics = self.carnot_ledger(torch.tensor(1.0), torch.tensor(5.0), admr_res, depth_N=4)
        return {
            "eta_stack": metrics["eta_stack"].item(),
            "p_waste": metrics["p_waste"].item(),
            "thermal_runaway": bool(metrics["thermal_runaway"]),
            "market_friction": metrics["topological_friction"].item()
        }

    # ---------------------------------------------------------------------
    # Cleanroom Mechanics Hooks (Inside cleanroom_mechanics.py)
    # ---------------------------------------------------------------------
    def hook_cleanroom_radar(
        self,
        radar_radius: int = 64,
        show_death_fossils: bool = True,
        center_pos: Tuple[float, float, float] = (0.0, 64.0, 0.0)
    ) -> Dict[str, Any]:
        """
        Hook for JourneyMap Topological Radar.
        Scans subterranean entity blips and maintains death waypoints.
        """
        mock_entities = [
            {"name": "AlliedAgent", "pos": (center_pos[0] + 12.0, center_pos[1], center_pos[2] + 8.0), "is_hostile": False},
            {"name": "ResonanceAdversary", "pos": (center_pos[0] - 24.0, center_pos[1] - 10.0, center_pos[2] + 15.0), "is_hostile": True},
            {"name": "SubterraneanDrone", "pos": (center_pos[0] + 5.0, center_pos[1] - 30.0, center_pos[2] - 18.0), "is_hostile": True}
        ]
        blips = self.radar.tick_radar(
            entities=mock_entities,
            player_pos=center_pos,
            radar_radius=radar_radius,
            detect_subterranean=True
        )
        return {
            "radar_radius": radar_radius,
            "center_pos": center_pos,
            "blip_count": len(blips),
            "blips": blips,
            "waypoints": [wp.name for wp in self.radar.waypoints if not wp.is_death_point or show_death_fossils]
        }

    def hook_cleanroom_jer(
        self,
        material_id: int = 10,
        y_height: int = 16,
        looting_level: int = 0
    ) -> Dict[str, Any]:
        """
        Hook for Just Enough Resources (JER) ore distribution and loot inspection.
        Evaluates Chebyshev polynomial density curve at altitude y.
        """
        density = self.jer.get_ore_density_at_height(material_id=material_id, y=y_height)
        drops = self.jer.roll_harvest_debris(entity_class="bipedal_adversary", looting_level=looting_level)
        return {
            "material_id": material_id,
            "y_height": y_height,
            "ore_density": density,
            "looting_level": looting_level,
            "mob_drop_count": len(drops),
            "mob_drops": [{"item_id": d.item_id, "count": d.count} for d in drops]
        }

    def hook_cleanroom_multi_mine(
        self,
        coord: Tuple[int, int, int] = (10, 64, 10),
        tool_power: float = 0.35
    ) -> Dict[str, Any]:
        """
        Hook for AtomicStryker Multi-Mine progressive multi-block fracturing.
        Accumulates mining damage and breaks block into canonical inventory mass upon yield.
        """
        broken = self.multi_mine.mine_block(
            coord=coord,
            damage_delta=tool_power,
            octree=self.octree,
            graph=None,
            inventory=self.inventory
        )
        dmg = self.multi_mine.block_damage.get(coord, 0.0)
        return {
            "coord": coord,
            "damage": dmg,
            "broken": broken,
            "inventory_mass": dict(self.inventory.block_masses)
        }

    def hook_cleanroom_mekanism(
        self,
        tier: int = 2,
        raw_ore: int = 8,
        chemical_reagent_mb: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Hook for Mekanism 1x-5x metallurgical refining pipeline.
        Monitors thermodynamic waste and entropy via CarnotMobiusLedger.
        """
        tier = max(1, min(5, tier))
        reagent_remaining = chemical_reagent_mb
        if tier == 1:
            out_ingots = self.mekanism.process_tier_1_smelt(raw_ore)
        elif tier == 2:
            out_ingots = self.mekanism.process_tier_2_enrichment(raw_ore)
        elif tier == 3:
            out_ingots, reagent_remaining = self.mekanism.process_tier_3_purification(raw_ore, chemical_reagent_mb)
        elif tier == 4:
            out_ingots, reagent_remaining = self.mekanism.process_tier_4_chemical_injection(raw_ore, chemical_reagent_mb)
        else:
            out_ingots, reagent_remaining = self.mekanism.process_tier_5_chemical_dissolution(raw_ore, chemical_reagent_mb)

        admr = torch.tensor([float(tier * raw_ore) / 100.0])
        thermo = self.mekanism.carnot_ledger(torch.tensor(1.0), torch.tensor(tier * 2.0), admr, depth_N=tier)

        return {
            "tier": tier,
            "raw_ore_in": raw_ore,
            "ingots_out": out_ingots,
            "reagent_remaining_mb": reagent_remaining,
            "thermodynamic_eta": thermo["eta_stack"].item(),
            "p_waste": thermo["p_waste"].item(),
            "thermal_runaway": bool(thermo["thermal_runaway"])
        }

    def hook_cleanroom_sem_tech(
        self,
        tailings_count: int = 128,
        saltwater_mb: float = 5000.0,
        energy_fe: float = 1500.0
    ) -> Dict[str, Any]:
        """
        Hook for SEM TECH closed-loop saltwater + electricity tailings extraction.
        Electrodeposits precious metals (Au, Ag, PGMs), base metals (Cu, Ni), and rare earths.
        """
        res = self.sem_extractor.extract_from_tailings(
            tailings_count=tailings_count,
            saltwater_mb=saltwater_mb,
            energy_fe=energy_fe
        )
        return {
            "precious_metal_units": res.precious_metal_units,
            "base_metal_units": res.base_metal_units,
            "rare_earth_units": res.rare_earth_units,
            "tailings_consumed": res.tailings_consumed,
            "spent_energy_fe": res.spent_energy_fe,
            "recycled_saltwater_mb": res.recycled_saltwater_mb,
            "efficiency_ratio": res.efficiency_ratio,
            "membrane_durability": self.sem_extractor.membrane_durability
        }

    def hook_cleanroom_aeronautics(
        self,
        balloon_volume_m3: float = 1200.0,
        gas_type: str = "helium",
        propeller_rpm: float = 1200.0,
        dt: float = 0.05
    ) -> Dict[str, Any]:
        """
        Hook for Create: Aeronautics free-flying rigid contraption.
        Calculates Archimedes buoyant envelope lift, propeller thrust, and Drucker-Prager structural stress.
        """
        self.aeronautics.balloon_volume_m3 = balloon_volume_m3
        self.aeronautics.gas_type = gas_type
        self.aeronautics.engine.throttle = min(1.0, propeller_rpm / 2000.0)
        phys = self.aeronautics.update_physics_step(dt=dt, inventory=self.inventory)
        return phys

    def hook_cleanroom_nutrition(
        self,
        food_id: str = "hearty_stew",
        nutrition: float = 8.0,
        saturation: float = 12.0
    ) -> Dict[str, Any]:
        """
        Hook for Farmer's Delight & Spice of Life Carrot/Onion mechanics.
        Carrot: permanent max HP milestones for unique foods.
        Onion: rolling Shannon entropy buffs/malaise penalties.
        """
        res = self.nutrition.consume_food(food_id=food_id, nutrition=nutrition, saturation=saturation)
        return {
            "food_id": food_id,
            "unique_foods_eaten": len(self.nutrition.foods_eaten_unique),
            "bonus_hearts": res["bonus_hearts"],
            "max_hp": res["max_hp"],
            "shannon_entropy": res["shannon_entropy"],
            "diversity_buff_active": res["diversity_buff_active"],
            "monotony_penalty_active": res["monotony_penalty_active"]
        }

    def hook_cleanroom_conduits_rednet(
        self,
        channel_color: str = "red",
        signal_strength: int = 255,
        power_active: bool = True,
        fluid_active: bool = True
    ) -> Dict[str, Any]:
        """
        Hook for EnderIO composite multi-voxel conduits and RedNet 16-channel bundled cables.
        """
        self.rednet.set_signal(channel_color, signal_strength)
        sig = self.rednet.get_signal(channel_color)
        transferred_fe = 0.0
        transferred_mb = 0.0
        if power_active:
            transferred_fe = self.conduit.transfer_power(500.0)
        if fluid_active:
            transferred_mb = self.conduit.transfer_fluid(250.0)

        return {
            "channel_color": channel_color,
            "channel_signal": sig,
            "bundled_mask": self.rednet.get_bundled_signal_mask(),
            "transferred_fe": transferred_fe,
            "transferred_mb": transferred_mb,
            "conduit_fe_stored": self.conduit.power_fe_stored,
            "conduit_fluid_stored": self.conduit.fluid_mb_stored
        }

    def hook_cleanroom_logistics(
        self,
        belt_tier: int = 2,
        inserter_speed: float = 1.5,
        stack_size: int = 4,
        dt: float = 0.05
    ) -> Dict[str, Any]:
        """
        Hook for Factorio transport belts and directional inserters.
        """
        self.transport_belt.belt_tier = belt_tier
        self.inserter.swing_speed = inserter_speed
        self.inserter.stack_capacity = stack_size
        self.transport_belt.advance_items(dt)
        throughput = 15.0 * belt_tier
        inserter_rate = inserter_speed * stack_size
        return {
            "belt_tier": belt_tier,
            "belt_throughput_items_per_sec": throughput,
            "inserter_rate_items_per_sec": inserter_rate,
            "left_lane_items": len(self.transport_belt.left_lane),
            "right_lane_items": len(self.transport_belt.right_lane)
        }

    def hook_cleanroom_mob(
        self,
        base_hp: float = 40.0,
        attack_damage: float = 6.0,
        affix_1: str = "Bulwark",
        affix_2: str = "Berserk",
        subtype: EnemySubtype = EnemySubtype.NONE,
        ambulatory_class: AmbulatoryClass = AmbulatoryClass.BIPED
    ) -> Dict[str, Any]:
        """
        Hook for Mob Properties and AtomicStryker's Infernal Mobs affixes.
        Composes with AdaptiveSkeletonHarness morphology.
        """
        self.mob_properties.base_max_health = base_hp
        self.mob_properties.attack_damage = attack_damage
        self.mob_properties.subtype = subtype
        self.mob_properties.ambulatory = ambulatory_class
        self.mob_properties.affixes.clear()
        for aff_str in (affix_1, affix_2):
            if aff_str and aff_str != "None":
                try:
                    self.mob_properties.affixes.append(InfernalAffix(aff_str))
                except ValueError:
                    pass
        self.mob_properties.apply_subtype_mutations()
        return {
            "mob_id": self.mob_properties.mob_id,
            "subtype": self.mob_properties.subtype.name,
            "ambulatory": self.mob_properties.ambulatory.name,
            "effective_hp": self.mob_properties.effective_max_health,
            "attack_damage": self.mob_properties.attack_damage,
            "movement_speed": self.mob_properties.movement_speed,
            "active_affixes": [a.value for a in self.mob_properties.affixes]
        }

    def hook_cleanroom_fauna_genetics(
        self,
        mutation_rate: float = 0.05
    ) -> Dict[str, Any]:
        """
        Hook for Animal Husbandry Mendelian phenotypic inheritance.
        """
        self.fauna_genetics.mutation_rate = mutation_rate
        partner = FaunaGeneticsComponent(
            allele_speed=(1.2, 0.9),
            allele_jump=(1.1, 1.3),
            allele_yield=(1.4, 1.0),
            mutation_rate=mutation_rate
        )
        offspring = self.fauna_genetics.breed_with(partner)
        return {
            "parent_speed": self.fauna_genetics.expressed_speed,
            "parent_jump": self.fauna_genetics.expressed_jump,
            "parent_yield": self.fauna_genetics.expressed_yield,
            "offspring_speed": offspring.expressed_speed,
            "offspring_jump": offspring.expressed_jump,
            "offspring_yield": offspring.expressed_yield,
            "mutation_rate": mutation_rate
        }

    def hook_cleanroom_inventory(
        self,
        item_id: int = 1,
        count: int = 64
    ) -> Dict[str, Any]:
        """
        Hook for ExpandedInventorySystem (ItemStacks, FluidStacks, GasStacks).
        """
        stack = ItemStack(item_id=item_id, count=count)
        self.expanded_inventory.insert_item(stack)
        return {
            "total_slots": self.expanded_inventory.size,
            "used_slots": len(self.expanded_inventory.slots),
            "fluid_tanks": {k: v.amount_mb for k, v in self.expanded_inventory.fluid_tanks.items()},
            "gas_tanks": {k: v.amount_mb for k, v in self.expanded_inventory.gas_tanks.items()}
        }

    # ---------------------------------------------------------------------
    # Exterior Mechanics Hooks (Outside cleanroom_mechanics.py)
    # ---------------------------------------------------------------------
    def hook_chisel_octree(
        self,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        carve: bool = True,
        material_id: int = 1
    ) -> Dict[str, Any]:
        """
        Hook for Chisels & Bits Morton-encoded PointerlessOctree bit operations.
        """
        if carve:
            retrieved = self.chisel_carve_bit(x, y, z, material_id)
            action = "carve"
        else:
            self.chisel_place_bit(x, y, z, material_id)
            retrieved = material_id
            action = "place"
        return {
            "action": action,
            "coord": (x, y, z),
            "material_id": retrieved,
            "total_morton_bits": len(self.octree.morton_grid),
            "inventory_mass": dict(self.inventory.block_masses)
        }

    def hook_adaptive_rig(
        self,
        subtype: EnemySubtype = EnemySubtype.NONE,
        ambulatory_class: AmbulatoryClass = AmbulatoryClass.BIPED,
        difficulty_scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Hook for procedural character and enemy rig generation via AdaptiveSkeletonHarness.
        """
        return self.generate_enemy_rig(subtype, ambulatory_class, difficulty_scale)

    def hook_dual_yield_stress(
        self,
        pressure_val: float = 100.0,
        shear_val: float = 50.0,
        friction_angle: float = 30.0,
        cohesion: float = 20.0
    ) -> Dict[str, Any]:
        """
        Hook for geotechnical dual-regime plasticity (Mohr-Coulomb & Drucker-Prager).
        """
        mc = MohrCoulombProjection(friction_angle=friction_angle, cohesion=cohesion)
        dp = DruckerPragerProjection(alpha=0.25, k=cohesion * 2.0)
        p_tensor = torch.tensor([[pressure_val, shear_val, pressure_val * 0.5]])
        load_tensor = torch.tensor([[shear_val, pressure_val, shear_val * 0.5]])
        mc_out = mc(p_tensor, load_tensor)
        dp_out = dp(p_tensor, load_tensor)
        return {
            "mohr_coulomb_flow": mc_out.mean().item(),
            "drucker_prager_flow": dp_out.mean().item(),
            "pressure": pressure_val,
            "shear": shear_val
        }

    # ---------------------------------------------------------------------
    # Extended Cleanroom Systems Hooks
    # ---------------------------------------------------------------------
    def hook_create_rotational_network(
        self,
        capacity_su: float = 2048.0,
        stress_su: float = 512.0,
        rpm: float = 64.0
    ) -> Dict[str, Any]:
        """Hook for Create Mod Rotational Kinetic Network (Stress Units, RPM, Torques)."""
        from src.environment.cleanroom_mechanics import RotationalNode
        motor = RotationalNode(name="waterwheel", rpm=rpm, stress_capacity_su=capacity_su, direction=1)
        crusher = RotationalNode(name="crushing_wheel", rpm=rpm, stress_impact_su=stress_su / rpm if rpm > 0 else 0.0)
        self.rotational_net.add_node(motor)
        self.rotational_net.add_node(crusher)
        return self.rotational_net.compute_network_state()

    def hook_tconstruct_smeltery(
        self,
        tool_type: str = "pickaxe",
        head_material: str = "cobalt",
        handle_material: str = "wood",
        redstone_mod: int = 2,
        quartz_mod: int = 15
    ) -> Dict[str, Any]:
        """Hook for Tinkers' Construct Modular Tools and Smeltery Alloying."""
        from src.environment.cleanroom_mechanics import ToolMaterial, ToolPartType, MaterialTrait
        cobalt_trait = MaterialTrait(name="Lightweight", description="Swings faster", stat_multipliers={"speed_mult": 1.15})
        wood_trait = MaterialTrait(name="Ecological", description="Regenerates durability", stat_multipliers={})
        
        head = ToolMaterial(name=head_material, head_durability=800, mining_speed=12.0, attack_damage=5.0, harvest_level=4, traits=[cobalt_trait])
        handle = ToolMaterial(name=handle_material, head_durability=100, handle_modifier=1.1, traits=[wood_trait])
        
        self.modular_tool.tool_type = tool_type
        self.modular_tool.parts = {ToolPartType.HEAD: head, ToolPartType.HANDLE: handle}
        self.modular_tool.modifiers = {"redstone": redstone_mod, "quartz": quartz_mod}
        self.modular_tool.recalculate_stats()
        
        # Smeltery casting test
        self.smeltery.add_molten_fluid("molten_copper", 300.0)
        self.smeltery.add_molten_fluid("molten_tin", 100.0)
        bronze_cast = self.smeltery.cast_part("pickaxe_head", "molten_bronze", required_mb=288.0)
        
        return {
            "tool_type": self.modular_tool.tool_type,
            "max_durability": self.modular_tool.max_durability,
            "mining_speed": self.modular_tool.effective_mining_speed,
            "attack_damage": self.modular_tool.effective_attack_damage,
            "harvest_level": self.modular_tool.harvest_level,
            "cast_result": bronze_cast,
            "molten_tanks": dict(self.smeltery.molten_tank)
        }

    def hook_rustic_delight(
        self,
        raw_beans: int = 10,
        hot_water_mb: float = 1000.0,
        raw_cotton: int = 5
    ) -> Dict[str, Any]:
        """Hook for Rustic Delight (Coffee roasting & brewing, cotton ginning)."""
        roasted = self.rustic_delight.process_coffee_roasting(raw_beans)
        coffee_brew = self.rustic_delight.brew_coffee(roasted, hot_water_mb)
        strings, seeds = self.rustic_delight.gin_cotton(raw_cotton)
        return {
            "roasted_coffee_beans": roasted,
            "brewed_coffee": coffee_brew,
            "cotton_strings_yield": strings,
            "cotton_seeds_yield": seeds
        }

    def hook_jade_raycast(
        self,
        block_id: int = 50,
        block_name: str = "deepslate_iron_ore",
        crop_age: int = 4
    ) -> Dict[str, Any]:
        """Hook for Jade (WAILA/HWYLA) HUD raycast inspector."""
        data = self.jade.inspect_voxel(block_id=block_id, block_name=block_name, player_tool=self.modular_tool, crop_age=crop_age)
        return {
            "block_name": data.block_name,
            "harvest_tool": data.harvest_tool,
            "harvest_level": data.harvest_level,
            "can_harvest": data.can_harvest,
            "hardness": data.current_hardness,
            "crop_growth_percent": data.crop_growth_percent
        }

    def hook_ferritecore_fastmap(
        self,
        property_key: str = "facing",
        property_val: str = "north"
    ) -> Dict[str, Any]:
        """Hook for FerriteCore blockstate deduplication FastMap."""
        props = {property_key: property_val, "waterlogged": False, "powered": True}
        state_id = self.fast_map.intern_state(props)
        mask = self.fast_map.pack_neighbor_occlusion([True, False, True, False, False, True])
        return {
            "interned_state_id": state_id,
            "retrieved_props": self.fast_map.get_properties(state_id),
            "bitpacked_occlusion_mask": mask
        }

    def hook_waystones(
        self,
        action: str = "warp",
        current_pos: Tuple[int, int, int] = (0, 64, 0),
        target_name: str = "spawn",
        xp_level: int = 30
    ) -> Dict[str, Any]:
        """Hook for Waystones dimensional teleportation network."""
        from src.environment.cleanroom_mechanics import WaystoneNode
        self.waystones.activate_waystone("player_1", target_name)
        warp_res = self.waystones.warp_player(
            player_uuid="player_1",
            current_pos=current_pos,
            target_id=target_name,
            current_xp_level=xp_level
        )
        return warp_res

    def hook_lootr_containers(
        self,
        container_id: str = "dungeon_chest_1",
        player_uuid: str = "player_alpha",
        tier: int = 2
    ) -> Dict[str, Any]:
        """Hook for Lootr per-player unique container instancing."""
        loot_items = self.lootr.get_or_generate_loot(container_id, player_uuid, loot_tier=tier)
        return {
            "container_id": container_id,
            "player_uuid": player_uuid,
            "item_count": len(loot_items),
            "items": [{"id": item.item_id, "count": item.count, "name": item.metadata.display_name} for item in loot_items]
        }

    def hook_easy_anvils(
        self,
        current_dura: int = 50,
        max_dura: int = 250,
        ingots: int = 2,
        enchants: int = 3
    ) -> Dict[str, Any]:
        """Hook for Easy Anvils prior work penalty elimination."""
        from src.environment.cleanroom_mechanics import EasyAnvilsSystem
        return EasyAnvilsSystem.calculate_repair_cost(
            current_durability=current_dura,
            max_durability=max_dura,
            material_count=ingots,
            enchantment_count=enchants
        )

    def hook_all_the_heads(
        self,
        subtype: EnemySubtype = EnemySubtype.VOID_STALKER,
        ambulatory: AmbulatoryClass = AmbulatoryClass.BIPED,
        impact_impulse_J: float = 75.0,
        beheading_lvl: int = 2
    ) -> Dict[str, Any]:
        """Hook for Cranial Joint Shear rupture and trophy fossil collection."""
        from src.environment.cleanroom_mechanics import CranialJointShearRegistry
        head = CranialJointShearRegistry.roll_decapitation(
            subtype=subtype,
            ambulatory=ambulatory,
            impact_impulse_J=impact_impulse_J,
            beheading_level=beheading_lvl
        )
        return {
            "subtype": subtype.name if hasattr(subtype, 'name') else str(subtype),
            "ambulatory": ambulatory.name if hasattr(ambulatory, 'name') else str(ambulatory),
            "head_dropped": head is not None,
            "head_name": head.metadata.display_name if head else None
        }

    def hook_enchantment_industry(
        self,
        blank_books: int = 1,
        book_name: str = "Efficiency V",
        fluid_xp_mb: float = 500.0
    ) -> Dict[str, Any]:
        """Hook for Create: Enchantment Industry (liquid XP, book copying, hyper-enchanting)."""
        from src.environment.cleanroom_mechanics import ItemStack
        book_stack = ItemStack(item_id=340, count=blank_books)
        printed, cost = self.enchantment_industry.print_enchanted_book(book_stack, book_name, fluid_xp_mb)
        hyper_lvl, hyper_cost = self.enchantment_industry.hyper_enchant(5, 5, fluid_xp_mb - cost)
        return {
            "printed_book": printed.metadata.display_name if printed else None,
            "xp_consumed_mb": cost,
            "hyper_enchant_level": hyper_lvl,
            "hyper_cost_mb": hyper_cost
        }

    def hook_climate_rivers(
        self,
        biome_type: str = "alpine_rapids",
        slope: float = 0.05
    ) -> Dict[str, Any]:
        """Hook for Climate Rivers biome-specific hydrologic velocity vectors."""
        from src.environment.cleanroom_mechanics import ClimateRiverSegment, RiverBiomeType
        b_enum = getattr(RiverBiomeType, biome_type.upper(), RiverBiomeType.ALPINE_RAPIDS)
        seg = ClimateRiverSegment(biome_type=b_enum, slope_gradient=slope)
        vec = seg.compute_flow_vector()
        return {
            "biome": b_enum.value,
            "slope": slope,
            "flow_velocity_vector_mps": vec,
            "channel_width_m": seg.channel_width_m
        }

    def hook_combat_nouveau(
        self,
        category_str: str = "sword",
        charge_ratio: float = 1.0,
        is_crit: bool = True
    ) -> Dict[str, Any]:
        """Hook for Combat Nouveau (Jeb combat test: weapon reach, charge, sweep interrupt)."""
        from src.environment.cleanroom_mechanics import CombatNouveauProfile, WeaponCategory
        w_enum = getattr(WeaponCategory, category_str.upper(), WeaponCategory.SWORD)
        profile = CombatNouveauProfile.create(w_enum)
        return profile.calculate_attack_strike(charge_ratio=charge_ratio, is_critical=is_crit)

    def hook_hotbar_swapper(
        self,
        swap_row: int = 1
    ) -> Dict[str, Any]:
        """Hook for Hotbar Swapper / Hotbar Keybinds inventory paging."""
        self.hotbar_swapper.swap_with_row(swap_row - 1)
        return {
            "active_hotbar_page": self.hotbar_swapper.active_page,
            "hotbar_slots_occupied": sum(1 for s in self.hotbar_swapper.hotbar if s is not None)
        }

    def hook_projecte_emc(
        self,
        action: str = "burn",
        item_id: str = "diamond",
        count: int = 2,
        player_uuid: str = "player_alpha"
    ) -> Dict[str, Any]:
        """Hook for ProjectE Equivalent Exchange recursive EMC solver."""
        if action == "burn":
            gain = self.project_e.burn_item_for_emc(player_uuid, item_id, count)
            return {
                "action": "burn",
                "emc_gained": gain,
                "current_stored_emc": self.project_e.player_emc.get(player_uuid, 0),
                "learned_items_count": len(self.project_e.learned_items.get(player_uuid, set()))
            }
        else:
            transmuted = self.project_e.transmute_item(player_uuid, item_id, count)
            return {
                "action": "transmute",
                "item_transmuted": item_id,
                "count": transmuted,
                "current_stored_emc": self.project_e.player_emc.get(player_uuid, 0)
            }

    def hook_cauldron_brewing(
        self,
        reagent: str = "nether_wart",
        stir_dir: bool = True,
        heat_active: bool = True
    ) -> Dict[str, Any]:
        """Hook for Diegetic Cauldron Brewing pipeline."""
        self.cauldron.heat_source_active = heat_active
        self.cauldron.heat_tick(dt=2.0)
        self.cauldron.stir(direction_cw=stir_dir)
        state = self.cauldron.add_reagent(reagent)
        return {
            "cauldron_temp_k": self.cauldron.temperature_k,
            "brew_state": state,
            "stir_count": self.cauldron.stir_count,
            "reagents": list(self.cauldron.added_reagents)
        }

    def hook_multipart_subgrid(
        self,
        side: str = "north",
        subgrid_yaw: float = 45.0
    ) -> Dict[str, Any]:
        """Hook for Multipart voxel cells (vertical slabs) & subgrid bundled wire transforms."""
        from src.environment.cleanroom_mechanics import MultipartVoxelCell
        cell = MultipartVoxelCell(coord=(0, 64, 0))
        placed = cell.place_vertical_slab(side=side, material_id=1)
        wire_vec = self.wire_harness.transform_signal_vector((1.0, 0.0, 0.0), subgrid_yaw_deg=subgrid_yaw)
        return {
            "vertical_slab_placed": placed,
            "sub_parts_count": len(cell.sub_parts),
            "subgrid_wire_world_vector": wire_vec
        }

    def hook_bag_of_holding(
        self,
        is_nested_bag: bool = False
    ) -> Dict[str, Any]:
        """Hook for Bag of Holding dimensional storage & void safeguard."""
        from src.environment.cleanroom_mechanics import ItemStack, ItemMetadata
        item = ItemStack(item_id=999, metadata=ItemMetadata(custom_nbt={"is_bag_of_holding": is_nested_bag}))
        res = self.bag_of_holding.insert_item(item)
        return {
            "bag_tier": self.bag_of_holding.tier.name,
            "insert_result": res,
            "is_void_collapsed": self.bag_of_holding.is_void_collapsed
        }

    def hook_crop_breeding_giant(
        self,
        action: str = "breed",
        crop_species: str = "wheat",
        g1: int = 10,
        g2: int = 8
    ) -> Dict[str, Any]:
        """Hook for IC2 crop breeding genetics & 3x3 giant crop multi-blocks."""
        from src.environment.cleanroom_mechanics import IC2CropGenome
        partner = IC2CropGenome(crop_species=crop_species, growth=g2, gain=g2, resistance=g1)
        offspring = self.crop_genome.cross_breed(partner)
        
        # Test 3x3 giant crop
        grid = {(x, z): 7 for x in (-1, 0, 1) for z in (-1, 0, 1)}
        can_fuse = self.giant_crop.check_and_fuse_3x3(grid, 0, 0)
        return {
            "parent_growth": self.crop_genome.growth,
            "offspring_growth": offspring.growth,
            "offspring_gain": offspring.gain,
            "offspring_resistance": offspring.resistance,
            "can_fuse_3x3_giant_crop": can_fuse
        }


    # ---------------------------------------------------------------------
    # Continuous Headless Tick
    # ---------------------------------------------------------------------
    def evaluate_tick(
        self,
        dt: float = 0.05,
        current_pressure: float = 0.5,
        current_mischief: float = 0.2,
        radar_radius: int = 64,
        show_death: bool = True,
        jer_y: int = 16,
        jer_looting: int = 0,
        mek_tier: int = 2,
        mek_ore: int = 8,
        mek_reagent: float = 1000.0,
        sem_tailings: int = 128,
        sem_saltwater: float = 5000.0,
        sem_energy: float = 1500.0,
        sem_membrane: float = 1.0,
        aero_volume: float = 1200.0,
        aero_gas: str = "helium",
        aero_rpm: float = 1200.0,
        food_id: str = "hearty_stew",
        food_nutr: float = 8.0,
        food_sat: float = 12.0,
        rednet_color: str = "red",
        rednet_signal: int = 255,
        conduit_power: bool = True,
        conduit_fluid: bool = True,
        belt_tier: int = 2,
        inserter_speed: float = 1.5,
        inserter_stack: int = 4,
        mob_hp: float = 40.0,
        mob_dmg: float = 6.0,
        affix_1: str = "Bulwark",
        affix_2: str = "Berserk",
        ambulatory_class: AmbulatoryClass = AmbulatoryClass.BIPED,
        subtype: EnemySubtype = EnemySubtype.NONE,
        enemy_diff: float = 1.0,
        mm_power: float = 0.25,
        fauna_mut: float = 0.05,
        chisel_coord: Tuple[int, int, int] = (0, 0, 0),
        dy_pressure: float = 100.0,
        dy_shear: float = 50.0
    ) -> Dict[str, Any]:
        veh = self.tick_vehicle_kinetics(dt)
        life = self.tick_life_and_hunger(dt, current_pressure=current_pressure, current_mischief=current_mischief)
        econ = self.evaluate_economy_thermodynamics()
        self.master_group.evaluate()

        # Hooks inside cleanroom_mechanics
        radar_res = self.hook_cleanroom_radar(radar_radius=radar_radius, show_death_fossils=show_death)
        jer_res = self.hook_cleanroom_jer(material_id=10, y_height=jer_y, looting_level=jer_looting)
        mm_res = self.hook_cleanroom_multi_mine(coord=(10, 64, 10), tool_power=mm_power)
        mek_res = self.hook_cleanroom_mekanism(tier=mek_tier, raw_ore=mek_ore, chemical_reagent_mb=mek_reagent)
        self.sem_extractor.membrane_durability = sem_membrane
        sem_res = self.hook_cleanroom_sem_tech(tailings_count=sem_tailings, saltwater_mb=sem_saltwater, energy_fe=sem_energy)
        aero_res = self.hook_cleanroom_aeronautics(balloon_volume_m3=aero_volume, gas_type=aero_gas, propeller_rpm=aero_rpm, dt=dt)
        nutr_res = self.hook_cleanroom_nutrition(food_id=food_id, nutrition=food_nutr, saturation=food_sat)
        conduit_res = self.hook_cleanroom_conduits_rednet(channel_color=rednet_color, signal_strength=rednet_signal, power_active=conduit_power, fluid_active=conduit_fluid)
        logistics_res = self.hook_cleanroom_logistics(belt_tier=belt_tier, inserter_speed=inserter_speed, stack_size=inserter_stack, dt=dt)
        mob_res = self.hook_cleanroom_mob(base_hp=mob_hp, attack_damage=mob_dmg, affix_1=affix_1, affix_2=affix_2, subtype=subtype, ambulatory_class=ambulatory_class)
        fauna_res = self.hook_cleanroom_fauna_genetics(mutation_rate=fauna_mut)
        inv_res = self.hook_cleanroom_inventory(item_id=1, count=1)

        # Hooks outside cleanroom_mechanics
        chisel_res = self.hook_chisel_octree(x=chisel_coord[0], y=chisel_coord[1], z=chisel_coord[2], carve=True)
        rig_res = self.hook_adaptive_rig(subtype=subtype, ambulatory_class=ambulatory_class, difficulty_scale=enemy_diff)
        dy_res = self.hook_dual_yield_stress(pressure_val=dy_pressure, shear_val=dy_shear)

        # Extended Cleanroom Systems
        rot_res = self.hook_create_rotational_network()
        tcon_res = self.hook_tconstruct_smeltery()
        rustic_res = self.hook_rustic_delight()
        jade_res = self.hook_jade_raycast()
        fc_res = self.hook_ferritecore_fastmap()
        way_res = self.hook_waystones()
        lootr_res = self.hook_lootr_containers()
        anvil_res = self.hook_easy_anvils()
        heads_res = self.hook_all_the_heads()
        ench_res = self.hook_enchantment_industry()
        river_res = self.hook_climate_rivers()
        combat_res = self.hook_combat_nouveau()
        hotbar_res = self.hook_hotbar_swapper()
        emc_res = self.hook_projecte_emc()
        brew_res = self.hook_cauldron_brewing()
        multi_sub_res = self.hook_multipart_subgrid()
        bag_res = self.hook_bag_of_holding()
        crop_res = self.hook_crop_breeding_giant()

        return {
            "vehicle": veh,
            "life": life,
            "economy": econ,
            "master_group_cog": self.master_group.outputs["out_cog_offset"].value,
            "radar": radar_res,
            "jer": jer_res,
            "multi_mine": mm_res,
            "mekanism": mek_res,
            "sem_tech": sem_res,
            "aeronautics": aero_res,
            "nutrition": nutr_res,
            "conduit_rednet": conduit_res,
            "logistics": logistics_res,
            "mob": mob_res,
            "fauna_genetics": fauna_res,
            "inventory": inv_res,
            "chisels_octree": chisel_res,
            "adaptive_rig": rig_res,
            "dual_yield": dy_res,
            "rotational_network": rot_res,
            "tconstruct": tcon_res,
            "rustic_delight": rustic_res,
            "jade": jade_res,
            "ferritecore": fc_res,
            "waystones": way_res,
            "lootr": lootr_res,
            "easy_anvils": anvil_res,
            "all_the_heads": heads_res,
            "enchantment_industry": ench_res,
            "climate_rivers": river_res,
            "combat_nouveau": combat_res,
            "hotbar_swapper": hotbar_res,
            "projecte_emc": emc_res,
            "cauldron_brewing": brew_res,
            "multipart_subgrid": multi_sub_res,
            "bag_of_holding": bag_res,
            "crop_breeding_giant": crop_res
        }

    # ---------------------------------------------------------------------
    # DearPyGui Node Setup & Evaluation Worker
    # ---------------------------------------------------------------------
    def _link_callback(self, sender, app_data):
        dpg.add_node_link(app_data[0], app_data[1], parent=sender)
        self.virtual_links.append((app_data[0], app_data[1]))
        logger.info(f"[Node Editor] Virtual link established: {app_data[0]} -> {app_data[1]}")

    def _delink_callback(self, sender, app_data):
        dpg.delete_item(app_data)
        self.virtual_links = [l for l in self.virtual_links if l != app_data]

    def _setup_nodes(self):
        with dpg.window(label="Gyroidic Node Scripting - Dataflow Console", width=1200, height=800):
            with dpg.node_editor(callback=self._link_callback, delink_callback=self._delink_callback, id="node_editor"):
                
                # 1. BSpline Mod Generator
                with dpg.node(label="BSpline Mod Generator", tag="node_bspline"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_bspline_out"):
                        dpg.add_text("BSpline Tensor Out (Vector)")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Latent Dim", default_value=3, min_value=1, max_value=10, tag="slider_latent_dim")
                        dpg.add_slider_int(label="Resolution", default_value=20, min_value=5, max_value=100, tag="slider_resolution")
                        
                # 2. Dark Matter Attractor
                with dpg.node(label="Dark Matter Attractor", tag="node_dark_matter"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_dark_matter_out"):
                        dpg.add_text("Dark Matter Fossil Out")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Iterations", default_value=500, min_value=100, max_value=2000, tag="slider_iterations")
                        dpg.add_slider_float(label="Time Step", default_value=0.05, min_value=0.01, max_value=0.2, tag="slider_dt")
                
                # 3. Global Physical Properties
                with dpg.node(label="Global Physical Properties", tag="node_physics"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Mass Cost Mod", default_value=1.0, min_value=0.1, max_value=5.0, tag="slider_mass_cost")
                        dpg.add_slider_float(label="Topological Persistence", default_value=0.5, min_value=0.1, max_value=1.0, tag="slider_topo_persist")
                        dpg.add_slider_float(label="Resonance Frequency (Hz)", default_value=432.0, min_value=1.0, max_value=1000.0, tag="slider_resonance")
                        dpg.add_slider_float(label="Quantum Tunnel Prob", default_value=0.05, min_value=0.0, max_value=1.0, tag="slider_quantum_tunnel")

                # 4. Master Node Group (Blender-Style Attribute Injector)
                with dpg.node(label="Master Node Group: Attribute Injector", tag="node_master_group"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="attr_mesh_in"):
                        dpg.add_text("Mesh Geometry In (Vector)")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="phys_hardness", default_value=10.0, min_value=1.0, max_value=50.0, tag="slider_phys_hardness")
                        dpg.add_slider_float(label="phys_friction", default_value=0.85, min_value=0.0, max_value=1.0, tag="slider_phys_friction")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_mesh_out"):
                        dpg.add_text("Baked Attributes Out (Vector/COG)")

                # 5. Vehicle Subsystems (Plasma Air Induction & Sleeve Timings)
                with dpg.node(label="Vehicle Subsystems & Flight", tag="node_vehicle_components"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="attr_vehicle_in"):
                        dpg.add_text("Chassis Mass In (Float)")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Engine Throttle", default_value=0.5, min_value=0.0, max_value=1.0, tag="slider_engine_throttle")
                        dpg.add_slider_float(label="Sleeve Timing Deg", default_value=12.5, min_value=0.0, max_value=30.0, tag="slider_sleeve_timing")
                        dpg.add_checkbox(label="Plasma Air Induction", default_value=True, tag="check_plasma_air")
                        dpg.add_checkbox(label="Reinforced Chassis", default_value=False, tag="check_chassis_reinforced")
                        dpg.add_checkbox(label="Energy Harvester", default_value=False, tag="check_harvester")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_vehicle_out"):
                        dpg.add_text("Kinematics & Delta-v Out (Vector)", tag="txt_vehicle_status")

                # 6. Character Rig & Enemy Subtype Generator
                with dpg.node(label="Character Rig & Enemy Subtypes", tag="node_character_enemy"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="attr_character_in"):
                        dpg.add_text("Archetype Seed In")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_combo(label="Ambulatory Class", items=[c.name for c in AmbulatoryClass], default_value="BIPED", tag="combo_ambulatory_class")
                        dpg.add_combo(label="Enemy Subtype", items=[s.name for s in EnemySubtype], default_value="NONE", tag="combo_enemy_subtype")
                        dpg.add_slider_float(label="Difficulty Scale", default_value=1.0, min_value=0.5, max_value=3.0, tag="slider_enemy_diff")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_character_out"):
                        dpg.add_text("Rigged Skeleton Out (Vector)", tag="txt_character_status")

                # 7. Admin Shop & Carnot-Leontief Economy
                with dpg.node(label="Admin Shop & Economy Sim", tag="node_economy_shop"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="attr_economy_in"):
                        dpg.add_text("Transaction Demand In")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_combo(label="Role Select", items=[r.name for r in Role], default_value="VISITOR", tag="combo_role_select")
                        dpg.add_slider_float(label="Admin Margin", default_value=0.15, min_value=0.0, max_value=0.5, tag="slider_admin_margin")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_economy_out"):
                        dpg.add_text("Market Clearance Out", tag="txt_economy_status")

                # 8. Voxelboxter Sink Node
                with dpg.node(label="Voxelboxter Graph Sink", tag="node_sink"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="attr_sink_in"):
                        dpg.add_text("Compiled Mod In")

                # 9. JourneyMap Topological Radar & Waypoints
                with dpg.node(label="JourneyTopoRadar (JourneyMap)", tag="node_journeymap"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Radar Radius", default_value=64, min_value=16, max_value=256, tag="slider_radar_radius")
                        dpg.add_checkbox(label="Show Death Fossils", default_value=True, tag="check_radar_death")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_radar_out"):
                        dpg.add_text("Entity Radar Blips (Vector)", tag="txt_radar_status")

                # 10. Just Enough Resources (JER) Inspector
                with dpg.node(label="Resource Distribution (JER)", tag="node_jer"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Inspect Y Height", default_value=16, min_value=-64, max_value=320, tag="slider_jer_y")
                        dpg.add_slider_int(label="Looting Level", default_value=0, min_value=0, max_value=5, tag="slider_jer_looting")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_jer_out"):
                        dpg.add_text("Ore Density & Drop Odds (Float)", tag="txt_jer_status")

                # 11. Mekanism Multi-Tier Metallurgical Refinery
                with dpg.node(label="Mekanism Ore Refinery (1x-5x)", tag="node_mekanism"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Processing Tier", default_value=2, min_value=1, max_value=5, tag="slider_mek_tier")
                        dpg.add_slider_int(label="Raw Ore In", default_value=8, min_value=1, max_value=64, tag="slider_mek_ore")
                        dpg.add_slider_float(label="Chemical Reagent (mB)", default_value=1000.0, min_value=0.0, max_value=5000.0, tag="slider_mek_reagent")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_mek_out"):
                        dpg.add_text("Refined Ingots Out (Int)", tag="txt_mek_status")

                # 11b. SEM TECH Saltwater Electrolyzer (Rowow Open-Source CMU)
                with dpg.node(label="SEM Tech Electrolyzer (Saltwater + Electricity)", tag="node_sem_tech"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Tailings / Gangue Stockpile", default_value=128, min_value=1, max_value=2048, tag="slider_sem_tailings")
                        dpg.add_slider_float(label="Saltwater Saline (mB)", default_value=5000.0, min_value=100.0, max_value=20000.0, tag="slider_sem_saltwater")
                        dpg.add_slider_float(label="Electricity (FE)", default_value=1500.0, min_value=100.0, max_value=10000.0, tag="slider_sem_energy")
                        dpg.add_slider_float(label="Membrane Durability", default_value=1.0, min_value=0.05, max_value=1.0, tag="slider_sem_membrane")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_sem_out"):
                        dpg.add_text("Precious Metals & Critical Minerals (Powder/Foils)", tag="txt_sem_status")

                # 12. Create: Aeronautics Airship Buoyancy & Flight
                with dpg.node(label="Create: Aeronautics Contraption", tag="node_aeronautics"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Balloon Volume (m3)", default_value=1200.0, min_value=100.0, max_value=5000.0, tag="slider_aero_volume")
                        dpg.add_combo(label="Buoyant Gas", items=["helium", "hot_air"], default_value="helium", tag="combo_aero_gas")
                        dpg.add_slider_float(label="Propeller RPM", default_value=1200.0, min_value=0.0, max_value=3000.0, tag="slider_aero_rpm")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_aero_out"):
                        dpg.add_text("Net Lift & Thrust (Vector)", tag="txt_aero_status")

                # 13. Farmer's Delight & Spice of Life Nutrition
                with dpg.node(label="Spice of Life & Nutrition", tag="node_nutrition"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_input_text(label="Food Item ID", default_value="hearty_stew", tag="text_food_id")
                        dpg.add_slider_float(label="Nutrition", default_value=8.0, min_value=1.0, max_value=20.0, tag="slider_food_nutr")
                        dpg.add_slider_float(label="Saturation", default_value=12.0, min_value=1.0, max_value=20.0, tag="slider_food_sat")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_nutrition_out"):
                        dpg.add_text("Max HP & Dynamic Buffs", tag="txt_nutrition_status")

                # 14. EnderIO Conduit & 16-Color RedNet
                with dpg.node(label="EnderIO & RedNet Bundled Cable", tag="node_conduit_rednet"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_combo(label="RedNet Subnet Color", items=["white", "red", "blue", "green", "black"], default_value="red", tag="combo_rednet_color")
                        dpg.add_slider_int(label="Channel Signal (0-255)", default_value=255, min_value=0, max_value=255, tag="slider_rednet_signal")
                        dpg.add_checkbox(label="Power Conduit Active", default_value=True, tag="check_conduit_power")
                        dpg.add_checkbox(label="Fluid Conduit Active", default_value=True, tag="check_conduit_fluid")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_conduit_out"):
                        dpg.add_text("Bundled Bus Signal (Int)", tag="txt_conduit_status")

                # 15. Factorio Transport Belts & Inserters
                with dpg.node(label="Factorio Logistics Network", tag="node_factorio"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Belt Tier", default_value=2, min_value=1, max_value=3, tag="slider_belt_tier")
                        dpg.add_slider_float(label="Inserter Swing Speed", default_value=1.5, min_value=0.5, max_value=5.0, tag="slider_inserter_speed")
                        dpg.add_slider_int(label="Stack Size", default_value=4, min_value=1, max_value=12, tag="slider_inserter_stack")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_factorio_out"):
                        dpg.add_text("Logistics Throughput (Items/s)", tag="txt_factorio_status")

                # 16. Adversary Morphology & Affix Invariants
                with dpg.node(label="Adversary Morphology & Affix Invariants", tag="node_infernal_mobs"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Base HP", default_value=40.0, min_value=10.0, max_value=200.0, tag="slider_mob_hp")
                        dpg.add_slider_float(label="Attack Damage", default_value=6.0, min_value=1.0, max_value=30.0, tag="slider_mob_damage")
                        dpg.add_combo(label="Affix Invariant 1", items=["1UP", "Berserk", "Bulwark", "Lifesteal", "Storm", "Webbing", "Alchemist", "Rust"], default_value="Bulwark", tag="combo_affix_1")
                        dpg.add_combo(label="Affix Invariant 2", items=["None", "1UP", "Berserk", "Blastoff", "Fiery", "Regen", "Sprint"], default_value="Berserk", tag="combo_affix_2")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_mob_out"):
                        dpg.add_text("Evaluated Adversary Rig Out", tag="txt_mob_status")

                # 17. AtomicStryker Multi-Mine Progressive Fracture
                with dpg.node(label="Multi-Mine Progressive Fracture", tag="node_multi_mine"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Tool Mining Power", default_value=0.35, min_value=0.05, max_value=1.0, tag="slider_mm_power")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_multi_mine_out"):
                        dpg.add_text("Fracture Damage & Yield", tag="txt_multi_mine_status")

                # 18. Animal Husbandry & Mendelian Genetics
                with dpg.node(label="Animal Husbandry & Genetics", tag="node_fauna_genetics"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Mutation Rate", default_value=0.05, min_value=0.01, max_value=0.5, tag="slider_fauna_mut_rate")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_fauna_genetics_out"):
                        dpg.add_text("Expressed Alleles Out", tag="txt_fauna_genetics_status")

                # 19. Chisels & Bits Micro-Octree Carving
                with dpg.node(label="Chisels & Bits Micro-Octree", tag="node_chisels_octree"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Bit X", default_value=0, min_value=0, max_value=15, tag="slider_chisel_x")
                        dpg.add_slider_int(label="Bit Y", default_value=0, min_value=0, max_value=15, tag="slider_chisel_y")
                        dpg.add_slider_int(label="Bit Z", default_value=0, min_value=0, max_value=15, tag="slider_chisel_z")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_chisels_out"):
                        dpg.add_text("Morton Spatial Hash Out", tag="txt_chisels_status")

                # 20. Geotechnical Dual-Yield Plasticity
                with dpg.node(label="Geotechnical Dual-Yield (MC & DP)", tag="node_dual_yield"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Hydrostatic Pressure", default_value=100.0, min_value=10.0, max_value=500.0, tag="slider_dy_pressure")
                        dpg.add_slider_float(label="Shear Stress", default_value=50.0, min_value=5.0, max_value=300.0, tag="slider_dy_shear")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_dual_yield_out"):
                        dpg.add_text("Plastic Flow & Rupture", tag="txt_dual_yield_status")

    def _evaluation_worker(self):
        """
        D-Wave Collective Computing Pool
        Asynchronously evaluates the graph to prevent DSP processing from
        blocking the UI thread. Educated by left-of-field SOMA mechanics.
        """
        import time
        import torch
        from src.core.structural_blueprints import MangostienBSplineMod, DarkMatterAttractorLayer
        from src.core.honest_jitter import harvest_honest_jitter
        from src.core.hardware_monitor import has_headroom
        from src.p2p.bonfire_consensus import BonfireNomadicRing
        from src.p2p.freenet_ws_client import FreenetClient
        from src.p2p.zk_aggregator import ZKAggregator
        from src.surrogates.calm_predictor import CALM

        try:
            dummy_freenet = FreenetClient(ws_url="ws://localhost:8080/bonfire")
            bonfire = BonfireNomadicRing(freenet_client=dummy_freenet)
        except Exception:
            bonfire = None
            
        zk_agg = ZKAggregator()
        calm = CALM(dim=8)
        history_buffer = torch.zeros(1, 8, 8) 
        
        while self.running:
            if not has_headroom():
                time.sleep(0.5)
                continue
                
            time.sleep(0.5) 
            
            try:
                latent_dim = dpg.get_value("slider_latent_dim") if dpg.does_item_exist("slider_latent_dim") else 3
                resolution = dpg.get_value("slider_resolution") if dpg.does_item_exist("slider_resolution") else 20
                iterations = dpg.get_value("slider_iterations") if dpg.does_item_exist("slider_iterations") else 500
                dt = dpg.get_value("slider_dt") if dpg.does_item_exist("slider_dt") else 0.05
                mass_cost = dpg.get_value("slider_mass_cost") if dpg.does_item_exist("slider_mass_cost") else 1.0
                topo_persist = dpg.get_value("slider_topo_persist") if dpg.does_item_exist("slider_topo_persist") else 0.5
                resonance = dpg.get_value("slider_resonance") if dpg.does_item_exist("slider_resonance") else 432.0
                quantum_tunnel = dpg.get_value("slider_quantum_tunnel") if dpg.does_item_exist("slider_quantum_tunnel") else 0.05
                current_links = list(self.virtual_links)
                
                # Sync GUI parameters to canonical engines
                if dpg.does_item_exist("slider_engine_throttle"):
                    self.engine.throttle = dpg.get_value("slider_engine_throttle")
                if dpg.does_item_exist("slider_sleeve_timing"):
                    self.engine.electrical_timing_degrees = dpg.get_value("slider_sleeve_timing")
                if dpg.does_item_exist("check_plasma_air"):
                    self.battery.plasma_air_induction = dpg.get_value("check_plasma_air")
                if dpg.does_item_exist("check_chassis_reinforced"):
                    self.rigid_body.reinforced_alloy_chassis = dpg.get_value("check_chassis_reinforced")
                if dpg.does_item_exist("check_harvester"):
                    self.rigid_body.impact_energy_harvester = dpg.get_value("check_harvester")
                if dpg.does_item_exist("slider_admin_margin"):
                    self.admin_margin = dpg.get_value("slider_admin_margin")

                # Cleanroom GUI parameters
                radar_radius = dpg.get_value("slider_radar_radius") if dpg.does_item_exist("slider_radar_radius") else 64
                show_death = dpg.get_value("check_radar_death") if dpg.does_item_exist("check_radar_death") else True
                jer_y = dpg.get_value("slider_jer_y") if dpg.does_item_exist("slider_jer_y") else 16
                jer_looting = dpg.get_value("slider_jer_looting") if dpg.does_item_exist("slider_jer_looting") else 0
                mek_tier = dpg.get_value("slider_mek_tier") if dpg.does_item_exist("slider_mek_tier") else 2
                mek_ore = dpg.get_value("slider_mek_ore") if dpg.does_item_exist("slider_mek_ore") else 8
                mek_reagent = dpg.get_value("slider_mek_reagent") if dpg.does_item_exist("slider_mek_reagent") else 1000.0
                sem_tailings = dpg.get_value("slider_sem_tailings") if dpg.does_item_exist("slider_sem_tailings") else 128
                sem_saltwater = dpg.get_value("slider_sem_saltwater") if dpg.does_item_exist("slider_sem_saltwater") else 5000.0
                sem_energy = dpg.get_value("slider_sem_energy") if dpg.does_item_exist("slider_sem_energy") else 1500.0
                sem_membrane = dpg.get_value("slider_sem_membrane") if dpg.does_item_exist("slider_sem_membrane") else 1.0
                aero_volume = dpg.get_value("slider_aero_volume") if dpg.does_item_exist("slider_aero_volume") else 1200.0
                aero_gas = dpg.get_value("combo_aero_gas") if dpg.does_item_exist("combo_aero_gas") else "helium"
                aero_rpm = dpg.get_value("slider_aero_rpm") if dpg.does_item_exist("slider_aero_rpm") else 1200.0
                food_id = dpg.get_value("text_food_id") if dpg.does_item_exist("text_food_id") else "hearty_stew"
                food_nutr = dpg.get_value("slider_food_nutr") if dpg.does_item_exist("slider_food_nutr") else 8.0
                food_sat = dpg.get_value("slider_food_sat") if dpg.does_item_exist("slider_food_sat") else 12.0
                rednet_color = dpg.get_value("combo_rednet_color") if dpg.does_item_exist("combo_rednet_color") else "red"
                rednet_signal = dpg.get_value("slider_rednet_signal") if dpg.does_item_exist("slider_rednet_signal") else 255
                conduit_power = dpg.get_value("check_conduit_power") if dpg.does_item_exist("check_conduit_power") else True
                conduit_fluid = dpg.get_value("check_conduit_fluid") if dpg.does_item_exist("check_conduit_fluid") else True
                belt_tier = dpg.get_value("slider_belt_tier") if dpg.does_item_exist("slider_belt_tier") else 2
                inserter_speed = dpg.get_value("slider_inserter_speed") if dpg.does_item_exist("slider_inserter_speed") else 1.5
                inserter_stack = dpg.get_value("slider_inserter_stack") if dpg.does_item_exist("slider_inserter_stack") else 4
                mob_hp = dpg.get_value("slider_mob_hp") if dpg.does_item_exist("slider_mob_hp") else 40.0
                mob_dmg = dpg.get_value("slider_mob_damage") if dpg.does_item_exist("slider_mob_damage") else 6.0
                affix_1 = dpg.get_value("combo_affix_1") if dpg.does_item_exist("combo_affix_1") else "Bulwark"
                affix_2 = dpg.get_value("combo_affix_2") if dpg.does_item_exist("combo_affix_2") else "Berserk"
                amb_str = dpg.get_value("combo_ambulatory_class") if dpg.does_item_exist("combo_ambulatory_class") else "BIPED"
                sub_str = dpg.get_value("combo_enemy_subtype") if dpg.does_item_exist("combo_enemy_subtype") else "NONE"
                enemy_diff = dpg.get_value("slider_enemy_diff") if dpg.does_item_exist("slider_enemy_diff") else 1.0
                mm_power = dpg.get_value("slider_mm_power") if dpg.does_item_exist("slider_mm_power") else 0.35
                fauna_mut = dpg.get_value("slider_fauna_mut_rate") if dpg.does_item_exist("slider_fauna_mut_rate") else 0.05
                chisel_x = dpg.get_value("slider_chisel_x") if dpg.does_item_exist("slider_chisel_x") else 0
                chisel_y = dpg.get_value("slider_chisel_y") if dpg.does_item_exist("slider_chisel_y") else 0
                chisel_z = dpg.get_value("slider_chisel_z") if dpg.does_item_exist("slider_chisel_z") else 0
                dy_p = dpg.get_value("slider_dy_pressure") if dpg.does_item_exist("slider_dy_pressure") else 100.0
                dy_tau = dpg.get_value("slider_dy_shear") if dpg.does_item_exist("slider_dy_shear") else 50.0

                amb_enum = getattr(AmbulatoryClass, amb_str, AmbulatoryClass.BIPED)
                sub_enum = getattr(EnemySubtype, sub_str, EnemySubtype.NONE)

                sim_res = self.evaluate_tick(
                    dt=dt,
                    radar_radius=radar_radius,
                    show_death=show_death,
                    jer_y=jer_y,
                    jer_looting=jer_looting,
                    mek_tier=mek_tier,
                    mek_ore=mek_ore,
                    mek_reagent=mek_reagent,
                    sem_tailings=sem_tailings,
                    sem_saltwater=sem_saltwater,
                    sem_energy=sem_energy,
                    sem_membrane=sem_membrane,
                    aero_volume=aero_volume,
                    aero_gas=aero_gas,
                    aero_rpm=aero_rpm,
                    food_id=food_id,
                    food_nutr=food_nutr,
                    food_sat=food_sat,
                    rednet_color=rednet_color,
                    rednet_signal=rednet_signal,
                    conduit_power=conduit_power,
                    conduit_fluid=conduit_fluid,
                    belt_tier=belt_tier,
                    inserter_speed=inserter_speed,
                    inserter_stack=inserter_stack,
                    mob_hp=mob_hp,
                    mob_dmg=mob_dmg,
                    affix_1=affix_1,
                    affix_2=affix_2,
                    ambulatory_class=amb_enum,
                    subtype=sub_enum,
                    enemy_diff=enemy_diff,
                    mm_power=mm_power,
                    fauna_mut=fauna_mut,
                    chisel_coord=(chisel_x, chisel_y, chisel_z),
                    dy_pressure=dy_p,
                    dy_shear=dy_tau
                )

                # Push live simulation telemetry back to DearPyGui output labels
                if dpg.does_item_exist("txt_radar_status"):
                    dpg.set_value("txt_radar_status", f"Blips: {sim_res['radar']['blip_count']} | Waypoints: {len(sim_res['radar']['waypoints'])}")
                if dpg.does_item_exist("txt_jer_status"):
                    dpg.set_value("txt_jer_status", f"Ore Density: {sim_res['jer']['ore_density']:.3f} | Drops: {sim_res['jer']['mob_drop_count']}")
                if dpg.does_item_exist("txt_mek_status"):
                    dpg.set_value("txt_mek_status", f"Tier {sim_res['mekanism']['tier']} -> {sim_res['mekanism']['ingots_out']} Ingots (Eta: {sim_res['mekanism']['thermodynamic_eta']:.2f})")
                if dpg.does_item_exist("txt_sem_status"):
                    dpg.set_value("txt_sem_status", f"Precious: {sim_res['sem_tech']['precious_metal_units']:.3f} | Recycled H2O: {sim_res['sem_tech']['recycled_saltwater_mb']:.0f}mB")
                if dpg.does_item_exist("txt_aero_status"):
                    dpg.set_value("txt_aero_status", f"Lift: {sim_res['aeronautics']['buoyant_lift_N']:.1f}N | Speed: {sim_res['aeronautics']['speed']:.1f}m/s | Yield: {sim_res['aeronautics']['structural_yield']:.2f}")
                if dpg.does_item_exist("txt_nutrition_status"):
                    dpg.set_value("txt_nutrition_status", f"Max HP: {sim_res['nutrition']['max_hp']:.1f} (+{sim_res['nutrition']['bonus_hearts']}H) | Entropy: {sim_res['nutrition']['shannon_entropy']:.2f}")
                if dpg.does_item_exist("txt_conduit_status"):
                    dpg.set_value("txt_conduit_status", f"RedNet: {sim_res['conduit_rednet']['channel_signal']} | FE: {sim_res['conduit_rednet']['transferred_fe']:.0f} | Fluid: {sim_res['conduit_rednet']['transferred_mb']:.0f}")
                if dpg.does_item_exist("txt_factorio_status"):
                    dpg.set_value("txt_factorio_status", f"Belt: {sim_res['logistics']['belt_throughput_items_per_sec']:.1f} it/s | Inserter: {sim_res['logistics']['inserter_rate_items_per_sec']:.1f} it/s")
                if dpg.does_item_exist("txt_mob_status"):
                    dpg.set_value("txt_mob_status", f"HP: {sim_res['mob']['effective_hp']:.1f} | Dmg: {sim_res['mob']['attack_damage']:.1f} | Affixes: {','.join(sim_res['mob']['active_affixes'])}")
                if dpg.does_item_exist("txt_multi_mine_status"):
                    dpg.set_value("txt_multi_mine_status", f"Block {sim_res['multi_mine']['coord']}: {sim_res['multi_mine']['damage']:.1%} broken")
                if dpg.does_item_exist("txt_fauna_genetics_status"):
                    dpg.set_value("txt_fauna_genetics_status", f"Child Speed: {sim_res['fauna_genetics']['offspring_speed']:.2f} | Jump: {sim_res['fauna_genetics']['offspring_jump']:.2f}")
                if dpg.does_item_exist("txt_chisels_status"):
                    dpg.set_value("txt_chisels_status", f"Morton Bits: {sim_res['chisels_octree']['total_morton_bits']}")
                if dpg.does_item_exist("txt_dual_yield_status"):
                    dpg.set_value("txt_dual_yield_status", f"MC Flow: {sim_res['dual_yield']['mohr_coulomb_flow']:.2f} | DP Flow: {sim_res['dual_yield']['drucker_prager_flow']:.2f}")
                if dpg.does_item_exist("txt_vehicle_status"):
                    dpg.set_value("txt_vehicle_status", f"Speed: {sim_res['vehicle']['speed']:.2f} | Battery: {sim_res['vehicle']['battery_charge']:.1f} | Delta-v: {sim_res['vehicle']['accumulated_delta_v']:.2f}")
                if dpg.does_item_exist("txt_character_status"):
                    dpg.set_value("txt_character_status", f"Rig Bones: {len(sim_res['adaptive_rig']['bones'])}")
                if dpg.does_item_exist("txt_economy_status"):
                    dpg.set_value("txt_economy_status", f"Eta: {sim_res['economy']['eta_stack']:.2f} | Waste: {sim_res['economy']['p_waste']:.2f} | Friction: {sim_res['economy']['market_friction']:.2f}")
            except Exception as e:
                logger.debug(f"[Evaluation Worker] Tick exception: {e}")
                continue

            state_tuple = (latent_dim, resolution, iterations, dt, mass_cost, topo_persist, resonance, quantum_tunnel, len(current_links))
            if not hasattr(self, 'last_state_tuple') or self.last_state_tuple != state_tuple:
                self.last_state_tuple = state_tuple
                
                current_state_tensor = torch.tensor([[latent_dim, resolution, iterations, dt, mass_cost, topo_persist, resonance, quantum_tunnel]])
                history_buffer = calm.update_buffer(history_buffer, current_state_tensor)
                
                abort_score, _, _, _, _, _ = calm(history_buffer, h_mischief=0.9, dt=dt)
                if abort_score.item() > 0.8 and not getattr(calm, 'meditation_active', False):
                    logger.warning(f"[CALM] Trajectory vetoed (score: {abort_score.item():.2f}). Entropic collapse detected. Aborting compile.")
                    continue
                
                consensus_kelly = 1.0
                if bonfire:
                    consensus_kelly = bonfire.compute_egalitarian_consensus()
                effective_mass_cost = mass_cost * (2.0 - consensus_kelly)
                
                connected_bspline = False
                connected_dark_matter = False
                for link in current_links:
                    if link[0] == "attr_bspline_out" and link[1] == "attr_sink_in":
                        connected_bspline = True
                    elif link[0] == "attr_dark_matter_out" and link[1] == "attr_sink_in":
                        connected_dark_matter = True
                        
                if self.patch_state:
                    with self.patch_state.lock:
                        self.patch_state.routine.layers = [l for l in self.patch_state.routine.layers if not getattr(l, 'name', '').startswith("NodeCompiledMod_")]
                        jitter_id = str(harvest_honest_jitter(torch.Size([1])).item())
                        
                        if connected_bspline:
                            dummy_gauge = torch.zeros(1)
                            proof = zk_agg.prove_chern_simons_invariant(current_state_tensor, dummy_gauge)
                            if not zk_agg.verify_proof("chern_simons", proof):
                                logger.error("[ZKAggregator] Topological leak detected. Refusing BSpline injection.")
                                continue
                                
                            logger.info(f"[D-Wave] Compiling BSpline layer (Dim: {latent_dim}, Res: {resolution})...")
                            new_layer = MangostienBSplineMod(f"NodeCompiledMod_BSpline_{jitter_id}", latent_dim=int(latent_dim), resolution=int(resolution))
                            new_layer.settings.mass_cost_modifier = effective_mass_cost
                            new_layer.settings.topological_persistence = topo_persist
                            new_layer.settings.resonance_frequency = resonance
                            new_layer.settings.quantum_tunnel_prob = quantum_tunnel
                            self.patch_state.routine.layers.append(new_layer)
                            self.patch_state.graph.dirty = True

                        if connected_dark_matter:
                            proof = zk_agg.prove_chern_simons_invariant(current_state_tensor, torch.zeros(1))
                            if not zk_agg.verify_proof("chern_simons", proof):
                                logger.error("[ZKAggregator] Topological leak detected. Refusing Dark Matter injection.")
                                continue
                                
                            logger.info(f"[D-Wave] Compiling Dark Matter Attractor (Iter: {iterations}, dt: {dt})...")
                            new_layer = DarkMatterAttractorLayer(f"NodeCompiledMod_DarkMatter_{jitter_id}", iterations=int(iterations), dt=dt)
                            new_layer.settings.mass_cost_modifier = effective_mass_cost
                            new_layer.settings.topological_persistence = topo_persist
                            new_layer.settings.resonance_frequency = resonance
                            new_layer.settings.quantum_tunnel_prob = quantum_tunnel
                            self.patch_state.routine.layers.append(new_layer)
                            self.patch_state.graph.dirty = True

    def _run_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title='Voxelboxter - Physical Scripting Console', width=1200, height=800)
        dpg.setup_dearpygui()
        self._setup_nodes()
        dpg.show_viewport()
        while dpg.is_dearpygui_running() and self.running:
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_dpg, daemon=True)
            self.thread.start()
            self.eval_thread = threading.Thread(target=self._evaluation_worker, daemon=True)
            self.eval_thread.start()
            logger.info("[Physical Scripting] DearPyGui window and D-Wave Evaluation Pool activated.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.eval_thread:
            self.eval_thread.join(timeout=2.0)
