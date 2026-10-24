import pytest
import torch
import math

from src.scripting.node_environment import (
    PhysicalNodeEditor,
    OreDictionary,
    ResourceRegistry,
    MasterNodeGroup,
    DataflowNode,
    NodeSocket,
    SocketType,
    DifficultyMode,
    Role
)
from src.core.adaptive_skeleton_harness import AmbulatoryClass, EnemySubtype
from src.ui.voxelboxter_simulation import InventoryComponent, RigidBody, AirBreathingBattery, VehicleEngine


def test_blender_dataflow_sockets_and_master_group():
    # 1. Test Typed Sockets & Evaluation
    node_in = DataflowNode("Input_Hardness")
    sock_out = NodeSocket(name="Hardness", socket_type=SocketType.FLOAT, is_output=True, value=10.0, is_field=False)
    node_in.outputs["out"] = sock_out

    node_math = DataflowNode("Scale_Math", compute_fn=lambda inputs: {"out": inputs.get("in", 0.0) * 1.5})
    sock_math_in = NodeSocket(name="In", socket_type=SocketType.FLOAT, is_output=False, default_value=1.0)
    sock_math_out = NodeSocket(name="Out", socket_type=SocketType.FLOAT, is_output=True)
    node_math.inputs["in"] = sock_math_in
    node_math.outputs["out"] = sock_math_out

    sock_math_in.connect(sock_out)
    result = node_math.evaluate()
    assert result["out"] == 15.0

    # 2. Test Master Node Group (COG Midpoint Offset & Named Attribute Baking)
    master = MasterNodeGroup("Vehicle_Attribute_Injector")
    master.inputs["in_dimensions"].value = (2.0, 4.0, 6.0)
    master.inputs["in_hardness"].value = 10.0
    master.inputs["in_friction"].value = 0.85
    master.inputs["in_hp"].value = 500.0

    outputs = master.evaluate()
    # Center of gravity midpoint should be dimensions * -0.5
    assert outputs["out_cog_offset"] == (-1.0, -2.0, -3.0)
    # Baked attributes for game engine export
    baked = master.baked_attributes
    assert baked["phys_hardness"] == 10.0
    assert baked["phys_friction"] == 0.85
    assert baked["phys_hp"] == 500.0
    assert baked["cog_offset"] == (-1.0, -2.0, -3.0)


def test_oredict_unification():
    oredict = OreDictionary()
    assert oredict.matches_tag(10, "oreIron")
    assert oredict.matches_tag(20, "ingotIron")
    assert not oredict.matches_tag(10, "gemDiamond")

    oredict.register_ore("customFuel", 999)
    assert oredict.matches_tag(999, "customFuel")
    assert 999 in oredict.get_materials("customFuel")


def test_native_crafting_via_addon_routine():
    editor = PhysicalNodeEditor()
    editor.inventory.block_masses[20] = 5

    # Craft addon layer
    success, msg = editor.craft_addon_layer(
        layer_name="ReinforcedArmor_Plate",
        material_id=20,
        quantity=2,
        mass_cost=10.0,
        energy_cost=25.0
    )
    assert success
    assert editor.inventory.block_masses[20] == 3
    assert len(editor.patch_state.routine.layers) == 1
    assert editor.patch_state.routine.layers[0].name == "ReinforcedArmor_Plate"

    # Attempt crafting without sufficient inventory
    fail_success, fail_msg = editor.craft_addon_layer(
        layer_name="AnotherLayer",
        material_id=20,
        quantity=10,
        mass_cost=50.0
    )
    assert not fail_success
    assert "Insufficient material" in fail_msg


def test_chisels_and_bits_native_carving():
    editor = PhysicalNodeEditor()
    macro_pos = (0, 0, 0)

    # Carve a bit out of the octree
    scrapped = editor.chisel_carve_bit(macro_pos, 0, 0, 0, material_id=1)
    assert scrapped == 1
    assert editor.bit_scraps[1] == 1

    # Place a bit back
    placed = editor.chisel_place_bit(macro_pos, 0, 0, 0, material_id=20)
    assert placed
    assert editor.bit_scraps[20] == 0


def test_delta_v_collision_dynamics_and_augments():
    editor = PhysicalNodeEditor()
    editor.inventory.block_masses[1] = 100 # Add structural blocks

    # 1. Low Tier Collision: Impulse 10 Ns on 100 kg body -> Delta-v = 0.1 m/s (~0.22 mph)
    res_low = editor.simulate_collision_impulse(impulse_magnitude=10.0, delta_t=0.05)
    assert res_low["tier"] == "LOW"
    assert res_low["blocks_sheared"] == 0
    assert "Cosmetic scuffs" in res_low["description"]

    # 2. Mid Tier Collision: Impulse 350 Ns on 100 kg body -> Delta-v = 3.5 m/s (~7.8 mph)
    res_mid = editor.simulate_collision_impulse(impulse_magnitude=350.0, delta_t=0.05)
    assert res_mid["tier"] == "MID"
    assert res_mid["blocks_sheared"] == 1
    assert editor.inventory.block_masses[1] == 99

    # 3. High Tier Collision: Impulse 800 Ns on 100 kg body -> Delta-v = 8.0 m/s (~17.9 mph)
    res_high = editor.simulate_collision_impulse(impulse_magnitude=800.0, delta_t=0.01)
    assert res_high["tier"] == "HIGH"
    assert res_high["blocks_sheared"] >= 3
    assert "Catastrophic structural shearing" in res_high["description"]

    # 4. Reinforced Alloy Chassis Augment: doubles thresholds
    editor.rigid_body.reinforced_alloy_chassis = True
    res_reinforced = editor.simulate_collision_impulse(impulse_magnitude=350.0, delta_t=0.05)
    # At 7.8 mph, normally MID tier, but reinforced chassis shifts threshold up to 10 mph for MID
    assert res_reinforced["tier"] == "LOW"

    # 5. Impact Energy Harvester Augment: converts collision energy into kinetic boost
    editor.rigid_body.impact_energy_harvester = True
    v_before = editor.rigid_body.linear_velocity[2]
    editor.simulate_collision_impulse(impulse_magnitude=800.0, delta_t=0.01)
    v_after = editor.rigid_body.linear_velocity[2]
    assert v_after > v_before


def test_plasma_air_induction_and_sleeve_valve_timings():
    editor = PhysicalNodeEditor()
    editor.battery.current_charge = 1000.0

    # 1. Test Plasma Air Induction in AirBreathingBattery
    assert editor.battery.plasma_air_induction is True
    assert editor.battery.plasma_boost_factor == 2.4
    ox_plasma = editor.battery.calculate_atmospheric_oxidation()

    # Disable plasma air induction
    editor.battery.plasma_air_induction = False
    ox_standard = editor.battery.calculate_atmospheric_oxidation()
    assert ox_plasma > ox_standard
    assert math.isclose(ox_plasma, ox_standard * 2.4, rel_tol=1e-3)

    # 2. Test Sleeve Valve Electrical Advance Timings in VehicleEngine
    assert editor.engine.electrical_timing_degrees == 12.5
    assert editor.engine.sleeve_port_overlap_efficiency == 1.35

    # Run vehicle kinetics tick
    telemetry = editor.tick_vehicle_kinetics(dt=0.1)
    assert telemetry["net_thrust"] > 0.0
    assert telemetry["speed"] > 0.0
    assert telemetry["battery_charge"] < 1000.0


def test_life_hunger_valence_and_ego_death():
    editor = PhysicalNodeEditor()
    editor.difficulty_mode = DifficultyMode.SURVIVAL

    res = editor.tick_life_and_hunger(dt=0.5, current_pressure=0.5, current_mischief=0.2)
    assert res["health"] == 100.0
    assert res["saturation"] < 20.0
    assert not res["starving"]
    assert "valence_hunger" in res
    assert not res["ego_death"]

    # Trigger starvation
    editor.saturation = 0.0
    editor.hunger = 0.0
    res_starve = editor.tick_life_and_hunger(dt=1.0, current_pressure=0.9, current_mischief=0.9)
    assert res_starve["starving"]
    assert res_starve["health"] < 100.0


def test_enemy_subtype_generation():
    editor = PhysicalNodeEditor()

    # Demiurgic Titan
    titan = editor.generate_enemy_rig(
        subtype=EnemySubtype.DEMIURGIC_TITAN,
        ambulatory_class=AmbulatoryClass.NON_AMBULATORY,
        difficulty_scale=1.5
    )
    assert titan["subtype"] == EnemySubtype.DEMIURGIC_TITAN
    assert titan["damage_type"] == "dt_dilation"
    assert titan["max_health"] == 600.0 * 1.5
    assert titan["skeleton"] is not None

    # Abstracted Glitch
    glitch = editor.generate_enemy_rig(
        subtype=EnemySubtype.ABSTRACTED_GLITCH,
        ambulatory_class=AmbulatoryClass.BIPED,
        difficulty_scale=1.0
    )
    assert glitch["subtype"] == EnemySubtype.ABSTRACTED_GLITCH
    assert glitch["damage_type"] == "phase_noise"


def test_economy_and_carnot_leontief_admin_shop():
    editor = PhysicalNodeEditor()
    editor.inventory.block_masses[10] = 5

    # Visitor buys item
    buy_ok, bal_after_buy, msg = editor.execute_admin_trade(
        material_id=10, quantity=2, is_buy=True, role=Role.VISITOR
    )
    assert buy_ok
    assert editor.inventory.block_masses[10] == 7
    assert bal_after_buy < 100.0

    # Visitor sells item
    sell_ok, bal_after_sell, msg = editor.execute_admin_trade(
        material_id=10, quantity=3, is_buy=False, role=Role.VISITOR
    )
    assert sell_ok
    assert editor.inventory.block_masses[10] == 4
    assert bal_after_sell > bal_after_buy

    # Thermodynamic evaluation
    thermo = editor.evaluate_economy_thermodynamics()
    assert "eta_stack" in thermo
    assert "p_waste" in thermo
    assert "market_friction" in thermo


def test_physical_node_editor_headless_evaluation():
    editor = PhysicalNodeEditor()
    res = editor.evaluate_tick(dt=0.05)
    assert "vehicle" in res
    assert "life" in res
    assert "economy" in res
    assert "master_group_cog" in res
    assert res["master_group_cog"] == (-0.5, -0.5, -0.5)
