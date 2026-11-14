"""
Unit tests for the cleanroom synthetic diesel fuel pipeline,
Mekanism industrial processing tiers, and Voxelboxter vehicle kinetics.
"""

import pytest
import math
import torch

from src.environment.cleanroom_mechanics import (
    MekanismProcessingPipeline,
    SyntheticDieselResult,
    SEMElectrochemicalExtractor,
    SEMPrecipitateResult,
    ExpandedInventorySystem,
    FluidStack,
    GasStack,
    ProjectEEMCSolver
)
from src.ui.voxelboxter_simulation import VehicleEngine, RigidBody, Propulsor
from src.scripting.node_environment import PhysicalNodeEditor


def test_sem_tech_hydrogen_generation():
    extractor = SEMElectrochemicalExtractor(membrane_durability=1.0)
    res: SEMPrecipitateResult = extractor.extract_from_tailings(
        tailings_count=64,
        saltwater_mb=4000.0,
        energy_fe=10000.0,
        feedstock_grade=1.0
    )
    assert res.tailings_consumed == 64
    assert res.hydrogen_gas_mb > 0.0
    # 64 tailings * 50 mB water = 3200 mB water * 2.0 * eff
    assert res.hydrogen_gas_mb >= 5000.0
    assert res.precious_metal_units > 0.0


def test_mekanism_synthetic_diesel_synthesis_ft():
    pipeline = MekanismProcessingPipeline()
    # Test Tier 2 Fischer-Tropsch
    res: SyntheticDieselResult = pipeline.process_synthetic_diesel_synthesis(
        carbon_count=20,
        hydrogen_mb=5000.0,
        route="gasification_fischer_tropsch",
        tier=2,
        energy_fe=3000.0
    )
    assert res.consumed_carbon_count == 20
    assert res.consumed_hydrogen_mb == 4000.0  # 20 * 200 mB
    assert res.remaining_hydrogen_mb == 1000.0
    assert res.remaining_carbon_count == 0
    # Tier 2 yield = 20 * 50 mB * 2 = 2000 mB
    assert res.diesel_fuel_mb == 2000.0
    assert res.route == "gasification_fischer_tropsch"
    assert res.tier == 2
    assert res.waste_heat_flux >= 0.0
    assert res.thermodynamic_eta <= 1.0


def test_mekanism_synthetic_diesel_synthesis_pyrolysis_and_acid_boost():
    pipeline = MekanismProcessingPipeline()
    # Test Tier 5 Chemical Dissolution with H2SO4 acid reagent
    res: SyntheticDieselResult = pipeline.process_synthetic_diesel_synthesis(
        carbon_count=10,
        hydrogen_mb=2000.0,
        route="pyrolysis_hydroprocessing",
        tier=5,
        energy_fe=5000.0,
        acid_reagent_mb=500.0
    )
    assert res.consumed_carbon_count == 10
    assert res.consumed_hydrogen_mb == 1500.0  # 10 * 150 mB
    assert res.remaining_hydrogen_mb == 500.0
    # Base yield = 10 * 40 mB = 400 mB; Tier 5 multiplier = 5.0 * 1.25 = 6.25 -> 2500 mB
    assert res.diesel_fuel_mb >= 2000.0
    assert res.route == "pyrolysis_hydroprocessing"


def test_expanded_inventory_fluid_storage():
    inv = ExpandedInventorySystem()
    fluid = FluidStack(fluid_id="synthetic_diesel", amount_mb=2500.0)
    leftover = inv.add_fluid(fluid)
    assert leftover == 0.0
    assert inv.get_fluid("synthetic_diesel") is not None
    assert inv.get_fluid("synthetic_diesel").amount_mb == 2500.0

    removed = inv.remove_fluid("synthetic_diesel", 1000.0)
    assert removed == 1000.0
    assert inv.get_fluid("synthetic_diesel").amount_mb == 1500.0


def test_vehicle_engine_diesel_combustion():
    engine = VehicleEngine(
        throttle=1.0,
        is_sleeve_valve=True,
        electrical_timing_degrees=12.5,
        sleeve_port_overlap_efficiency=1.35
    )
    # Add synthetic diesel fuel
    leftover = engine.add_diesel_fuel(100.0)
    assert leftover == 0.0
    assert engine.diesel_fuel_mb == 100.0

    # Consume over dt = 1.0s (burn rate = 5.0 mB/s)
    is_combusting, consumed, boost = engine.consume_diesel_fuel(1.0)
    assert is_combusting is True
    assert consumed == 5.0
    assert engine.diesel_fuel_mb == 95.0
    # Boost factor > 1.0 due to synthetic diesel and sleeve valve timing advance
    assert boost > 1.2


def test_node_environment_diesel_pipeline_integration():
    editor = PhysicalNodeEditor()
    # 1. Electrolyze tailings to generate hydrogen
    sem_res = editor.hook_cleanroom_sem_tech(tailings_count=32, saltwater_mb=2000.0, energy_fe=5000.0)
    assert sem_res["hydrogen_gas_mb"] > 0.0
    assert editor.expanded_inventory.get_gas("hydrogen") is not None

    # 2. Synthesize synthetic diesel and deposit to vehicle
    diesel_res = editor.hook_cleanroom_synthetic_diesel(
        carbon_count=8,
        hydrogen_mb=2000.0,
        route="gasification_fischer_tropsch",
        tier=3,
        deposit_to_vehicle=True
    )
    assert diesel_res["diesel_fuel_mb"] > 0.0
    assert editor.engine.diesel_fuel_mb == diesel_res["diesel_fuel_mb"]
    assert editor.expanded_inventory.get_fluid("synthetic_diesel") is not None

    # 3. Step vehicle kinetics and verify combustion boost
    kinetics = editor.tick_vehicle_kinetics(dt=0.1)
    assert kinetics["diesel_consumed_mb"] > 0.0
    assert kinetics["combustion_boost"] > 1.0
    assert kinetics["net_thrust"] > 0.0
