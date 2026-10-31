import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import pytest

from src.environment.cleanroom_mechanics import (
    CleanroomCapabilityRegistry, CleanroomCapability, CapabilityType,
    ParallelRegionTicker, CubicRegion16, SovereignRuptureOperator, RuptureStatus
)
from src.ui.voxelboxter_simulation import VoxelboxterEngine

def test_weft_capability_registry():
    """Verify Weft-inspired Engine-as-a-Mod capability bus."""
    registry = CleanroomCapabilityRegistry.get_instance()
    
    # Register an energy-voltage rated handler
    cap = CleanroomCapability(
        capability_type=CapabilityType.ENERGY_VOLTAGE_RATED,
        properties={"max_voltage": 512.0, "nominal_current": 16.0},
        handler=lambda payload: {
            "status": "DELIVERED",
            "voltage": payload.get("voltage", 128.0),
            "safe": payload.get("voltage", 128.0) <= 512.0
        }
    )
    registry.register("machine_macerator_0", cap)
    
    retrieved = registry.get_capability("machine_macerator_0", CapabilityType.ENERGY_VOLTAGE_RATED)
    assert retrieved is not None
    assert retrieved.properties["max_voltage"] == 512.0
    
    # Test dispatch
    res = registry.dispatch("machine_macerator_0", CapabilityType.ENERGY_VOLTAGE_RATED, {"voltage": 240.0})
    assert res["status"] == "DELIVERED"
    assert res["safe"] is True

def test_weft_parallel_region_ticker_and_catchup():
    """Verify parallel cubic region partitioning and deterministic dormant catch-up ticking."""
    ticker = ParallelRegionTicker()
    
    # Access a block at (35, 10, 48) -> should map to region (2, 0, 3)
    region = ticker.get_or_create_region((35, 10, 48))
    assert region.region_coord == (2, 0, 3)
    assert region.is_active is False
    assert region.dormant_tick_debt == 0.0
    
    # Simulate time passing with dormant region
    ticker.tick_all(dt=1.0)
    ticker.tick_all(dt=1.5)
    assert region.dormant_tick_debt >= 2.5
    
    # Activate region (player re-enters) -> executes catch-up tick
    ticker.set_region_active((35, 10, 48), active=True)
    assert region.is_active is True
    assert region.dormant_tick_debt == 0.0  # Debt amortized

    report = ticker.tick_all(dt=0.05)
    assert report["active_regions"] >= 1
    assert report["tick_index"] == 3

def test_sovereign_rupture_operator_mandy_veto():
    """Verify Mandy's Sovereign Refusal breaking sycophantic Nash-Kuiper corrugation."""
    op = SovereignRuptureOperator(shear_cohesion_c=200.0, friction_angle_phi_deg=30.0, max_corrugation_depth=5)
    
    # 1. Nominal stress below yield -> no rupture, elastic balance
    status_nom = op.evaluate_sycophancy_vs_rupture(
        normal_stress_sigma=50.0,
        shear_stress_tau=100.0,
        corrugation_iterations=2
    )
    assert status_nom.is_ruptured is False
    assert status_nom.mandy_veto_triggered is False
    assert status_nom.yield_mechanism == "NOMINAL_ELASTIC_EQUILIBRIUM"

    # 2. Extreme shear stress exceeding Mohr-Coulomb yield -> Mandy's Sovereign Refusal
    # tau_yield = 200 + 50 * tan(30) = ~228.87. Stress is 450.0 -> rupture
    status_rupt = op.evaluate_sycophancy_vs_rupture(
        normal_stress_sigma=50.0,
        shear_stress_tau=450.0,
        corrugation_iterations=2
    )
    assert status_rupt.is_ruptured is True
    assert status_rupt.mandy_veto_triggered is True
    assert status_rupt.yield_mechanism == "MANDY_MOHR_COULOMB_VETO"
    assert status_rupt.billy_mischief_score > 0.0

    # 3. Endless corrugation avoidance loop (Teacher's Pet / Sycophancy trap)
    status_syc = op.evaluate_sycophancy_vs_rupture(
        normal_stress_sigma=10.0,
        shear_stress_tau=50.0,
        corrugation_iterations=12,
        external_sycophancy_pressure=0.9
    )
    # Exceeds max_corrugation_depth and high pressure -> Mandy triggers veto
    assert status_syc.mandy_veto_triggered is True

def test_voxelboxter_engine_region_ticker_integration():
    """Verify VoxelboxterEngine instantiates region ticker and ticks cleanly."""
    engine = VoxelboxterEngine(device="cpu")
    assert hasattr(engine, "region_ticker")
    assert hasattr(engine, "capability_bus")
    assert hasattr(engine, "rupture_operator")

    # Tick engine
    engine.tick(dt=0.05)
    assert engine.region_ticker.total_processed_ticks == 1

if __name__ == "__main__":
    print("Running test_weft_capability_registry...")
    test_weft_capability_registry()
    print("Running test_weft_parallel_region_ticker_and_catchup...")
    test_weft_parallel_region_ticker_and_catchup()
    print("Running test_sovereign_rupture_operator_mandy_veto...")
    test_sovereign_rupture_operator_mandy_veto()
    print("Running test_voxelboxter_engine_region_ticker_integration...")
    test_voxelboxter_engine_region_ticker_integration()
    print("ALL WEFT CLEANROOM AND RUPTURE TESTS PASSED!")
