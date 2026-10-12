"""
tests/test_cleanroom_factor_systems.py
Verification suite for Sovereign Cleanroom Factor Systems in Gyroidic Reasoner Ecosystem:
- Validates mathematical & physical AI foundations without Minecraft tropes.
- Tests CranialJointShearRegistry via AdaptiveSkeletonHarness cervical joint strains.
- Tests ResourceDistributionInspector Chebyshev ore curves and physical harvest debris.
- Tests AdversaryEntityProfile morphological mutations and non-dual affix invariants.
- Tests ProjectEEMCSolver Leontief input-output matrix transmutations.
- Tests BagOfHolding MetaPolytopeMatrioshka dimensional pocket safeguards.
- Tests RotationalKineticNetwork gyroscopic torque precession via TopologicalGyrocompass.
"""

import unittest
import torch
from src.core.adaptive_skeleton_harness import EnemySubtype, AmbulatoryClass
from src.environment.cleanroom_mechanics import (
    CranialJointShearRegistry,
    ResourceDistributionInspector,
    AdversaryEntityProfile,
    InfernalAffix,
    ProjectEEMCSolver,
    BagOfHolding,
    BagTier,
    ItemStack,
    ItemMetadata,
    RotationalKineticNetwork,
    RotationalNode,
    SubgridRotationalWireHarness,
    EnchantmentIndustryPipeline,
    ClimateRiverSegment,
    RiverBiomeType,
    CombatNouveauProfile,
    WeaponCategory,
    GasTubeTier,
    GasDumpingMode,
    GasStack,
    CompositeVoxelConduit,
    MekanismScubaTank,
    RotaryCondensentrator,
    TracerOxygenAtmosphereEngine,
    U1GaugeAtmosphereBridge
)
from src.ui.voxelboxter_simulation import EnvironmentalAtmosphere, AirBreathingBattery
from src.scripting.node_environment import PhysicalNodeEditor


class TestCleanroomFactorSystems(unittest.TestCase):
    def test_cranial_joint_shear_registry(self):
        # Under normal impulse, joint strain is within tolerance -> None
        trophy_low = CranialJointShearRegistry.roll_decapitation(
            subtype=EnemySubtype.VOID_STALKER,
            ambulatory=AmbulatoryClass.BIPED,
            impact_impulse_J=10.0,
            beheading_level=0
        )
        self.assertIsNone(trophy_low)

        # Under high kinetic impulse + beheading bonus -> shear rupture triggers fossil extraction
        trophy_high = CranialJointShearRegistry.roll_decapitation(
            subtype=EnemySubtype.DEMIURGIC_TITAN,
            ambulatory=AmbulatoryClass.BIPED,
            impact_impulse_J=100.0,
            beheading_level=3
        )
        self.assertIsNotNone(trophy_high)
        self.assertIn("Cranial_Fossil_DEMIURGIC_TITAN_BIPED", trophy_high.metadata.display_name)

    def test_resource_distribution_inspector(self):
        inspector = ResourceDistributionInspector()
        # Chebyshev density at peak height should be high (> 0.5)
        peak_density = inspector.get_ore_density_at_height(material_id=10, y=16)
        self.assertGreater(peak_density, 0.5)

        # Beyond bounds should be 0.0
        out_of_bounds = inspector.get_ore_density_at_height(material_id=10, y=500)
        self.assertEqual(out_of_bounds, 0.0)

        # Harvest debris returns physical discrete tokens
        debris = inspector.roll_harvest_debris(entity_class="bipedal_adversary", looting_level=1)
        self.assertIsInstance(debris, list)

    def test_adversary_entity_profile(self):
        profile = AdversaryEntityProfile(
            entity_id="adv_test",
            subtype=EnemySubtype.DEMIURGIC_TITAN,
            base_max_health=40.0,
            attack_damage=10.0
        )
        # DEMIURGIC_TITAN subtype applies 5x HP multiplier and BULWARK affix
        self.assertEqual(profile.max_hp, 200.0)
        self.assertEqual(profile.attack_damage, 30.0)
        self.assertIn(InfernalAffix.BULWARK, profile.affixes)

    def test_projecte_leontief_solver(self):
        solver = ProjectEEMCSolver()
        gain = solver.burn_item_for_emc("player_1", "iron_ingot", count=2)
        self.assertEqual(gain, 512)
        self.assertEqual(solver.player_emc["player_1"], 512)

        # Transmute
        transmuted = solver.transmute_item("player_1", "iron_ingot", count=1)
        self.assertEqual(transmuted, 1)
        self.assertEqual(solver.player_emc["player_1"], 256)

        # Singularity crystal exists without netherite
        self.assertIn("singularity_crystal", solver.emc_values)

    def test_bag_of_holding_matrioshka_safeguard(self):
        bag = BagOfHolding(tier=BagTier.LEATHER)
        self.assertEqual(len(bag.slots), 9)

        # Normal insertion
        item = ItemStack(item_id=1, count=16)
        res = bag.insert_item(item)
        self.assertTrue(res["success"])
        self.assertEqual(bag.slots[0], item)

        # Recursive dimensional nesting -> triggers void safeguard
        nested_bag_item = ItemStack(
            item_id=999,
            count=1,
            metadata=ItemMetadata(custom_nbt={"is_bag_of_holding": True})
        )
        nest_res = bag.insert_item(nested_bag_item)
        self.assertFalse(nest_res["success"])
        self.assertTrue(nest_res["void_safeguard_triggered"])
        self.assertTrue(bag.is_void_collapsed)

    def test_rotational_kinetics_gyrocompass(self):
        net = RotationalKineticNetwork()
        gen = RotationalNode("motor", rpm=32.0, stress_capacity_su=1024.0)
        load = RotationalNode("crusher", rpm=32.0, stress_impact_su=16.0)
        net.add_node(gen)
        net.add_node(load)

        state = net.compute_network_state()
        self.assertFalse(state["is_overstressed"])
        self.assertEqual(state["total_capacity_su"], 1024.0)
        self.assertEqual(state["total_stress_su"], 512.0)
        self.assertGreater(state["precessed_torque_norm"], 0.0)

    def test_combat_nouveau_profile(self):
        profile = CombatNouveauProfile.create(WeaponCategory.SWORD)
        self.assertEqual(profile.base_damage, 6.0)
        self.assertEqual(profile.attack_reach_meters, 3.0)

        hit = profile.calculate_attack_strike(charge_ratio=1.0, is_critical=True)
        self.assertTrue(hit["sweeping_hit"])
        self.assertTrue(hit["is_critical"])
        self.assertAlmostEqual(hit["damage"], 9.0)

    def test_mekanism_pressurized_tubes_and_dumping_modes(self):
        # 1. Tier properties
        self.assertEqual(GasTubeTier.BASIC.capacity_mb, 1000.0)
        self.assertEqual(GasTubeTier.ADVANCED.capacity_mb, 4000.0)
        self.assertEqual(GasTubeTier.ELITE.capacity_mb, 16000.0)
        self.assertEqual(GasTubeTier.ULTIMATE.capacity_mb, 64000.0)

        conduit = CompositeVoxelConduit()
        conduit.has_gas_conduit = True
        conduit.gas_tier = GasTubeTier.ADVANCED
        conduit.dumping_mode = GasDumpingMode.IDLE

        # Normal gas transfer
        gas_in = GasStack(gas_id="oxygen", amount_mb=3000.0)
        transferred = conduit.transfer_gas(gas_in)
        self.assertEqual(transferred, 3000.0)
        self.assertIsNotNone(conduit.gas_stored)
        self.assertEqual(conduit.gas_stored.amount_mb, 3000.0)

        # Incompatible gas rejects
        diff_gas = GasStack(gas_id="hydrogen", amount_mb=500.0)
        rejected = conduit.transfer_gas(diff_gas)
        self.assertEqual(rejected, 0.0)

        # Dumping mode vents to atmosphere
        conduit.dumping_mode = GasDumpingMode.DUMPING
        atmos = EnvironmentalAtmosphere(oxygen_density=1.0, ambient_pressure=101.3)
        vented = conduit.transfer_gas(GasStack(gas_id="oxygen", amount_mb=1000.0), atmosphere=atmos)
        self.assertEqual(vented, 1000.0)
        self.assertGreater(atmos.oxygen_density, 1.0)
        self.assertGreater(atmos.ambient_pressure, 101.3)

        # Dumping excess mode preserves 95% buffer and vents remainder
        conduit_excess = CompositeVoxelConduit()
        conduit_excess.has_gas_conduit = True
        conduit_excess.gas_tier = GasTubeTier.BASIC  # capacity 1000.0, 95% = 950.0
        conduit_excess.dumping_mode = GasDumpingMode.DUMPING_EXCESS

        atmos_excess = EnvironmentalAtmosphere(oxygen_density=1.0, ambient_pressure=101.3)
        conduit_excess.transfer_gas(GasStack(gas_id="oxygen", amount_mb=1000.0), atmosphere=atmos_excess)
        self.assertEqual(conduit_excess.gas_stored.amount_mb, 950.0)
        self.assertGreater(atmos_excess.ambient_pressure, 101.3)

    def test_mekanism_scuba_tank_and_rotary_condensentrator(self):
        # 1. Scuba tank
        scuba = MekanismScubaTank()
        self.assertFalse(scuba.is_active)
        scuba.toggle_active()
        self.assertTrue(scuba.is_active)

        filled = scuba.fill(GasStack(gas_id="oxygen", amount_mb=10000.0))
        self.assertEqual(filled, 10000.0)

        supported, consumed = scuba.consume_breathing_gas(dt=1.0)
        self.assertTrue(supported)
        self.assertEqual(consumed, 10.0)
        self.assertEqual(scuba.gas_stack.amount_mb, 9990.0)

        # Air-breathing battery in vacuum utilizing scuba backup
        battery = AirBreathingBattery()
        rate_vacuum_no_scuba = battery.calculate_atmospheric_oxidation(ambient_oxygen_density=0.0)
        self.assertGreater(rate_vacuum_no_scuba, 0.0)  # Uses internal reserve initially
        battery.stored_oxygen_reserve = 0.0
        rate_empty_reserve = battery.calculate_atmospheric_oxidation(ambient_oxygen_density=0.0)
        self.assertEqual(rate_empty_reserve, 0.0)

        rate_with_scuba = battery.calculate_atmospheric_oxidation(ambient_oxygen_density=0.0, scuba_tank=scuba)
        self.assertGreater(rate_with_scuba, 0.0)

        # 2. Rotary Condensentrator (Gas -> Fluid)
        condensentrator = RotaryCondensentrator(energy_per_mb_fe=5.0)
        o2_gas = GasStack(gas_id="oxygen", amount_mb=1000.0)
        fluid, rem_gas, used_fe = condensentrator.condensentrate(o2_gas, available_energy_fe=2500.0)

        self.assertIsNotNone(fluid)
        self.assertEqual(fluid.fluid_id, "liquid_oxygen")
        self.assertEqual(fluid.amount_mb, 500.0)
        self.assertEqual(rem_gas, 500.0)
        self.assertEqual(used_fe, 2500.0)

        # Decondensentrate (Fluid -> Gas)
        rev_gas, rem_fluid, rev_fe = condensentrator.decondensentrate(fluid, available_energy_fe=2500.0)
        self.assertIsNotNone(rev_gas)
        self.assertEqual(rev_gas.gas_id, "oxygen")
        self.assertEqual(rev_gas.amount_mb, 500.0)
        self.assertEqual(rem_fluid, 0.0)

    def test_tracer_oxygen_atmosphere_engine(self):
        engine = TracerOxygenAtmosphereEngine(
            hull_bounds=(-10.0, 10.0, -5.0, 5.0, -10.0, 10.0),
            emitter_pos=(0.0, 0.0, 0.0),
            num_molecules=8,
            molecule_speed=10.0
        )
        self.assertFalse(engine.is_depressurized)
        self.assertEqual(len(engine.molecules), 8)

        # Add brick partition between emitter and obstructed player
        engine.add_brick_obstacle(1.0, 2.0, -3.0, 3.0, -3.0, 3.0)

        consumers = {
            "player_open": (0.5, 0.0, 0.0),       # Unobstructed
            "player_blocked": (4.0, 0.0, 0.0)     # Obstructed by partition
        }

        # Step multiple iterations to accumulate saturation
        for _ in range(5):
            res = engine.step(dt=0.05, consumer_positions=consumers)

        self.assertFalse(res["is_depressurized"])
        # Dynamic envelope expanded beyond emitter
        self.assertLess(res["active_envelope"][0], 0.0)
        self.assertGreater(res["active_envelope"][1], 0.0)

        # Player behind brick partition receives less or zero line-of-sight oxygen
        sat_open = res["consumer_saturation"]["player_open"]
        sat_blocked = res["consumer_saturation"]["player_blocked"]
        self.assertGreaterEqual(sat_open, sat_blocked)

        # Hull breach depressurization test
        engine.molecules[0].position = [15.0, 0.0, 0.0]  # Outside hull max_x=10.0
        breach_step = engine.step(dt=0.05, consumer_positions=consumers)
        self.assertTrue(breach_step["is_depressurized"])
        self.assertIsNotNone(breach_step["leak_location"])
        self.assertFalse(engine.is_breathable("player_open"))

    def test_u1_gauge_atmosphere_bridge(self):
        bridge = U1GaugeAtmosphereBridge(c_speed=1.0)

        A_mu = torch.tensor([101.3, 2.0, 1.0, 0.0], dtype=torch.float32)
        dA_dx = torch.tensor([-0.4, 0.0, 0.5, 0.0], dtype=torch.float32)
        dA_dy = torch.tensor([0.0, -0.5, 0.0, 0.2], dtype=torch.float32)
        dA_dz = torch.tensor([0.0, 0.0, -0.2, 0.0], dtype=torch.float32)
        dA_dt = torch.tensor([0.0, 0.1, 0.0, 0.0], dtype=torch.float32)

        fields = bridge.compute_field_tensor(A_mu, dA_dx, dA_dy, dA_dz, dA_dt)
        F = fields["F_mu_nu"]
        E = fields["electric_vector_E"]
        B = fields["magnetic_pseudovector_B"]

        # Anti-symmetry: F_mu_nu = -F_nu_mu
        self.assertTrue(torch.allclose(F, -F.T, atol=1e-5))
        # Diagonal elements must be zero
        for i in range(4):
            self.assertAlmostEqual(F[i, i].item(), 0.0)

        # The Six Ways: 3 electric and 3 magnetic components correctly embedded
        self.assertAlmostEqual(F[1, 0].item(), E[0].item())
        self.assertAlmostEqual(F[2, 0].item(), E[1].item())
        self.assertAlmostEqual(F[3, 0].item(), E[2].item())
        self.assertAlmostEqual(F[3, 2].item(), B[0].item())
        self.assertAlmostEqual(F[1, 3].item(), B[1].item())
        self.assertAlmostEqual(F[2, 1].item(), B[2].item())

        # Invariant and Action density
        B_sq = torch.sum(B * B).item()
        E_sq = torch.sum(E * E).item()
        expected_inv = 2.0 * (B_sq - E_sq)
        self.assertAlmostEqual(fields["invariant_F_squared"].item(), expected_inv, places=4)

        # Continuity & Chern-Simons leak repair
        conserved_res = bridge.check_continuity_and_plug_leak(drho_dt=-0.05, div_j=0.05)
        self.assertTrue(conserved_res["is_conserved"])
        self.assertFalse(conserved_res["has_leak"])

        leaked_res = bridge.check_continuity_and_plug_leak(drho_dt=-0.5, div_j=0.1)
        self.assertFalse(leaked_res["is_conserved"])
        self.assertTrue(leaked_res["has_leak"])
        self.assertIsNotNone(leaked_res["topological_repair"])

    def test_node_environment_cleanroom_gasses_hook(self):
        editor = PhysicalNodeEditor()
        res = editor.hook_cleanroom_gasses(
            gas_id="oxygen",
            gas_amount_mb=3000.0,
            tube_tier=2,
            dumping_mode="IDLE",
            simulate_tracer=True,
            test_scuba=True,
            test_rotary_condensentrator=True,
            dt=0.1
        )
        self.assertEqual(res["conduit_gas_tier"], "ADVANCED")
        self.assertEqual(res["gas_transferred_mb"], 3000.0)
        self.assertTrue(res["scuba_telemetry"]["is_breathing"])
        self.assertEqual(res["rotary_telemetry"]["produced_fluid_id"], "liquid_oxygen")
        self.assertFalse(res["tracer_telemetry"]["is_depressurized"])
        self.assertTrue(res["gauge_is_conserved"])


if __name__ == "__main__":
    unittest.main()
