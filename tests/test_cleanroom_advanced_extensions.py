import unittest
import math
import torch

from src.environment.cleanroom_advanced_extensions import (
    OpenGliderKinematics, GliderConfig, GliderDeploymentState,
    StratosphericBalloon, BalloonSize, LargeVeinWorldGenerator,
    RailNetworkSystem, RailType, MinecartState,
    EntangledBindingSystem, AutoPlantManager, BiomeReactor,
    ClarityFogParameters, AnalogLever, SlurpInteraction,
    MetalAnvil, MetalAnvilTier, DiagonalArchitectureBlock,
    DespawnCustomizer, LootBagSystem, LootBagTier,
    BalancedToolTracker, PortableUtilitiesKit, BlockPeripheralAdapter,
    RemoteCursorItem, AutonomousLuggageStorage, SlimalyzerSensor,
    SonicGlassesSensor, RecursiveSphericalPortal, Hyperdimensional4DEngine
)
from src.ui.voxelboxter_simulation import VoxelboxterEngine
from src.ui.voxelboxter_client import PatchStateResource


class TestCleanroomAdvancedExtensions(unittest.TestCase):
    def test_01_openglider_kinematics_and_wind(self):
        cfg = GliderConfig(wind_sensitivity=0.8, base_lift=10.0)
        glider = OpenGliderKinematics(config=cfg)
        self.assertEqual(glider.state, GliderDeploymentState.STOWED)
        
        # Toggle deployment
        st = glider.toggle_deployment()
        self.assertEqual(st, GliderDeploymentState.DEPLOYING)
        
        # Step through flight against headwind
        vel = (5.0, 0.0, 0.0)
        orient = (5.0, 0.0, 0.0)  # slight pitch down
        wind = (-3.0, 0.5, 0.0)   # opposing wind
        new_vel, telem = glider.update_flight_kinematics(vel, orient, wind, dt=0.5)
        self.assertTrue(telem["airspeed"] > 0.0)
        self.assertTrue(glider.durability <= 100.0)

    def test_02_kerballoons_stratospheric_ascent(self):
        balloon = StratosphericBalloon(size=BalloonSize.STANDARD, payload_mass_kg=50.0)
        telem = balloon.update_ascent(ambient_pressure_kpa=101.3, dt=1.0)
        self.assertTrue(telem["altitude_m"] > 0.0)
        self.assertIn("pressure", telem["data_summary"])

    def test_03_large_veins_generator(self):
        gen = LargeVeinWorldGenerator()
        voxels = gen.generate_vein_voxels((0, 10, 0), "copper_porphyry")
        self.assertTrue(len(voxels) > 10)
        pos, ore = voxels[0]
        self.assertEqual(len(pos), 3)

    def test_04_rails_locking_direction_teleport(self):
        rails = RailNetworkSystem()
        # Locking rail
        rails.register_rail((0, 1, 0), RailType.LOCKING, facing_direction=(1, 0, 0))
        cart = MinecartState(cart_id="cart1", position=(0.5, 1.0, 0.5), velocity=(2.0, 0.0, 0.0))
        c1 = rails.process_cart_interaction(cart, (0, 1, 0))
        self.assertTrue(c1.held_by_rail)
        self.assertEqual(c1.velocity, (0.0, 0.0, 0.0))

        # Power rail
        rails.trigger_redstone_signal((0, 1, 0), True)
        c2 = rails.process_cart_interaction(cart, (0, 1, 0))
        self.assertFalse(c2.held_by_rail)
        self.assertTrue(c2.velocity[0] > 0.0)

        # Teleporting rail
        rails.register_rail((10, 1, 10), RailType.TELEPORTING, frequency_color=(255, 0, 0))
        rails.register_rail((50, 1, 50), RailType.TELEPORTING, frequency_color=(255, 0, 0))
        cart_tp = MinecartState(cart_id="cart_tp", position=(10.5, 1.0, 10.5), velocity=(1.0, 0.0, 0.0))
        c_tp = rails.process_cart_interaction(cart_tp, (10, 1, 10))
        self.assertAlmostEqual(c_tp.position[0], 50.5, places=1)
        self.assertAlmostEqual(c_tp.position[2], 50.5, places=1)

    def test_05_entangled_blocks(self):
        ent = EntangledBindingSystem()
        self.assertTrue(ent.bind_blocks((0, 0, 0), (100, 50, 100)))
        success, amount = ent.route_interaction((0, 0, 0), "energy", 5000.0)
        self.assertTrue(success)
        self.assertEqual(amount, 5000.0)

    def test_06_autoplant_rescue(self):
        ap = AutoPlantManager()
        res = ap.evaluate_despawning_item("oak_sapling", (12.3, 64.0, 15.7), ground_material_id=2)
        self.assertIsNotNone(res)
        coord, item = res
        self.assertEqual(coord, (12, 65, 15))
        self.assertEqual(item, "oak_sapling")

    def test_07_biome_reactor_and_clarity(self):
        br = BiomeReactor()
        br.charge_energy(500000.0)
        result = br.activate_transmutation("cactus")
        self.assertIsNotNone(result)
        self.assertEqual(result["new_biome"], "desert")
        self.assertEqual(result["radius"], 7)

        clarity = ClarityFogParameters()
        self.assertLess(clarity.fog_density, 0.05)

    def test_08_analog_lever_slurp_anvil(self):
        lever = AnalogLever(0)
        self.assertEqual(lever.step_up(), 1)
        self.assertEqual(lever.set_level(15), 15)

        slurp = SlurpInteraction.slurp_water(source_purity=0.5)
        self.assertEqual(slurp["hydration_added"], 25.0)

        anvil = MetalAnvil(tier=MetalAnvilTier.STEEL)
        self.assertTrue(anvil.hammer_repair(50.0))

    def test_09_lootbags_tools_and_peripherals(self):
        lb = LootBagSystem()
        items = lb.open_bag(LootBagTier.RARE)
        self.assertIn("diamond", items)

        tool = BalancedToolTracker("mining_hammer")
        self.assertTrue(tool.add_xp(120.0))
        self.assertEqual(tool.level, 2)

        cur = RemoteCursorItem()
        cur.bind_to_block((10, 20, 30))
        self.assertEqual(cur.trigger_remote_click(), (10, 20, 30))

        luggage = AutonomousLuggageStorage("player1")
        new_pos = luggage.follow_step((5.0, 0.0, 0.0), dt=0.1)
        self.assertTrue(new_pos[0] > 0.0)

        self.assertTrue(isinstance(SlimalyzerSensor.is_slime_chunk(16, 32), bool))
        pings = SonicGlassesSensor.project_acoustic_vibrations((0.0, 0.0, 0.0), [((5.0, 0.0, 0.0), 10.0)])
        self.assertEqual(len(pings), 1)

    def test_10_recursive_portal_dark_energy(self):
        portal = RecursiveSphericalPortal(inner_radius=1.0, outer_radius=5.0)
        pt, scale = portal.transform_coordinates((2.0, 0.0, 0.0), recursion_depth=2)
        self.assertEqual(scale, 0.25)
        force, l_proxy = portal.compute_recursive_gravity_dark_energy(100.0, (3.0, 0.0, 0.0))
        self.assertTrue(force[0] > 0.0)  # Outward dark energy acceleration

    def test_11_hyperdimensional_4d_engine(self):
        eng4d = Hyperdimensional4DEngine()
        # Verify 4D rotational planes
        rot_xw = eng4d.get_rotation_matrix_4d("XW", math.pi / 4.0)
        self.assertEqual(rot_xw.shape, (4, 4))
        proj = eng4d.rotate_and_project_tesseract("XW", math.pi / 4.0)
        self.assertEqual(len(proj), 16)
        self.assertEqual(len(proj[0]), 3)

    def test_12_voxelboxter_engine_and_client_integration(self):
        engine = VoxelboxterEngine(device="cpu")
        self.assertIsNotNone(engine.glider)
        self.assertIsNotNone(engine.large_veins)
        self.assertIsNotNone(engine.rail_network)
        self.assertIsNotNone(engine.recursive_portal)
        self.assertIsNotNone(engine.engine_4d)

        client_state = PatchStateResource()
        opts = client_state.update_ingame_options_gui({"wind_sensitivity": 0.5, "fog_density": 0.01})
        self.assertEqual(opts["wind_sensitivity"], 0.5)
        self.assertEqual(opts["status"], "CONFIG_APPLIED")


if __name__ == "__main__":
    unittest.main()

