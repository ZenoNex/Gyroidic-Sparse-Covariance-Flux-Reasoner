import unittest
import torch
import math
import numpy as np

from src.topology.einstein_monotile import (
    EinsteinMonotile,
    EinsteinMonotileTessellation,
    get_einstein_tessellation
)
from src.ui.voxelboxter_simulation import PointerlessOctree
from src.core.zeitgeist_router import ZeitgeistRouter, ZeitgeistState
from src.models.resonance_cavity import ResonanceCavity
from src.core.garden_statistical_attractors import CerumenPotIsolation, MemoryPalaceRoom
from src.p2p.bonfire_consensus import FractalDiscontinuityEmbedding
from src.core.orchestrator import UniversalOrchestrator
from src.core.situational_batching import SituationalBatchSampler
from src.augmentation.mandelbulb_gyroidic_augmenter import MandelbulbEmbedder
from src.core.diegetic_physics_engine import DiegeticPhysicsEngine


class TestEinsteinMonotileIntegration(unittest.TestCase):
    def setUp(self):
        self.tessellation = get_einstein_tessellation(base_scale=1.0)

    def test_monotile_geometry(self):
        """Test canonical Hat polykite geometry and properties."""
        hat = EinsteinMonotile(scale=1.0)
        self.assertEqual(len(hat.vertices), 14)
        self.assertGreater(hat.radius, 0.5)

        # Centroid should be zero
        centroid = np.mean(hat.vertices, axis=0)
        np.testing.assert_allclose(centroid, [0.0, 0.0], atol=1e-6)

        # Transformed vertices
        trans = hat.get_transformed_vertices(center=(2.0, 3.0), angle=math.pi / 6.0)
        np.testing.assert_allclose(np.mean(trans, axis=0), [2.0, 3.0], atol=1e-5)

    def test_domain_1_voxelboxter_octree(self):
        """Domain 1: PointerlessOctree aperiodic subvoxel coordinates and shear fracture."""
        octree = PointerlessOctree(max_depth=6)
        self.assertIsNotNone(octree.einstein_tessellation)

        # Test subvoxel coordinate projection
        snapped, angle, tile_id = octree.get_monotile_subvoxel_coord(1.23, 4.56, 7.89)
        self.assertIsInstance(snapped, tuple)
        self.assertEqual(len(snapped), 3)
        self.assertIsInstance(angle, float)
        self.assertIsInstance(tile_id, int)

        # Test shear fracture with aperiodic monotile orientation
        sheared = octree.shear_voxels(use_monotile=True)
        self.assertGreaterEqual(sheared, 1)
        self.assertGreaterEqual(octree.betti_1, 1)

    def test_domain_2_zeitgeist_router_facet_grazing(self):
        """Domain 2: ZeitgeistRouter non-periodic facet grazing phase modulation."""
        dim = 16
        router = ZeitgeistRouter(dim=dim, moduli=[3, 5, 7])
        self.assertIsNotNone(router.einstein_tessellation)

        x = torch.randn(2, dim)
        g = router._facet_projections(x)
        mask_iter0 = router._grazing_mask(g, iteration=0)
        mask_iter1 = router._grazing_mask(g, iteration=1)
        self.assertEqual(mask_iter0.shape, g.shape)
        self.assertEqual(mask_iter1.shape, g.shape)

        # Modulations across iterations should be strictly non-periodic
        mod0 = router.einstein_tessellation.compute_aperiodic_facet_modulation(0, iteration=0)
        mod10 = router.einstein_tessellation.compute_aperiodic_facet_modulation(0, iteration=10)
        self.assertNotEqual(mod0, mod10)

    def test_domain_3_resonance_cavity_mode_lock_break(self):
        """Domain 3: ResonanceCavity spatial mode-lock breaking via monotile wall reflection."""
        dim = 32
        cavity = ResonanceCavity(hidden_dim=dim, num_modes=8)
        self.assertIsNotNone(cavity.einstein_tessellation)

        initial_modes = cavity.M[0].clone()
        perturbed = cavity.einstein_tessellation.break_spatial_mode_lock(initial_modes, coupling_gain=0.05)
        self.assertEqual(perturbed.shape, initial_modes.shape)
        # Nodal lines should be slightly perturbed to disrupt degenerate standing waves
        diff = torch.norm(perturbed - initial_modes).item()
        self.assertGreater(diff, 1e-4)

    def test_domain_4_cerumen_pot_isolation(self):
        """Domain 4: CerumenPotIsolation non-linear pot distribution blocking diffusion."""
        isolation = CerumenPotIsolation(threshold=0.5)
        self.assertIsNotNone(isolation.einstein_tessellation)

        # Generate aperiodic pot centers
        centers = isolation.generate_aperiodic_pot_centers(num_pots=16, dim=32, pot_spacing=2.0)
        self.assertEqual(centers.shape, (16, 32))

        # Mathematically audit straight-line diffusion channels
        is_secure, min_dist = isolation.verify_diffusion_barrier(centers, num_probe_rays=32)
        self.assertTrue(is_secure)
        self.assertGreater(min_dist, 0.0)

        # Verify MemoryPalaceRoom records the aperiodic monotile barrier
        room = MemoryPalaceRoom(room_id="room_1", center=centers[0])
        self.assertTrue(room.metadata.get("meliponini_aperiodic_monotile_barrier", False))

    def test_domain_5_bonfire_scale_invariant_sightlines(self):
        """Domain 5: FractalDiscontinuityEmbedding continuous sightlines across Matrioshka depths."""
        embedder = FractalDiscontinuityEmbedding()
        
        # Peer A at depth 3, Peer B at depth 7 (x16 scale difference)
        pos_a = (1.0, 2.0, 0.5)
        pos_b = (1.2, 2.1, 0.6)
        sightline_res = embedder.compute_monotile_sightline(
            peer_a_pos=pos_a,
            peer_a_depth=3,
            peer_b_pos=pos_b,
            peer_b_depth=7
        )
        self.assertTrue(sightline_res["is_continuous"])
        self.assertEqual(sightline_res["depth_gap"], 4)
        self.assertGreater(sightline_res["seam_coherence"], 0.0)
        self.assertGreater(sightline_res["sightline_distance"], 0.0)

    def test_compute_affordance_rehybridization(self):
        """Test compute affordance rehybridization across all 8 channels."""
        orchestrator = UniversalOrchestrator(dim=32)
        telemetry = orchestrator.get_compute_affordance_telemetry()
        
        # Verify all 8 typed channels are reported
        expected_keys = [
            "silicon_ok", "thermal_ok", "gpu_sovereign_ok", "topological_ok",
            "p2p_mesh_ok", "diegetic_ok", "algorithmic_ok", "compute_batch_ok"
        ]
        for k in expected_keys:
            self.assertIn(k, telemetry)
            self.assertIsInstance(telemetry[k], bool)

        # Test SituationalBatchSampler adapts batch size under affordance
        sampler = SituationalBatchSampler(num_samples=20, batch_size=4)
        batches = list(sampler)
        self.assertGreater(len(batches), 0)

        # Test Mandelbulb iteration with affordance check
        mandelbulb = MandelbulbEmbedder(power=8, max_iterations=20)
        coords = torch.randn(2, 4, 3)
        res = mandelbulb._mandelbulb_iteration(coords)
        self.assertEqual(res.shape, coords.shape)

        # Test DiegeticPhysicsEngine returns diegetic_affordance_ok
        physics = DiegeticPhysicsEngine(state_dim=32)
        v_state = {"angular_velocity": (0.0, 0.0, 0.0)}
        t_state = {}
        c_input = {"throttle": 0.5, "yaw": 0.1}
        phys_out = physics.process_input(v_state, t_state, c_input)
        self.assertIn("diegetic_affordance_ok", phys_out)
        self.assertIsInstance(phys_out["diegetic_affordance_ok"], bool)


if __name__ == "__main__":
    unittest.main()
