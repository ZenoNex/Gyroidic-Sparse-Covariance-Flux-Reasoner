import unittest
import torch
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestSplatAffordanceIntegration(unittest.TestCase):
    """Test suite verifying DMCA-safe generative splat interception and compute affordance integration."""

    def test_compute_affordance_integration(self):
        """Verify UniversalOrchestrator and hardware monitor compute affordance bindings."""
        from src.core.hardware_monitor import (
            has_compute_affordance,
            ComputeAffordanceType,
            check_compute_affordance,
            get_compute_affordance_telemetry
        )
        from src.core.orchestrator import UniversalOrchestrator

        self.assertTrue(callable(has_compute_affordance))
        self.assertTrue(callable(check_compute_affordance))
        self.assertTrue(callable(get_compute_affordance_telemetry))

        # Check telemetry
        telemetry = get_compute_affordance_telemetry()
        self.assertIn("affordances", telemetry)
        self.assertIn("gauges", telemetry)

        # Check orchestrator affordance wiring
        orchestrator = UniversalOrchestrator(dim=16)
        self.assertTrue(hasattr(orchestrator, "affordance_gauges"))
        self.assertTrue(hasattr(orchestrator, "affordance_status"))
        self.assertTrue(hasattr(orchestrator, "check_affordance"))
        
        # Silicon/CPU affordance should be available
        silicon_affordance = orchestrator.check_affordance(ComputeAffordanceType.SILICON)
        self.assertTrue(silicon_affordance)

    def test_dmca_safe_generative_splat_interception(self):
        """Verify that VoxelboxterEngine intercepts raw Gaussian Splats via AdaptiveSkeletonHarness."""
        from src.ui.voxelboxter_simulation import VoxelboxterEngine

        engine = VoxelboxterEngine()

        # Mock Gaussian Splat records
        raw_splats = [
            {
                "x": 1.0, "y": 2.0, "z": 3.0,
                "r": 200, "g": 100, "b": 50,
                "scale_x": 0.2, "scale_y": 0.2, "scale_z": 0.2,
                "opacity": 0.9,
                "pressure_potential": 0.75
            },
            {
                "x": 1.5, "y": 2.5, "z": 3.5,
                "r": 50, "g": 150, "b": 220,
                "scale_x": 0.3, "scale_y": 0.3, "scale_z": 0.3,
                "opacity": 0.85,
                "pressure_potential": 0.82
            }
        ]

        result = engine.ingest_gaussian_splats(raw_splats, default_material=5)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["dmca_safe_generative_intercept"])
        self.assertEqual(result["pipeline"], "AdaptiveSkeletonHarness_Bouligand_Ganbreeder")
        self.assertIn("blocks_placed", result)
        self.assertIn("bone_voxel_count", result)
        self.assertIn("flesh_voxel_count", result)
        self.assertGreater(result["blocks_placed"], 0)

    def test_matrioshka_mandelbulb_partitioning_affordance(self):
        """Verify Matrioshka Mandelbulb partitioning and compute batching affordance positioning."""
        from src.core.hardware_monitor import (
            evaluate_matrioshka_mandelbulb_affordance,
            can_afford_matrioshka_partitioning,
            ComputeAffordanceType,
            get_typed_affordance
        )
        from src.core.adaptive_skeleton_harness import AdaptiveSkeletonHarness

        # 1. Hardware monitor positioning
        mb_aff = evaluate_matrioshka_mandelbulb_affordance(task_complexity=1.5, dt=0.5, demand_dim=32)
        self.assertIn("can_afford_partitioning", mb_aff)
        self.assertIn("optimal_matrioshka_shells", mb_aff)
        self.assertIn("batch_capacity", mb_aff)
        self.assertIn("amortization_factor", mb_aff)
        self.assertTrue(mb_aff["can_afford_partitioning"])
        self.assertGreaterEqual(mb_aff["optimal_matrioshka_shells"], 2)

        # 2. Typed batch affordance
        batch_tok = get_typed_affordance(ComputeAffordanceType.COMPUTE_BATCH)
        self.assertTrue(bool(batch_tok))
        self.assertGreater(batch_tok.gauge, 0.0)

        # 3. Adaptive skeleton harness rig synthesis without veto
        harness = AdaptiveSkeletonHarness(state_dim=32, device="cpu")
        demand_test = torch.ones(32) * 5.0
        harness.register_tag_coordinate("high_value_target", demand_test)
        rig = harness.build_superposed_rig(weights={"high_value_target": 1.0})
        self.assertIsNotNone(rig)
        self.assertEqual(rig.shape, torch.Size([32]))

    def test_canonical_demand_synthesis_from_splats(self):
        """Verify that synthesize_demand_from_splats leverages ConjugateMomentTransport."""
        from src.core.adaptive_skeleton_harness import AdaptiveSkeletonHarness

        harness = AdaptiveSkeletonHarness(state_dim=32, device="cpu")
        splats = [
            {"x": 1.0, "y": 2.0, "z": 3.0, "r": 255, "g": 100, "b": 50, "pressure_potential": 0.8},
            {"x": 2.0, "y": 4.0, "z": 6.0, "r": 50, "g": 200, "b": 150, "pressure_potential": 0.6},
            {"x": 3.0, "y": 6.0, "z": 9.0, "r": 20, "g": 30, "b": 240, "pressure_potential": 0.4},
        ]

        demand = harness.synthesize_demand_from_splats(splats)
        self.assertEqual(demand.shape, torch.Size([32]))
        # Spatial moments (mean_x, mean_y, mean_z) = (2.0, 4.0, 6.0)
        self.assertAlmostEqual(demand[0].item(), 2.0, places=4)
        self.assertAlmostEqual(demand[1].item(), 4.0, places=4)
        self.assertAlmostEqual(demand[2].item(), 6.0, places=4)
        # Moments beyond 6 should contain OKLab and Log-Cholesky covariance invariant signatures
        self.assertTrue(torch.any(demand[6:15] != 0.0))

    def test_octree_splat_ingestion_and_pipeline(self):
        """Verify PointerlessOctree splat ingestion and GltfSplatIngestionPipeline extraction."""
        from src.ui.voxelboxter_simulation import PointerlessOctree
        from src.data.gltf_splat_ingestor import GltfSplatIngestionPipeline

        pipeline = GltfSplatIngestionPipeline()
        # Test nonexistent file graceful fallback generates safe fallback splats
        fallback_primitives = pipeline.extract_splat_primitives("nonexistent_test_model.glb")
        self.assertGreater(len(fallback_primitives), 0)
        self.assertIn("x", fallback_primitives[0])

        octree = PointerlessOctree(max_depth=6)
        splats = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "r": 180, "g": 90, "b": 45, "scale_x": 0.1, "opacity": 0.95},
            {"x": 1.0, "y": 1.0, "z": 1.0, "r": 90, "g": 180, "b": 45, "scale_x": 0.1, "opacity": 0.95},
        ]
        res = octree.ingest_gaussian_splats(splats, default_material=2, construct_name="test_octree_construct")
        self.assertEqual(res["status"], "success")
        self.assertTrue(len(octree.morton_grid) > 0)


if __name__ == '__main__':
    unittest.main()

