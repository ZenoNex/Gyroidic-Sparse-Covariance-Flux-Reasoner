import unittest
import torch
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestRehybridizationIslands(unittest.TestCase):
    """Test suite verifying the reintegration of the 6 isolated architectural islands."""

    def test_island_1_physical_scripting(self):
        """Island 1: Physical Scripting & Simulation."""
        from src.scripting.node_environment import ResourceRegistry, OreDictionary
        registry = ResourceRegistry()
        resources = registry.all_resources()
        self.assertGreater(len(resources), 0)
        
        ore_dict = OreDictionary()
        iron_ores = ore_dict.get_materials("oreIron")
        self.assertIn(10, iron_ores)
        self.assertTrue(ore_dict.matches_tag(10, "oreIron"))

    def test_island_2_adaptive_training_energy(self):
        """Island 2: Adaptive Training & Structural Energy Scaffolding."""
        from src.training.trainer import StructuralAdaptor, ConstraintDataset
        from src.core.energy_monitor import StructuralEnergyMonitor
        from src.core.negentropic_manifold import NTMOperator
        from src.core.polynomial_scaffold import PolynomialCoefficientFunctional

        # Verify Energy Monitor
        monitor = StructuralEnergyMonitor(device='cpu')
        current_p = torch.tensor([0.2])
        alt_p = torch.tensor([[0.5, 0.8]])
        monitor.update(current_p, alt_p)
        self.assertGreater(monitor.get_gap(), 0.0)
        metrics = monitor.get_metrics()
        self.assertIn('energy_gap', metrics)
        self.assertIn('free_energy', metrics)

        # Verify NTM and Scaffolding
        ntm = NTMOperator(dim=32, degree=4, device='cpu')
        scaffold = PolynomialCoefficientFunctional(device='cpu')
        self.assertIsNotNone(ntm)
        self.assertIsNotNone(scaffold)

    def test_island_3_signal_dsp_kinematics(self):
        """Island 3: Signal DSP, Enhanced Bezout CRT & High-Speed Kinematics."""
        from src.core.enhanced_bezout_crt import EnhancedBezoutCRT, CrossbarIKSolver
        from src.models.modular_embeddings import get_canonical_embedder
        
        crt = EnhancedBezoutCRT(state_dim=32, num_moduli=5)
        self.assertEqual(len(crt.moduli), 5)
        self.assertEqual(len(crt.crt_coefficients), 5)
        
        ik_solver = CrossbarIKSolver(num_harmonics=4)
        t = torch.linspace(0, 1, 10)
        traj = ik_solver(t)
        self.assertEqual(traj.shape, (10, 2))

        # Canonical embedder bridge
        embedder = get_canonical_embedder(text_dim=32, graph_dim=32, num_dim=32, hidden_dim=32)
        self.assertIsNotNone(embedder)

    def test_island_4_safety_oracle_evidence(self):
        """Island 4: Safety & Subversive Oracle."""
        from src.safety.subversive_oracle import ResonantSVNNOracle
        from src.core.evidence_modules import TripartiteEvidenceModule
        
        evidence = TripartiteEvidenceModule(hidden_dim=32)
        state = torch.randn(1, 32)
        ev_metrics = evidence(state)
        self.assertIn('data_evidence', ev_metrics)
        self.assertIn('architectural_evidence', ev_metrics)
        self.assertIn('adversarial_evidence', ev_metrics)
        self.assertIn('is_blind', ev_metrics)

        oracle = ResonantSVNNOracle(hidden_dim=32)
        voxel_state = torch.randn(1, 4, 32)
        subversion, toxicity = oracle(voxel_state)
        self.assertEqual(subversion.shape, (1,))
        self.assertEqual(toxicity.shape, (1,))

    def test_island_5_secondary_ui_ingestion(self):
        """Island 5: Secondary UI & Unrooted Ingestion."""
        from src.ui.wikipedia_integration import wikipedia_integration
        title = wikipedia_integration.extract_title_from_url("https://en.wikipedia.org/wiki/Chinese_remainder_theorem")
        self.assertIn("remainder", title.lower())

        from src.data.webp_prompt_extractor import WebPPromptExtractor
        extractor = WebPPromptExtractor("data")
        self.assertIsNotNone(extractor)

    def test_island_6_augmentation_vision_surgery(self):
        """Island 6: Augmentation & Vision Surgery."""
        from src.codec.vision_surgery import IntercosaminationOperator, MirrorTestProbe
        from src.augmentation.mandelbulb_pipeline import create_mandelbulb_pipeline

        surgeon = IntercosaminationOperator(cnn_dim=32, gyroid_dim=32)
        mirror = MirrorTestProbe(cnn_dim=32, gyroid_dim=32)
        flesh = torch.randn(1, 32)
        bone = torch.randn(1, 32)
        
        interlaced = surgeon(flesh, bone)
        self.assertEqual(interlaced.shape, (1, 32))
        
        pas_h, valid = mirror(interlaced, bone)
        self.assertTrue(torch.is_tensor(pas_h))

        # Dataset pipeline
        mock_data = [(torch.randn(32), torch.tensor(0.5)) for _ in range(5)]
        loader = create_mandelbulb_pipeline(mock_data, batch_size=2)
        self.assertIsNotNone(loader)


if __name__ == '__main__':
    unittest.main()
