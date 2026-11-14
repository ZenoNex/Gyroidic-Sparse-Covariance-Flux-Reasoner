"""
tests/test_manifold_bridge.py
Verification suite for the Ingestion Manifold Bridge:
- Clean, non-crashing initialization of DatasetIngestionSystem
- Silicon Sovereignty device resolution (PyOpenCL host staging / CPU substrate independence)
- Fallback and direct operational capability of LightweightManifoldBridge
- Manifold-aware (thick) sample preprocessing with residue vectors and step tracking
"""

import unittest
import torch
from dataset_ingestion_system import DatasetIngestionSystem, DatasetConfig, LightweightManifoldBridge


class TestManifoldBridge(unittest.TestCase):
    def test_lightweight_manifold_bridge_standalone(self):
        bridge = LightweightManifoldBridge(dim=256, k=5, device='cpu')
        self.assertEqual(bridge.dim, 256)
        self.assertEqual(bridge.k, 5)
        self.assertEqual(bridge.meta_state.shape, (1, 256))
        
        result = bridge.process_input("Topological covariance flux across gyroidic boundary", ingestion_mode=True)
        self.assertIn("residue_vector", result)
        self.assertIn("manifold_step", result)
        self.assertEqual(result["manifold_step"], 1)
        self.assertEqual(len(result["residue_vector"]), 256)
        self.assertTrue(torch.isfinite(torch.tensor(result["residue_vector"])).all())

    def test_dataset_ingestion_system_initialization(self):
        system = DatasetIngestionSystem(device='auto')
        self.assertIsNotNone(system.engine)
        self.assertEqual(str(system.device), 'cpu')
        self.assertTrue(hasattr(system.engine, 'process_input'))
        self.assertTrue(hasattr(system.engine, 'iteration'))

    def test_manifold_aware_sample_preprocessing(self):
        system = DatasetIngestionSystem(device='auto')
        config = DatasetConfig(
            name="test_manifold_dataset",
            source_type="local",
            source_path="data",
            preprocessing="text",
            manifold_aware=True
        )
        sample = {
            "text": "The Riemann curvature tensor vanishes on flat affine connections.",
            "source": "differential_geometry"
        }
        processed = system._preprocess_sample(sample, "text", config=config)
        self.assertIsNotNone(processed)
        self.assertIn("metadata", processed)
        metadata = processed["metadata"]
        self.assertIn("residue_vector", metadata)
        self.assertIn("manifold_step", metadata)
        self.assertIsInstance(metadata["residue_vector"], list)
        self.assertEqual(len(metadata["residue_vector"]), 256)


if __name__ == "__main__":
    unittest.main()
