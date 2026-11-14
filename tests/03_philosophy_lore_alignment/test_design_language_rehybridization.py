import unittest
import torch
import tempfile
from pathlib import Path

from src.core.orchestrator import UniversalOrchestrator
from src.ui.diegetic_backend import DiegeticPhysicsEngine, HybridPhysicsBackend
from src.models.polynomial_embeddings import PolynomialFunctionalEmbedder
from src.codec.vision_utilities import project_multimodal_fingerprint_to_96

class TestDesignLanguageRehybridization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dim = 32
        cls.orchestrator = UniversalOrchestrator(dim=cls.dim)

    def test_poly_embedder_bound_and_callable(self):
        self.assertTrue(hasattr(self.orchestrator, 'poly_embedder'))
        self.assertIsNotNone(self.orchestrator.poly_embedder)
        self.assertIsInstance(self.orchestrator.poly_embedder, PolynomialFunctionalEmbedder)
        
        # Test direct text call
        out = self.orchestrator.poly_embedder("Topological manifold resonance test")
        self.assertIn("fused_hidden", out)
        fused = out["fused_hidden"]
        self.assertFalse(torch.all(fused == 0), "Fused embedding should be non-zero for real text")

    def test_ingest_knowledge_dyad_with_text_string(self):
        dyad_tensor = self.orchestrator.ingest_knowledge_dyad(
            text_raw="Non-Abelian holonomy in gyroidic boundary shells",
            modality="TextOnly",
            fossilize=False
        )
        self.assertIsInstance(dyad_tensor, torch.Tensor)
        self.assertEqual(dyad_tensor.shape[-1], self.orchestrator.poly_config.k)
        self.assertFalse(torch.all(dyad_tensor == 0), "Text ingestion must not collapse to zero tensor")

    def test_ingest_knowledge_dyad_with_multimodal_fingerprint(self):
        # Dict with streamlines/primitives
        streamlines_payload = {
            "streamlines": [
                [[0.0, 0.0], [0.1, 0.2], [0.3, 0.5]],
                [[0.2, 0.1], [0.4, 0.3], [0.6, 0.7]]
            ]
        }
        dyad_tensor = self.orchestrator.ingest_knowledge_dyad(
            text_raw="Streamline filaments of phase velocity",
            image_emb=streamlines_payload,
            modality="Streamlines",
            fossilize=False
        )
        self.assertIsInstance(dyad_tensor, torch.Tensor)
        self.assertEqual(dyad_tensor.shape[-1], self.orchestrator.poly_config.k)
        self.assertFalse(torch.all(dyad_tensor == 0))

    def test_bio_governance_captured_in_forward(self):
        state_input = torch.randn(1, self.dim)
        out_dict = self.orchestrator.step(state_input)
        self.assertIsInstance(out_dict, dict)
        
        # Verify bio_governance was captured in telemetry_state
        self.assertIn("bio_governance", self.orchestrator.telemetry_state)
        bio_gov = self.orchestrator.telemetry_state["bio_governance"]
        self.assertIsInstance(bio_gov, dict)
        self.assertIn("precision_matrix", bio_gov)
        self.assertIn("neuro_bus", bio_gov)
        self.assertIn("jax_rigidity", bio_gov)
        self.assertIn("panic", bio_gov)
        self.assertIn("noradrenaline", bio_gov["neuro_bus"])
        self.assertIn("gaba", bio_gov["neuro_bus"])

    def test_universal_artifact_ingestion(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Universal Topology Ingestion Verification Artifact with non-Abelian resonance.")
            f_path = f.name
            
        try:
            tensor = self.orchestrator.ingest_universal_artifact(f_path, fossilize=False)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape[-1], self.orchestrator.poly_config.k)
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_diegetic_backend_poly_embedder_binding(self):
        engine = DiegeticPhysicsEngine(dim=self.dim, device='cpu')
        self.assertTrue(hasattr(engine, 'poly_embedder'))
        self.assertIsNotNone(engine.poly_embedder)
        
        backend = HybridPhysicsBackend(state_dim=self.dim, device='cpu')
        backend.bind_engine(self.orchestrator)
        self.assertTrue(hasattr(backend, 'poly_embedder'))
        self.assertEqual(backend.poly_embedder, self.orchestrator.poly_embedder)

if __name__ == '__main__':
    unittest.main()
