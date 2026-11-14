import unittest
import torch
import os
import shutil
from src.core.orchestrator import UniversalOrchestrator
from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer

class TestKnowledgeDyadRehybridization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_encodings_dir = "data/test_encodings_dyad_rehyb"
        os.makedirs(cls.test_encodings_dir, exist_ok=True)
        cls.orchestrator = UniversalOrchestrator(dim=32, device="cpu")
        cls.orchestrator.fossilizer = DyadFossilizer(storage_dir=cls.test_encodings_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_encodings_dir):
            shutil.rmtree(cls.test_encodings_dir, ignore_errors=True)

    def test_standard_dyad_ingest_and_fossilization(self):
        """Verify standard ingestion builds KnowledgeDyad and persists to fossilizer."""
        text_emb = torch.randn(1, 32)
        image_emb = torch.randn(1, 32)
        
        diag = self.orchestrator.ingest_knowledge_dyad_diagnostics(
            text_emb,
            image_emb,
            text_raw="Test dyad description for topological embedding",
            fossilize=True
        )
        
        self.assertIn("residues", diag)
        self.assertEqual(diag["residues"].shape, (1, 5))
        self.assertIsNotNone(diag["dyad"])
        self.assertIsInstance(diag["dyad"], KnowledgeDyad)
        self.assertEqual(diag["dyad"].linguistic_description, "Test dyad description for topological embedding")
        self.assertIsNotNone(diag["fossil_path"])
        self.assertTrue(os.path.exists(diag["fossil_path"]))

    def test_raw_knowledge_dyad_ingest(self):
        """Verify direct ingestion of pre-formed KnowledgeDyad instances."""
        dyad = KnowledgeDyad(
            linguistic_description="Preformed Raw Knowledge Dyad",
            image_fingerprint=torch.randn(32),
            metadata={"source": "test_preformed"}
        )
        residues = self.orchestrator.ingest_raw_dyad(dyad, fossilize=False)
        self.assertEqual(residues.shape, (1, 5))
        self.assertEqual(self.orchestrator.last_dyad.linguistic_description, "Preformed Raw Knowledge Dyad")

    def test_arxiv_dyad_ingest(self):
        """Verify ArXiv preprint ingestion via specialized bridge."""
        residues = self.orchestrator.ingest_arxiv_dyad(
            arxiv_id="2603.13370",
            title="Non-Abelian CRT Holonomy on Minimal Surfaces",
            abstract="Topological proof of non-commutative energy conservation.",
            categories=["math.DG", "cs.AI"],
            fossilize=True
        )
        self.assertEqual(residues.shape, (1, 5))
        self.assertIsNotNone(self.orchestrator.last_dyad)
        self.assertIn("2603.13370", self.orchestrator.last_dyad.linguistic_description)
        self.assertEqual(self.orchestrator.last_dyad.metadata["source"], "arxiv")
        self.assertIsNotNone(self.orchestrator.last_fossil_path)
        self.assertTrue(os.path.exists(self.orchestrator.last_fossil_path))

    def test_conversational_dyad_ingest(self):
        """Verify dialogue dyad ingestion via specialized bridge."""
        residues = self.orchestrator.ingest_conversational_dyad(
            user_text="What is the non-Abelian commutator gap?",
            assistant_text="The non-Abelian commutator gap measures ||A*B - B*A|| on GL(n).",
            conversation_id="conv_42",
            fossilize=True
        )
        self.assertEqual(residues.shape, (1, 5))
        self.assertIn("User:", self.orchestrator.last_dyad.linguistic_description)
        self.assertEqual(self.orchestrator.last_dyad.metadata["conversation_id"], "conv_42")

    def test_investor_news_dyad_ingest(self):
        """Verify financial news topology ingestion via specialized bridge."""
        residues = self.orchestrator.ingest_investor_news_dyad(
            headline="Sovereign Bond Yield Curve Normalization",
            content="Topological stress metrics contract across short-duration debt.",
            source_feed="macro_daily",
            fossilize=True
        )
        self.assertEqual(residues.shape, (1, 5))
        self.assertIn("MARKET TOPOLOGY:", self.orchestrator.last_dyad.linguistic_description)
        self.assertEqual(self.orchestrator.last_dyad.metadata["feed"], "macro_daily")

    def test_video_dyad_ingest(self):
        """Verify video bitstream ingestion via specialized bridge."""
        # Provide simulated video dummy base64
        import base64
        dummy_bytes = b"ftypmp42\x00\x00\x00\x00moov\x00\x00\x00\x10mdat\x01\x02\x03\x04"
        dummy_b64 = base64.b64encode(dummy_bytes).decode("ascii")
        
        residues = self.orchestrator.ingest_video_dyad(
            video_b64=dummy_b64,
            description="Quantum Gyroidic Simulation Video",
            extract_audio=False,
            fossilize=True
        )
        self.assertEqual(residues.shape, (1, 5))
        self.assertIn("Quantum Gyroidic Simulation Video", self.orchestrator.last_dyad.linguistic_description)
        self.assertEqual(self.orchestrator.last_dyad.metadata["source"], "video")

if __name__ == "__main__":
    unittest.main()
