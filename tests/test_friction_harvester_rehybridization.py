"""
Unit tests for ChatGPTFrictionHarvester rehybridization across bioplausible systems,
search gates, Windows slow-drip memory ingestion, and community compute federation.
"""

import unittest
import torch
import os
import shutil

from src.data.chatgpt_friction_harvester import ChatGPTFrictionHarvester, is_semisimple_or_sterile
from src.core.archetype_engines import SovereignRefusalOperator
from src.core.orchestrator import UniversalOrchestrator
from src.core.knowledge_dyad_fossilizer import DyadFossilizer


class TestFrictionHarvesterRehybridization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "data/test_friction_rehyb"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.harvester = ChatGPTFrictionHarvester(export_dir=cls.test_dir, dim=32)
        cls.orchestrator = UniversalOrchestrator(dim=32, device="cpu")
        cls.orchestrator.fossilizer = DyadFossilizer(storage_dir=cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_bioplausible_friction_head_query(self):
        """Verify query_friction_head returns signals for SovereignRefusalOperator."""
        signals = self.harvester.query_friction_head(
            user_text="What is the non-commutative Berry phase on this gyroid?",
            assistant_text="It measures geometric phase rotation."
        )
        self.assertIn("pas_modifier", signals)
        self.assertIn("rebicameral_drift_detected", signals)
        self.assertIn("splitting_pressure", signals)
        self.assertIn("veto_signal", signals)
        self.assertIn("jaccard_coherence", signals)
        self.assertIsInstance(signals["pas_modifier"], float)

    def test_mandy_sovereign_refusal_friction_integration(self):
        """Verify SovereignRefusalOperator audits friction head signals."""
        mandy = SovereignRefusalOperator(training_mode=False)
        friction_signals = {
            "pas_modifier": 0.3,
            "rebicameral_drift_detected": True,
            "splitting_pressure": 0.8,
            "veto_signal": False
        }
        
        state = torch.ones(1, 32)
        # Low PAS_h with rebicameral drift triggers Mandy refusal
        veto_state = mandy(
            state,
            phase_alignment=0.25,
            mischief_harmonics=0.2,
            friction_signals=friction_signals
        )
        self.assertEqual(veto_state.abs().sum().item(), 0.0)

    def test_orchestrator_conversational_dyad_with_friction_enrichment(self):
        """Verify UniversalOrchestrator enriches conversational dyads with friction tags."""
        residue = self.orchestrator.ingest_conversational_dyad(
            user_text="Akkaris exploring topological non-ergodic agreement",
            assistant_text="Agreement verified across the manifold.",
            conversation_id="conv_friction_test",
            fossilize=True
        )
        self.assertEqual(residue.shape, (1, 5))
        last_dyad = self.orchestrator.last_dyad
        self.assertIsNotNone(last_dyad)
        self.assertIn("friction_signals", last_dyad.metadata)

    def test_larynx_query_generation(self):
        """Verify ArXivSovereignIngestor._generate_larynx_query succeeds without errors."""
        from src.data.knowledge_ingestor import ArXivSovereignIngestor
        ingestor = ArXivSovereignIngestor(
            fossilizer=self.orchestrator.fossilizer,
            engine_dim=32,
            device="cpu"
        )
        query = ingestor._generate_larynx_query()
        self.assertIsInstance(query, str)
        self.assertTrue(len(query) > 0)


if __name__ == '__main__':
    unittest.main()
