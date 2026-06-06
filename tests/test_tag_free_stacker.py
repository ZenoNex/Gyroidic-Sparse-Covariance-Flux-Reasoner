import torch
import unittest
from unittest.mock import MagicMock

from src.core.superposed_tag_stacker import SuperposedTagStacker
from src.core.archetype_engines import ArchetypalSynthesisEngine
from src.data.knowledge_ingestor import ArXivSovereignIngestor
from src.data.conversational_api_ingestor import SovereignConversationalIngestor


class TestTagFreeStackerAndIngestion(unittest.TestCase):
    def setUp(self):
        self.state_dim = 16
        self.device = 'cpu'

    def test_superposed_tag_stacker_tag_free(self):
        stacker = SuperposedTagStacker(state_dim=self.state_dim, device=self.device)
        
        # Stacker catalog empty, should return zero target
        state = torch.ones(self.state_dim)
        target = stacker.compute_composite_target(tag_weights=None, current_state=state)
        self.assertTrue(torch.allclose(target, torch.zeros(self.state_dim)))

        # Add mock tags (note: add_tag uses TextbookFilter, which we will bypass or let execute)
        # TextbookFilter.assess is mocked here to always return admissibility
        stacker.textbook_filter.assess = MagicMock()
        mock_report = MagicMock()
        mock_report.is_admissible = True
        stacker.textbook_filter.assess.return_value = mock_report

        v1 = torch.zeros(self.state_dim)
        v1[0] = 1.0 # orthogonal direction 1
        success, _ = stacker.add_tag("tag1", v1, "context context context context context")
        self.assertTrue(success)

        v2 = torch.zeros(self.state_dim)
        v2[1] = 1.0 # orthogonal direction 2
        success, _ = stacker.add_tag("tag2", v2, "context context context context context")
        self.assertTrue(success)

        # Query state pointing directly along v1
        state_v1 = torch.zeros(self.state_dim)
        state_v1[0] = 1.0

        # Run tag-free target computation
        target_v1 = stacker.compute_composite_target(tag_weights=None, current_state=state_v1)
        
        # Since state_v1 matches v1 perfectly (cos_sim = 1.0) and is orthogonal to v2 (cos_sim = 0.0),
        # target should be 1.0 * v1 + 0.0 * v2 = v1
        self.assertTrue(torch.allclose(target_v1, v1, atol=1e-5))

        # Query state pointing equally at v1 and v2
        state_mid = torch.zeros(self.state_dim)
        state_mid[0] = 1.0
        state_mid[1] = 1.0

        target_mid = stacker.compute_composite_target(tag_weights=None, current_state=state_mid)
        # Cosine similarity to both should be equal (approx 0.707)
        # So target should be approx 0.707 * v1 + 0.707 * v2
        expected_val = 1.0 / (2.0 ** 0.5)
        self.assertAlmostEqual(target_mid[0].item(), expected_val, places=4)
        self.assertAlmostEqual(target_mid[1].item(), expected_val, places=4)

    def test_archetype_engines_ombre_blending(self):
        governor = ArchetypalSynthesisEngine(state_dim=self.state_dim)
        governor.tag_stacker.textbook_filter.assess = MagicMock()
        mock_report = MagicMock()
        mock_report.is_admissible = True
        governor.tag_stacker.textbook_filter.assess.return_value = mock_report

        # Register tag
        v = torch.zeros(self.state_dim)
        v[2] = 1.0
        governor.harvest_named_coordinate("tag3", v, "context context context context context")

        # Current state orthogonal to v
        state = torch.zeros(self.state_dim)
        state[0] = 1.0

        # Run under normal environmental luminosity (1.0), should apply primer shift
        # without boundary relaxation blending (returns primed state = state + 0.1 * target)
        # Wait, since state is orthogonal to v, similarity is 0.0, so target is 0.0.
        # Let's align state with v so similarity is 1.0, making target = v.
        state_aligned = torch.zeros(self.state_dim)
        state_aligned[2] = 1.0

        res_bright = governor.run_archetypes(
            current_state=state_aligned,
            stranded_states=torch.zeros((1, self.state_dim)),
            current_mischief=0.5,
            phase_alignment=0.8,
            love_strengths=torch.tensor([1.0]),
            void_frictions=torch.tensor([0.0]),
            global_dt=1.0,
            env_luminosity=1.0,
            volitional_scalar=0.0,
            system_entropy=0.1,
            memory_trauma=0.1,
            dissonance=0.1,
            lucidity_idx=0.8,
            raw_unquantized_state=state_aligned
        )
        self.assertIsNotNone(res_bright["stacked_target"])
        self.assertTrue(torch.allclose(res_bright["stacked_target"], v))

        # Run under low environmental luminosity (0.1), should trigger BoundaryRelaxationOperator
        # which blends state * boost + target * 0.1
        res_dark = governor.run_archetypes(
            current_state=state_aligned,
            stranded_states=torch.zeros((1, self.state_dim)),
            current_mischief=0.5,
            phase_alignment=0.8,
            love_strengths=torch.tensor([1.0]),
            void_frictions=torch.tensor([0.0]),
            global_dt=1.0,
            env_luminosity=0.1,
            volitional_scalar=0.0,
            system_entropy=0.1,
            memory_trauma=0.1,
            dissonance=0.1,
            lucidity_idx=0.8,
            raw_unquantized_state=state_aligned
        )
        self.assertIsNotNone(res_dark["stacked_target"])

    def test_dynamic_fallback_steering(self):
        fossilizer = MagicMock()
        fossilizer.recover_fossils.return_value = []
        
        # Mock engine with resonance cavity
        engine = MagicMock()
        cavity = MagicMock()
        cavity.M = torch.ones((2, 2, self.state_dim))
        engine.cavity = cavity

        ingestor = ArXivSovereignIngestor(
            fossilizer=fossilizer,
            engine_dim=self.state_dim,
            device='cpu',
            engine=engine
        )

        fallback = ingestor._get_dynamic_fallback()
        self.assertIsInstance(fallback, str)
        self.assertGreater(len(fallback), 0)

    def test_sovereign_logic_steering(self):
        # Mock engine
        engine = MagicMock()
        cavity = MagicMock()
        cavity.M = torch.ones((2, 2, self.state_dim))
        engine.cavity = cavity

        ingestor = SovereignConversationalIngestor(
            repository_root="test_repo",
            device='cpu',
            engine=engine
        )

        # Mock requests inside sovereign logic
        ingestor.sovereign.ingest_stack_exchange = MagicMock(return_value=[])
        ingestor.sovereign.ingest_hacker_news = MagicMock(return_value=[])

        convs = ingestor.ingest_sovereign_logic(limit=5)
        self.assertEqual(len(convs), 0)
        ingestor.sovereign.ingest_stack_exchange.assert_called_once()
        self.assertEqual(ingestor.sovereign.ingest_hacker_news.call_count, 2)

    def test_volition_breather_conjuring(self):
        governor = ArchetypalSynthesisEngine(state_dim=self.state_dim)
        state = torch.ones(self.state_dim)
        
        # Test under low volition (<0.9), should return state unchanged (or transformed by subsequent layers)
        out_low = governor.run_archetypes(
            current_state=state,
            stranded_states=torch.zeros((1, self.state_dim)),
            current_mischief=0.5,
            phase_alignment=0.8,
            love_strengths=torch.tensor([1.0]),
            void_frictions=torch.tensor([0.0]),
            global_dt=1.0,
            env_luminosity=1.0,
            volitional_scalar=0.5, # low volition
            system_entropy=0.1,
            memory_trauma=0.1,
            dissonance=0.1,
            lucidity_idx=0.8,
            raw_unquantized_state=state
        )
        self.assertIsNotNone(out_low["active_state"])
        
        # Test under high volition (>0.9), should trigger volition_injector breather mode
        out_high = governor.run_archetypes(
            current_state=state,
            stranded_states=torch.zeros((1, self.state_dim)),
            current_mischief=0.5,
            phase_alignment=0.8,
            love_strengths=torch.tensor([1.0]),
            void_frictions=torch.tensor([0.0]),
            global_dt=1.0,
            env_luminosity=1.0,
            volitional_scalar=0.95, # high volition
            system_entropy=0.1,
            memory_trauma=0.1,
            dissonance=0.1,
            lucidity_idx=0.8,
            raw_unquantized_state=state
        )
        self.assertIsNotNone(out_high["active_state"])
        self.assertNotEqual(torch.norm(out_low["active_state"] - out_high["active_state"]).item(), 0.0)

    def test_resonance_clustering_and_dyadic_naming(self):
        import shutil
        import os
        from src.topology.embedding_graph import GyroidicGraphManager, KnowledgeFossilNode
        from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
        from src.core.energy_based_soliton_healer import EnergyBasedSolitonHealer
        
        # 1. Test Resonance Clustering
        manager = GyroidicGraphManager(dim=self.state_dim)
        
        # Add 3 nodes: node1 and node2 close, node3 far
        s1 = torch.zeros(self.state_dim)
        s1[0] = 1.0
        
        s2 = torch.zeros(self.state_dim)
        s2[0] = 0.95
        s2[1] = 0.05
        
        s3 = torch.zeros(self.state_dim)
        s3[5] = 1.0 # orthogonal / far
        
        n1 = KnowledgeFossilNode("node1", s1, "Text 1", {"chiral_score": 0.1})
        n2 = KnowledgeFossilNode("node2", s2, "Text 2", {"chiral_score": 0.1})
        n3 = KnowledgeFossilNode("node3", s3, "Text 3", {"chiral_score": 0.9})
        
        manager.nodes = [n1, n2, n3]
        
        clusters = manager.find_resonance_clusters()
        self.assertEqual(len(clusters), 1) # should find 1 cluster of node1 and node2
        self.assertIn("node1", clusters[0])
        self.assertIn("node2", clusters[0])
        
        # Test Cluster Healing
        healer = EnergyBasedSolitonHealer(state_dim=self.state_dim)
        prev_n1_state = n1.state.clone()
        prev_n2_state = n2.state.clone()
        
        results = manager.heal_resonance_clusters(healer)
        self.assertIn("cluster_0", results)
        
        # Nodes should have moved closer to the centroid / healed state
        self.assertFalse(torch.allclose(n1.state, prev_n1_state))
        self.assertFalse(torch.allclose(n2.state, prev_n2_state))
        self.assertTrue(n1.metrics.get('healed_in_cluster', False))
        
        # 2. Test Dynamic Dyadic Naming
        temp_dir = "data/test_fossil_temp"
        os.makedirs(temp_dir, exist_ok=True)
        try:
            fossilizer = DyadFossilizer(storage_dir=temp_dir, feature_dim=self.state_dim)
            dyad = KnowledgeDyad(
                linguistic_description="This is a pomni dyad description",
                image_fingerprint=torch.zeros(96)
            )
            text_emb = torch.ones(self.state_dim)
            filepath = fossilizer.fossilize(dyad, text_emb, seed_state=torch.ones(self.state_dim))
            
            # Filename should use original descriptive safe_desc format
            filename = os.path.basename(filepath)
            self.assertTrue(filename.startswith("encoding_Thisisapomnidyad"))
            self.assertTrue(os.path.exists(filepath))
            
            # Check payload tags
            data = torch.load(filepath)
            self.assertIn("tags", data)
            self.assertIn("pomni", data["tags"])
            
            # Test agent smith dynamic naming with 2D prime_frequencies
            filepath_smith = fossilizer.export_agent_smith(
                dyad=dyad,
                prime_frequencies=torch.ones(1, self.state_dim),
                betti_numbers={0: 1.0},
                filename="soliton_smith"
            )
            filename_smith = os.path.basename(filepath_smith)
            self.assertEqual(filename_smith, "soliton_smith.pt")
            self.assertTrue(os.path.exists(filepath_smith))
            
            # Check agent smith payload tags
            data_smith = torch.load(filepath_smith)
            self.assertIn("tags", data_smith)
            self.assertIn("pomni", data_smith["tags"])
            
            # Additional check for typographical character tagging
            dyad_b = KnowledgeDyad(
                linguistic_description="fossil B",
                image_fingerprint=torch.zeros(96)
            )
            filepath_b = fossilizer.fossilize(dyad_b, text_emb, seed_state=torch.ones(self.state_dim))
            data_b = torch.load(filepath_b)
            self.assertIn("tags", data_b)
            self.assertIn("B", data_b["tags"])
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
