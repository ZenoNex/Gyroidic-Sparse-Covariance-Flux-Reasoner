import os
import sys
import unittest
import torch
import torch.nn.functional as F

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.audience_mapping import AudienceProjection
from src.core.garden_statistical_attractors import GardenOrchestrator
from src.training.temporal_association_trainer import TemporalAssociationDataset, TemporalAssociationTrainer


class TestConceptModelingAudienceIntegration(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.feature_dim = 64
        self.dataset = TemporalAssociationDataset(device=self.device)
        self.trainer = TemporalAssociationTrainer(model=None, dataset=self.dataset)
        self.audience_mapper = AudienceProjection(input_dim=self.feature_dim, audience_dim=self.feature_dim)
        self.garden = GardenOrchestrator(num_attractors=4, feature_dim=self.feature_dim, device=self.device)

    def test_audience_projection_concept_alignment(self):
        """Verify AudienceProjection handles arbitrary dimensions and preserves Betti-1 homology."""
        # 1. Exact match dimension
        exact_state = torch.randn(1, self.feature_dim)
        proj_exact = self.audience_mapper.project_concept(exact_state)
        self.assertEqual(proj_exact.shape[-1], self.feature_dim)

        # 2. Arbitrary higher dimension (e.g., 768)
        high_dim_state = torch.randn(1, 768)
        proj_high = self.audience_mapper.project_concept(high_dim_state)
        self.assertEqual(proj_high.shape[-1], self.feature_dim)

        # 3. Arbitrary lower dimension (e.g., 32)
        low_dim_state = torch.randn(1, 32)
        proj_low = self.audience_mapper.project_concept(low_dim_state)
        self.assertEqual(proj_low.shape[-1], self.feature_dim)

        # 4. Standard forward call alignment
        out_fwd = self.audience_mapper(high_dim_state)
        self.assertEqual(out_fwd.shape[-1], self.feature_dim)

    def test_temporal_association_concept_prediction(self):
        """Verify TemporalAssociationDataset and Trainer predict_next_concept and trajectories."""
        # Seed state
        current_state = torch.randn(1, self.feature_dim)

        # Single step next concept prediction
        next_emb, next_id, affinity = self.trainer.predict_next_concept(current_state, context_type="default")
        self.assertIsNotNone(next_id)
        self.assertGreater(affinity, 0.0)
        self.assertEqual(next_emb.shape[-1], 768)

        # Project predicted 768-dim concept through audience mapper into feature_dim space
        proj_audience = self.audience_mapper.project_concept(next_emb)
        self.assertEqual(proj_audience.shape[-1], self.feature_dim)

        # Multi-step trajectory prediction
        steps = 4
        trajectory = self.trainer.predict_concept_trajectory(current_state, num_steps=steps, context_type="default")
        self.assertEqual(len(trajectory), steps)
        for i, step_info in enumerate(trajectory):
            self.assertEqual(step_info["step"], i + 1)
            self.assertIn("concept_id", step_info)
            self.assertIn("affinity", step_info)
            self.assertEqual(len(step_info["embedding"]), 768)

    def test_garden_evolution_and_audience_coupling(self):
        """Verify GardenOrchestrator evolves concepts and projects into audience space."""
        concepts = torch.randn(2, self.feature_dim)
        garden_res = self.garden.evolve_garden(concepts)

        self.assertIn('evolved_concepts', garden_res)
        self.assertIn('entropy', garden_res)
        self.assertIn('resonance_strength', garden_res)
        self.assertEqual(garden_res['evolved_concepts'].shape, concepts.shape)

        # Project evolved concepts into audience space
        audience_state = self.audience_mapper(garden_res['evolved_concepts'])
        self.assertEqual(audience_state.shape, concepts.shape)
        self.assertTrue(torch.isfinite(audience_state).all())

    def test_memory_palace_room_planting(self):
        """Verify paired association plants an isolated room in MemoryPalace."""
        assoc_vector = torch.randn(self.feature_dim)
        room_id = "test_assoc_chamber_01"
        room = self.garden.memory_palace.plant_room(
            room_id=room_id,
            center=assoc_vector,
            metadata={"source": "gyroidic", "target": "flux"}
        )
        self.assertIn(room_id, self.garden.memory_palace.rooms)
        self.assertEqual(self.garden.memory_palace.rooms[room_id].center.shape[0], self.feature_dim)
        self.assertEqual(self.garden.memory_palace.rooms[room_id].metadata["source"], "gyroidic")


if __name__ == '__main__':
    unittest.main()
