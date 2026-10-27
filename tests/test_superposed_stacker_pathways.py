import torch
import unittest
from unittest.mock import MagicMock

from src.core.superposed_tag_stacker import SuperposedTagStacker
from src.core.speculative_coprime_gate import SpeculativeCoprimeGate


class TestSuperposedStackerPathways(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.device = 'cpu'
        self.stacker = SuperposedTagStacker(state_dim=self.dim, device=self.device)
        self.stacker.textbook_filter.assess = MagicMock()
        mock_report = MagicMock()
        mock_report.is_admissible = True
        self.stacker.textbook_filter.assess.return_value = mock_report

        v1 = torch.zeros(self.dim)
        v1[0] = 1.0
        self.stacker.add_tag("alpha_pole", v1, "instructive textbook context")

        v2 = torch.zeros(self.dim)
        v2[1] = 1.0
        self.stacker.add_tag("beta_pole", v2, "instructive textbook context")

    def test_multi_scalar_superposition_and_subtraction(self):
        target = self.stacker.compute_composite_target(tag_weights={"alpha_pole": 0.8, "beta_pole": -0.5})
        self.assertAlmostEqual(target[0].item(), 0.8, places=4)
        self.assertAlmostEqual(target[1].item(), -0.5, places=4)

    def test_hyperbolic_sine_quotient_stacking(self):
        # Val as dictionary with u and gamma
        target = self.stacker.compute_composite_target(tag_weights={"alpha_pole": {"u": 0.5, "gamma": 2.0}})
        import math
        expected = math.sinh(2.0 * 0.5) / math.sinh(2.0)
        self.assertAlmostEqual(target[0].item(), expected, places=4)

    def test_breed_into_state_pathway(self):
        state = torch.ones(self.dim)
        bred = self.stacker.breed_into_state(
            active_state=state,
            tag_weights={"alpha_pole": 1.0},
            alpha=0.5
        )
        self.assertEqual(bred.shape, state.shape)
        self.assertFalse(torch.isnan(bred).any())

    def test_route_to_speculative_gate_pathway(self):
        gate = SpeculativeCoprimeGate(dim=self.dim)
        converged = torch.randn(2, self.dim)
        recovered, metrics = self.stacker.route_to_speculative_gate(
            gate=gate,
            converged_state=converged,
            tag_weights={"alpha_pole": 1.0, "beta_pole": 0.5},
            mode='PLAY'
        )
        self.assertEqual(recovered.shape, converged.shape)
        self.assertIn('chiral_score', metrics)

    def test_nested_macro_coordinate_stacking(self):
        success, report = self.stacker.save_macro_coordinate(
            macro_name="composite_macro",
            tag_weights={"alpha_pole": 0.6, "beta_pole": 0.8},
            context_text="Nested composite macro stack text"
        )
        self.assertTrue(success)
        self.assertIn("composite_macro", self.stacker)
        target = self.stacker.compute_composite_target(tag_weights={"composite_macro": 1.0})
        self.assertAlmostEqual(target[0].item(), 0.6, places=4)
        self.assertAlmostEqual(target[1].item(), 0.8, places=4)


if __name__ == "__main__":
    unittest.main()
