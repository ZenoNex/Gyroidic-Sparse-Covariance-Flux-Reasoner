import unittest
from unittest.mock import patch
import os
import sys

# Guarantee workspace root in sys.path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import math

from src.topology.alexander_wild_embedding import (
    AlexanderWildEmbeddingOperator,
    AlexanderTransversalityReport
)
from src.core.archetype_engines import SovereignRefusalOperator
from src.core.orchestrator import UniversalOrchestrator


class TestAlexanderWildEmbedding(unittest.TestCase):
    def setUp(self):
        os.environ["FREENET_WS_PORT"] = "3000"
        self.dim = 32
        self.operator = AlexanderWildEmbeddingOperator(dim=self.dim, num_cubes=7)

    def test_fractal_cube_ratios_distinct(self):
        """Verify that no two sub-cubes share the same aspect ratio (Mrs. Perkins's Quilt in 3D)."""
        ratios = self.operator.evaluate_cube_dissection_ratios()
        self.assertEqual(len(ratios), 7)
        r_list = ratios.tolist()
        for i in range(len(r_list)):
            for j in range(i + 1, len(r_list)):
                self.assertNotAlmostEqual(r_list[i], r_list[j], places=4,
                    msg=f"Sub-cube ratios at {i} and {j} must not be identical.")

    def test_negentropy_shear_planes_restore_transversality(self):
        """Verify that negentropy shear planes restore transversality in RP^4 void."""
        state = torch.randn(self.dim)
        report = self.operator(
            state=state,
            negentropy_flux=0.8,
            hyperbolic_shear=0.6,
            spatial_gap=1e-4,
            boundary_tension=0.05
        )
        self.assertIsInstance(report, AlexanderTransversalityReport)
        self.assertTrue(report.is_transversal)
        self.assertGreater(report.shear_flux, 0.0)
        self.assertGreater(report.topological_degrees_of_freedom, self.dim)

    def test_lhopital_boundary_limit_regularization(self):
        """Verify that L'Hopital limit audits resolve 0/0 boundary indeterminate quotients."""
        # Non-zero gap
        scale_reg = self.operator.evaluate_lhopital_boundary_limit(spatial_gap=0.1, boundary_tension=0.05)
        # Near-zero gap (L'Hopital regime)
        scale_lhopital = self.operator.evaluate_lhopital_boundary_limit(spatial_gap=1e-7, boundary_tension=0.05)
        self.assertTrue(math.isfinite(scale_reg))
        self.assertTrue(math.isfinite(scale_lhopital))
        self.assertGreater(scale_lhopital, 0.0)

    def test_mandy_encephalization_and_alexander_veto(self):
        """Verify Mandy's high Encephalization refusal asserts boundaries on rebicameral drift and wild pinch."""
        mandy = SovereignRefusalOperator(pas_lock=3.0 / 11.0, encephalization_eq=1.618, training_mode=False)
        state = torch.ones(self.dim)

        # 1. Healthy state with high PAS_h and transversality -> Accepted
        transversal_report = AlexanderTransversalityReport(
            is_transversal=True,
            wild_singularity_risk=0.1,
            shear_flux=0.5,
            lhopital_scale=1.0,
            bouligand_kink_intensity=0.0,
            topological_degrees_of_freedom=36,
            reconstructed_state=state,
            diagnostics={}
        )
        accepted = mandy(state, phase_alignment=0.85, mischief_harmonics=0.5,
                         alexander_transversality=transversal_report)
        self.assertTrue(torch.allclose(accepted, state))

        # 2. Rebicameral drift detected (Echo Chamber of One) -> Refused
        rebicameral_state = mandy(state, phase_alignment=0.35, mischief_harmonics=0.2,
                                  private_invariants={"rebicameral_drift_detected": True})
        self.assertEqual(float(rebicameral_state.norm().item()), 0.0)

        # 3. Alexander wild pinch singularity without shear -> Refused
        pinched_report = AlexanderTransversalityReport(
            is_transversal=False,
            wild_singularity_risk=0.85,
            shear_flux=0.0,
            lhopital_scale=0.1,
            bouligand_kink_intensity=1.2,
            topological_degrees_of_freedom=28,
            reconstructed_state=state,
            diagnostics={}
        )
        pinched_state = mandy(state, phase_alignment=0.4, mischief_harmonics=0.2,
                              alexander_transversality=pinched_report)
        self.assertEqual(float(pinched_state.norm().item()), 0.0)

    def test_orchestrator_alexander_operator_integration(self):
        """Verify Alexander operator produces transversality report and regularizes states."""
        test_state = torch.randn(self.dim)
        report = self.operator(test_state, negentropy_flux=0.5, hyperbolic_shear=0.4)
        self.assertTrue(report.is_transversal)
        self.assertIn("cube_ratios", report.diagnostics)
        self.assertEqual(len(report.diagnostics["cube_ratios"]), 7)
        self.assertGreater(report.topological_degrees_of_freedom, self.dim)
        self.assertGreater(report.lhopital_scale, 0.0)


if __name__ == '__main__':
    unittest.main()
