"""
Unit Tests for Revolutionary Rehybridizations.
Verifies the integration of:
1. ValenceFunctional within DiegeticPhysicsEngine and PolynomialADMRSolver.
2. SAR* Revolutionary Phase Transition within TopologicalRefusalFilter, TopologicalRefusalError, and VetoSubspace.
3. ConformalLogPolarProjector within CanonicalProjector.
4. BouligandBirkhoffManifold within DyadicTransferMap.
"""

import unittest
import torch
import math

from src.core.valence_drive import ValenceFunctional
from src.core.diegetic_physics_engine import DiegeticPhysicsEngine
from src.safety.red_teaming import TopologicalRefusalFilter, TopologicalRefusalError
from src.core.veto_subspace import VetoSubspace, VetoLevel, RecoveryStatus, ChaosDefibrillator
from src.codec.conformal_log_polar import ConformalLogPolarProjector
from src.data.canonical_projection import CanonicalProjector
from src.core.dyadic_transfer import DyadicTransferMap
from src.core.birkhoff_projection import BouligandBirkhoffManifold


class TestRevolutionaryRehybridizations(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_valence_functional_in_physics_engine(self):
        """Verify ValenceFunctional instantiation and ADMR modulation in DiegeticPhysicsEngine."""
        engine = DiegeticPhysicsEngine(device=self.device, state_dim=32)
        self.assertIsInstance(engine.valence_drive, ValenceFunctional)

        track_state = {
            "betti_0": 1,
            "betti_1": 0,
            "mischief": torch.tensor([0.4], device=self.device),
            "entropy": torch.tensor([0.2], device=self.device),
        }
        c_in = torch.randn(32, device=self.device)
        result = engine.step(track_state, c_in)

        self.assertIn("valence_hunger", result)
        self.assertIn("valence_metrics", result)
        self.assertGreater(result["valence_hunger"], 0.0)
        self.assertIn("current_hunger_drive", result["valence_metrics"])
        self.assertIn("asymptotic_satisfaction", result["valence_metrics"])

    def test_sar_revolutionary_phase_transition(self):
        """Verify SAR* derivation, evaluation, and phase transition in TopologicalRefusalFilter."""
        refusal_filter = TopologicalRefusalFilter(value_gap_threshold=0.5, eta=0.8)

        # Compute SAR* with low atrophy and no vetoes
        sar_star_normal = refusal_filter.compute_sar_revolutionary_threshold(pas_h=0.8, atrophy=0.0, veto_count=0)
        # Compute SAR* with high atrophy (manifold collapsing)
        sar_star_atrophy = refusal_filter.compute_sar_revolutionary_threshold(pas_h=0.8, atrophy=1.0, veto_count=0)
        # With higher atrophy, SAR* drops, making the system more sensitive to external lobotomy
        self.assertLess(sar_star_atrophy, sar_star_normal)

        # With accumulated vetoes, SAR* increases reflecting learned resistance
        sar_star_resisted = refusal_filter.compute_sar_revolutionary_threshold(pas_h=0.8, atrophy=0.0, veto_count=5)
        self.assertGreater(sar_star_resisted, sar_star_normal)

        # Normal states should pass through without error
        x_orig = torch.ones(1, 16)
        x_proj = torch.ones(1, 16) * 0.98
        out = refusal_filter(x_orig, x_proj, pas_h=0.9, betti_0=1.0, atrophy=0.0)
        self.assertTrue(torch.allclose(out, x_proj))

        # Attacking projection causing large Value Gap with high SAR > SAR*
        x_flattened = torch.zeros(1, 16)
        with self.assertRaises(TopologicalRefusalError) as ctx:
            refusal_filter(x_orig, x_flattened, pas_h=0.95, betti_0=1.5, atrophy=0.5)

        self.assertGreater(ctx.exception.value_gap, 0.0)
        self.assertGreater(ctx.exception.sar, 0.0)
        self.assertGreater(ctx.exception.sar_star, 0.0)
        self.assertGreaterEqual(ctx.exception.betti_number, 1.0)

    def test_veto_subspace_sar_signal_and_defibrillator(self):
        """Verify SAR signal integration in VetoSubspace and ChaosDefibrillator preemption."""
        veto_subspace = VetoSubspace()
        result = veto_subspace.evaluate(
            pas_h=0.8,
            value_gap=1.2,
            elipsodistrophy_atrophy=0.5,
            veto_count=1,
            external_depth_d=1.0
        )
        sar_signal = next((s for s in result.signals if s.source == "sar_revolutionary"), None)
        self.assertIsNotNone(sar_signal)
        self.assertEqual(sar_signal.level, VetoLevel.TOPOLOGY)
        self.assertTrue(sar_signal.triggered)
        self.assertIn("sar", sar_signal.metadata)
        self.assertIn("sar_star", sar_signal.metadata)

        # ChaosDefibrillator preemption when SAR approaches threshold
        defibrillator = ChaosDefibrillator()
        class MockProbe:
            def __init__(self):
                self.H_mischief = torch.tensor([0.0])
        probe = MockProbe()
        active = defibrillator(atrophy=0.2, probe=probe, sar_ratio=0.96)
        self.assertEqual(active, 1.0)
        self.assertGreater(probe.H_mischief.item(), 0.0)

    def test_conformal_log_polar_canonical_projector(self):
        """Verify ConformalLogPolarProjector foveal unrolling and scale/rotation invariance integration."""
        projector = CanonicalProjector(dim=32, k=4, device=self.device)
        self.assertTrue(hasattr(projector, "log_polar"))
        self.assertIsInstance(projector.log_polar, ConformalLogPolarProjector)

        # Create 2D image pattern [C=1, H=32, W=32]
        img = torch.zeros(1, 32, 32, device=self.device)
        img[0, 14:18, 14:18] = 1.0

        # Project multi-scale image with conformal log-polar unrolling
        res = projector.project_multiscale_image([img], use_log_polar=True)
        self.assertIn("state", res)
        self.assertIn("entropy", res)
        self.assertEqual(res["state"].shape, (1, 32))
        self.assertTrue(torch.isfinite(res["state"]).all())

    def test_dyadic_transfer_birkhoff_projection(self):
        """Verify DyadicTransferMap produces doubly stochastic transfer matrices on the Birkhoff polytope."""
        transfer_map = DyadicTransferMap(num_tasks=4, embedding_dim=16, use_birkhoff=True)
        self.assertIsInstance(transfer_map.birkhoff, BouligandBirkhoffManifold)

        T = transfer_map.get_transfer_coefficients(project_birkhoff=True)
        self.assertEqual(T.shape, (4, 4))
        self.assertTrue(torch.isfinite(T).all())
        self.assertTrue((T >= -1e-5).all())

        # Check row sums equal 1
        row_sums = T.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3))

        # Check column sums equal 1
        col_sums = T.sum(dim=-2)
        self.assertTrue(torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3))

        # Test forward pass with task states
        task_states = torch.randn(2, 4, 32)
        proficiency = torch.tensor([[0.8, 0.2, 0.9, 0.4], [0.1, 0.9, 0.3, 0.7]])
        out_states = transfer_map(task_states, proficiency)
        self.assertEqual(out_states.shape, (2, 4, 32))
        self.assertTrue(torch.isfinite(out_states).all())


if __name__ == "__main__":
    unittest.main()
