import unittest
import os
import sys

# Guarantee workspace root in sys.path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

from src.topology import (
    SolitonStability,
    ApproximatePHProbe,
    TriadicReciprocityChecker,
    MoebiusFiberBundle,
    LeyLineGeodesicMetric,
    HomologyPressure,
    PersistentHomologyComputer,
    UnknowledgeDomain
)
from src.core.pressure_typing import StructuralPressure
from src.core.relational_kappa import RelationalKappa
from src.core.gluing_operator import GluingOperator
from src.core.carnot_mobius_ledger import CarnotMobiusLedger
from src.core.ley_line_tracker import LeyLineTracker
from src.core.cayley_cubic_probe import CayleyCubicProbe
from src.core.invariants import APAS_Zeta


class TestTopologyCoreRehybridization(unittest.TestCase):
    def test_soliton_stability_binds_relational_kappa(self):
        """Verifies SolitonStability delegates dynamic thresholding to RelationalKappa."""
        sol = SolitonStability(eta=0.5)
        self.assertIsInstance(sol.relational_kappa, RelationalKappa)
        
        residue = torch.randn(2, 16)
        constraint = torch.randn(2, 16)
        res = sol(residue, constraint)
        self.assertIn("is_soliton", res)
        self.assertIn("ratio", res)
        self.assertIn("kappa_threshold", res)
        self.assertTrue(torch.is_tensor(res["kappa_threshold"]))

    def test_approximate_ph_binds_apas_zeta(self):
        """Verifies ApproximatePHProbe binds APAS_Zeta invariant for drift bounding."""
        probe = ApproximatePHProbe(relative_threshold=0.25)
        self.assertIsInstance(probe.apas_bound, APAS_Zeta)
        
        pts1 = torch.randn(20, 16)
        pts2 = torch.randn(20, 16)
        r1 = probe(pts1)
        r2 = probe(pts2)
        self.assertIn("is_rupture", r2)
        self.assertIn("apas_drift", r2)
        self.assertIn("barcode_signature", r2)

    def test_triadic_reciprocity_binds_cayley_probe(self):
        """Verifies TriadicReciprocityChecker evaluates Cayley cubic singularities."""
        triad = TriadicReciprocityChecker()
        self.assertIsInstance(triad.cayley_probe, CayleyCubicProbe)
        
        fa = torch.randn(2, 16)
        fb = torch.randn(2, 16)
        fc = torch.randn(2, 16)
        res = triad.check_cayley_variety(fa, fb, fc, cycle_debt=0.6)
        self.assertIn("cayley_residual", res)
        self.assertIn("projected_triad", res)
        self.assertIn("reciprocity_score", res)
        self.assertEqual(res["projected_triad"].shape[-1], 3)

    def test_moebius_fiber_bundle_binds_gluing_and_carnot(self):
        """Verifies MoebiusFiberBundle binds GluingOperator and CarnotMobiusLedger."""
        moebius = MoebiusFiberBundle(dim=16, fiber_dim=8)
        self.assertIsInstance(moebius.gluing_operator, GluingOperator)
        self.assertIsInstance(moebius.carnot_ledger, CarnotMobiusLedger)
        
        x = torch.randn(2, 16)
        twist_gate = torch.tensor([1.0, 0.0])
        fiber_out = moebius(x, twist_gate)
        self.assertEqual(fiber_out.shape, (2, 8))

    def test_ley_line_geodesic_metric_binds_tracker(self):
        """Verifies LeyLineGeodesicMetric binds LeyLineTracker and calculates anisotropic distance."""
        ley = LeyLineGeodesicMetric(dim=16, num_samples=16)
        self.assertIsInstance(ley.tracker, LeyLineTracker)
        
        x1 = torch.randn(16)
        x2 = torch.randn(16)
        dist = ley.geodesic_distance(x1, x2)
        self.assertTrue(dist.item() > 0.0)

    def test_homology_pressure_forward_typed(self):
        """Verifies HomologyPressure emits domain-isolated StructuralPressure."""
        hp = HomologyPressure()
        typed_res = hp.forward_typed([[0, 1]], torch.tensor([0.9, 0.8]), domain="homology_ker_obstruction")
        self.assertIsInstance(typed_res, StructuralPressure)
        self.assertEqual(typed_res.domain, "homology_ker_obstruction")
        self.assertTrue(typed_res.value.item() > 0.0)

    def test_persistent_homology_quantum_betti(self):
        """Verifies PersistentHomologyComputer delegates to QuantumBettiApproximator."""
        phc = PersistentHomologyComputer(max_dimension=1)
        adj = torch.eye(8)
        betti_approx = phc.approximate_betti_quantum(adj)
        self.assertIn(0, betti_approx)
        self.assertIn(1, betti_approx)
        self.assertEqual(betti_approx[0], 8)


if __name__ == "__main__":
    unittest.main()
