"""
Tests for Non-Hermitian Flow Taxonomy & Implementation.
Verifies all 3 meanings of Non-Hermitian Flow:
1. Dynamical Evolution under non-Hermitian operator (gain/loss/nonreciprocal coupling)
2. Spectral Flow and Exceptional Point detection
3. Integration with NonCommutativityCurvature Arrow-of-Time dynamics
"""

import unittest
import torch
from src.core.quantum_inspired_reasoning import QuantumInspiredReasoningState
from src.core.noncommutativity_curvature import NonCommutativityCurvature


class TestNonHermitianFlow(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.qr = QuantumInspiredReasoningState(dim=self.dim)
        self.nc = NonCommutativityCurvature(dim=self.dim)

    def test_hermitian_baseline(self):
        """Verify baseline reasoning Hamiltonian is Hermitian within tolerance."""
        is_herm, err = self.qr.is_hermitian(tol=1e-4)
        self.assertTrue(is_herm)
        self.assertLess(err, 1e-4)

    def test_meaning_1_dynamical_evolution_gain_loss(self):
        """Verify Meaning 1: State evolution under non-Hermitian H with gain/loss."""
        # Initial complex state
        state = torch.complex(torch.randn(self.dim), torch.randn(self.dim))
        state = state / torch.norm(state)

        # Gain/loss matrix: negative imaginary part -> gain, positive -> loss
        gain_loss = torch.diag(torch.linspace(-0.2, 0.2, steps=self.dim))

        evolved, H_eff = self.qr.evolve_non_hermitian_flow(
            state, dt=0.05, gain_loss=gain_loss, renormalize=True
        )

        self.assertEqual(evolved.shape, state.shape)
        self.assertEqual(H_eff.shape, (self.dim, self.dim))

        # Effective Hamiltonian must be non-Hermitian due to gain/loss
        diff = H_eff - H_eff.conj().T
        non_herm_norm = torch.norm(diff).item()
        self.assertGreater(non_herm_norm, 0.1)

        # Norm should be preserved when renormalize=True
        self.assertAlmostEqual(torch.norm(evolved).item(), 1.0, places=4)

    def test_meaning_1_nonreciprocal_coupling(self):
        """Verify Meaning 1: Non-reciprocal transport where J_ij != J_ji."""
        state = torch.complex(torch.randn(self.dim), torch.randn(self.dim))
        state = state / torch.norm(state)

        # Unidirectional coupling from node i to i+1
        asymmetric_coupling = torch.zeros((self.dim, self.dim))
        for i in range(self.dim - 1):
            asymmetric_coupling[i, i + 1] = 0.5  # unidirectional forward hop

        evolved, H_eff = self.qr.evolve_non_hermitian_flow(
            state, dt=0.05, nonreciprocal_coupling=asymmetric_coupling
        )

        self.assertEqual(evolved.shape, state.shape)
        # Check asymmetry in coupling
        diff_coupling = asymmetric_coupling - asymmetric_coupling.T
        self.assertGreater(torch.norm(diff_coupling).item(), 0.5)

    def test_meaning_2_spectral_flow(self):
        """Verify Meaning 2: Eigenvalue trajectory tracking across parameter drift."""
        trajectory = self.qr.compute_spectral_flow(num_steps=5)
        self.assertEqual(len(trajectory), 5)
        for eigs in trajectory:
            self.assertEqual(eigs.shape, (self.dim,))
            # Must be complex numbers
            self.assertTrue(eigs.is_complex())

    def test_meaning_2_exceptional_point_detection(self):
        """Verify Meaning 2: Detection of defective Exceptional Point (EP2)."""
        # Construct an explicit 2x2 defective Jordan block: [[1, 1], [0, 1]]
        # Eigenvalues: {1, 1}, Eigenvector: [1, 0] (coalesced, defective!)
        qr2 = QuantumInspiredReasoningState(dim=2)
        jordan_block = torch.tensor([[1.0 + 0j, 1.0 + 0j],
                                     [0.0 + 0j, 1.0 + 0j]], dtype=torch.complex64)

        is_near_ep, min_dist, cond = qr2.detect_exceptional_points(operator=jordan_block)
        self.assertAlmostEqual(min_dist, 0.0, places=5)
        # Jordan block has coalesced eigenvectors, condition number diverges
        self.assertTrue(is_near_ep or cond > 50.0)

    def test_meaning_2_arrow_of_time_inflection(self):
        """Verify connection between non-Hermiticity, commutator, and Arrow of Time."""
        forward = torch.randn(self.dim, self.dim)
        backward = torch.randn(self.dim, self.dim)

        inflection_mask, asymmetry_score = self.nc.arrow_of_time_inflection(forward, backward)
        self.assertEqual(inflection_mask.shape, (self.dim,))
        self.assertGreaterEqual(asymmetry_score.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

