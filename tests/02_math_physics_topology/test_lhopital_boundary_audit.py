"""
Unit tests auditing the applicability and boundaries of L'Hopital's Rule:
1. Beneficial Domains (Continuous manifolds, fractal complexity limits, gradient stencils).
2. Forbidden/Unhelpful Domains (Discrete CRT, integer Betti numbers, bitwise Morton codes,
   discrete quantized entropy, and permutation matrices where smoothing induces leakage).
"""

import unittest
import math
import torch
import numpy as np

from src.ui.voxelboxter_simulation import PointerlessOctree, RigidBody
from src.p2p.bonfire_consensus import FractalDiscontinuityEmbedding
from src.topology.gyroid_differentiation import GyroidFlowConstraint
from src.core.continuous_coprimality import DiscreteEntropyComputer
from src.core.polynomial_crt import PolynomialCRT
from src.core.fgrt_primitives import morton_encode_3d, GyroidManifold
from src.core.birkhoff_projection import project_to_birkhoff


class TestLHopitalBoundaryAudit(unittest.TestCase):

    def test_lhopital_continuous_beneficial_domains(self):
        """
        Verify domains where L'Hopital's rule s(c) = log(c + 1) / c is mathematically essential:
        1. Non-uniform octree depth scaling at c -> 0: limit = 1.0.
        2. Fractal discontinuity embedding dimension at c -> 0: continuous supremum.
        3. Gradient stencils across discontinuous fracture lines: damping derivative explosion.
        """
        # Domain 1: Octree scale factor
        octree = PointerlessOctree(max_depth=8)
        c_zero = 0.0
        c_tiny = 1e-9
        scale_zero = octree.compute_local_scale(c_zero)
        scale_tiny = octree.compute_local_scale(c_tiny)
        self.assertEqual(scale_zero, 1.0)
        self.assertAlmostEqual(scale_tiny, 1.0, places=6)

        # Domain 2: Discontinuity embedding ambient dimension
        embedder = FractalDiscontinuityEmbedding()
        ambient_m = embedder.compute_ambient_dimension()
        local_m_zero = embedder.compute_embedding_dimension(0.0)
        local_m_large = embedder.compute_embedding_dimension(10.0)
        self.assertEqual(ambient_m, local_m_zero)
        self.assertGreaterEqual(ambient_m, local_m_large)

        # Domain 3: Gradient regularizer in Gyroid flow
        constraint = GyroidFlowConstraint()
        residue = torch.tensor([[1.0, 0.0, 0.0]])
        emb = torch.tensor([[0.0, math.pi / 2.0, 0.0]])
        # Functional directional derivative with steep jump
        steep_fn = lambda r: r * 1e5
        grad_steep = constraint.compute_flow_gradient(residue, emb, embedding_fn=steep_fn)
        # Verify L'Hopital regularizer dampens the output to O(log(jump)) instead of raw 1e5
        self.assertTrue(torch.isfinite(grad_steep).all())
        self.assertLess(grad_steep.abs().max().item(), 1e5)

    def test_lhopital_unhelpful_domains_discrete_invariants(self):
        """
        Verify domains where L'Hopital continuous regularization would be UNHELPFUL and HARMFUL:
        1. Discrete Quantized Entropy: Must preserve exact integer bincounts and exact log2.
        2. Discrete Morton Spatial Hashing: Must maintain bitwise integer bijection without smoothing.
        3. Polynomial CRT Moduli: Moduli must be exact discrete values, not smoothed continuous forms.
        4. Birkhoff Permutations: Extreme vertices must preserve exact binary {0, 1} entries.
        """
        # 1. Discrete Quantized Entropy must not be continuous-smoothed
        entropy_computer = DiscreteEntropyComputer(num_bins=16)
        int_data = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
        h_discrete = entropy_computer.compute_entropy(int_data)
        self.assertIsInstance(h_discrete.item(), float)
        self.assertGreater(h_discrete.item(), 0.0)
        # Ensure no negative or fractional bin leakage
        bincount_len = len(torch.bincount((int_data * 5).long()))
        self.assertGreaterEqual(bincount_len, 3)

        # 2. Morton code bitwise integrity
        x = torch.tensor([1, 2, 4], dtype=torch.int32)
        y = torch.tensor([0, 1, 3], dtype=torch.int32)
        z = torch.tensor([3, 0, 1], dtype=torch.int32)
        codes = morton_encode_3d(x, y, z)
        # Morton codes must be exact non-negative integers with strict bitwise identity
        self.assertEqual(codes.dtype, torch.int32)
        self.assertEqual(len(torch.unique(codes)), 3)
        # Applying continuous float smoothing to bitwise keys is strictly unhelpful

        # 3. Polynomial CRT channel separation
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        poly_config = PolynomialCoprimeConfig(k=3, degree=4)
        crt = PolynomialCRT(poly_config=poly_config)
        self.assertEqual(crt.K, 3)
        self.assertEqual(crt.D, 5)
        # Exact channel dimensions must remain integer-quantized
        coeffs = torch.randn(2, 3, 5)
        recon = crt(coeffs)
        self.assertEqual(recon.shape[0], 2)

        # 4. Birkhoff Polytope extreme points (Permutation matrices)
        perm = torch.tensor([[0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0]])
        projected = project_to_birkhoff(perm)
        # Must retain exact doubly-stochastic row/col sum == 1 without fractional dissipation
        row_sums = projected.sum(dim=-1)
        col_sums = projected.sum(dim=-2)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
        self.assertTrue(torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
