import torch
import unittest
import numpy as np
from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
from src.core.spectral_coherence_repair import apply_energy_based_stabilization
from src.core.admr_solver import PolynomialADMRSolver
from src.ui.diegetic_backend import DiegeticPhysicsEngine

class TestEndogenousCoupling(unittest.TestCase):
    def test_elliptical_lattice_evaluation(self):
        engine = SiliconSovereigntyEngine(use_gpu=False)
        coords = np.array([[1.0, 1.0], [0.1, 0.1], [5.0, 5.0]], dtype=np.float32)
        lattice = [
            {
                'center': [0.0, 0.0],
                'axes': [1.2, 1.2],
                'rotation': 0.0,
                'thickness': 0.5
            },
            {
                'center': [5.0, 5.0],
                'axes': [1.0, 1.0],
                'rotation': 0.5,
                'thickness': 0.2
            }
        ]
        
        bitmask, pressure = engine.evaluate_elliptical_hash_lattice(coords, lattice)
        self.assertEqual(len(bitmask), 3)
        self.assertEqual(len(pressure), 3)
        self.assertTrue(np.all(pressure >= 0.0))

    def test_lazarus_overtone_rehydration(self):
        state = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('inf')]])
        overtones = torch.tensor([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
        
        stabilized = apply_energy_based_stabilization(state, cavity_overtones=overtones)
        self.assertFalse(torch.isnan(stabilized).any())
        self.assertFalse(torch.isinf(stabilized).any())
        self.assertGreater(stabilized[0, 1].abs().item(), 1e-6)

    def test_viscoelastic_boundary_drag(self):
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        config = PolynomialCoprimeConfig(k=5, degree=4, learnable=True, device='cpu')
        solver = PolynomialADMRSolver(config, 16, device='cpu')
        
        states = torch.randn(2, 16)
        neighbor_states = torch.randn(2, 3, 16)
        adjacency_weight = torch.ones(2, 3) / 3.0
        
        boundary_near = states.clone() + 1e-4
        out_near = solver.fractional_stochastic_differential_step(
            states=states,
            neighbor_states=neighbor_states,
            adjacency_weight=adjacency_weight,
            boundary_state=boundary_near
        )
        self.assertEqual(out_near.shape, states.shape)
        
        boundary_far = states.clone() + 10.0
        out_far = solver.fractional_stochastic_differential_step(
            states=states,
            neighbor_states=neighbor_states,
            adjacency_weight=adjacency_weight,
            boundary_state=boundary_far
        )
        self.assertEqual(out_far.shape, states.shape)

    def test_diegetic_engine_jaccard_bubble(self):
        engine = DiegeticPhysicsEngine(dim=64, k=5, device='cpu')
        
        res1 = engine._process_input_internal("hello world physical resonance simulation")
        self.assertIn("response", res1)
        
        res2 = engine._process_input_internal("completely unrelated mathematics and geometry of the golden ratio")
        self.assertIn("response", res2)
        
        res3 = engine._process_input_internal("completely unrelated mathematics and geometry of golden spiral")
        self.assertIn("response", res3)

if __name__ == "__main__":
    unittest.main()
