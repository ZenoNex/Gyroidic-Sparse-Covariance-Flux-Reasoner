import sys
import os
sys.path.append(os.getcwd())

import torch
import unittest
import torch.nn.functional as F
from src.surrogates.calm_predictor import CALM
from src.core.admr_solver import PolynomialADMRSolver
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.surrogates.kagh_networks import KAGHBlock, KANLayer

class TestMAMLAdaptation(unittest.TestCase):
    def setUp(self):
        self.state_dim = 16
        self.history_len = 8
        self.device = 'cpu'
        
    def test_calm_adaptation(self):
        """Verify CALM predictor online inner-loop adaptation."""
        calm = CALM(dim=self.state_dim, history_len=self.history_len)
        calm.eval()
        
        # Create dummy support history and targets
        support_history = torch.randn(2, self.history_len, self.state_dim)
        support_targets = torch.randn(2, self.state_dim) # target for forcing head
        
        # Test original prediction
        with torch.no_grad():
            orig_outputs = calm(support_history)
            orig_forcing = orig_outputs[3]
            orig_loss = F.mse_loss(orig_forcing, support_targets).item()
            
        # Adapt model
        entropy = torch.tensor([0.5])
        adapted_calm = calm.adapt(support_history, support_targets, steps=2, lr=0.05, entropy=entropy)
        
        # Verify adaptation
        with torch.no_grad():
            new_outputs = adapted_calm(support_history)
            new_forcing = new_outputs[3]
            new_loss = F.mse_loss(new_forcing, support_targets).item()
            
        # Loss on support set should decrease after gradient steps
        self.assertLess(new_loss, orig_loss)
        
        # Test functional forward
        params = {name: param for name, param in adapted_calm.named_parameters()}
        func_outputs = calm.functional_forward(support_history, params=params)
        self.assertTrue(torch.allclose(func_outputs[3], new_forcing))

    def test_admr_solver_meta_optimization(self):
        """Verify ADMR solver meta-optimization on constraint violation."""
        from unittest.mock import patch
        
        def mock_jitter(shape, device=None, scaled=True):
            return torch.zeros(shape, device=device)
            
        with patch('src.core.admr_solver.harvest_honest_jitter', mock_jitter), \
             patch('src.core.polynomial_coprime.harvest_honest_jitter', mock_jitter):
             
            poly_config = PolynomialCoprimeConfig(k=5, degree=4, basis_type='chebyshev')
            solver = PolynomialADMRSolver(poly_config=poly_config, state_dim=self.state_dim, device=self.device)
            solver.eval()
            
            # Create dummy states, neighbors, and weights (shifted to bypass sidechain dropout)
            states = torch.randn(4, self.state_dim) + 15.0
            neighbor_states = torch.randn(4, 3, self.state_dim) + 15.0
            adjacency_weight = torch.ones(4, 3) / 3.0
            
            # Initial constraint violation
            with torch.no_grad():
                _, orig_violation = solver.stochastic_differential_step(
                    states, neighbor_states, adjacency_weight, return_violation=True
                )
                orig_loss = torch.norm(orig_violation, p=2, dim=-1).mean().item()
                
            # Adapt solver transition operator A
            entropy = torch.tensor([0.8])
            adapted_solver = solver.meta_optimize_admm_step(
                states, neighbor_states, adjacency_weight, steps=3, lr=0.05, entropy=entropy
            )
            
            # Adapted constraint violation
            with torch.no_grad():
                _, new_violation = adapted_solver.stochastic_differential_step(
                    states, neighbor_states, adjacency_weight, return_violation=True
                )
                new_loss = torch.norm(new_violation, p=2, dim=-1).mean().item()
                
            # Violation loss should decrease
            self.assertLess(new_loss, orig_loss)
            # Parameter A should have changed
            self.assertFalse(torch.allclose(solver.A, adapted_solver.A))

    def test_kagh_block_adaptation(self):
        """Verify KAGH Block online adaptation and functional forward."""
        kagh = KAGHBlock(n_in=self.state_dim, n_out=self.state_dim, width=32, depth=2)
        kagh.eval()
        
        # Create dummy support states and targets
        support_states = torch.randn(4, self.state_dim)
        support_targets = torch.randn(4, self.state_dim)
        
        # Test original prediction loss
        with torch.no_grad():
            orig_preds = kagh(support_states)
            orig_loss = F.mse_loss(orig_preds, support_targets).item()
            
        # Adapt KAGH Block online
        entropy = torch.tensor([0.2])
        adapted_kagh = kagh.adapt_online(support_states, support_targets, steps=2, lr=0.05, entropy=entropy)
        
        # Adapted loss
        with torch.no_grad():
            new_preds = adapted_kagh(support_states)
            new_loss = F.mse_loss(new_preds, support_targets).item()
            
        # Loss should decrease
        self.assertLess(new_loss, orig_loss)
        
        # Test functional forward on KAGHBlock
        params = {name: param for name, param in adapted_kagh.named_parameters()}
        func_preds = kagh.functional_forward(support_states, params=params)
        self.assertTrue(torch.allclose(func_preds, new_preds))
        
        # Test functional forward on KANLayer
        kan_layer = KANLayer(in_features=self.state_dim, out_features=self.state_dim)
        x = torch.randn(2, self.state_dim)
        res_orig = kan_layer(x)
        params_kan = {name: param for name, param in kan_layer.named_parameters()}
        res_func = kan_layer.functional_forward(x, params=params_kan)
        self.assertTrue(torch.allclose(res_orig, res_func))

if __name__ == '__main__':
    unittest.main()
