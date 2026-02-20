"""
Verification script for Polynomial ADMR Solver.
"""

import torch
import torch.nn as nn
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.admr_solver import PolynomialADMRSolver

def test_polynomial_admr():
    print("Testing Polynomial ADMR Solver...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    state_dim = 64
    k = 5
    degree = 4
    
    # 1. Setup Config
    poly_config = PolynomialCoprimeConfig(k=k, degree=degree, device=device)
    
    # 2. Setup Solver
    solver = PolynomialADMRSolver(poly_config=poly_config, state_dim=state_dim, device=device)
    
    # 3. Dummy Data
    batch_size = 4
    num_neighbors = 3
    states = torch.randn(batch_size, state_dim)
    neighbor_states = torch.randn(batch_size, num_neighbors, state_dim)
    adjacency_weight = torch.softmax(torch.randn(batch_size, num_neighbors), dim=-1)
    
    # 4. Forward Pass
    print("Running forward pass...")
    output = solver(states, neighbor_states, adjacency_weight)
    
    assert output.shape == (batch_size, state_dim), f"Expected shape {(batch_size, state_dim)}, got {output.shape}"
    print(f"Output shape verified: {output.shape}")
    
    # 5. Test Scaffold Update
    print("Testing scaffold update...")
    negentropy = torch.tensor([0.5])
    dt = torch.tensor(0.1)
    solver.update_scaffold(negentropy, dt)
    assert solver.tau == 0.1
    print("Scaffold update verified.")
    
    # 6. Test Coherence Metrics
    print("Testing coherence metrics...")
    metrics = solver.get_coherence_metrics(output)
    print(f"Metrics: {metrics}")
    assert 'polynomial_coherence' in metrics
    assert 'local_functional_entropy' in metrics
    
    print("Polynomial ADMR Solver verification PASSED.")

if __name__ == "__main__":
    test_polynomial_admr()

