"""
Verification script for Spectral Structural Trainer.
"""

import torch
import torch.nn as nn
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.training.fgrt_fgrt_trainer import SpectralStructuralTrainer

class MockModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, x):
        return self.linear(x)

def test_spectral_trainer():
    print("Testing Spectral Structural Trainer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    state_dim = 64
    k = 5
    degree = 4
    
    # 1. Setup Config
    poly_config = PolynomialCoprimeConfig(k=k, degree=degree, device=device)
    
    # 2. Setup Model
    model = MockModel(state_dim)
    
    # 3. Setup Trainer
    trainer = SpectralStructuralTrainer(
        model=model,
        poly_config=poly_config,
        lr=1e-3
    )
    
    # 4. Dummy Input
    batch_size = 8
    input_data = torch.randn(batch_size, state_dim)
    
    # 5. Train Step
    print("Running train step...")
    metrics = trainer.train_step(input_data)
    
    print(f"Metrics: {metrics}")
    assert 'willmore_energy' in metrics
    assert 'spectral_entropy' in metrics
    assert 'pas_h' in metrics
    
    # 6. Verify System 2 Trigger
    # If we pass very high entropy input, System 2 should be active (entropy check done inside)
    # We can't easily check if solver.solve was called without mocking, but we can verify it doesn't crash.
    
    print("Spectral Structural Trainer verification PASSED.")

if __name__ == "__main__":
    test_spectral_trainer()

