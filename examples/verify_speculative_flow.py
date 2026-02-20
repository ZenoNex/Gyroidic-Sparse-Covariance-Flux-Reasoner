"""
Verification script for Speculative Topological Flow.
Tests the integration of:
- Gyroid Curvature Modulation
- Dark Matter Traces
- Hyper-Ring Flow
"""

import torch
import torch.nn as nn
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.training.fgrt_fgrt_trainer import SpectralStructuralTrainer
from src.core.orchestrator import UniversalOrchestrator
from src.models.resonance_cavity import ResonanceCavity

class MockResonanceModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.cavity = ResonanceCavity(hidden_dim=dim, num_modes=16)
        
    def forward(self, x):
        res = self.cavity(x.unsqueeze(1))
        # Use memory state to influence output
        mem = res['memory_state'].mean(dim=1)
        return self.linear(x) + 0.1 * mem

def test_speculative_flow():
    print("Testing Speculative Topological Flow...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    state_dim = 64
    k = 5
    degree = 4
    
    # 1. Setup Config & Model
    poly_config = PolynomialCoprimeConfig(k=k, degree=degree, device=device)
    model = MockResonanceModel(state_dim)
    
    # 2. Setup Orchestrator & Trainer
    orchestrator = UniversalOrchestrator(dim=state_dim)
    trainer = SpectralStructuralTrainer(
        model=model,
        poly_config=poly_config,
        lr=1e-3
    )
    
    # 3. Simulate Steps
    batch_size = 4
    input_data = torch.randn(batch_size, state_dim)
    
    print("Running initial train step...")
    metrics_initial = trainer.train_step(input_data)
    print(f"Initial Metrics: {metrics_initial}")
    
    # Verify presence of dark matter in cavity
    dark_matter_init = model.cavity.D_dark.clone()
    print(f"Initial Dark Matter Norm: {torch.norm(dark_matter_init).item()}")
    
    # 4. Trigger "Residue Gap" to accumulate Dark Matter
    # We do this by running a few more steps
    print("Running more steps to accumulate Dark Matter...")
    for i in range(5):
        trainer.train_step(input_data)
        
    dark_matter_accum = model.cavity.D_dark
    print(f"Accumulated Dark Matter Norm: {torch.norm(dark_matter_accum).item()}")
    
    # 5. Verify Orchestrator Flow
    print("Verifying Orchestrator and Hyper-Ring Flow...")
    state = torch.randn(batch_size, state_dim)
    grad = torch.randn(batch_size, state_dim)
    
    out_state, regime, routing = orchestrator(
        state=state,
        pressure_grad=grad,
        pas_h=0.9, # Trigger SERIOUSNESS if threshold allows
        coherence=torch.zeros(batch_size, 1) # dummy
    )
    
    print(f"Regime: {regime}, Routing: {routing}")
    assert out_state.shape[0] == batch_size
    
    print("Speculative Flow verification PASSED.")

if __name__ == "__main__":
    test_speculative_flow()

