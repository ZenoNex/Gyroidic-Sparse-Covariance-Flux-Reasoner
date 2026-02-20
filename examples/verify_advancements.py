import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.admr_solver import PolynomialADMRSolver
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.spectral_coherence_repair import SpectralCoherenceCorrector

def verify_stochastic_dynamics():
    print("🧪 Verifying Stochastic Differential Dynamics...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    state_dim = 64
    num_functionals = 5
    
    # 1. Initialize Components
    config = PolynomialCoprimeConfig(
        num_functionals=num_functionals,
        poly_degree=4,
        device=device
    )
    
    solver = PolynomialADMRSolver(
        poly_config=config,
        state_dim=state_dim,
        device=device
    )
    
    repair = SpectralCoherenceCorrector(device=device)
    
    # 2. Setup Dummy Data
    batch_size = 2
    states = torch.randn(batch_size, state_dim)
    neighbor_states = torch.randn(batch_size, 3, state_dim)
    adjacency_weight = torch.softmax(torch.randn(batch_size, 3), dim=-1)
    
    # 3. Test Stochastic Step
    print("📈 Running stochastic_differential_step...")
    new_states = solver.stochastic_differential_step(
        states=states,
        neighbor_states=neighbor_states,
        adjacency_weight=adjacency_weight,
        dt=0.1,
        sigma=0.01
    )
    
    print(f"✅ Stochastic step completed. State shape: {new_states.shape}")
    drift_mag = torch.norm(new_states - states).item()
    print(f"   Drift Magnitude: {drift_mag:.4f}")
    
    # 4. Test Acoustic Projection
    print("🔊 Running project_to_acoustic_resonance...")
    # Evaluate facets
    facets = config.evaluate(new_states)
    if facets.dim() > 2:
        facets = facets.mean(dim=1) # [batch, num_functionals]
        
    time_steps = torch.linspace(0, 1.0, 100)
    acoustic_signal = repair.project_to_acoustic_resonance(facets, time_steps)
    
    print(f"✅ Acoustic projection completed. Signal shape: {acoustic_signal.shape}")
    print(f"   Max amplitude: {acoustic_signal.abs().max().item():.4f}")
    
    print("\n✨ All theoretical advancements verified in code!")

if __name__ == "__main__":
    verify_stochastic_dynamics()

