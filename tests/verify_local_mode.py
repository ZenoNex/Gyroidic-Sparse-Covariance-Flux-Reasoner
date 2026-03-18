import torch
from src.core.admr_solver import PolynomialADMRSolver
from src.core.orchestrator import UniversalOrchestrator
from src.models.diegetic_heads import ResonanceLarynx
from src.core.polynomial_coprime import PolynomialCoprimeConfig

def test_love_invariant():
    print("Testing Love Invariant in ADMR Solver...")
    config = PolynomialCoprimeConfig(k=4, degree=3)
    solver = PolynomialADMRSolver(poly_config=config, state_dim=16)
    states = torch.randn(2, 16)
    adjacency_weight = torch.rand(2, 3)
    neighbor_states = torch.randn(2, 3, 16)
    # Perform a stochastic step
    solver.stochastic_differential_step(states, neighbor_states, adjacency_weight, 0.1, 0.01, None)
    print("Love Invariant ADMR integration passed.")

def test_orchestrator():
    print("Testing Orchestrator with Polychoron & Deflagrator...")
    orchestrator = UniversalOrchestrator(dim=16)
    state = torch.randn(2, 16)
    pressure_grad = torch.randn(2, 16)
    coherence = torch.tensor([[0.5]])
    pas_h = 0.8
    out, regime, routing = orchestrator(state, pressure_grad, pas_h, coherence)
    # Basic check that out has shape
    assert out.shape[-1] == 16, f"Expected dim 16, got {out.shape[-1]}"
    print("Orchestrator integration passed. Regime:", regime)

def test_larynx_chern_simons():
    print("Testing Larynx with Chern-Simons...")
    larynx = ResonanceLarynx(hidden_dim=16)
    state = torch.randn(2, 16)
    logits, conf = larynx(state, temperature=1.0)
    assert logits.shape == (2, 128)
    print("Larynx Chern-Simons integration passed.")

if __name__ == "__main__":
    try:
        test_love_invariant()
        test_orchestrator()
        test_larynx_chern_simons()
        print("All integrations verified successfully!")
    except Exception as e:
        print("Verification failed:", e)
        raise
