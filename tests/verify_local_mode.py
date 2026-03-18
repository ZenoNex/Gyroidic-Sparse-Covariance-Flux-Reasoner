import torch
from src.core.admr_solver import PolynomialADMRSolver
from src.core.orchestrator import UniversalOrchestrator
from src.models.diegetic_heads import ResonanceLarynx

def test_love_invariant():
    print("Testing Love Invariant in ADMR Solver...")
    solver = PolynomialADMRSolver(state_dim=16, num_functionals=4, poly_degree=3)
    states = torch.randn(2, 16)
    drift = torch.randn(2, 16)
    noise = torch.randn(2, 16) * 0.01
    negotiation = torch.zeros(2, 16)
    # Perform a stochastic step (should print nothing, just not crash)
    solver.stochastic_differential_step(states, drift, noise, negotiation, 0.1, None)
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
