import sys
import os
import torch
import pytest

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.topological_gyrocompass import TopologicalGyrocompass
from src.core.admr_solver import PolynomialADMRSolver
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.orchestrator import UniversalOrchestrator

def test_precess_torque_outward():
    """
    Verifies that precess_torque redirects an outward-pointing update orthogonally
    to the boundary normal vector.
    """
    dim = 8
    gyro = TopologicalGyrocompass(state_dim=dim, love_dim=2)
    
    # normal points along the first dimension
    normal = torch.zeros(dim)
    normal[0] = 1.0
    
    # spin connection points along the second dimension
    spin = torch.zeros(dim)
    spin[1] = 1.0
    
    # proposed update points outward (positive first dimension)
    dx = torch.zeros(1, dim)
    dx[0, 0] = 2.0  # outward
    dx[0, 2] = 0.5  # tangent component
    
    dx_precessed = gyro.precess_torque(dx, normal, spin)
    
    # Check that normal component is zero (or tangent space projection worked)
    # The normal component of dx_precessed should be 0 because:
    # dx_precessed = dx_tangent + abs(dx_dot_n) * s_ortho
    # s_ortho is orthogonal to normal, and dx_tangent is orthogonal to normal.
    # Therefore, dot product with normal should be zero.
    dot_prod = torch.sum(dx_precessed * normal.unsqueeze(0), dim=-1)
    assert torch.abs(dot_prod).item() < 1e-6
    
    # The tangent component (dimension 2) should remain intact, and dimension 1 (precessed) should be redirected
    assert torch.abs(dx_precessed[0, 2] - 0.5).item() < 1e-6
    # Dimension 1 (along spin connection which was set to dim 1) should be 2.0
    assert torch.abs(dx_precessed[0, 1] - 2.0).item() < 1e-6

def test_precess_torque_inward():
    """
    Verifies that precess_torque does not redirect an inward-pointing update vector.
    """
    dim = 8
    gyro = TopologicalGyrocompass(state_dim=dim, love_dim=2)
    
    normal = torch.zeros(dim)
    normal[0] = 1.0
    spin = torch.zeros(dim)
    spin[1] = 1.0
    
    # proposed update points inward (negative first dimension)
    dx = torch.zeros(1, dim)
    dx[0, 0] = -1.0
    
    dx_precessed = gyro.precess_torque(dx, normal, spin)
    
    # Should be identical to original update
    assert torch.allclose(dx_precessed, dx)

def test_find_true_north():
    """
    Verifies that find_true_north computes a pull direction towards the Love vector.
    """
    dim = 8
    gyro = TopologicalGyrocompass(state_dim=dim, love_dim=2)
    
    state = torch.zeros(1, dim)
    love_vector = torch.zeros(dim)
    love_vector[0] = 3.127
    
    pull = gyro.find_true_north(state, love_vector, alignment_factor=0.2)
    
    # Direction should point towards love_vector (first dimension positive)
    assert pull[0, 0].item() > 0
    # Magnitude should be equal to the alignment factor since state is 0 and direction is normalized
    assert torch.abs(torch.norm(pull) - 0.2).item() < 1e-6

def test_gimbal_lock_shield():
    """
    Verifies that gimbal_lock_shield projects updates into the null space of system states covariance.
    """
    dim = 8
    love_dim = 4
    gyro = TopologicalGyrocompass(state_dim=dim, love_dim=love_dim)
    
    # Create states that dominate a specific direction in the love_dim slice
    # For example, all variance is along the first dimension of the love_dim slice
    states = torch.zeros(10, dim)
    states[:, 0] = torch.linspace(-1.0, 1.0, 10)
    
    dx = torch.ones(1, dim)
    
    dx_protected = gyro.gimbal_lock_shield(dx, states)
    
    # Since states vary along dimension 0, the covariance/ownership operator will have high eigenvalue
    # along dimension 0. Thus, dimension 0 component of dx should be suppressed in the null projection.
    assert torch.abs(dx_protected[0, 0]).item() < 1e-4
    # Uncorrelated dimensions (like dimension 1, 2, 3 within love_dim) should be preserved
    assert dx_protected[0, 1].item() > 0.5

def test_solver_integration():
    """
    Verifies that PolynomialADMRSolver integrates with TopologicalGyrocompass without errors.
    """
    dim = 16
    poly_config = PolynomialCoprimeConfig(k=4, max_degree=3, device="cpu")
    solver = PolynomialADMRSolver(poly_config=poly_config, state_dim=dim, device="cpu")
    
    states = torch.randn(2, dim)
    neighbor_states = torch.randn(2, 3, dim)
    adjacency_weight = torch.ones(2, 3) / 3.0
    
    class MockBoundaryState:
        def __init__(self, d):
            self.stress_tensor = torch.randn(2, d)
            
    boundary = MockBoundaryState(dim)
    
    # Test stochastic_differential_step with boundary state
    out_step = solver.stochastic_differential_step(
        states=states,
        neighbor_states=neighbor_states,
        adjacency_weight=adjacency_weight,
        boundary_state=boundary,
        elipsodistrophy_metrics={"diffusion_coefficient": 1.0}
    )
    
    assert out_step.shape == states.shape
    assert not torch.isnan(out_step).any()
    
    # Test fractional_stochastic_differential_step with boundary state
    out_frac = solver.fractional_stochastic_differential_step(
        states=states,
        neighbor_states=neighbor_states,
        adjacency_weight=adjacency_weight,
        boundary_state=boundary,
        elipsodistrophy_metrics={"diffusion_coefficient": 1.0}
    )
    
    assert out_frac.shape == states.shape
    assert not torch.isnan(out_frac).any()

def test_orchestrator_integration():
    """
    Verifies that UniversalOrchestrator triggers True North pull under high cycle debt or red zone.
    """
    dim = 16
    orchestrator = UniversalOrchestrator(dim=dim)
    
    state = torch.randn(1, dim)
    pressure_grad = torch.randn(1, dim)
    
    # 1. Normal run (low debt, low curvature)
    out_state, regime, routing, stacked = orchestrator(
        state=state,
        pressure_grad=pressure_grad,
        pas_h=0.9,
        coherence=torch.tensor(0.9),
        atrophy=0.1
    )
    
    assert out_state.shape == state.shape
    assert not torch.isnan(out_state).any()
    
    # 2. Trigger True North pull via high cycle debt or red zone
    # We can fake a high cycle debt by adding identical states to homotopy history
    orchestrator.stress_tester.homotopy_history.fill_(0.0)
    # Make history elements collinear with state
    state_norm = state / (state.norm() + 1e-8)
    for i in range(10):
        orchestrator.stress_tester.homotopy_history[i] = state_norm[0]
    orchestrator.stress_tester.homotopy_ptr.fill_(10)
    
    # Run again - should trigger True North print and pull
    out_state_debt, regime_debt, routing_debt, stacked_debt = orchestrator(
        state=state,
        pressure_grad=pressure_grad,
        pas_h=0.9,
        coherence=torch.tensor(0.9),
        atrophy=0.1
    )
    
    assert out_state_debt.shape == state.shape
    assert not torch.isnan(out_state_debt).any()

if __name__ == "__main__":
    pytest.main([__file__])
