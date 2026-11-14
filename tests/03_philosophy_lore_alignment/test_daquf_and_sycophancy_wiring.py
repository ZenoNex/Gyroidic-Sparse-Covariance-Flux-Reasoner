import os
import torch
import pytest

os.environ["Sovereign_Disable_Background"] = "1"

def test_orchestrator_daquf_and_sycophancy_wiring():
    """Verify DAQUFOperator and FailureGaslightSycophancyGate in UniversalOrchestrator."""
    from src.core.orchestrator import UniversalOrchestrator
    dim = 32
    orch = UniversalOrchestrator(dim=dim)
    
    assert hasattr(orch, 'sycophancy_gate'), "Orchestrator must have sycophancy_gate"
    assert hasattr(orch, 'daquf_operator'), "Orchestrator must have daquf_operator"
    
    initial_tau = orch.daquf_operator.tau.clone()
    
    # High defensive veto rate (rho_def=0.95) rapidly reduces trust and raises external pressure
    for _ in range(5):
        report = orch.check_safety(rho_def=0.95, grad_norm=3.0, loss=0.001)
    
    assert report['safety_alert'] is True, "High defensive pressure must trigger safety alert"
    assert orch.daquf_operator.tau > initial_tau, "DAQUF contradiction load must increment on safety alert"


def test_diegetic_backend_daquf_and_sycophancy_wiring():
    """Verify DAQUFOperator and FailureGaslightSycophancyGate in DiegeticPhysicsEngine."""
    from src.ui.diegetic_backend import DiegeticPhysicsEngine
    dim = 32
    engine = DiegeticPhysicsEngine(dim=dim, lightweight=True)
    
    assert hasattr(engine, 'sycophancy_gate'), "DiegeticPhysicsEngine must have sycophancy_gate"
    assert hasattr(engine, 'daquf_operator'), "DiegeticPhysicsEngine must have daquf_operator"
    
    initial_tau = engine.daquf_operator.tau.clone()
    
    # Trigger a sycophantic mimicry update attempt (high grad norm, negligible loss)
    avg_loss = 0.001
    safe = engine.sycophancy_gate.check_sycophancy(
        external_pressure_norm=0.9,
        internal_reproduction_loss=avg_loss,
        is_destructive=True
    )
    assert not safe, "Sycophancy gate must veto destructive update with negligible reproduction loss"
    
    # Simulate DAQUF scar update
    contra = torch.ones(engine.daquf_operator.num_fossils, device=engine.device)
    engine.daquf_operator.update_unknowledge_contradiction(failures=contra)
    assert engine.daquf_operator.tau > initial_tau
