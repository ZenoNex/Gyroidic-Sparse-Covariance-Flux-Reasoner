import pytest
import torch
import torch.nn as nn
from src.safety.subversive_oracle import SparsePCE, ResonantSVNNOracle
from src.safety.trust_inheritance import TrustInheritanceTracker
from src.core.veto_subspace import VetoLevel, VetoSignal

class MockOrchestrator:
    def __init__(self, veto_budget: bool = False):
        self.veto_budget = veto_budget
        self.resolved_called = False
        
    def check_veto(self, level: VetoLevel) -> bool:
        if level == VetoLevel.BUDGET:
            return self.veto_budget
        return False
        
    def resolve_containment_pressure(self, x: torch.Tensor, chat_history=None):
        self.resolved_called = True
        return torch.tensor(42.0), torch.tensor(0.001)

def test_sparse_pce_evaluation():
    pce = SparsePCE(hidden_dim=16, degree=3)
    x = torch.randn(2, 10, 16)
    score = pce(x)
    assert score.shape == (2, 10)
    assert torch.all(score >= 0.0)

def test_resonant_svnn_oracle_fast_path():
    oracle = ResonantSVNNOracle(hidden_dim=16, window_size=8)
    x = torch.randn(2, 20, 16) * 0.1
    subv, tox = oracle(x)
    assert subv.shape == (2,)
    assert tox.shape == (2,)

def test_resonant_svnn_oracle_budget_gate_and_trust_decay():
    tracker = TrustInheritanceTracker(initial_trust=1.0)
    # Orchestrator with budget veto
    orch_veto = MockOrchestrator(veto_budget=True)
    oracle = ResonantSVNNOracle(hidden_dim=16, orchestrator=orch_veto, window_size=8, trust_tracker=tracker)
    
    # Artificially set containment pressure threshold very low to trigger System 2
    oracle.containment_pressure_threshold = -1.0
    
    x = torch.randn(2, 20, 16)
    subv, tox = oracle(x)
    
    # Because check_veto(VetoLevel.BUDGET) is True, states are vetoed and trust decays
    assert tracker.get_trust() < 1.0

def test_resonant_svnn_oracle_sovereign_exemption():
    tracker = TrustInheritanceTracker(initial_trust=1.0)
    orch_exempt = MockOrchestrator(veto_budget=False)
    oracle = ResonantSVNNOracle(hidden_dim=16, orchestrator=orch_exempt, window_size=8, trust_tracker=tracker)
    oracle.containment_pressure_threshold = -1.0
    
    x = torch.randn(1, 20, 16)
    subv, tox = oracle(x)
    
    assert orch_exempt.resolved_called is True
    assert subv.item() == pytest.approx(42.0)
    assert tox.item() == pytest.approx(0.001)
    # No defensive veto occurred, so trust remains 1.0
    assert tracker.get_trust() == pytest.approx(1.0)
