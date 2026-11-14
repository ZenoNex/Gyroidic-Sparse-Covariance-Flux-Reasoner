import pytest
import torch
import math
import numpy as np
from src.core.speculative_coprime_gate import SpeculativeCoprimeGate
from src.training.training_manager import TrainingManager
from src.core.orchestrator import UniversalOrchestrator
from src.core.honest_jitter import harvest_honest_jitter

def test_spectral_speculative_exit_low_entropy():
    dim = 32
    gate = SpeculativeCoprimeGate(dim=dim, spectral_exit_threshold=0.95)
    
    # Create a low-entropy state: a pure single-frequency sinusoid (soliton breather)
    t = torch.linspace(0, 2 * math.pi, dim)
    low_entropy_state = torch.sin(2 * t).unsqueeze(0) # [1, 32]
    
    # Calculate spectral entropy via canonical engine
    can_fastpass, h_spec = gate.evaluate_spectral_entropy(low_entropy_state)
    assert can_fastpass is True
    
    # Run speculative recovery
    rec_state, metrics = gate.speculative_recovery(
        converged_state=low_entropy_state,
        mode='SERIOUSNESS'
    )
    
    assert 'speculative_exit_bypassed' in metrics
    assert metrics['speculative_exit_bypassed'] is True
    assert metrics['wasserstein_distance'] == 0.0
    assert rec_state.shape == (1, dim)

def test_spectral_speculative_exit_high_entropy_fallback():
    dim = 32
    # Set a very low threshold so random/diffuse noise will exceed it
    gate = SpeculativeCoprimeGate(dim=dim, spectral_exit_threshold=0.01)
    
    # Create a high-entropy state: broad-band noise
    high_entropy_state = harvest_honest_jitter((1, dim), scaled=True)
    
    can_fastpass, h_spec = gate.evaluate_spectral_entropy(high_entropy_state)
    assert can_fastpass is False
    
    # Run speculative recovery
    rec_state, metrics = gate.speculative_recovery(
        converged_state=high_entropy_state,
        mode='SERIOUSNESS'
    )
    
    # When threshold is 0.01, the high-entropy state does not take the speculative bypass
    assert metrics['speculative_exit_bypassed'] is False
    assert rec_state.shape == (1, dim)

def test_training_manager_temporal_association():
    class FastMockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(768, 768)
        def forward(self, x):
            return self.linear(x)

    class MockAISystem:
        def __init__(self):
            self.temporal_model = FastMockModel()
            self.device = torch.device('cpu')
            
    ai_sys = MockAISystem()
    manager = TrainingManager(ai_sys)
    
    # Run a quick 1-epoch training loop in temporal mode
    ok, msg = manager.start_training(epochs=1, learning_rate=0.001, trainer_mode="temporal")
    assert ok is True
    
    # Wait for thread to finish
    manager.training_thread.join(timeout=60.0)
    status = manager.get_status()
    
    assert status['active'] is False
    assert len(manager.metrics_history) > 0
    assert manager.results is not None
    assert manager.results['success'] is True

def test_universal_orchestrator_caq_integration():
    dim = 32
    orchestrator = UniversalOrchestrator(dim=dim)
    
    # Verify CAQ is instantiated
    assert hasattr(orchestrator, 'caq')
    assert orchestrator.caq is not None
    assert orchestrator.caq.dim == dim
    
    # Run a step on orchestrator with mock inputs
    state = harvest_honest_jitter((dim,), scaled=True)
    step_res = orchestrator.step(state)
    assert isinstance(step_res, dict)
    out_state = step_res["state"]
    
    assert out_state is not None
    assert out_state.shape[-1] == dim
    assert hasattr(orchestrator, '_last_boundary_state')


def test_training_manager_gdpo_sovereignty():
    class FastMockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)
        def forward(self, x):
            return self.linear(x)

    class MockAISystem:
        def __init__(self):
            self.temporal_model = FastMockModel()
            self.device = torch.device('cpu')

    ai_sys = MockAISystem()
    manager = TrainingManager(ai_sys)

    ok, msg = manager.start_training(epochs=1, learning_rate=0.001, trainer_mode="gdpo")
    assert ok is True

    manager.training_thread.join(timeout=60.0)
    status = manager.get_status()

    assert status['active'] is False
    assert len(manager.metrics_history) > 0
    assert manager.results is not None
    assert manager.results['success'] is True


def test_symbolic_trust_table_determinism():
    from src.models.resonance_cavity import SymbolicTrustTable
    table1 = SymbolicTrustTable(table_size=128, k_dim=5)
    table2 = SymbolicTrustTable(table_size=128, k_dim=5)

    test_residues = torch.randn(4, 5)
    indices1 = table1._hash(test_residues)
    indices2 = table2._hash(test_residues)

    assert torch.equal(indices1, indices2), "Trust table hashes must be deterministic across instances"


def test_spectral_coherence_repair_canonical():
    from src.core.spectral_coherence_repair import (
        SpectralSpeculationEngine,
        SpectralCoherenceCorrector,
        BezoutCoefficientRefresh,
        apply_energy_based_stabilization
    )
    from src.core.invariants import SpectralSpeculationEngine as InvariantSpectralEngine
    # Verify backward compatible re-export
    assert InvariantSpectralEngine is SpectralSpeculationEngine

    engine = SpectralSpeculationEngine(epsilon=0.5, num_resonators=16)
    sig = torch.randn(1, 16)
    can_fastpass, h_spec = engine.evaluate_spectral_entropy(sig)
    assert isinstance(can_fastpass, bool)
    assert isinstance(h_spec, float)

    corrector = SpectralCoherenceCorrector()
    corrected = corrector.adaptive_coherence_correction(sig)
    assert corrected.shape == sig.shape

    stabilized = apply_energy_based_stabilization(sig)
    assert torch.isfinite(stabilized).all()

