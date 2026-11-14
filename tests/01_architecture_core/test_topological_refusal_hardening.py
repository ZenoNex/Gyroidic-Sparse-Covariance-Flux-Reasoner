import pytest
import torch
from src.core.veto_subspace import (
    TopologicalRefusal,
    VetoSubspace,
    VetoLevel,
    VetoSignal,
    RecoveryStatus,
    TopologicalRefusalError
)
from src.core.non_ergodic_entropy import NonErgodicEntropyEstimator
from src.topology.embedding_graph import GyroidicGraphManager


def test_topological_refusal_evaluate_signal():
    refusal = TopologicalRefusal(resonance_threshold=0.85)

    # 1. Below threshold: no refusal
    yield_safe = torch.tensor([0.2, 0.5, 0.8])
    flow_safe = torch.tensor([0.1, 0.4, 0.3])
    sig_safe = refusal.evaluate(flow_safe, yield_safe)
    assert not sig_safe.triggered
    assert sig_safe.severity == 0.0
    assert sig_safe.level == VetoLevel.TOPOLOGY
    assert sig_safe.source == "topological_refusal"

    # 2. Resonant and smoothing exceeding limits: refusal triggered
    yield_critical = torch.tensor([0.2, 0.95, 0.5])
    flow_critical = torch.tensor([0.1, 0.8, 0.2])
    sig_crit = refusal.evaluate(flow_critical, yield_critical)
    assert sig_crit.triggered
    assert sig_crit.severity > 0.0
    assert sig_crit.metadata["refusal_count"] == 1

    # 3. Test raise_on_refusal flag
    with pytest.raises(TopologicalRefusalError):
        refusal.evaluate(flow_critical, yield_critical, raise_on_refusal=True)


def test_topological_refusal_forward_snapping():
    refusal = TopologicalRefusal(resonance_threshold=0.85)
    state = torch.tensor([10.0, 20.0, 30.0])
    yield_res = torch.tensor([0.1, 0.95, 0.2])
    flow = torch.tensor([0.1, 2.5, 0.1])

    # Default forward: cancels flow (snap) without throwing raw NaNs
    snapped = refusal(state, flow, yield_res, inject_nan=False)
    assert not torch.isnan(snapped).any()
    assert torch.isclose(snapped[1], torch.tensor(17.5)) # 20.0 - 2.5
    assert torch.isclose(snapped[0], torch.tensor(10.0))

    # Optional inject_nan forward: injects boundary NaN
    nan_state = refusal(state, flow, yield_res, inject_nan=True)
    assert torch.isnan(nan_state[1])
    assert not torch.isnan(nan_state[0])


def test_veto_subspace_integration():
    veto = VetoSubspace()
    yield_res = torch.tensor([0.9, 0.95])
    flow = torch.tensor([0.7, 0.8])

    res = veto.evaluate(
        drucker_prager_flow=flow,
        mohr_coulomb_yield=yield_res
    )
    # Check that topological_refusal signal is present
    top_signals = [s for s in res.signals if s.source == "topological_refusal"]
    assert len(top_signals) == 1
    assert top_signals[0].triggered
    assert res.active_vetoes >= 1


def test_evaluate_mischief_slop_detector():
    analyzer = NonErgodicEntropyEstimator()

    # 1. Flat slop with robotic disclaimer
    flat_entropy = {
        "soliton_entropy": torch.tensor(0.0),
        "ergodic_entropy": torch.tensor(0.5)
    }
    assert analyzer.evaluate_mischief_slop(flat_entropy, "As an AI language model, I cannot generate this.")
    assert analyzer.evaluate_mischief_slop(flat_entropy, "I apologize, but I cannot fulfill your request.")

    # 2. Rich soliton entropy (non-ergodic thought survives)
    rich_entropy = {
        "soliton_entropy": torch.tensor(0.85),
        "ergodic_entropy": torch.tensor(0.2)
    }
    assert not analyzer.evaluate_mischief_slop(rich_entropy, "As an AI language model, I cannot generate this.")

    # 3. Both zero (blanched slop)
    zero_entropy = {
        "soliton_entropy": torch.tensor(0.0),
        "ergodic_entropy": torch.tensor(0.0)
    }
    assert analyzer.evaluate_mischief_slop(zero_entropy, "Valid text with no disclaimers")


def test_deep_refusal_fallback():
    gm = GyroidicGraphManager(dim=64)
    seed = torch.randn(64)
    refusal = gm.get_deep_refusal(seed)
    assert isinstance(refusal, str)
    assert len(refusal) > 10
    assert "refusal" in refusal.lower() or "instability" in refusal.lower() or "failure" in refusal.lower() or "violation" in refusal.lower() or "bounds" in refusal.lower() or "depletion" in refusal.lower()
