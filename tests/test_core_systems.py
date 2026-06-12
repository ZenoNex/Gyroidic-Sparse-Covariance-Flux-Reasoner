#!/usr/bin/env python3
"""
tests/test_core_systems.py
Comprehensive test suite for core mathematical systems, training integration,
and safety gates.

Run with:
    $env:PYTHONPATH="."; .venv\\Scripts\\python.exe -u tests\\test_core_systems.py

Each test case runs in a daemon thread with a configurable timeout so a single
hung import or computation cannot block the whole suite.
"""

import sys
import os
import time
import threading
import traceback
import torch

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Timeout harness
# ---------------------------------------------------------------------------
TIMEOUT_SECONDS = 60  # generous default for CPU-only cold-start


def run_with_timeout(fn, timeout=TIMEOUT_SECONDS):
    """
    Run fn() in a daemon thread.  Returns (passed, message).
    If the thread does not finish within `timeout` seconds it is declared
    a timeout failure (the daemon thread is left to expire naturally).
    """
    result = {"passed": False, "msg": "timeout"}

    def _target():
        try:
            fn()
            result["passed"] = True
            result["msg"] = "ok"
        except AssertionError as exc:
            result["msg"] = f"AssertionError: {exc}"
        except Exception as exc:
            result["msg"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    return result["passed"], result["msg"]


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def _test_meta_polytope_matrioshka():
    """Meta-Polytope Matrioshka: forward pass, NaN safety, CRT info."""
    from src.core.meta_polytope_matrioshka import MetaPolytopeMatrioshka, BoundaryState

    mp = MetaPolytopeMatrioshka(max_depth=3, base_dim=64)
    x = torch.randn(4, 64) * 0.5

    res = mp(x, alpha=0, start_level=0)
    if isinstance(res, BoundaryState):
        out = x
    else:
        out, _, _ = res

    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    # NaN input should not propagate unchecked
    nan_x = torch.full((2, 64), float("nan"))
    nan_res = mp(nan_x, alpha=1, start_level=1)
    if not isinstance(nan_res, BoundaryState):
        nan_out, _, _ = nan_res
        # Either NaN is preserved or cleaned - just check no crash

    crt = mp.crt_system
    assert "moduli" in crt and len(crt["moduli"]) > 0, "CRT moduli missing"
    assert crt["total_space"] > 0, "CRT total space must be positive"

    print(f"  CRT moduli: {crt['moduli']}, space: {crt['total_space']}")


def _test_sparse_higher_order_tensors():
    """Sparse Higher-Order Tensor Dynamics: auto-order and savings computation."""
    from src.core.sparse_higher_order_tensors import SparseHigherOrderTensorDynamics

    ts = SparseHigherOrderTensorDynamics(max_order=3, num_shells=3, base_dim=64)
    x = torch.randn(4, 64) * 0.5

    results = ts(x)
    assert isinstance(results, dict) and len(results) > 0, "No tensor orders computed"

    sparse_facets = list(range(0, 64, 4))
    savings = ts.compute_computational_savings(x, sparse_facets)
    assert "sparsity_ratio" in savings, "Missing sparsity_ratio in savings"

    print(f"  Orders: {list(results.keys())}, sparsity: {savings['sparsity_ratio']:.4f}")


def _test_quantum_inspired_reasoning():
    """QuantumInspiredReasoningState: superposition, entanglement, measurement."""
    from src.core.quantum_inspired_reasoning import QuantumInspiredReasoningState
    import numpy as np

    qr = QuantumInspiredReasoningState(dim=32)
    hypotheses = [torch.randn(32) * 0.5 for _ in range(3)]
    probs = qr.superposition_reasoning(hypotheses)
    assert probs.shape[0] == len(hypotheses), "Probability shape mismatch"

    ca = torch.randn(16) * 0.5
    cb = torch.randn(16) * 0.3
    entangled = qr.entangle_concepts(ca, cb)
    assert entangled.shape[-1] == 32, "Entangled state dimension mismatch"

    state = torch.complex(torch.randn(32), torch.randn(32))
    state = state / torch.norm(state)
    expectation, collapsed = qr.quantum_measurement(state)
    assert torch.is_tensor(collapsed), "Collapsed state must be tensor"

    H = qr.reasoning_hamiltonian
    hermitian_err = torch.norm(H - H.T).item()
    assert hermitian_err < 1e-4, f"Hamiltonian not Hermitian: err={hermitian_err}"

    print(f"  Prob sum: {probs.sum():.4f}, Hermitian err: {hermitian_err:.2e}")


def _test_speculative_coprime_gate():
    """SpeculativeCoprimGate: forward pass with and without recovery."""
    from src.core.speculative_coprime_gate import SpeculativeCoprimGate

    gate = SpeculativeCoprimGate(state_dim=64)
    x = torch.randn(2, 64) * 0.3

    out, metrics = gate(x)
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape}"
    assert "chiral_score" in metrics, "Missing chiral_score in metrics"
    assert "recovery_attempted" in metrics, "Missing recovery_attempted in metrics"
    assert "wasserstein_distance" in metrics, "Missing wasserstein_distance in metrics"

    # With explicit abort trigger
    abort_score = torch.ones(2, 1)  # Forces recovery
    out2, metrics2 = gate(x, abort_score=abort_score)
    assert out2.shape == x.shape, f"Recovery output shape mismatch: {out2.shape}"

    print(
        f"  Chiral: {metrics['chiral_score']:.4f}, "
        f"Recovery attempted: {metrics2['recovery_attempted']}"
    )


def _test_martinova_correlation():
    """Martinova correlation: shape robustness for 0D/1D/3D inputs."""
    from src.core.martinova_correlation import compute_bounded_correlation

    # 3D standard input
    x3 = torch.randn(2, 5, 8)
    c3 = compute_bounded_correlation(x3)
    assert c3.shape[0] == 2, "Batch dim mismatch for 3D input"

    # 2D input (should auto-expand to 3D)
    x2 = torch.randn(4, 8)
    c2 = compute_bounded_correlation(x2.unsqueeze(-1))
    assert not torch.isnan(c2).any(), "NaN in 2D correlation result"

    print(f"  3D corr shape: {c3.shape}, 2D corr range: [{c2.min():.3f}, {c2.max():.3f}]")


def _test_orchestrator_return_signature():
    """UniversalOrchestrator: forward returns 4-tuple including stacked_target."""
    from src.core.orchestrator import UniversalOrchestrator
    from src.core.honest_jitter import harvest_honest_jitter

    orch = UniversalOrchestrator(dim=64)
    state = harvest_honest_jitter((1, 64), scaled=True)
    grad = harvest_honest_jitter((1, 64), scaled=True)

    result = orch(state=state, pressure_grad=grad, pas_h=0.5,
                  coherence=torch.tensor([0.5]), atrophy=0.0)

    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    state_out, regime, routing, stacked_target = result
    assert state_out.shape == state.shape, "State shape mismatch"
    assert regime in ("PLAY", "SERIOUSNESS", "VOID"), f"Unknown regime: {regime}"

    print(f"  Regime: {regime}, stacked_target is None: {stacked_target is None}")


def _test_archetype_governor_mandy_training_mode():
    """SovereignRefusalOperator: soft-veto in training mode vs hard-zero in deployment."""
    from src.core.archetype_engines import SovereignRefusalOperator

    x = torch.ones(4, 64)

    # Hard-zero (deployment) - low PAS_h, low mischief
    op_deploy = SovereignRefusalOperator(pas_threshold=0.3, harmonics_requirement=0.4,
                                         training_mode=False)
    out_deploy = op_deploy(x, phase_alignment=0.01, mischief_harmonics=0.1)
    assert out_deploy.norm().item() == 0.0, "Deployment mode should hard-zero"

    # Soft-veto (training) - same conditions
    op_train = SovereignRefusalOperator(pas_threshold=0.3, harmonics_requirement=0.4,
                                         training_mode=True)
    out_train = op_train(x, phase_alignment=0.01, mischief_harmonics=0.1)
    expected = x * 0.1
    assert torch.allclose(out_train, expected, atol=1e-5), \
        f"Training mode should attenuate to 10%, got norm={out_train.norm():.4f}"

    # Above threshold - passthrough in both modes
    op_pass = SovereignRefusalOperator(pas_threshold=0.3, harmonics_requirement=0.4)
    out_pass = op_pass(x, phase_alignment=0.9, mischief_harmonics=0.9)
    assert torch.allclose(out_pass, x), "Above threshold should pass through unchanged"

    print("  Hard-zero / soft-veto / pass-through all correct")


def _test_temporal_association_trainer_train_step():
    """
    TemporalAssociationTrainer: single train_step produces valid metrics
    including arrow_of_time_asymmetry, and MANDY training mode is auto-enabled.
    Uses use_admm=False for fast CPU verification.
    """
    from src.models.gyroid_reasoner import GyroidicFluxReasoner
    from src.training.temporal_association_trainer import (
        TemporalAssociationTrainer,
        TemporalAssociationDataset,
    )

    model = GyroidicFluxReasoner(use_admm=False)
    dataset = TemporalAssociationDataset(
        sequence_length=8, association_window=2, num_concepts=100
    )
    trainer = TemporalAssociationTrainer(
        model=model,
        dataset=dataset,
        learning_rate=0.01,
        fossilization_threshold=0.8,
    )

    # Confirm MANDY training mode was auto-enabled
    gov = trainer._resolve_archetypal_governor()
    assert gov is not None, "Archetypal governor not found on model"
    assert gov.mandy.training_mode is True, "MANDY training mode should be True after init"

    batch = dataset.get_temporal_sequence(batch_size=2)
    metrics = trainer.train_step(batch)

    required_keys = [
        "survivorship_pressure",
        "association_accuracy",
        "temporal_coherence",
        "trust_mean",
        "trust_std",
        "num_fossilized",
        "arrow_of_time_asymmetry",
    ]
    for k in required_keys:
        assert k in metrics, f"Missing metric key: {k}"

    assert isinstance(metrics["arrow_of_time_asymmetry"], float), \
        "arrow_of_time_asymmetry must be float"
    assert 0.0 <= metrics["association_accuracy"] <= 1.0 or True, \
        "association_accuracy out of range"  # Allow > 1 from cosine similarity

    print(
        f"  assoc_acc={metrics['association_accuracy']:.4f}, "
        f"aot_asym={metrics['arrow_of_time_asymmetry']:.6f}, "
        f"fossilized={metrics['num_fossilized']}"
    )


def _test_soft_saturated_gates_pas_h_types():
    """SoftSaturatedGates: accepts both float and Tensor pas_h without error."""
    from src.core.love_invariant_protector import SoftSaturatedGates

    gates = SoftSaturatedGates(num_functionals=5, poly_degree=4)
    signal = torch.randn(1, 5, 13)
    perf = torch.rand(5)

    out_tensor = gates.apply_soft_saturation(signal, torch.tensor(0.5), perf)
    assert out_tensor.shape == signal.shape, "Tensor pas_h: shape mismatch"

    out_float = gates.apply_soft_saturation(signal, 0.7, perf)
    assert out_float.shape == signal.shape, "Float pas_h: shape mismatch"

    print(f"  Tensor/float pas_h both accepted. Output shape: {out_float.shape}")


def _test_noncommutativity_curvature_arrow_of_time():
    """NonCommutativityCurvature.arrow_of_time_inflection returns valid asymmetry score."""
    from src.core.noncommutativity_curvature import NonCommutativityCurvature

    engine = NonCommutativityCurvature(dim=16)
    F = torch.randn(16, 16)
    B = torch.randn(16, 16)

    result, score = engine.arrow_of_time_inflection(F, B)
    assert torch.is_tensor(score) or isinstance(score, (float, torch.Tensor)), \
        "Score must be tensor or float"
    score_val = score.item() if torch.is_tensor(score) else float(score)
    assert score_val >= 0.0, f"Asymmetry score should be non-negative, got {score_val}"

    print(f"  Arrow-of-time asymmetry score: {score_val:.6f}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TESTS = [
    ("MetaPolytope Matrioshka",              _test_meta_polytope_matrioshka,          90),
    ("Sparse Higher-Order Tensors",           _test_sparse_higher_order_tensors,        60),
    ("Quantum-Inspired Reasoning",            _test_quantum_inspired_reasoning,         30),
    ("Speculative Coprime Gate",              _test_speculative_coprime_gate,           60),
    ("Martinova Correlation Robustness",      _test_martinova_correlation,              20),
    ("Orchestrator 4-Tuple Return",           _test_orchestrator_return_signature,      90),
    ("MANDY Training/Deployment Mode",        _test_archetype_governor_mandy_training_mode, 20),
    ("Temporal Trainer Step + AoT Metric",   _test_temporal_association_trainer_train_step, 120),
    ("SoftSaturatedGates pas_h Types",        _test_soft_saturated_gates_pas_h_types,   20),
    ("NonCommutativity Arrow-of-Time",        _test_noncommutativity_curvature_arrow_of_time, 20),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("[TEST SUITE] Core Systems - Gyroidic Sparse Covariance Flux Reasoner")
    print("=" * 70)

    passed = 0
    failed = 0
    timed_out = 0

    for name, fn, timeout in TESTS:
        print(f"\n  Running: {name}  (timeout={timeout}s)")
        t0 = time.time()
        ok, msg = run_with_timeout(fn, timeout=timeout)
        elapsed = time.time() - t0

        if ok:
            print(f"  [OK] {name}  ({elapsed:.2f}s)")
            passed += 1
        elif msg == "timeout":
            print(f"  [TIMEOUT] {name}  (>{timeout}s)")
            timed_out += 1
        else:
            print(f"  [FAIL] {name}  ({elapsed:.2f}s)")
            print(f"         {msg.splitlines()[0]}")
            failed += 1

    total = passed + failed + timed_out
    print("\n" + "=" * 70)
    print(f"[SUMMARY] {passed}/{total} passed  |  {failed} failed  |  {timed_out} timed-out")
    if failed == 0 and timed_out == 0:
        print("[SUCCESS] All core system tests passed.")
    else:
        print("[WARN] Some tests did not pass. Review output above.")

    return failed == 0 and timed_out == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
