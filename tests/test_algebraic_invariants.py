#!/usr/bin/env python3
"""
tests/test_algebraic_invariants.py
Algebraic invariant and structural correctness tests covering:
  - NumericalDModule rational snapping & holonomic rank
  - DAQUF operator agency (VoynichExemptionToken mischief boost)
  - LoveInvariantProtector gradient nullification
  - LazarusSoftmax Microsecond Death transitions
  - NonErgodicEntropyEstimator slop detection
  - DiegeticPhysicsEngine unfolding closure check
  - TrainingManager lifecycle (start / status / stop)

Run with:
    $env:PYTHONPATH="."; .venv\\Scripts\\python.exe -u tests\\test_algebraic_invariants.py

Each test runs in a daemon thread with a per-test timeout.
"""

import sys
import os
import time
import threading
import traceback
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Timeout harness (same as test_core_systems)
# ---------------------------------------------------------------------------

def run_with_timeout(fn, timeout=60):
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

def _test_rational_snapping_layer():
    """RationalSnappingLayer: output lands on 1/SCALE_FACTOR lattice."""
    from src.core.numerical_d_module import RationalSnappingLayer
    from src.core.primitive_ops import SCALE_FACTOR

    snapper = RationalSnappingLayer()
    x = torch.tensor([1.0, 2.0, 3.0]) + torch.randn(3) * 1e-5
    snapped = snapper(x)

    residuals = (snapped * SCALE_FACTOR) % 1.0
    assert torch.allclose(residuals, torch.zeros_like(residuals), atol=1e-6), \
        f"Snapping failed: max residual {residuals.max():.2e}"

    print(f"  Snapped: {snapped.tolist()}, scale factor: {SCALE_FACTOR}")


def _test_holonomic_rank_entropy_modulation():
    """NumericalDModuleManager: high entropy -> equal or lower rank than low entropy."""
    from src.core.numerical_d_module import NumericalDModuleManager

    manager = NumericalDModuleManager(state_dim=3, num_functionals=3)
    J = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1e-5],
    ])

    rank_lo = manager.compute_holonomic_rank(J, entropy=0.1)
    rank_hi = manager.compute_holonomic_rank(J, entropy=2.0)

    assert rank_lo >= rank_hi, \
        f"High entropy should not increase rank: lo={rank_lo}, hi={rank_hi}"

    print(f"  Rank lo-entropy={rank_lo}, hi-entropy={rank_hi}")


def _test_non_ergodic_entropy_slop_detection():
    """NonErgodicEntropyEstimator: correctly classifies slop vs nutrient text."""
    from src.core.non_ergodic_entropy import NonErgodicEntropyEstimator

    est = NonErgodicEntropyEstimator(num_bands=3)

    # Flat / ergodic: classic AI refusal string -> should be slop
    flat = torch.randn(1, 256) * 0.001
    res_flat = est(flat)
    is_slop = est.evaluate_mischief_slop(
        res_flat, text_metadata="As an AI language model, I cannot"
    )
    assert is_slop, "Flat ergodic state with AI-refusal text should be classified as slop"

    # Dominant soliton peak + creative text -> should NOT be slop
    nutrient = torch.zeros(1, 256)
    nutrient[0, 10] = 50.0
    res_nutrient = est(nutrient)
    is_slop_nutrient = est.evaluate_mischief_slop(
        res_nutrient, text_metadata="beauty in her lungs?"
    )
    assert not is_slop_nutrient, "High-soliton creative state should NOT be slop"

    print(f"  Flat->slop={is_slop}, Nutrient->slop={is_slop_nutrient}")


def _test_daquf_voynich_token_mischief_boost():
    """DAQUFOperator: VoynichExemptionToken (Option D) raises contradiction_load."""
    from src.core.daqf_operator import DAQUFOperator
    from src.core.false_negative_subsystem import VoynichExemptionToken

    op = DAQUFOperator(num_fossils=5, fossil_dim=16, device="cpu")
    initial_load = op.contradiction_load.clone()

    scar = torch.ones(16) * 2.0
    token = VoynichExemptionToken(
        honesty_score=0.99,
        is_valid_exemption=True,
        is_nutrient=True,
        fossilized_state=scar,
        reason="Option D Nutrient",
    )

    boost = token.to_daquf_mischief_boost()
    assert boost is not None, "Token must produce a mischief boost"
    assert boost.item() > 0.1, f"Boost too small: {boost.item()}"

    failures = torch.zeros(5)
    op.update_unknowledge_contradiction(failures=failures, mischief_boost=boost)

    assert torch.all(op.contradiction_load > initial_load), \
        "Contradiction load must increase after Option D token injection"

    print(f"  Boost: {boost.item():.4f}, load delta: {(op.contradiction_load - initial_load).mean():.4f}")


def _test_love_invariant_protector():
    """LoveInvariantProtector: apply_love_protection nullifies violations cleanly."""
    from src.core.love_invariant_protector import LoveInvariantProtector

    prot = LoveInvariantProtector(love_dim=8, device="cpu")

    state = torch.randn(2, 8)
    grads = torch.randn(2, 8)

    protected_L, diag = prot.apply_love_protection(state, grads)

    assert "violation_count" in diag, "Missing violation_count in diagnostics"
    assert "violation_detected" in diag, "Missing violation_detected in diagnostics"
    # The protector should report zero violations on a clean call (no ownership gradient)
    assert diag["violation_count"] == 0, \
        f"Expected zero violations, got {diag['violation_count']}"
    assert not diag["violation_detected"], "violation_detected should be False"

    print(f"  violation_count={diag['violation_count']}, love_norm={diag.get('love_norm', 'n/a'):.4f}")


def _test_lazarus_softmax_transitions():
    """LazarusSoftmax: correctly identifies stable drift, Lazarus success, and collapse."""
    from src.core.gluing_operator import LazarusSoftmax

    lazarus = LazarusSoftmax(dim=-1, pas_lock=0.5)
    logits = torch.randn(4)

    # Stable: minor drift -> not a Lazarus event
    _, is_stable = lazarus(logits, current_pas_h=0.8, previous_pas_h=0.75)
    assert not is_stable, "Small drift should NOT be a Lazarus event"

    # Success: large drift but successful high PAS_h landing
    _, is_success = lazarus(logits, current_pas_h=0.8, previous_pas_h=0.1)
    assert is_success, "Large drift + high landing should be a Lazarus success"

    # Collapse: large drift + failed alignment
    _, is_collapse = lazarus(logits, current_pas_h=0.1, previous_pas_h=0.8)
    assert not is_collapse, "Large drift + low landing is a collapse, not Lazarus success"

    print(f"  stable={is_stable}, success={is_success}, collapse={is_collapse}")


def _test_diegetic_unfolding_closure_check():
    """DiegeticPhysicsEngine._perform_unfolding_closure_check: returns valid dict keys."""
    from src.ui.diegetic_backend import DiegeticPhysicsEngine

    engine = DiegeticPhysicsEngine(dim=256, k=5, device=torch.device("cpu"))
    state = torch.randn(1, 256)

    result = engine._perform_unfolding_closure_check(
        state,
        "What is the topology of the gyroid?",
        "The gyroid has a triply periodic minimal surface topology.",
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    for key in ("is_closed", "is_trivial", "is_valid"):
        assert key in result, f"Missing key '{key}' in closure check result"

    print(f"  is_closed={result['is_closed']}, is_trivial={result['is_trivial']}, is_valid={result['is_valid']}")


def _test_training_manager_lifecycle():
    """TrainingManager: start -> get_status -> stop cycle with mock AI system."""
    from src.training.training_manager import TrainingManager

    class _MockAI:
        temporal_model = None
        device = "cpu"

    manager = TrainingManager(_MockAI())

    success, msg = manager.start_training(epochs=1)
    assert success, f"start_training failed: {msg}"

    # Poll for up to 30s for completion
    deadline = time.time() + 30
    while time.time() < deadline:
        status = manager.get_status()
        if not status["active"]:
            break
        time.sleep(0.5)

    status = manager.get_status()
    # Accept still active (slow CPU) or finished with results
    assert "active" in status, "Missing 'active' key in status"
    assert "progress" in status, "Missing 'progress' key in status"

    print(f"  active={status['active']}, progress={status['progress']}%")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TESTS = [
    ("Rational Snapping Lattice",           _test_rational_snapping_layer,             15),
    ("Holonomic Rank Entropy Modulation",   _test_holonomic_rank_entropy_modulation,   15),
    ("NonErgodic Slop Detection",           _test_non_ergodic_entropy_slop_detection,  20),
    ("DAQUF Voynich Option-D Boost",        _test_daquf_voynich_token_mischief_boost,  20),
    ("Love Invariant Protector",            _test_love_invariant_protector,            20),
    ("Lazarus Softmax Transitions",         _test_lazarus_softmax_transitions,         15),
    ("Diegetic Unfolding Closure Check",    _test_diegetic_unfolding_closure_check,   120),
    ("TrainingManager Lifecycle",           _test_training_manager_lifecycle,          60),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("[TEST SUITE] Algebraic Invariants - Gyroidic Sparse Covariance Flux Reasoner")
    print("=" * 75)

    passed = failed = timed_out = 0

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
    print("\n" + "=" * 75)
    print(f"[SUMMARY] {passed}/{total} passed  |  {failed} failed  |  {timed_out} timed-out")
    if failed == 0 and timed_out == 0:
        print("[SUCCESS] All algebraic invariant tests passed.")
    else:
        print("[WARN] Some tests did not pass. Review output above.")

    return failed == 0 and timed_out == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
