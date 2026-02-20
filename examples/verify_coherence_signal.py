"""
Verify Pointer #4: Coherence Loss Is a Signal, Not a Fault

Tests that MetaInvariant treats H_1 expansion as a SIGNAL (logged, no penalty)
and H_1 contraction as a VIOLATION (penalty pressure).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.core.meta_invariant import MetaInvariant


def test_coherence_as_signal():
    """
    Verify Pointer #4: Coherence Loss Is a Signal, Not a Fault
    """
    print("=" * 60)
    print("Verifying Pointer #4: Coherence-as-Signal")
    print("=" * 60)
    
    meta = MetaInvariant(expansion_threshold=0.01)
    all_passed = True
    
    # Test 1: Expansion should NOT produce violation pressure
    print("\n[1] Testing topology EXPANSION (H_1: 5 -> 7)...")
    meta.prev_h1_dim = torch.tensor(5.0)
    meta.step_count = torch.tensor(1)
    is_valid, rate, violation = meta.check_invariant(torch.tensor(7.0))
    
    print(f"    Rate: {rate.item():.2f}")
    print(f"    is_valid: {is_valid}")
    print(f"    violation_pressure: {violation.item():.4f}")
    
    if is_valid and violation.item() == 0.0:
        print("    ✓ PASSED: Expansion treated as signal, not fault")
    else:
        print("    ✗ FAILED: Expansion should be valid with zero violation!")
        all_passed = False
    
    # Test 2: Contraction SHOULD produce violation pressure
    print("\n[2] Testing topology CONTRACTION (H_1: 7 -> 4)...")
    meta.prev_h1_dim = torch.tensor(7.0)
    meta.step_count = torch.tensor(2)
    is_valid, rate, violation = meta.check_invariant(torch.tensor(4.0))
    
    print(f"    Rate: {rate.item():.2f}")
    print(f"    is_valid: {is_valid}")
    print(f"    violation_pressure: {violation.item():.4f}")
    
    if not is_valid and violation.item() > 0.0:
        print("    ✓ PASSED: Contraction treated as violation")
    else:
        print("    ✗ FAILED: Contraction should produce violation pressure!")
        all_passed = False
    
    # Test 3: Stasis should be acceptable
    print("\n[3] Testing topology STASIS (H_1: 5.0 -> 5.005)...")
    meta.prev_h1_dim = torch.tensor(5.0)
    meta.step_count = torch.tensor(3)
    is_valid, rate, violation = meta.check_invariant(torch.tensor(5.005))
    
    print(f"    Rate: {rate.item():.4f}")
    print(f"    is_valid: {is_valid}")
    print(f"    violation_pressure: {violation.item():.4f}")
    
    if is_valid and violation.item() == 0.0:
        print("    ✓ PASSED: Stasis is acceptable")
    else:
        print("    ✗ FAILED: Stasis should be valid with zero violation!")
        all_passed = False
    
    # Test 4: Expansion logging
    print("\n[4] Checking expansion event logging...")
    meta.prev_h1_dim = torch.tensor(5.0)
    meta.step_count = torch.tensor(4)
    meta.check_invariant(torch.tensor(8.0))  # Large expansion
    
    if hasattr(meta, '_expansion_log') and len(meta._expansion_log) > 0:
        print(f"    Logged events: {len(meta._expansion_log)}")
        print(f"    Last event: rate={meta._expansion_log[-1]['rate']:.2f}, dim={meta._expansion_log[-1]['new_dim']:.2f}")
        print("    ✓ PASSED: Expansion events are logged")
    else:
        print("    ✗ FAILED: Expansion events should be logged!")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_coherence_as_signal()
    sys.exit(0 if success else 1)
