"""
Verify Pointer #3: Saturation-Gated Fossilization

Tests that fossilization is blocked until saturation is achieved.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.core.polynomial_coprime import PolynomialCoprimeConfig


def test_saturation_gated_fossilization():
    """
    Verify Pointer #3: Irreversibility Hardens Early Bias
    Fossilization should be gated on saturation detection.
    """
    print("=" * 60)
    print("Verifying Pointer #3: Saturation-Gated Fossilization")
    print("=" * 60)
    
    config = PolynomialCoprimeConfig(k=5, degree=4)
    all_passed = True
    
    # Test 1: Pre-saturation fossilization should be blocked
    print("\n[1] Attempting fossilization BEFORE saturation...")
    result = config.fossilize(0)
    
    if not result:
        print("    ✓ PASSED: Fossilization blocked (not saturated)")
    else:
        print("    ✗ FAILED: Fossilization should be blocked before saturation!")
        all_passed = False
    
    # Test 2: Build up pressure history to simulate saturation
    print("\n[2] Simulating pressure history (bounded oscillation)...")
    for i in range(25):
        # Simulate bounded oscillation (small variance)
        pressure = 0.5 + 0.01 * (i % 3 - 1)  # Oscillates between 0.49 and 0.51
        config.update_pressure_history(0, pressure)
    
    is_saturated = config._is_saturated(0)
    print(f"    Saturation status for k=0: {is_saturated}")
    
    if is_saturated:
        print("    ✓ PASSED: Bounded oscillation detected as saturation")
    else:
        print("    ✗ FAILED: Bounded oscillation should indicate saturation!")
        all_passed = False
    
    # Test 3: Post-saturation fossilization should succeed
    print("\n[3] Attempting fossilization AFTER saturation...")
    result = config.fossilize(0)
    
    if result and config.is_fossilized[0]:
        print("    ✓ PASSED: Fossilization succeeded after saturation")
    else:
        print("    ✗ FAILED: Fossilization should succeed after saturation!")
        all_passed = False
    
    # Test 4: High variance should NOT be saturated
    print("\n[4] Testing high-variance history (NOT saturated)...")
    for i in range(25):
        # High variance oscillation
        pressure = 0.5 + 0.5 * (i % 2)  # Oscillates between 0.5 and 1.0
        config.update_pressure_history(1, pressure)
    
    is_saturated_1 = config._is_saturated(1)
    print(f"    Saturation status for k=1: {is_saturated_1}")
    
    if not is_saturated_1:
        print("    ✓ PASSED: High variance correctly identified as NOT saturated")
    else:
        print("    ✗ FAILED: High variance should not be saturated!")
        all_passed = False
    
    # Test 5: Get saturation status for all
    print("\n[5] Checking saturation status overview...")
    status = config.get_saturation_status()
    print(f"    Status: {status}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_saturation_gated_fossilization()
    sys.exit(0 if success else 1)
