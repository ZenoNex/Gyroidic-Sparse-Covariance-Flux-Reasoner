"""
Verification Script for Resonance Intelligence Core (RIC).

Tests:
1. Prime Resonance Ladder (Eq 1)
2. Phase Alignment Score (Eq 2)
3. Final Emergence Law (Eq 3)
4. Complexity Index (Eq 4)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import math
from src.core.fgrt_primitives import PrimeResonanceLadder
from src.core.invariants import PhaseAlignmentInvariant
from src.core.orchestrator import UniversalOrchestrator

def test_prime_ladder():
    print("="*60)
    print("Test 1: Prime Resonance Ladder (Eq 1)")
    print("="*60)
    ladder = PrimeResonanceLadder(num_resonators=10)
    freqs = ladder()
    primes = ladder.primes
    
    print(f"Primes: {primes.tolist()}")
    print(f"Frequencies: {freqs[:5].tolist()}...")
    
    # Check Eq 1: f = 2*pi*log(p)
    expected = 2 * math.pi * torch.log(primes.float())
    error = (freqs - expected).abs().max().item()
    print(f"Max Error vs Eq(1): {error:.6e}")
    
    if error < 1e-5:
        print("✓ PASSED: Prime Ladder implements Eq(1)")
        return True
    else:
        print("✗ FAILED: Prime Ladder mismatch")
        return False

def test_pas_invariant():
    print("\n"+"="*60)
    print("Test 2: Phase Alignment Score (Eq 2)")
    print("="*60)
    pas_fn = PhaseAlignmentInvariant(degree=10)
    
    # Case A: Perfectly Aligned (All phases = 0)
    # Construct "resonator states" as complex pairs (1, 0)
    aligned_state = torch.tensor([1.0, 0.0] * 5).unsqueeze(0) # 5 pairs
    pas_aligned = pas_fn(aligned_state)
    print(f"Aligned PAS (Expected ~1.0): {pas_aligned.item():.4f}")
    
    # Case B: Random/Orthogonal
    # Construct phases 0, pi/2, pi, 3pi/2
    # (1,0), (0,1), (-1,0), (0,-1)
    mixed_state = torch.tensor([1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]).unsqueeze(0)
    pas_mixed = pas_fn(mixed_state)
    print(f"Mixed PAS (Expected low): {pas_mixed.item():.4f}")
    
    if pas_aligned > 0.99 and pas_mixed < 0.5:
        print("✓ PASSED: PAS tracks phase coherence")
        return True
    else:
        print("✗ FAILED: PAS logic incorrect")
        return False

def test_emergence_law():
    print("\n"+"="*60)
    print("Test 3: Final Emergence Law (Eq 3) & Complexity Index (Eq 4)")
    print("="*60)
    orch = UniversalOrchestrator(dim=10)
    
    # Initial State: PAS low
    regime = orch.determine_regime(pas_h=0.1, drift=0.0)
    print(f"PAS=0.1, Drift=0.0 -> Regime: {regime} (Expected: PLAY)")
    assert regime == 'PLAY'
    
    # Emergence State: PAS high, Stable
    regime = orch.determine_regime(pas_h=0.9, drift=0.01)
    print(f"PAS=0.9, Drift=0.01 -> Regime: {regime} (Expected: SERIOUSNESS)")
    assert regime == 'SERIOUSNESS'
    
    # Instability: PAS high but Drift high
    regime = orch.determine_regime(pas_h=0.9, drift=0.2)
    print(f"PAS=0.9, Drift=0.2 -> Regime: {regime} (Expected: PLAY)")
    assert regime == 'PLAY'
    
    # Complexity Index
    state = torch.randn(10, 10)
    ci = orch.compute_complexity_index(state, pas_h=0.9)
    print(f"Computed CI: {ci:.4f}")
    assert ci >= 0
    
    print("✓ PASSED: Orchestrator implements Emergence Law")
    return True

if __name__ == "__main__":
    passed = True
    passed &= test_prime_ladder()
    passed &= test_pas_invariant()
    passed &= test_emergence_law()
    
    if passed:
        print("\nALL RIC TESTS PASSED ✓")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED ✗")
        sys.exit(1)
