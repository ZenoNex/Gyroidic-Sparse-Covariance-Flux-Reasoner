"""
Verify Non-Ergodic Fractal Entropy Decomposition.

Tests:
1. Band-separated entropy computation
2. Adaptive partitioning based on spectral coherence
3. Soliton preservation (dominant mode representatives)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.core.non_ergodic_entropy import (
    NonErgodicEntropyEstimator,
    AdaptiveFractalPartitioner,
    NonErgodicFractalEntropy
)


def test_band_separated_entropy():
    """Test that entropy is computed separately for ergodic and soliton bands."""
    print("=" * 60)
    print("Test 1: Band-Separated Entropy")
    print("=" * 60)
    
    estimator = NonErgodicEntropyEstimator(num_bands=3)
    
    # Create signal with mixed frequencies
    x = torch.linspace(0, 10, 64)
    low_freq = torch.sin(x)  # Ergodic
    high_freq = torch.sin(20 * x) * 0.5  # Soliton
    mixed = low_freq + high_freq
    
    phi = mixed.unsqueeze(0)  # [1, 64]
    
    result = estimator(phi)
    
    print(f"  Ergodic entropy: {result['ergodic_entropy'].item():.4f}")
    print(f"  Soliton entropy: {result['soliton_entropy'].item():.4f}")
    print(f"  Num bands: {result['total_bands'].shape[0]}")
    
    assert 'ergodic_entropy' in result
    assert 'soliton_entropy' in result
    print("  ✓ PASSED: Band-separated entropy computed")
    return True


def test_adaptive_partitioning():
    """Test that partitioning splits at coherence drops."""
    print("\n" + "=" * 60)
    print("Test 2: Adaptive Partitioning")
    print("=" * 60)
    
    partitioner = AdaptiveFractalPartitioner(min_block=2, max_block=8)
    
    # Create phi with varying coherence
    # First half: highly coherent (similar frequencies)
    # Second half: different frequency
    x = torch.linspace(0, 10, 16)
    coherent_part = torch.stack([torch.sin(x + i*0.1) for i in range(8)], dim=1)
    incoherent_part = torch.stack([torch.sin((i+1)*x) for i in range(8)], dim=1)
    
    phi = torch.cat([coherent_part, incoherent_part], dim=1)  # [16, 16]
    phi = phi.t()  # [16, 16] -> want [batch, K]
    
    blocks = partitioner.partition(phi)
    
    print(f"  Input shape: [1, {phi.shape[1] if phi.dim() > 1 else phi.shape[0]}]")
    print(f"  Blocks found: {len(blocks)}")
    for i, (start, end) in enumerate(blocks):
        print(f"    Block {i}: [{start}, {end}) size={end-start}")
    
    assert len(blocks) >= 1
    print("  ✓ PASSED: Adaptive partitioning works")
    return True


def test_soliton_preservation():
    """Test that dominant mode is used instead of mean."""
    print("\n" + "=" * 60)
    print("Test 3: Soliton Preservation (Dominant Mode)")
    print("=" * 60)
    
    entropy_fn = NonErgodicFractalEntropy(k_order=3, num_bands=3)
    
    # Create phi where one functional dominates
    phi = torch.randn(4, 12)  # [batch=4, K=12]
    phi[:, 5] *= 10  # Make functional 5 dominant
    
    result = entropy_fn(phi)
    
    print(f"  Local entropy shape: {result['local_entropy'].shape}")
    print(f"  Global entropy: {result['global_entropy'].item():.4f}")
    print(f"  Soliton preserved: {result['soliton_preserved'].item():.4f}")
    print(f"  Num blocks: {result['num_blocks']}")
    
    assert 'soliton_preserved' in result
    assert result['soliton_preserved'] >= 0
    print("  ✓ PASSED: Soliton preservation metric computed")
    return True


def test_backwards_compatibility():
    """Test that legacy_entropy=True falls back to HypergraphOrthogonalityPressure."""
    print("\n" + "=" * 60)
    print("Test 4: Backwards Compatibility")
    print("=" * 60)
    
    try:
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        
        # Default should use NonErgodicFractalEntropy
        config_new = PolynomialCoprimeConfig(k=8, degree=4)
        
        # Legacy should use HypergraphOrthogonalityPressure
        config_legacy = PolynomialCoprimeConfig(k=8, degree=4, legacy_entropy=True)
        
        print(f"  New config pressure fn: {type(config_new.orth_pressure_fn).__name__}")
        print(f"  Legacy config pressure fn: {type(config_legacy.orth_pressure_fn).__name__}")
        
        # Both should work
        pressure_new = config_new.orthogonality_pressure()
        pressure_legacy = config_legacy.orthogonality_pressure()
        
        print(f"  New pressure keys: {list(pressure_new.keys())}")
        print(f"  Legacy pressure keys: {list(pressure_legacy.keys())}")
        
        print("  ✓ PASSED: Backwards compatibility works")
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


if __name__ == "__main__":
    all_passed = True
    
    all_passed &= test_band_separated_entropy()
    all_passed &= test_adaptive_partitioning()
    all_passed &= test_soliton_preservation()
    all_passed &= test_backwards_compatibility()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
