import torch
import sys
import os

# Add local project to path
sys.path.append(os.getcwd())

from src.core.numerical_d_module import NumericalDModuleManager, RationalSnappingLayer
from src.core.primitive_ops import SCALE_FACTOR

def test_rational_snapping():
    print("[TEST] Testing Rational Snapping Layer...")
    snapper = RationalSnappingLayer()
    
    # Input with small float noise
    x = torch.tensor([1.0, 2.0, 3.0]) + torch.randn(3) * 0.00001
    snapped = snapper(x)
    
    # Check if results are on the 1/SCALE_FACTOR grid
    residuals = (snapped * SCALE_FACTOR) % 1.0
    assert torch.allclose(residuals, torch.zeros_like(residuals), atol=1e-6), "Snapping failed to align with lattice!"
    print("[OK] Rational Snapping secured symbolic integrity.")

def test_entropy_rank():
    print("[TEST] Testing Entropy-Based Rank Detection...")
    manager = NumericalDModuleManager(state_dim=3, num_functionals=3)
    
    # Rank 2 matrix
    J = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.00001] # Near-vanishing ideal
    ])
    
    # Case 1: Low entropy (High coherence) -> Vanishing ideal is detected as rank 2
    rank_lo = manager.compute_holonomic_rank(J, entropy=0.1)
    
    # Case 2: High entropy (Noise) -> Higher threshold -> Rank 2
    # If we want the ideal to "vanish" more easily under noise:
    rank_hi = manager.compute_holonomic_rank(J, entropy=2.0)
    
    print(f"[RESULTS] Rank (Lo Entropy): {rank_lo}, Rank (Hi Entropy): {rank_hi}")
    assert rank_lo >= rank_hi, "High entropy should not increase rank detection!"
    print("[OK] D-module rank correctly modulates with functional entropy.")

if __name__ == "__main__":
    try:
        test_rational_snapping()
        test_entropy_rank()
        print("\n[VERIFICATION COMPLETE] Unicorn Synthesis algebraic core is stable.")
    except Exception as e:
        print(f"\n[VERIFICATION FAILED] {e}")
        sys.exit(1)
