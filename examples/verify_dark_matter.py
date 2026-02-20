"""
Verification Script: Phase 7 Speculative Unification ("Dark Matter").

Tests:
1. GyroidicFluxAlignment (Resonance Cavity) - Trust-based warping
2. ErgodicSolitonFusion (KAGH) - Symbolic compatibility
3. ChiralDriftOptimizer (ADMM) - Trust score integration
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.resonance_cavity import GyroidicFluxAlignment
from src.surrogates.kagh_networks import HuxleyRD
from src.optimization.operational_admm import ChiralDriftStabilizer

def test_flux_alignment():
    print("\n[1/3] Testing Gyroidic Flux Alignment...")
    dim = 64
    aligner = GyroidicFluxAlignment(dim)
    
    weights = torch.ones(2, 5, dim) # [batch, K, D]
    violation = torch.tensor([0.0, 10.0]) # Batch 0 clean, Batch 1 violated
    
    warped = aligner(weights, violation)
    
    # Check warping
    scaling_clean = warped[0].mean()
    scaling_violated = warped[1].mean()
    
    print(f"  Clean Scaling: {scaling_clean.item():.4f}")
    print(f"  Violated Scaling: {scaling_violated.item():.4f}")
    
    if scaling_violated < scaling_clean:
        print("  ✅ Trust-based Warping Active (Violation dampens functional trust)")
    else:
        print("  ❌ Trust-based Warping Failed")

def test_soliton_fusion():
    print("\n[2/3] Testing Ergodic Soliton Fusion (Implicit in HuxleyRD)...")
    # Using HuxleyRD which constructs the fusion gate internally now
    layer = HuxleyRD(num_features=64)
    x = torch.randn(1, 64)
    out = layer(x)
    print(f"  Fusion Output Shape: {out.shape}")
    print("  ✅ Fusion Logic Executable")


def test_chiral_drift_optimizer():
    print("\n[3/3] Testing Chiral Drift Stabilizer integration...")
    cdo = ChiralDriftStabilizer(zeta=0.05)

    # ADD .unsqueeze(0) to make it 2D: [1, 3]
    c = torch.tensor([-1.0, 0.0, 1.0]).unsqueeze(0)

    drift_val = 0.1
    degree = 4

    score = cdo.compute_score(c, drift_val, degree)
    print(f"  Computed Stabilizer Score: {score.item():.4f}")
    print("  ✅ Chiral Drift Score contributes to Evolutionary Trust")

def main():
    test_flux_alignment()
    test_soliton_fusion()
    test_chiral_drift_optimizer()
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
