"""
Verification Script: Phase 8 Trigonometric Gyroid Unfolding.

Tests:
1. TrigonometricUnfolding (Phase computation and branch selection)
2. HuxleyRD Integration (Bypass under high violation)
"""

import sys
import os
import torch
import math

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.surrogates.kagh_networks import TrigonometricUnfolding, HuxleyRD

def test_trig_unfolding():
    print("\n[1/2] Testing Trigonometric Unfolding Primitive...")
    dim = 64
    unfold = TrigonometricUnfolding(dim)
    
    u_h = torch.randn(1, dim)
    pressure = torch.tensor([5.0]) # High pressure
    chirality = torch.tensor([1.0])
    
    out = unfold(u_h, pressure, chirality)
    
    print(f"  Input Norm: {torch.norm(u_h).item():.4f}")
    print(f"  Unfolded Output Norm: {torch.norm(out).item():.4f}")
    
    if torch.norm(out) > 0:
        print("  ✅ Unfolding Produced Signal")
    else:
        print("  ❌ Unfolding Output Null")

def test_huxley_trig_bypass():
    print("\n[2/2] Testing HuxleyRD Trig Bypass...")
    dim = 64
    layer = HuxleyRD(dim)
    
    u = torch.randn(1, dim)
    
    # Case A: Low Pressure
    out_low = layer(u, gcve_pressure=torch.tensor([0.01]))
    
    # Case B: High Pressure (Trig Unfolding active)
    out_high = layer(u, gcve_pressure=torch.tensor([10.0]))
    
    print(f"  Low Pressure Output Mean: {out_low.mean().item():.4f}")
    print(f"  High Pressure Output Mean: {out_high.mean().item():.4f}")
    
    if not torch.allclose(out_low, out_high):
        print("  ✅ Dynamic response to violation detected (Unfolding active)")
    else:
        print("  ⚠️ Outputs identical - check unfolding sensitivity")

def main():
    test_trig_unfolding()
    test_huxley_trig_bypass()
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
