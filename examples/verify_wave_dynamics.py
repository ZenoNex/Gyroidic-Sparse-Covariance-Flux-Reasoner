"""
Verification Script: Phase 6.5 Harmonic Wave Dynamics.

Tests:
1. HarmonicWaveDecomposition (Spectral Gating)
2. HuxleyRD Dual Dynamics (Ergodic Diffusion vs Non-Ergodic Soliton)
3. CALM Positivity Constraints

"Let part of the harmonic wave decomposition... carry through non-ergodic sub-dynamics"
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.surrogates.kagh_networks import HuxleyRD, HarmonicWaveDecomposition, KAGHBlock
from src.surrogates.calm_predictor import CALM

def test_wave_decomposition():
    print("\n[1/3] Testing Harmonic Wave Decomposition...")
    
    dim = 64
    decomp = HarmonicWaveDecomposition(dim)
    
    # Create a mixed signal: Low Freq Sine + High Freq Noise
    x = torch.linspace(0, 10, dim)
    signal_low = torch.sin(x) # Ergodic candidate
    signal_high = torch.sin(20 * x) * 0.5 # Non-Ergodic candidate
    mixed = (signal_low + signal_high).unsqueeze(0) # [1, dim]
    
    ergodic, non_ergodic = decomp(mixed)
    
    print(f"  Input Energy: {mixed.pow(2).sum().item():.4f}")
    print(f"  Ergodic Energy: {ergodic.pow(2).sum().item():.4f}")
    print(f"  Non-Ergodic Energy: {non_ergodic.pow(2).sum().item():.4f}")
    
    # Check if separation occurred (heuristic check dependent on init gate)
    # Ideally, gate init (ones) means ergodic=all, non_ergodic=0 initially
    # until trained.
    print("  (Note: Initial gate is all 1s, so ergodic should capture most initially)")

def test_huxley_dynamics():
    print("\n[2/3] Testing HuxleyRD Dual Dynamics...")
    
    dim = 64
    layer = HuxleyRD(num_features=dim, tau=0.1)
    
    # Set soliton velocity for test
    layer.soliton_velocity.data.fill_(5.0)
    
    x = torch.zeros(1, dim)
    x[0, dim//2] = 1.0 # Impulse
    
    # Forward pass
    out = layer(x)
    
    print(f"  Input Peak Location: {x.argmax().item()}")
    print(f"  Output Peak Location: {out.argmax().item()}")
    
    # Peak shift implies soliton movement or diffusion spreading
    if out.argmax().item() != x.argmax().item():
        print("  ✅ Dynamics Active (Shift/Diffusion detected)")
    else:
        print("  ⚠️ No shift detected (Check velocity/tau)")

def test_calm_positivity():
    print("\n[3/3] Testing CALM Positivity...")
    
    calm = CALM(dim=16)
    history = torch.randn(4, 8, 16) # Random input
    
    next_c, rho = calm(history)
    
    min_val = next_c.min().item()
    print(f"  Min predicted value: {min_val:.6f}")
    
    if min_val >= 0:
        print("  ✅ Positivity Maintained (Softplus active)")
    else:
        print("  ❌ Positivity Violated!")

def main():
    test_wave_decomposition()
    test_huxley_dynamics()
    test_calm_positivity()
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
