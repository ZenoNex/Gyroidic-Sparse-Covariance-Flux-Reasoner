"""
Verify Spectral Speculative Decoding & Ranging Logic
"""
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.fractional_operators import frac_apply
from src.core.non_ergodic_entropy import HybridLassoQuantizer, NonErgodicEntropyEstimator

def test_spectral_ranging():
    print("="*60)
    print("Verifying Adaptive Ranging (Fractional Operators)")
    print("="*60)
    
    # 1. Coherent Signal (Simulated)
    # v sum is controlled to give PAS=1 (approx)
    # CODES uses math.cos(m * phase). max when phase=0.
    v_coherent = torch.zeros(10) # Sum = 0 -> Phase = 0 -> PAS = 1
    
    # M = Identity
    M = torch.eye(10)
    
    # With PAS=1, alpha should be alpha_base (e.g. 1.0)
    res_coherent = frac_apply(M, v_coherent, alpha=1.0, ranging_gamma=0.5)
    print("Coherent Call: Success")
    
    # 2. Incoherent Signal
    # We can't easily force PAS=0 without precise math on sum, but we can verify it runs
    v_random = torch.randn(10)
    res_incoherent = frac_apply(M, v_random, alpha=1.0, ranging_gamma=0.5)
    print("Incoherent Call: Success")
    
    return True

def test_hardened_quantization():
    print("\n" + "="*60)
    print("Verifying Hardened Windowing Quantization")
    print("="*60)
    
    quantizer = HybridLassoQuantizer(dim=10, lasso_lambda=0.1)
    
    x = torch.tensor([0.05, 0.2, 0.5]) # 0.05 < 0.1 (should be zeroed)
    
    # 1. Base Hardening (Factor=1)
    y_base = quantizer(x, hardening_factor=1.0)
    print(f"Input: {x}")
    print(f"Output (Base): {y_base}")
    # Expect: 0.05 -> 0, 0.2 -> quantized
    assert y_base[0] == 0.0, "Lasso failed at base"
    
    # 2. Hardened (Factor=3) -> lambda=0.3
    # 0.2 < 0.3 (should be zeroed now)
    y_hard = quantizer(x, hardening_factor=3.0)
    print(f"Output (Hardened): {y_hard}")
    assert y_hard[1] == 0.0, "Hardening failed to silence weak signal"
    
    print("Hardening Logic: Verified")
    return True

if __name__ == "__main__":
    test_spectral_ranging()
    test_hardened_quantization()
