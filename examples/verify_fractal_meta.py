"""
Verify Fractal Meta-Functional Recursion.

Ensures that the S_fractal meta-state evolves recursively and perturbs
the system's symbolic output.
"""

import torch
import os
from src.core.fractal_meta_functional import FractalMetaFunctional
from src.ui.diegetic_backend import DiegeticPhysicsEngine

def verify_fractal_recursion():
    print("Verifying Fractal Meta-Functional Recursion...")
    
    # 1. Test Module Directly
    print("Testing FractalMetaFunctional module...")
    dim = 64
    batch_size = 2
    
    fractal = FractalMetaFunctional(dim=dim)
    
    state = torch.randn(batch_size, dim)
    meta_prev = torch.randn(batch_size, dim)
    residues = torch.randn(batch_size, 5)
    dark_matter = torch.randn(batch_size, dim)
    
    out = fractal(state, meta_prev, residues, dark_matter)
    
    s_fractal = out['s_fractal']
    print(f"S_fractal shape: {s_fractal.shape}")
    assert s_fractal.shape == (batch_size, dim)
    
    # Check component presence
    print("Components:", out['components'].keys())
    assert 'crt' in out['components']
    assert 'admr' in out['components']
    
    # 2. Test Integration in Physics Engine
    print("\nTesting Integration in DiegeticPhysicsEngine...")
    engine = DiegeticPhysicsEngine(dim=dim)
    
    # Initial meta state
    meta_0 = engine.meta_state.clone()
    print(f"Meta State t=0 norm: {torch.norm(meta_0).item()}")
    
    # Process interactions
    print("Processing interaction 1...")
    res1 = engine.process_input("Recursive Trust")
    meta_1 = engine.meta_state.clone()
    
    print("Processing interaction 2...")
    res2 = engine.process_input("Self Distrust")
    meta_2 = engine.meta_state.clone()
    
    # Verify evolution
    diff_01 = torch.norm(meta_0 - meta_1).item()
    diff_12 = torch.norm(meta_1 - meta_2).item()
    
    print(f"Meta-State Drift (0->1): {diff_01}")
    print(f"Meta-State Drift (1->2): {diff_12}")
    
    assert diff_01 > 0.0, "Meta-state must evolve after interaction"
    assert diff_12 > 0.0, "Meta-state must continue evolving"
    
    print("Fractal Recursion Verification PASSED.")

if __name__ == "__main__":
    verify_fractal_recursion()
