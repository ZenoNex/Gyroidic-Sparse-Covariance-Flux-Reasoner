"""
Yield Flow Demo: Dual-Regime Plasticity in Cognition.

Demonstrates:
1. Mohr-Coulomb (MC): Sharp local rupture sites.
2. Drucker-Prager (DP): Smooth global adaptation envelope.
3. Love Vector (L): Non-ownable persistent invariant.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.yield_criteria import MohrCoulombProjection, DruckerPragerProjection
from src.core.love_vector import LoveVector

def run_yield_demo():
    print("--- Starting Yield Flow Demo ---")
    
    # 1. Initialize Components
    dim = 8
    mc = MohrCoulombProjection(friction_angle=30.0, cohesion=1.0)
    dp = DruckerPragerProjection(alpha=0.1, k=2.0)
    love = LoveVector(dim, intensity=0.5)
    
    # 2. Simulate Increasing Pressure
    print("\nSimulating Information Pressure...")
    
    base_state = torch.ones(1, dim) * 0.5
    
    for i in range(5):
        pressure_level = (i + 1) * 1.5
        # Simulate a directional load (anisotropic)
        load = torch.randn(1, dim) * pressure_level
        
        # A. Love Vector co-presence
        # L is invariant; it doesn't care about the pressure.
        state_with_love = love(base_state)
        
        # B. Mohr-Coulomb: Local situational yield
        # Handles sharp rupture.
        yielded_mc = mc(state_with_love, load)
        
        # C. Drucker-Prager: Global smooth envelope
        # Provides the navigability out of the rupture.
        final_state = dp(yielded_mc)
        
        # D. Diagnostics
        rupture_severity = torch.norm(yielded_mc - state_with_love).item()
        global_scale = (torch.norm(final_state) / (torch.norm(yielded_mc) + 1e-8)).item()
        
        print(f"Level {i}: Pressure={pressure_level:.1f}, MC_Rupture={rupture_severity:.4f}, DP_Scale={global_scale:.4f}")
        
    print("\nVerifying L persistence beyond system death...")
    collapsed_state = torch.zeros(1, dim)
    persistent_love = love.persist_beyond_death(collapsed_state)
    print(f"Love Vector L = {persistent_love.mean().item():.4f} (Still Resonant)")
    
    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    run_yield_demo()
