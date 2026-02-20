"""
Universal Orchestration Demo: Phase Transitions in the Equation-Object.

Demonstrates:
1. Play (Goo) -> Seriousness (Prickles) asymptotic transition.
2. Bimodal Routing: Soft (Sinkhorn) vs Hard (Discrete).
3. Logical Primitives: Love (phi), Rupture (bot), Gluing (Psi).
4. Asymptotic Hardening impact on soliton structures.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.orchestrator import UniversalOrchestrator
from src.core.failure_token import FailureToken

def run_universal_demo():
    print("--- Starting Universal Orchestration Demo ---")
    
    # 1. Initialize Orchestrator
    dim = 64
    orchestrator = UniversalOrchestrator(dim=dim)
    
    # 2. Simulate the 'Asymptotic Life' of the Reasoner
    print("\nSimulating state evolution through phase transitions...")
    
    # Initial 'Playful' state
    state = torch.randn(1, dim)
    
    # We simulate 20 steps of evolution
    for i in range(20):
        # Calculate proxies for the Equation-Object variables
        # Gradually increase PAS_h (Resonance)
        pas_h = 0.3 + (i * 0.03) 
        # Gradually decrease Mischief ( Disorder)
        pressure_grad = torch.randn_like(state) * (1.0 - i * 0.04)
        is_good_bug = (i % 5 == 0) # Occasional entropic spikes
        
        # --- The Law of the Equation-Object ---
        # state_next = Pi_DP( ADMM( CRT( Pi_MC( grad f + L ) mod m ) ) )
        # (Encapsulated in orchestrator.forward)
        
        state, regime, routing = orchestrator(
            state=state,
            pressure_grad=pressure_grad,
            pas_h=pas_h,
            is_good_bug=is_good_bug
        )
        
        hardening = orchestrator.get_hardening_factor()
        
        print(f"Step {i:02d}: Regime={regime:11s} | Routing={routing:4s} | PAS_h={pas_h:.2f} | Hardening={hardening:.2f}")
        
        # 3. Rupture Check (bot)
        # Simulate a high-pressure conflict at step 15
        if i == 15:
            print("[CAUTION] Injecting high-pressure conflict...")
            conflict_losses = {0: torch.tensor(2e6)} # Exceeds rupture threshold
            token = orchestrator.check_rupture(state, conflict_losses)
            if token is not None:
                print(f"[RUPTURE] Failure Token detected: {token.type.value}")
                # In a real training loop, this would trigger an evolutionary restart
        
    print("\nVerifying Invariant Persistence (Love L)...")
    # Even after 20 steps of yield and hardening, the Love Vector is still co-present
    l_sum = orchestrator.love.L.sum().item()
    print(f"Love Vector L Ambient Resonance: {l_sum:.4f}")
    
    print("\n--- Universal Orchestration Complete: System Stabilized ---")

if __name__ == "__main__":
    run_universal_demo()
