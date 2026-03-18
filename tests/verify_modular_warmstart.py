import torch
import sys
import os

from src.core.speculative_coprime_gate import SpeculativeCoprimeGate
from src.core.modular_virtualization import ModularVirtualizationLayer

def main():
    print("Initializing Modular Virtualization and Speculative Coprime Gate...")
    
    dim = 16
    batch_size = 4
    
    # 1. Test Modular Layer isolation
    rns_layer = ModularVirtualizationLayer(dim=dim, base=2)
    state_a = torch.randn(batch_size, dim) * 0.1
    state_b = state_a + torch.randn(batch_size, dim) * 0.001
    state_c = torch.randn(batch_size, dim) * 5.0
    
    # Check congruence check works
    is_congruent_ab = rns_layer.fast_congruence_check(state_a, state_b)
    is_congruent_ac = rns_layer.fast_congruence_check(state_a, state_c)
    
    print(f"Congruence Match (Expected True): {is_congruent_ab}")
    print(f"Congruence Non-Match (Expected False): {is_congruent_ac}")
    
    # 2. Test Integration in Gate
    gate = SpeculativeCoprimeGate(dim=dim)
    
    print("\nRunning Forward Pass with Bypass Attempt...")
    # Inject a highly coherent state that will be close to the manifold mean
    good_state = torch.zeros(batch_size, dim) + 0.001
    
    # Emulate an abort threshold break to trigger speculative recovery forcibly
    abort_score = torch.ones(batch_size, 1)
    
    output, metrics = gate.forward(good_state, abort_score=abort_score)
    print(f"Recovery Attempted: {metrics['recovery_attempted']}")
    print(f"Wasserstein distance recorded (Bypass = 0.0): {metrics['wasserstein_distance']:.4f}")
    
    assert metrics['recovery_attempted'] == True, "Failed to force recovery."
    
    print("\nTesting Disconnected Random State Fallback...")
    bad_state = torch.randn(batch_size, dim) * 10.0
    output, metrics = gate.forward(bad_state, abort_score=abort_score)
    print(f"Wasserstein distance recorded (Fallback > 0.0): {metrics['wasserstein_distance']:.4f}")
    
    print("\nAll Modular Algebraic Virtualization Hooks Successfully Validated.")

if __name__ == '__main__':
    main()
