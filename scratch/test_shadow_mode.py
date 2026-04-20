import torch
import sys
import os

sys.path.insert(0, os.getcwd())

from src.core.orchestrator import UniversalOrchestrator
from src.core.speculative_coprime_gate import SpeculativeCoprimeGate
from src.core.false_negative_subsystem import VoynichExemptionToken

def run_shadow_test():
    dim = 16
    print("Initializing Orchestrator and Gate...")
    orchestrator = UniversalOrchestrator(dim=dim)
    gate = SpeculativeCoprimeGate(dim=dim, num_heads=4)
    
    state = torch.randn(1, dim)
    pressure_grad = torch.randn(1, dim) * 0.1
    # Low coherence to simulate a glitch/entropy
    pas_h = 0.5 
    
    # 1. Test Orchestrator Shadow Logging
    print("\n--- Testing Orchestrator Shadow Logging ---")
    print("We expect a shadow log since we force is_good_bug=True but PAS_h is low.")
    # We call forward with is_good_bug=True, but pass all required args.
    # We need a coherence tensor for the probe.
    coherence = torch.tensor([0.5])
    
    out, regime, routing = orchestrator(state, pressure_grad, pas_h, coherence, is_good_bug=True)
    print("Orchestrator finished.")
    
    # 2. Test Gate Shadow Logging
    print("\n--- Testing SpeculativeCoprimeGate Shadow Logging ---")
    print("We expect shadow logs since we pass a valid VoynichExemptionToken but have parity violations.")
    
    token = VoynichExemptionToken() # Defaults to shadow_mode=True
    # The gate requires a dictionary for 'losses'
    losses = {
        'geometric_penalty': torch.tensor(1.0),
        'coprime_lock_failure': True
    }
    abort_score = torch.tensor([1.0]) # High abort score to mock parity issue
    
    refined_state = gate(
        state, 
        losses, 
        abort_score=abort_score, 
        chiral_score=0.1, 
        exemption_token=token
    )
    print("Gate evaluation finished.")
    
    print("\n--- SHADOW MODE TEST COMPLETE ---")

if __name__ == "__main__":
    run_shadow_test()
