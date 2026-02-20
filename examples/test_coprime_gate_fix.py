import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.core.speculative_coprime_gate import SpeculativeCoprimeGate

def test_gate():
    dim = 64
    gate = SpeculativeCoprimeGate(dim=dim, num_heads=8)
    
    # 1. Test normal pass through
    state = torch.randn(1, dim)
    out, metrics = gate(state)
    print("Normal Pass Metrics:", metrics)
    assert out.shape == state.shape
    
    # 2. Test recovery trigger (high abort score)
    abort_score = torch.tensor([[0.8]])
    out_rec, metrics_rec = gate(state, abort_score=abort_score)
    print("\nRecovery Pass Metrics:", metrics_rec)
    assert out_rec.shape == state.shape
    assert metrics_rec['recovery_attempted'] == True
    assert 'yield_pressure' in metrics_rec
    
    # 3. Test winding consistency
    winding = metrics_rec['winding_numbers']
    print("\nWinding Numbers:", winding)
    assert winding.shape == (8,) # This was (4,) before the fix!
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_gate()
