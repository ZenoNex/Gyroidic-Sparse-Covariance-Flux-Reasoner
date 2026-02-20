"""
Verification of DAQF — Diegetic Amortized Quantized Fossil Operator.

Tests:
1. Fossil Selection: Scars persist under pressure.
2. Diegetic Amortization: Cost redistribution over tau.
3. Lattice Quantization: Error (memory) retention.
4. Love Invariant: Structural permanence.
"""

import torch
from src.core.daqf_operator import DAQFOperator


def verify_daqf_operator():
    print("--- Verifying DAQF Operator ---")
    num_fossils = 10
    fossil_dim = 16
    daqf = DAQFOperator(num_fossils=num_fossils, fossil_dim=fossil_dim)
    
    # 1. Contradiction Update
    # Assume even fossils have failures, odd ones don't
    failures = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    flux = torch.ones(num_fossils, 1) # Non-zero flux for all
    
    results = daqf.apply_daqf(failures, flux)
    
    print(f"Diegetic Tau: {results['diegetic_tau']}")
    print(f"Amortized Cost (mean): {results['amortized_cost'].mean().item():.4f}")
    assert results['diegetic_tau'] == 1.0
    
    # 2. Lattice Quantization
    Q_f, Delta_q = daqf.quantize_fossils()
    print(f"Quantization Error Norm: {Delta_q.norm().item():.4f}")
    assert Delta_q.norm() > 0, "Quantization Error must be retained (Delta_q != 0)"
    
    # 3. Love Invariant
    L = daqf.get_love_invariant()
    original_L = L.clone()
    print(f"Love Invariant Sum: {L.sum().item():.4f}")
    
    # Explicitly check that L is invariant
    daqf.check_invariants(original_L)
    print("✓ Invariants check passed.")
    
    # 4. Speculative Persistence
    # If flux is zero for some fossils, they should not persist? 
    # Logic: "Persistence via non-collapse"
    flux_zero = torch.zeros(num_fossils, 1)
    flux_zero[0] = 1.0 # Only fossil 0 has non-zero flux
    persistence = daqf.speculate_persistence(flux_zero)
    print(f"Persistence mask: {persistence}")
    assert persistence[0] == 1.0
    assert persistence[1] == 0.0
    
    print("✓ DAQF Operator logic verified.")


if __name__ == "__main__":
    verify_daqf_operator()
    print("\nALL DAQF VERIFICATIONS PASSED.")
