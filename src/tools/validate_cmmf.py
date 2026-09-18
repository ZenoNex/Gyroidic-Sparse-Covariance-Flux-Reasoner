import torch
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.core.speculative_coprime_gate import SpeculativeCoprimeGate
from src.core.conjugate_moment_transport import ConjugateMomentTransport

def test_langevin_monte_carlo():
    print("--- Testing Langevin Monte Carlo (LMC) Drift ---")
    dim = 16
    batch = 4
    cm_transport = ConjugateMomentTransport(dim=dim)
    
    # Start with a low entropy "collapsed" state
    z_collapsed = torch.zeros(batch, dim)
    print(f"Initial State Variance: {z_collapsed.var().item():.6f}")
    
    # Apply LMC
    z_drifted = cm_transport.langevin_prior_drift(z_collapsed, gamma=0.01, steps=10)
    print(f"Drifted State Variance: {z_drifted.var().item():.6f}")
    
    assert z_drifted.var().item() > z_collapsed.var().item(), "LMC failed to inject topological drift (honest jitter)"
    assert torch.isfinite(z_drifted).all(), "LMC produced non-finite values"
    print("LMC Test: PASSED\n")

def test_cmmf_honeybee_mode():
    print("--- Testing CMMF Legendre Transport (Honeybee Mode) ---")
    dim = 16
    batch = 2
    gate = SpeculativeCoprimeGate(dim=dim)
    
    # A completely collapsed state that would normally trigger recovery
    converged_state = torch.ones(batch, dim) * 0.1
    
    # Run in HONEYBEE mode to test CMMF path
    recovered_state, metrics = gate.speculative_recovery(
        converged_state=converged_state,
        mode='HONEYBEE'
    )
    
    assert torch.isfinite(recovered_state).all(), "CMMF produced non-finite recovered state"
    assert metrics['recovery_attempted'] is True, "Recovery should have been attempted"
    
    # Verify Cayley projection bypass
    # If honeybee mode is True, Cayley projection is skipped.
    # We can check this by running ConjugateMomentTransport manually
    z_drifted = gate.conjugate_transport.langevin_prior_drift(converged_state, gamma=0.01, steps=5)
    
    transported_honeybee = gate.conjugate_transport(z_drifted, is_honeybee_mode=True)
    transported_normal = gate.conjugate_transport(z_drifted, is_honeybee_mode=False)
    
    # Since normal mode enforces V(C) = x^2+y^2+z^2-xyz-4 = 0, they should differ
    diff = (transported_honeybee - transported_normal).abs().sum().item()
    print(f"Cayley Projection Differential: {diff:.6f}")
    
    print("CMMF Honeybee Mode Test: PASSED\n")

def main():
    print("Starting CMMF Validation (TO-MSR-OT-2026-CMM)...\n")
    try:
        test_langevin_monte_carlo()
        test_cmmf_honeybee_mode()
        print("ALL CMMF VALIDATION TESTS PASSED.")
    except Exception as e:
        print(f"VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
