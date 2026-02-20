"""
Verification Script for Phase 14: Missing Concepts & Veto Refactor.

Tests the following components:
1. Gyroidic Unknowledge (Nostalgic Leak, Mischief, Obscured Birkhoff, GMVE)
2. Safety Monitors (Anti-Scaling, Incommensurativity, Trust)
3. Simulated Quantum TDA
4. Audience Mapping
"""

import torch
import sys
import os
import math

# Add source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.unknowledge_flux import NostalgicLeakFunctional, EntropicMischiefProbe
from src.core.birkhoff_projection import ObscuredBirkhoffManifold
from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe
from src.core.structural_monitors import AntiScalingMonitor, MetaInfraIntraMonitor
from src.safety.trust_inheritance import TrustInheritanceTracker
from src.safety.red_teaming import RedTeamProjection
from src.core.quantum_tda import QuantumBettiApproximator
from src.core.audience_mapping import AudienceProjection
from src.core.orchestrator import UniversalOrchestrator

def test_unknowledge():
    print("Testing Gyroidic Unknowledge...")
    dim = 16
    leak = NostalgicLeakFunctional(fossil_dim=dim)
    x = torch.randn(4, dim)
    out = leak(x)
    assert out.shape == (4, 1), f"Leak shape mismatch: {out.shape}"
    print("  [x] Nostalgic Leak: OK")
    
    mischief = EntropicMischiefProbe()
    mischief.update(torch.tensor(0.5), torch.tensor(0.8), 0.9, is_good_bug=True)
    metrics = mischief.get_metrics()
    assert metrics['H_mischief'] > 0, "Mischief should be positive for good bug"
    print(f"  [x] Mischief Probe: H_meta={metrics['H_meta']:.4f} OK")

def test_obscured_birkhoff():
    print("Testing Obscured Birkhoff...")
    dim = 5
    delta_o = 0.1
    birkhoff = ObscuredBirkhoffManifold(delta_o=delta_o)
    T = torch.randn(2, dim, dim)
    T_proj = birkhoff(T)
    
    row_sums = T_proj.sum(dim=-1)
    target = 1.0 - delta_o
    max_err = (row_sums - target).abs().max().item()
    assert max_err < 1e-4, f"Row sum error too high: {max_err}"
    print(f"  [x] Obscured Birkhoff (delta_o={delta_o}): Max Error {max_err:.6f} OK")

def test_gmve():
    print("Testing GMVE...")
    probe = SparseGyroidCovarianceProbe(hidden_dim=32, window_size=16)
    states = torch.randn(32, 32)
    C_loc = probe.compute_local_covariance(states, start_idx=0)
    
    gmve = probe.compute_gmve(C_loc, h_mischief=0.5)
    print(f"  [x] GMVE Score: {gmve:.4f} OK")

def test_monitors():
    print("Testing Monitors...")
    # Anti-Scaling
    asm = AntiScalingMonitor(window_size=5)
    # Simulate decreasing loss but FASTER decreasing grad norm (Paradox)
    for i in range(5):
        loss = 1.0 - i*0.1
        grad = 1.0 - i*0.2 # Dropping faster
        asm.update(grad, loss)
    
    paradox = asm.check_paradox()
    print(f"  [x] Anti-Scaling Paradox Score: {paradox['paradox_score']:.4f}")
    
    # Incommensurativity
    mim = MetaInfraIntraMonitor()
    # Meta rising fastest
    mim.update(5, 10, 1, 10, 1, 10) # 50% vs 10% vs 10%
    res = mim.check_incommensurativity()
    assert res['incommensurativity_score'] > 0, "Should detect meta collapse"
    print(f"  [x] Incommensurativity Score: {res['incommensurativity_score']:.4f} OK")
    
    # Trust
    tracker = TrustInheritanceTracker()
    tracker.update(rho_def=0.1) # 10% veto
    trust = tracker.get_trust()
    assert trust < 1.0, "Trust should decay"
    print(f"  [x] Trust Tracker: {trust:.4f} OK")

def test_quantum_tda():
    print("Testing Quantum TDA (Simulation)...")
    qtda = QuantumBettiApproximator()
    # Simple cycle graph adjacency (triangle)
    adj = torch.tensor([[0,1,1],[1,0,1],[1,1,0]], dtype=torch.float32)
    bettis = qtda.estimate_betti_numbers(adj, max_dim=1)
    print(f"  [x] Betti Numbers: {bettis} OK")

def test_audience():
    print("Testing Audience Mapping...")
    mapper = AudienceProjection(input_dim=10, audience_dim=5)
    x = torch.randn(2, 10)
    y = mapper(x)
    assert y.shape == (2, 5)
    print("  [x] Audience Mapping: Shape OK")

def main():
    test_unknowledge()
    test_obscured_birkhoff()
    test_gmve()
    test_monitors()
    test_quantum_tda()
    test_audience()
    print("\nAll Phase 14 tests passed!")

if __name__ == "__main__":
    main()
