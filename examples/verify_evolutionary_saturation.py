import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gyroid_reasoner import GyroidicFluxReasoner
from src.core.polynomial_coprime import PolynomialCoprimeConfig

def test_evolutionary_saturation():
    print("--- Verifying Evolutionary Saturation Framework ---")
    
    # 1. Initialize Model with Saturation
    model = GyroidicFluxReasoner(
        num_functionals=5,
        poly_degree=4,
        use_saturation=True,
        use_admm=True,
        use_resonance=True
    )
    
    # 2. Test Selection Pressure (Hypergraph Orthogonality)
    print("\n[1] Testing Selection Pressure (Hypergraph Orthogonality)...")
    batch_size = 10
    phi = torch.randn(batch_size, 5) # Dummy saturated outcomes
    selection_p = model.selection_pressure_fn(phi)
    print(f"  Selection Pressure (Entropy): {selection_p.item():.4f}")
    assert selection_p <= 0, "Entropy-based pressure should be negative or zero (maximizing entropy)"

    # 3. Test Bimodal Routing
    print("\n[2] Testing Bimodal Routing (Birkhoff Projection)...")
    X = torch.randn(2, 5, 5)
    # Mode 0: Soft
    out_soft = model.layers[0].attention.birkhoff(X, genome=torch.zeros(2))
    # Mode 1: Hard
    out_hard = model.layers[0].attention.birkhoff(X, genome=torch.ones(2))
    
    is_hard = torch.all((out_hard == 0) | (out_hard == 1))
    print(f"  Mode 1 (Hard) is discrete: {is_hard.item()}")
    assert is_hard, "Mode 1 should produce discrete permutation matrices"

    # 4. Test Majority-Symbol CRT
    print("\n[3] Testing Majority-Symbol CRT...")
    # Mock distributions: [batch, K, D]
    dist = torch.zeros(2, 5, 5)
    dist[:, :, 2] = 1.0 # Symbol '2' has majority everywhere
    recon = model.crt(dist, mode='majority')
    print(f"  Reconstruction shape: {recon.shape}")
    # Since all K picked '2', reconstruction should be roughly a basis vector or sum
    # The weight logic will sum them.

    # 5. Test Signal Sovereignty (Fossilization)
    print("\n[4] Testing Functional Fossilization...")
    normalizer = model.crt.normalizer if hasattr(model.crt, 'normalizer') else None
    if normalizer:
        print(f"  Initial Fossilization: {normalizer.is_fossilized}")
        # Mock high stability to trigger fossilization
        normalizer.fossil_threshold = 2 # Lower for test
        values = torch.randn(10, 5) # Constant variance
        for _ in range(3):
            _, _ = normalizer(values, weights=torch.ones(5))
        print(f"  Fossilization after stability: {normalizer.is_fossilized}")
    else:
        print("  Signal Sovereignty: Normalizer not found in CRT (skipped if not using Decoupled CRT)")

    # 6. Test Repair Trace Compression (Zero Gradient Leakage)
    print("\n[5] Testing Repair Trace Compression (Gradient Leakage)...")
    initial_c = torch.randn(1, 25).requires_grad_(True)
    
    # Run ADMM (Note: target/anchor removed, initial_c is the starting point)
    from src.optimization.operational_admm import OperationalAdmmPrimitive
    output = OperationalAdmmPrimitive.apply(
        initial_c, model.kagh_surrogate, 
        2.0, 0.1, 5, 0.05, 4, None, None
    )
    
    # Compute gradient of output w.r.t initial_c
    grad = torch.autograd.grad(output[0].sum(), initial_c, allow_unused=True)[0]
    is_leakage_free = (grad is None or torch.all(grad == 0))
    print(f"  Is Leakage Free: {is_leakage_free}")
    assert is_leakage_free, "System 2 should NOT leak smoothness (gradients) to initial_c"

    # 7. Test Homology Drift
    print("\n[6] Testing Residue Homology Drift...")
    import networkx as nx
    G1 = nx.cycle_graph(5)
    G2 = nx.complete_graph(5)
    
    drift1, trigger1 = model.homology_drift_tracker(G1)
    drift2, trigger2 = model.homology_drift_tracker(G2)
    print(f"  Drift on change G1->G2: {drift2.item():.4f}")
    print(f"  Triggered: {trigger2}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_evolutionary_saturation()
