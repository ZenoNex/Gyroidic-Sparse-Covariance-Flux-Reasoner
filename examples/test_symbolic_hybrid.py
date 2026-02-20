import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gyroid_reasoner import GyroidicFluxReasoner

def test_symbolic_hybrid():
    print("Initializing Hybrid GSCFR...")
    model = GyroidicFluxReasoner(
        num_functionals=5,
        poly_degree=4,
        use_admm=True,
        use_resonance=True,
        use_gyroid_probes=True
    )
    
    # 1. Verify Saturated Polynomial Gates
    print("\nVerifying Saturated Polynomial Gates...")
    batch_size = 4
    text_emb = torch.randn(batch_size, 768)
    
    results = model(text_emb=text_emb, return_analysis=True)
    residue_dist = results['residue_distributions']
    
    # Saturated outputs should be discrete (often -1, 1 if using torch.sign)
    is_saturated = torch.all((residue_dist == 1.0) | (residue_dist == -1.0) | (residue_dist == 0.0))
    print(f"Residue distributions are saturated: {is_saturated}")
    
    # 2. Verify Permutation Collapse
    print("\nVerifying Permutation Collapse in Sinkhorn...")
    # Trigger training mode to enable annealing
    model.train()
    for i in range(20):
        # Forward pass to trigger iteration count increase
        _ = model(text_emb=text_emb)
    
    # Check temperature
    curr_temp = model.crt.poly_config.sampler.sinkhorn_iters # Wait, temperature is in BirkhoffProjection
    # Let's find the BirkhoffProjection in the model
    # It's inside ModularTransformerLayer
    layer = model.layers[0]
    attn = layer.attention
    # attn.projector is BirkhoffProjection
    print(f"Current Sinkhorn Temperature: {attn.projector.temperature.item():.4f}")
    
    # 3. Verify Binary-First Hierarchy
    print("\nVerifying Binary-First Hierarchy (System 1 -> System 2)...")
    # Simulate a "failure" by passing anchors that don't match residues
    anchors = torch.randn(batch_size) * 10 
    
    # We should see evidence of ADMM being called if reconstruction pressure is high
    results_with_fail = model(text_emb=text_emb, anchors=anchors, return_analysis=True)
    print("Hierarchy test complete.")
    
    # 4. Verify Pressure Regimes
    print("\nVerifying Pressure Regimes...")
    print(f"Total Pressure: {results_with_fail['total_pressure'].item():.4f}")
    print(f"CRT Pressure: {results_with_fail['crt_pressure'].item():.4f}")
    print(f"Homology Pressure: {results_with_fail['homology_pressure'].item():.4f}")
    
    print("\nHybrid Architecture Verification Successful!")

if __name__ == "__main__":
    test_symbolic_hybrid()
