
import torch
import sys
import os
import time

# Adjust path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topology.speculative_homology import SpeculativeHomologyEngine

def verify_efficiency():
    print("Verifying Speculative Betti Decoding Efficiency...")
    
    # Initialize Engine
    # D=64 features
    engine = SpeculativeHomologyEngine(feature_dim=64, zeta=0.1)
    
    # Simulator
    batch_size = 1
    features = 64
    
    # Initial State (Stable)
    x = torch.randn(batch_size, features)
    prev_pas = torch.tensor(0.5) # Initial PAS
    
    start_time = time.time()
    
    print("\n--- Phase 1: Stable Regime (Draft Acceptance) ---")
    for i in range(10):
        # Small perturbations (Stable)
        x = x + torch.randn_like(x) * 0.01 
        betti, pas, used_draft = engine(x, prev_pas)
        print(f"Step {i}: Used Draft? {used_draft} | PAS: {pas.item():.4f}")
        prev_pas = pas
        
    print("\n--- Phase 2: Rupture Regime (Rejection) ---")
    # Large pertubation (Rupture)
    x = x + torch.randn_like(x) * 5.0 
    betti, pas, used_draft = engine(x, prev_pas)
    print(f"RUPTURE Step: Used Draft? {used_draft} | PAS: {pas.item():.4f}")
    prev_pas = pas
    
    print("\n--- Stats ---")
    stats = engine.get_stats()
    print(f"Accept Rate: {stats['accept_rate']:.2%}")
    print(f"Theoretical Speedup: {stats['speedup_proxy']:.2f}x")
    
    if stats['accept_rate'] > 0.8:
        print("PASS: High acceptance in stable regime verified.")
    else:
        print("FAIL: Acceptance rate too low.")

if __name__ == "__main__":
    verify_efficiency()
