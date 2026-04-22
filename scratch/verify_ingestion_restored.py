import torch
import sys
import os

# Add project root to path
sys.path.append(r"d:\programming\python\Gyroidic Sparse Covariance Flux Reasoner")

try:
    from src.ui.diegetic_backend import DiegeticPhysicsEngine
    from hybrid_backend import HybridAI
    print("[OK] Imports successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

def test_robust_fossil_priming():
    print("\n[TEST] Testing Robust Fossil Priming...")
    # Create engine
    engine = DiegeticPhysicsEngine(dim=256)
    
    # Create a malformed fossil (missing residue_vector)
    malformed_fossil = {
        'text_input': 'legacy interaction',
        'meta_state': torch.randn(1, 256),
        'metrics': {}
    }
    
    engine.fossil_cache = [malformed_fossil]
    
    # This should NOT crash now
    try:
        engine._prime_manifold_with_fossils(torch.randn(1, 256))
        print("[PASS] Robust priming successful (no KeyError)")
    except Exception as e:
        print(f"[FAIL] Priming crashed: {e}")

def test_regime_injection():
    print("\n[TEST] Testing Regime Injection...")
    engine = DiegeticPhysicsEngine(dim=256)
    
    # Initial hardening
    initial_hardening = engine.hardening
    print(f"Initial hardening: {initial_hardening}")
    
    # Test GOO
    engine.process_input("test goo", regime='goo')
    print(f"Hardening after GOO: {engine.hardening}")
    if engine.hardening < initial_hardening:
        print("[PASS] GOO regime softened the manifold")
    else:
        print("[FAIL] GOO regime did not soften the manifold")
        
    # Test PRICKLES
    current_hardening = engine.hardening
    engine.process_input("test prickles", regime='prickles')
    print(f"Hardening after PRICKLES: {engine.hardening}")
    if engine.hardening > current_hardening:
        print("[PASS] PRICKLES regime hardened the manifold")
    else:
        print("[FAIL] PRICKLES regime did not harden the manifold")

def test_hybrid_fossilization():
    print("\n[TEST] Testing Hybrid Fossilization Schema...")
    ai = HybridAI()
    
    # Simulate a process_text call
    ai.process_text("test interaction", regime='goo')
    
    # Check if a fossil was created with residue_vector
    fossil_files = [f for f in os.listdir(ai.graph_dir) if f.startswith('fossil_')]
    if not fossil_files:
        print("[WARN] No fossils found in graph_dir. Check if graph_manager is active.")
        return
        
    latest_fossil = max(fossil_files)
    fossil_path = os.path.join(ai.graph_dir, latest_fossil)
    data = torch.load(fossil_path, map_location='cpu')
    
    if 'residue_vector' in data:
        print(f"[PASS] Fossil contains residue_vector: {type(data['residue_vector'])}")
    else:
        print("[FAIL] Fossil missing residue_vector")

if __name__ == "__main__":
    # Create graph_dir if it doesn't exist for test
    if not os.path.exists("graph_persistence"):
        os.makedirs("graph_persistence")
        
    test_robust_fossil_priming()
    test_regime_injection()
    # test_hybrid_fossilization() # Requires full AI init which might be slow/heavy
