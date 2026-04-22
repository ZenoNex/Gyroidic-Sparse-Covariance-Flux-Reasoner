import torch
import sys
import os

# Add project root to path
sys.path.append(r"d:\programming\python\Gyroidic Sparse Covariance Flux Reasoner")

from src.ui.diegetic_backend import DiegeticPhysicsEngine

def test_chirality_metrics():
    print("\n[TEST] Testing Chirality Metrics...")
    engine = DiegeticPhysicsEngine(dim=256)
    
    # Process a simple input
    metrics = engine.process_input("test input", regime='goo')
    
    print(f"Metrics keys: {metrics.keys()}")
    
    if 'chiral_score' in metrics:
        print(f"[PASS] chiral_score found: {metrics['chiral_score']}")
    else:
        print("[FAIL] chiral_score missing")
        
    if 'chiral_torsion' in metrics:
        print(f"[PASS] chiral_torsion found: {metrics['chiral_torsion']}")
    else:
        print("[FAIL] chiral_torsion missing")
        
    if 'glyphlock' in metrics:
        print(f"[PASS] glyphlock found: {metrics['glyphlock']}")
    else:
        print("[FAIL] glyphlock missing")

    # Test Agent Smith Export
    print("\n[TEST] Testing Agent Smith Export...")
    from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
    fossilizer = DyadFossilizer(storage_dir="scratch/encodings")
    
    # Create a dummy dyad with the meta_state from the engine
    dyad = KnowledgeDyad(
        linguistic_description="test agent smith",
        meta_state=engine.meta_state.detach().cpu()
    )
    
    # Export
    filepath = fossilizer.export_agent_smith(
        dyad=dyad,
        prime_frequencies=torch.zeros(1, 256),
        betti_numbers={0: 1.0, 1: 0.0},
        filename="test_smith"
    )
    
    import json
    with open(filepath, 'r') as f:
        payload = json.load(f)
    
    fields = ['chiral_shift', 'chiral_torsion', 'glyphlock', 'spectral_entropy']
    for field in fields:
        if field in payload:
            print(f"[PASS] Agent Smith field '{field}' found: {payload[field]}")
        else:
            print(f"[FAIL] Agent Smith field '{field}' missing")

    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)

if __name__ == "__main__":
    test_chirality_metrics()
