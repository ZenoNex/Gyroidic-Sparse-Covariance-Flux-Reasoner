import os
import sys

# Anti-Stagnation Initialization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad

def test_derivation():
    print("--- Starting Topological Derivation Test ---")
    
    # Initialize Fossilizer
    fossilizer = DyadFossilizer(storage_dir="scratch/test_fossil", feature_dim=256)
    
    # Create a dummy KnowledgeDyad
    dyad = KnowledgeDyad(
        linguistic_description="ASSOCIATE: spectral ghost <-> fractal resonance",
        image_fingerprint=torch.randn(137),
        relevance_score=0.85,
        metadata={"response_text": "Resonance stabilized."}
    )
    
    # Dummy embeddings
    text_embedding = torch.randn(1, 256)
    seed_state = torch.randn(1, 256)
    
    print("Fossilizing with derivation...")
    fossil_path = fossilizer.fossilize(dyad, text_embedding, seed_state=seed_state)
    
    print(f"Fossil saved to: {fossil_path}")
    
    # Load and verify
    data = torch.load(fossil_path)
    
    print("\n--- Fossil Content Verification ---")
    keys_to_check = ['text_input', 'meta_state', 'betti_0', 'betti_1', 'chiral_score', 'spectral_entropy', 'metrics']
    for k in keys_to_check:
        if k in data:
            val = data[k]
            if k == 'metrics':
                print(f"Key: {k}, Soliton Entropy: {val.get('soliton_entropy')}, PAS_h: {val.get('pas_h')}")
            else:
                print(f"Key: {k}, Value: {val}")
        else:
            print(f"MISSING KEY: {k}")

    # Cleanup
    if os.path.exists(fossil_path):
        os.remove(fossil_path)
    if os.path.exists("scratch/test_fossil"):
        import shutil
        shutil.rmtree("scratch/test_fossil")

if __name__ == "__main__":
    test_derivation()
