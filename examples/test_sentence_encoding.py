
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.diegetic_backend import DiegeticPhysicsEngine

def main():
    print("--- [VERIFYING SEQUENCE-AWARE ENCODING] ---")
    
    engine = DiegeticPhysicsEngine(dim=64)
    
    s1 = "The cat is on the mat"
    s2 = "The mat is on the cat"
    s3 = "cat The is on the mat"
    
    t1 = engine._text_to_tensor(s1)
    t2 = engine._text_to_tensor(s2)
    t3 = engine._text_to_tensor(s3)
    
    sim12 = torch.cosine_similarity(t1, t2).item()
    sim13 = torch.cosine_similarity(t1, t3).item()
    
    print(f"Sentence 1: '{s1}'")
    print(f"Sentence 2: '{s2}'")
    print(f"Sentence 3: '{s3}'")
    
    print(f"\nSimilarity (1 vs 2): {sim12:.4f}")
    print(f"Similarity (1 vs 3): {sim13:.4f}")
    
    # In a bag-of-characters system, these would be ~1.0
    if sim12 < 0.95 and sim13 < 0.95:
        print("\nSUCCESS: Sequence-Aware encoding confirmed. Different word orders produce distinct embeddings.")
    else:
        print("\nWARNING: Encodings are too similar. Resolution might still be a bottleneck.")

if __name__ == "__main__":
    main()
