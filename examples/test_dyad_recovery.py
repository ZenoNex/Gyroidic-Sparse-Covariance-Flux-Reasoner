
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.diegetic_backend import DiegeticPhysicsEngine

def test_knowledge_dyad_recovery():
    print("Testing Knowledge Dyad Recovery & Speculative Memory...")
    
    dim = 64
    engine = DiegeticPhysicsEngine(dim=dim)
    
    # 1. Ingest a specific Dyad (Simulated via text for now)
    # Goal: Verify that the system "remembers" the structural impact of an ingestion
    label = "chiral_spiral_topology"
    print(f"Ingesting Dyad: {label}")
    
    # Process initial ingestion
    res_1 = engine.process_input(f"INGEST_DYAD: {label}")
    initial_chiral = res_1['chiral_score']
    initial_yield = res_1['yield_pressure']
    
    print(f"Initial Metrics - Chiral: {initial_chiral:.4f}, Yield: {initial_yield:.4f}")
    
    # 2. Simulate Trajectory Collapse
    # We force the engine to process "noise" that should trigger CALM abort and SCCCG recovery
    noise_input = "noise " * 20
    print("\nProcessing Noise to trigger Speculative Recovery...")
    
    res_2 = engine.process_input(noise_input)
    
    print(f"Recovery Metrics - Attempted: {res_2['recovery_count']}, Generative: {res_2['is_generative']}")
    print(f"Recovery Metrics - Chiral: {res_2['chiral_score']:.4f}, Yield: {res_2['yield_pressure']:.4f}")
    
    # 3. Check for "Knowledge Leakage"
    # Does the recovered state contain traces of the ingested dyad?
    # We compare the recovered meta_state to a projection of the label
    label_tensor = engine._text_to_tensor(label)
    recovered_state = engine.meta_state
    
    similarity = torch.cosine_similarity(label_tensor, recovered_state).item()
    print(f"\nSimilarity between Label and Recovered State: {similarity:.4f}")
    
    # 4. Check Persistent Encodings
    encoding_files = os.listdir("data/encodings")
    latest_encoding = sorted(encoding_files)[-1]
    print(f"\nLatest Persistent Encoding: {latest_encoding}")
    
    # Load and verify content
    data = torch.load(os.path.join("data/encodings", latest_encoding))
    print(f"Saved Input: {data['text_input']}")
    
    print("\nKnowledge Dyad Test COMPLETE.")

if __name__ == "__main__":
    if not os.path.exists("data/encodings"):
        os.makedirs("data/encodings")
    test_knowledge_dyad_recovery()
