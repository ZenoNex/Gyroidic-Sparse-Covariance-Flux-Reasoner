
import sys
import os
import torch

# Add src to path
sys.path.append(os.getcwd())

from src.topology.embedding_graph import GyroidicGraphManager

def main():
    print("--- [GYROIDIC MANIFOLD EXPORTER] ---")
    
    # Initialize Manager
    manager = GyroidicGraphManager(data_dir="data/encodings", dim=64)
    
    # Load recent fossils
    print("Scanning data/encodings for resonance fossils...")
    manager.load_fossils(limit=150)
    
    if not manager.nodes:
        print("ERROR: No fossils found in data/encodings. Ingest some dyads first!")
        return
        
    print(f"Loaded {len(manager.nodes)} nodes.")
    
    # Generate Mermaid Text
    print("Computing resonance edges and generating Mermaid structure...")
    mermaid_text = manager.generate_mermaid_text()
    
    # Save to file
    output_path = "docs/manifold_graph.mmd"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mermaid_text)
        
    print(f"SUCCESS: Manifold graph exported to {output_path}")
    print("\n[MERMAID PREVIEW (Top 5 lines)]")
    print("\n".join(mermaid_text.split('\n')[:10]))
    print("...")

if __name__ == "__main__":
    main()
