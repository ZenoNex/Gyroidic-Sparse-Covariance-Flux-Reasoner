
import os
import sys
import shutil
import json
import torch

# Add root to path
sys.path.append(os.getcwd())

from hybrid_backend import HybridAI

def verify_graph():
    print("🧪 Verifying Graph Integration...")
    
    # Clean up old test fossils if any
    test_dir = os.path.join(os.getcwd(), 'data', 'encodings')
    
    # Initialize System
    print("⚙️ Initializing HybridAI...")
    ai = HybridAI(use_spectral_correction=False)
    
    if not ai.graph_manager:
        print("❌ Graph Manager failed to initialize!")
        sys.exit(1)
        
    initial_count = len(ai.graph_manager.nodes)
    print(f"   Initial node count: {initial_count}")
    
    # Generate a fossil
    print("📝 Processing text to generate fossil...")
    response = ai.process_text("This is a test fossil for the graph topology.")
    
    # Check if node added
    new_count = len(ai.graph_manager.nodes)
    print(f"   New node count: {new_count}")
    
    if new_count <= initial_count:
        print("❌ Fossil not added to live graph!")
        sys.exit(1)
        
    # Check if file saved
    files = os.listdir(test_dir)
    if not any(f.startswith("fossil_") for f in files):
        print("❌ No fossil file found in data/encodings!")
        sys.exit(1)
        
    # Check JSON export
    print("📊 Exporting Graph JSON...")
    json_str = ai.graph_manager.export_graph_json()
    try:
        data = json.loads(json_str)
        nodes = len(data['nodes'])
        links = len(data['links'])
        print(f"   Export success: {nodes} nodes, {links} links")
    except Exception as e:
        print(f"❌ JSON Export failed: {e}")
        sys.exit(1)
        
    print("✅ Graph Verification Passed!")

if __name__ == "__main__":
    verify_graph()
