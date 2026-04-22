import torch
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

print("Script started...")

fossil_path = r"D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\data\encodings\fossil_1776815662691.pt"
state_path = r"D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\gyroid_state.pt"

def inspect_pt(path):
    print(f"\n--- Inspecting {os.path.basename(path)} ---")
    try:
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            for k, v in data.items():
                if k == 'metrics' and isinstance(v, dict):
                    print(f"Key: {k}, Type: Dict")
                    for mk, mv in v.items():
                        print(f"  - {mk}: {mv}")
                elif isinstance(v, torch.Tensor):
                    print(f"Key: {k}, Type: Tensor, Shape: {v.shape}")
                elif isinstance(v, dict):
                    print(f"Key: {k}, Type: Dict, Subkeys: {list(v.keys())}")
                else:
                    print(f"Key: {k}, Value Type: {type(v)}, Value: {str(v)[:300]}...")
        else:
            print(f"Data type: {type(data)}")
            print(f"Content: {str(data)[:500]}...")
    except Exception as e:
        print(f"Error loading {path}: {e}")

inspect_pt(fossil_path)
inspect_pt(state_path)
