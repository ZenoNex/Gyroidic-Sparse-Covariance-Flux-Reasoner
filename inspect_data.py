
import torch
import os

def inspect_pt(path):
    print(f"--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File does not exist.")
        return
    
    try:
        data = torch.load(path, map_location='cpu')
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: Tensor {v.shape} {v.dtype}")
                elif isinstance(v, dict):
                    print(f"  {k}: Dict with keys {list(v.keys())}")
                else:
                    print(f"  {k}: {type(v)}")
        elif isinstance(data, torch.Tensor):
            print(f"Tensor {data.shape} {data.dtype}")
        else:
            print(f"Data: {data}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Inspect gyroid_state.pt
inspect_pt('d:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/gyroid_state.pt')

# Inspect an encoding file
inspect_pt('d:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/data/encodings/encoding_1_1769786847.pt')
