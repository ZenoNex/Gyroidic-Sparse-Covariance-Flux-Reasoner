
import os
import sys

# ANTI-STAGNATION INITIALIZATION
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_JIT'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import time

def inspect_pt(path):
    print(f"\n--- Inspecting {path} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    try:
        # Check first few bytes
        with open(path, 'rb') as f:
            header = f.read(16)
            print(f"Header (hex): {header.hex()}")
            if header.startswith(b'PK\x03\x04'):
                print("Format: ZIP (Modern PyTorch)")
            elif header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'\x80\x04'):
                print("Format: Pickle (Legacy PyTorch)")
            else:
                print("Format: Unknown/Custom")
        
        # Try to load
        start_time = time.time()
        print("Attempting torch.load...")
        data = torch.load(path, map_location='cpu')
        end_time = time.time()
        print(f"Load successful in {end_time - start_time:.4f}s")
        
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: Tensor shape={v.shape}, dtype={v.dtype}, device={v.device}")
                elif isinstance(v, (list, tuple)):
                    print(f"  {k}: {type(v)} len={len(v)}")
                else:
                    item_str = str(v)
                    if len(item_str) > 100:
                        item_str = item_str[:100] + "..."
                    print(f"  {k}: {type(v)} = {item_str}")
        elif isinstance(data, torch.Tensor):
            print(f"Tensor shape={data.shape}, dtype={data.dtype}")
        else:
            print(f"Data summary: {str(data)[:200]}")
            
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")

if __name__ == "__main__":
    base_dir = r"d:\programming\python\Gyroidic Sparse Covariance Flux Reasoner"
    
    # Files to check
    files = [
        os.path.join(base_dir, "gyroid_state.pt"),
        os.path.join(base_dir, "data", "encodings", "encoding_18_1769790746.pt"),
        os.path.join(base_dir, "data", "encodings", "encoding_1_1769879422.pt")
    ]
    
    for f in files:
        inspect_pt(f)
