
import json
import os
import sys

file_path = r"D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\data\raw\quixiAI.dolfin\flan1m-sharegpt-deduped.json"

print(f"Testing file: {file_path}")

if not os.path.exists(file_path):
    print("File not found!")
    sys.exit(1)

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        print("Attempting json.load...")
        data = json.load(f)
        print(f"Success! Loaded {type(data)} with len {len(data)}")
except Exception as e:
    print(f"json.load failed: {e}")
    
    # Try reading as text to see start/end
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            start = f.read(100)
            f.seek(0, 2)
            end = f.read(100) # This might fail if file is small, but likely huge
            print(f"First 100 chars: {start}")
            print(f"Last 100 chars: {end}")
    except Exception as e2:
        print(f"Reading text failed: {e2}")

