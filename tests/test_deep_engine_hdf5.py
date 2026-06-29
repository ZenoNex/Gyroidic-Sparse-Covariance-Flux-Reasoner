import os
import sys
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.open_science_ingestor import OpenScienceIngestor
from dataset_ingestion_system import DatasetIngestionSystem

def main():
    hdf5_path = "datasets/H-H1_GWOSC_4KHZ_R1-1242442952-32.hdf5"
    if not os.path.exists(hdf5_path):
        print(f"File not found: {hdf5_path}")
        return

    import json
    from dataset_ingestion_system import DatasetConfig
    
    print("Initializing DatasetIngestionSystem...")
    dis = DatasetIngestionSystem()
    
    query = [{"type": "local_hdf5", "file_path": hdf5_path}]
    config = DatasetConfig(
        name="deep_test_gwosc",
        source_type="open_science",
        source_path=json.dumps(query),
        max_samples=100
    )
    
    print("Adding open science dataset source...")
    success = dis.add_dataset_source(config)
    
    if success:
        print("[SUCCESS] Real HDF5 data successfully passed PAS_LOCK and was ingested!")
    else:
        print("[FAIL] Even real HDF5 data was discarded by PAS_LOCK.")

if __name__ == "__main__":
    main()
