import os
import json
import torch
from pathlib import Path
from dataset_ingestion_system import DatasetIngestionSystem, DatasetConfig

def setup_mock_data():
    print("Setting up mock data...")
    data_dir = Path("test_raw_data")
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a small JSON file
    json_data = [{"text": f"json_sample_{i}"} for i in range(50)]
    with open(data_dir / "test1.json", "w") as f:
        json.dump(json_data, f)
        
    # Create a JSONL file
    with open(data_dir / "test2.jsonl", "w") as f:
        for i in range(50):
            f.write(json.dumps({"text": f"jsonl_sample_{i}"}) + "\n")
            
    # Create a CSV file
    with open(data_dir / "test3.csv", "w") as f:
        f.write("text,source\n")
        for i in range(50):
            f.write(f"csv_sample_{i},local\n")
            
    return data_dir

def test_ingestion_limit():
    raw_data_dir = setup_mock_data()
    system = DatasetIngestionSystem(device='cpu')
    
    # Use a small limit of 75 samples total
    config = DatasetConfig(
        name="test_limit_dataset",
        source_type="local",
        source_path=str(raw_data_dir),
        max_samples=75
    )
    
    print("\n--- Starting Ingestion Test ---")
    success = system.add_dataset_source(config)
    print("--- Ingestion Test Finished ---\n")
    
    # Load the processed data to verify count
    processed_path = Path("datasets/test_limit_dataset/processed_data.pt")
    if processed_path.exists():
        samples = torch.load(processed_path)
        print(f"Total samples collected: {len(samples)}")
        assert len(samples) == 75, f"Expected 75 samples, got {len(samples)}"
        print("[OK] SUCCESS: Sample limit was correctly enforced.")
    else:
        print("[FAIL] FAIL: Processed data file not found.")

if __name__ == "__main__":
    try:
        test_ingestion_limit()
    finally:
        # Cleanup
        import shutil
        if Path("test_raw_data").exists():
            shutil.rmtree("test_raw_data")
        if Path("datasets/test_limit_dataset").exists():
            shutil.rmtree("datasets/test_limit_dataset")
