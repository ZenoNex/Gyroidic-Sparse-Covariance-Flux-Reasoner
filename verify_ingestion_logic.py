
import sys
import unittest
from pathlib import Path
import json
import os
import shutil

# Mock the dataset system
sys.path.append(os.getcwd())

# Ensure we can import the module
try:
    from dataset_ingestion_system import DatasetIngestionSystem, DatasetConfig
except ImportError:
    print("Could not import dataset_ingestion_system")
    sys.exit(1)

# Helper config class if not imported
if 'DatasetConfig' not in globals():
    @dataclass
    class DatasetConfig:
        name: str
        source_type: str
        source_path: str
        preprocessing: str = 'text'
        augmentation: bool = True
        mandelbulb_augmentation: bool = False
        temporal_associations: bool = True
        max_samples: int = None
        validation_split: float = 0.2

class TestLargeFileHandling(unittest.TestCase):
    def setUp(self):
        # Create temp dir
        self.test_dir = Path("tests_temp_ingestion")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        self.system = DatasetIngestionSystem()

    def tearDown(self):
        # clean up
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_normal_json_load(self):
        """Test that normal small JSON files still load."""
        f_path = self.test_dir / "test_normal.json"
        data = [{"id": i, "text": f"sample {i}"} for i in range(50)]
        with open(f_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
        config = DatasetConfig(
            name="test_normal",
            source_type="local",
            source_path=str(f_path),
            preprocessing="text"
        )
        
        # Test normal load
        samples = self.system._process_local_file(f_path, config)
        print(f"Loaded {len(samples)} samples from normal file.")
        self.assertEqual(len(samples), 50)
        self.assertEqual(samples[0]['text'], "sample 0")

    def test_jsonl_load(self):
        """Test JSONL loading."""
        f_path = self.test_dir / "test.jsonl"
        with open(f_path, 'w', encoding='utf-8') as f:
            for i in range(10):
                f.write(json.dumps({"id": i, "text": f"line {i}"}) + "\n")
                
        config = DatasetConfig(
            name="test_jsonl",
            source_type="local",
            source_path=str(f_path),
            preprocessing="text"
        )
        samples = self.system._process_local_file(f_path, config)
        print(f"Loaded {len(samples)} samples from JSONL.")
        self.assertEqual(len(samples), 10), "Should generate 10 samples"


if __name__ == '__main__':
    unittest.main()
