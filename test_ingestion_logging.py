
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import io

# Add root to path
sys.path.append(os.getcwd())

class TestIngestionLogging(unittest.TestCase):
    
    @patch('dataset_ingestion_system.ijson')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='[{"id": 1}]')
    @patch('pathlib.Path.stat')
    def test_progress_logging(self, mock_stat, mock_file, mock_ijson):
        # Mock file size > 100MB
        mock_stat.return_value.st_size = 200 * 1024 * 1024
        
        # Mock ijson items to yield 2500 items
        mock_ijson.items.return_value = [{'id': i, 'text': f'sample {i}'} for i in range(2500)]
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            from dataset_ingestion_system import DatasetIngestionSystem, DatasetConfig
            
            system = DatasetIngestionSystem()
            config = DatasetConfig(
                name="test_dataset",
                source_type="local",
                source_path="test_large.json",
                preprocessing="text",
                max_samples=10000
            )
            
            # Run ingestion logic directly or via _process_local_file
            # We need to mock _preprocess_sample to return something
            with patch.object(system, '_preprocess_sample', side_effect=lambda x, y: x):
                 system._process_local_file(os.path.abspath("test_large.json"), config)
                 
            # Check output
            output = captured_output.getvalue()
            
            # Should see progress for 1000 and 2000
            self.assertIn("Streamed 1000 items", output)
            self.assertIn("Streamed 2000 items", output)
            self.assertIn("Streaming complete", output)
            
        finally:
            sys.stdout = sys.__stdout__

if __name__ == "__main__":
    unittest.main()
