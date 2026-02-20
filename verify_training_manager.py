
import sys
import os
import time

# Add project root to path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Add examples to path (as handled in TrainingManager, but good to ensure)
examples_dir = os.path.join(root_dir, 'examples')
if examples_dir not in sys.path:
    sys.path.append(examples_dir)

# Mock AI System
class MockAISystem:
    def __init__(self):
        self.temporal_model = None
        self.device = 'cpu'

try:
    from src.training.training_manager import TrainingManager
    print("[OK] TrainingManager imported")
except ImportError as e:
    print(f"[FAIL] Could not import TrainingManager: {e}")
    sys.exit(1)

def verify_training():
    ai = MockAISystem()
    manager = TrainingManager(ai)
    
    print("[TEST] Starting training (2 epochs)...")
    success, msg = manager.start_training(epochs=2)
    print(f"Start result: {success} - {msg}")
    
    if not success:
        return
        
    # Poll status
    for i in range(60):
        status = manager.get_status()
        print(f"Status check {i}: Active={status['active']}, Progress={status['progress']}%")
        if status['log']:
            print(f"Latest log: {status['log'][-1]}")
            # Print full new logs if needed
        
        if not status['active'] and status['results']:
            print(f"[DONE] Training finished. Results: {status['results']}")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    verify_training()
