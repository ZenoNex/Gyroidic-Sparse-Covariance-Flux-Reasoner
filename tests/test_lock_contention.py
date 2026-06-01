import sys
import os
import time
import threading
import torch

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

def test_lock_contention():
    print("Testing Lock Contention & Starvation Prevention...")
    
    from src.ui.diegetic_backend import DiegeticPhysicsEngine
    
    # Initialize Engine on CPU
    print("[TEST] Initializing DiegeticPhysicsEngine...")
    engine = DiegeticPhysicsEngine(dim=256, device=torch.device('cpu'))
    
    print("[TEST] Simulating lock acquisition by a main thread (e.g. heavy video dyad ingestion)...")
    # Acquire the processing lock on the main thread
    main_acquired = engine._processing_lock.acquire(blocking=True)
    if not main_acquired:
        print("[FAIL] Main thread could not acquire lock.")
        return False
    print("[TEST] Main thread successfully acquired lock.")
    
    # Try calling _train_mimicry_step in a background thread
    larynx_loss = []
    def background_larynx_step():
        print("[BACKGROUND] Attempting background training step...")
        start_time = time.time()
        # _train_mimicry_step has a timeout of 5s on acquiring self._processing_lock
        loss = engine._train_mimicry_step("test input text")
        duration = time.time() - start_time
        print(f"[BACKGROUND] training step completed in {duration:.2f}s with loss={loss}")
        larynx_loss.append((loss, duration))
        
    t = threading.Thread(target=background_larynx_step)
    t.start()
    
    # Wait for the background thread to finish. Since the timeout is 5 seconds, it should finish in ~5 seconds.
    t.join(timeout=10.0)
    
    if t.is_alive():
        print("[FAIL] Background larynx thread deadlocked and did not terminate within 10 seconds.")
        # Release lock to avoid leaving system in bad state
        engine._processing_lock.release()
        return False
        
    print("[TEST] Background thread finished successfully (no deadlock).")
    
    # Verify that the background thread timed out and returned None
    if len(larynx_loss) != 1:
        print("[FAIL] Background thread did not execute mimicry step.")
        engine._processing_lock.release()
        return False
        
    loss, duration = larynx_loss[0]
    if loss is not None:
        print(f"[FAIL] Background thread returned loss={loss} instead of None while lock was held.")
        engine._processing_lock.release()
        return False
        
    if duration < 4.0 or duration > 7.0:
        print(f"[WARN] Expected timeout duration to be around 5.0s, got {duration:.2f}s.")
        
    print("[TEST] Releasing main processing lock...")
    engine._processing_lock.release()
    
    # Now verify that when the lock is free, the background step succeeds
    print("[TEST] Testing background training step when lock is free...")
    loss = engine._train_mimicry_step("test input text")
    print(f"[TEST] Completed training step when lock is free with loss={loss}")
    if loss is None:
        print("[FAIL] Training step failed when lock was free.")
        return False
        
    print("[PASS] Lock contention and starvation verification test passed!")
    return True

if __name__ == "__main__":
    success = test_lock_contention()
    sys.exit(0 if success else 1)
