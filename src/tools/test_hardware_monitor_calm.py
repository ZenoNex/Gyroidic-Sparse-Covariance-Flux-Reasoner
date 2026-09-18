import psutil
from src.core.hardware_monitor import HardwareMonitor

def test_hardware_monitor_calm():
    print("Testing CALM + HardwareMonitor Integration...")
    
    hm = HardwareMonitor()
    
    # 1. Test Stable condition
    hm.update_topology(abort_score=0.1, gauge=0.2)
    stable_headroom = hm.has_headroom()
    print(f"[STABLE] abort_score=0.1 -> has_headroom: {stable_headroom} (Expected: False)")
    
    # 2. Test Panic Override (Collapse Imminent)
    hm.update_topology(abort_score=0.9, gauge=0.9)
    # Mocking psutil to simulate moderate load (should allow bypass to save manifold)
    original_cpu_percent = psutil.cpu_percent
    original_virtual_memory = psutil.virtual_memory
    
    try:
        class MockMem:
            available = 4.0 * (1024 ** 3) # 4GB free
            
        psutil.cpu_percent = lambda interval=None: 85.0 # High load, but not >95%
        psutil.virtual_memory = lambda: MockMem()
        
        panic_headroom = hm.has_headroom()
        print(f"[PANIC] abort_score=0.9, CPU=85.0% -> has_headroom: {panic_headroom} (Expected: True)")
        
        # 3. Test Critical Exhaustion (even Panic can't bypass)
        psutil.cpu_percent = lambda interval=None: 98.0 # Extreme load
        exhausted_headroom = hm.has_headroom()
        print(f"[EXHAUSTED] abort_score=0.9, CPU=98.0% -> has_headroom: {exhausted_headroom} (Expected: False)")
        
    finally:
        psutil.cpu_percent = original_cpu_percent
        psutil.virtual_memory = original_virtual_memory

if __name__ == "__main__":
    test_hardware_monitor_calm()
