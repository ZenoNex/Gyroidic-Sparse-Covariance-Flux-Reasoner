import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.diegetic_backend import DiegeticPhysicsEngine

def main():
    print("Initializing DiegeticPhysicsEngine...")
    engine = DiegeticPhysicsEngine(dim=128)
    
    print("\n[Test 1] Injecting High-Entropy Signal (Should trigger full SCCCG recovery if collapse imminent)")
    # Generate random noise (high entropy)
    high_entropy_meta = torch.randn(1, 128)
    engine.meta_state = high_entropy_meta
    
    # Force CALM to predict an abort (abort_score > 0.5)
    # The actual CALM assessment is internal to process_input.
    # We can just run process_input and see what happens.
    # To reliably trigger CALM abort, we might need a very destabilizing input.
    destabilizing_input = "💥" * 50
    try:
        metrics1 = engine.process_input(destabilizing_input)
        print(f"Metrics 1 Status: {metrics1.get('calm_diagnostics', {}).get('trajectory_status', 'UNKNOWN')}")
    except Exception as e:
        print(f"Error in Test 1: {e}")
        
    print("\n[Test 2] Injecting Low-Entropy (Highly Structured) Signal (Should trigger Spectral Early Exit)")
    # Generate a pure sine wave (low entropy)
    x = torch.linspace(0, 4*3.14159, 128)
    low_entropy_meta = torch.sin(x).unsqueeze(0)
    engine.meta_state = low_entropy_meta
    
    # Run process_input
    try:
        metrics2 = engine.process_input("What is the meaning of life? Explain calmly.")
        print(f"Metrics 2 Status: {metrics2.get('calm_diagnostics', {}).get('trajectory_status', 'UNKNOWN')}")
    except Exception as e:
        print(f"Error in Test 2: {e}")
        
    import time
    time.sleep(2)
        
if __name__ == "__main__":
    main()
