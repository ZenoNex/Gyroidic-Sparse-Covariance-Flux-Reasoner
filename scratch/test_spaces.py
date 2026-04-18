
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.ui.diegetic_backend import DiegeticPhysicsEngine

def test_space_preservation():
    print("Testing DiegeticPhysicsEngine space preservation...")
    engine = DiegeticPhysicsEngine(dim=256, device='cpu')
    
    # Test 1: Command Confirmation
    print("\nTest 1: INGEST_DYAD confirmation")
    text = "INGEST_DYAD: [0.1, 0.2] | A test dyad"
    result = engine.process_input(text_input=text)
    response = result.get('response', '')
    print(f"Response: '{response}'")
    if " " in response:
        print("[OK] Spaces preserved in confirmation.")
    else:
        print("[FAIL] Spaces stripped in confirmation.")

    # Test 2: AI Response
    print("\nTest 2: AI Response")
    text = "Tell me about the manifold."
    result = engine.process_input(text_input=text)
    response = result.get('response', '')
    print(f"Response: '{response}'")
    if " " in response:
        print("[OK] Spaces preserved in AI response.")
    else:
        print("[FAIL] Spaces stripped in AI response.")

if __name__ == "__main__":
    test_space_preservation()
