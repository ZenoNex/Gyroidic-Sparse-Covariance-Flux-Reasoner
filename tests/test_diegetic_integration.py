
import sys
import os
import torch

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

def test_integration():
    print("Testing HybridAI -> DiegeticPhysicsEngine Integration...")
    
    try:
        from hybrid_backend import HybridAI
    except ImportError:
        # It's possible hybrid_backend is in root, so we need to add root to path which we did
        # But let's check if the file exists
        if os.path.exists(os.path.join(project_root, "hybrid_backend.py")):
             from hybrid_backend import HybridAI
        else:
            print("FAIL: hybrid_backend.py not found.")
            return

    # Initialize HybridAI
    # It attempts to load DiegeticPhysicsEngine
    try:
        ai = HybridAI(use_spectral_correction=False) # Disable spectral to speed up init if possible
    except Exception as e:
        print(f"FAIL: HybridAI init failed: {e}")
        return

    # Check if engine is attached
    if hasattr(ai, 'engine') and ai.engine is not None:
        print("PASS: DiegeticPhysicsEngine attached successfully.")
    else:
        print("FAIL: DiegeticPhysicsEngine NOT attached.")
        return

    # Test processing
    test_msg = "Hello system context."
    try:
        result = ai.process_text(test_msg)
        print("PASS: process_text execution successful.")
    except Exception as e:
        print(f"FAIL: process_text failed: {e}")
        return

    # Verify backend flag
    if result.get('backend') == 'hybrid_diegetic_integrated':
        print("PASS: Backend flag confirms integration.")
    else:
        print(f"FAIL: Backend flag is {result.get('backend')}")
        
    # Verify diagnostics presence (indicating advanced engine)
    diags = result.get('diagnostics', {})
    if 'calm_diagnostics' in diags:
        print("PASS: CALM diagnostics present.")
    else:
        print("FAIL: CALM diagnostics missing.")

    if 'phase4_diagnostics' in diags:
        print("PASS: Phase 4 diagnostics present.")
    else:
        print("FAIL: Phase 4 diagnostics missing.")

    print(f"Response: {result.get('response')[:50]}...")
    print("Integration Test Complete.")

if __name__ == "__main__":
    test_integration()
