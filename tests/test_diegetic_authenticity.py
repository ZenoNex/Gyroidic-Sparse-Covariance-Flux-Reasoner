
import sys
import os
import torch
import torch.nn as nn
import json

# Ensure project root
sys.path.insert(0, os.getcwd())

def test_diegetic_authenticity():
    print("Testing Diegetic Authenticity & Dyadic Transfer...")
    
    from src.ui.diegetic_backend import DiegeticPhysicsEngine
    from src.core.love_vector import LoveVector
    from src.core.knowledge_dyad_fossilizer import DyadFossilizer
    from src.core.dyadic_transfer import DyadicTransferMap
    
    dim = 256
    engine = DiegeticPhysicsEngine(dim=dim, device='cpu')
    
    # 1. Verify LoveVector
    print("\n[1] LoveVector Verification:")
    if isinstance(engine.love_vector, LoveVector):
        print("PASS: LoveVector initialized correctly.")
    else:
        print("FAIL: LoveVector missing.")
    
    ownership = engine.love_vector.ownership_check().item()
    print(f"   Ownership Leak (expected 0.0): {ownership}")
    if ownership == 0.0:
        print("PASS: Ownership invariant maintained.")
    else:
        print("FAIL: Ownership invariant violated.")

    # 2. Verify Dyad Fossilizer
    print("\n[2] Dyad Fossilizer Verification:")
    if isinstance(engine.fossilizer, DyadFossilizer):
        print("PASS: DyadFossilizer initialized correctly.")
    else:
        print("FAIL: DyadFossilizer missing.")
        
    # Test INGEST_DYAD
    print("Ingesting test dyad...")
    ingest_cmd = 'INGEST_DYAD: {"r":[0.1,0.2], "g":[0.3,0.4], "b":[0.5,0.6], "l":[0.7,0.8], "texture":0.9} | Test multi-modal dyad'
    response = engine.process_input(ingest_cmd)
    print(f"   Response: {response['response']}")
    
    if "fossilized" in response['response'].lower():
        print("PASS: Ingestion handler fossilized correctly.")
    else:
        print("FAIL: Ingestion failed.")

    # 3. Verify Dyadic Transfer
    print("\n[3] Dyadic Transfer Verification:")
    if isinstance(engine.transfer_map, DyadicTransferMap):
        print("PASS: DyadicTransferMap initialized correctly.")
    else:
        print("FAIL: DyadicTransferMap missing.")
        
    # Check if transfer matrix is non-zero
    T = engine.transfer_map.get_transfer_coefficients()
    print(f"   Transfer Matrix Mean: {T.mean().item():.4f}")
    if T.mean() > 0:
        print("PASS: Transfer map active.")
    else:
        print("FAIL: Transfer map inactive.")

    # 4. Verify Affordance Trigger
    print("\n[4] Affordance Trigger Verification:")
    # Send high expandability text to trigger agentic ingestion
    # Keywords: generate, create, build, construct, pattern, template
    test_text = "generate create build construct pattern template template pattern build construct" 
    response = engine.process_input(test_text)
    print(f"   Response includes: {response['response'][:60]}...")
    
    if "fossilized" in response['response'].lower():
        print("PASS: Agentic Ingestion triggered by affordance gradients.")
    else:
        print("FAIL: Agentic trigger failed.")

    print("\nDiegetic Authenticity Verification Complete.")

if __name__ == "__main__":
    test_diegetic_authenticity()
