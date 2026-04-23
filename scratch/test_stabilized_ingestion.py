import torch
import sys
import os

# Mock the backend to test the logic
sys.path.append(os.getcwd())

from src.ui.diegetic_backend import DiegeticPhysicsEngine

def test_stabilized_ingestion():
    print("Initializing DiegeticPhysicsEngine...")
    engine = DiegeticPhysicsEngine()
    
    # Mock video data (base64)
    video_b64 = "AAAAGGZ0eXBtcDQyAAAAAGlzb21tcDQyAAAD+21vb3YAAABsbXZoZAAAAADbe6P723uj+wAAA+gAAAUAAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAI0dHJhawAAXHRraGQAAAAD23uj+9t7o/sAAAABAAAAAAAAUAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAALNtZGlhAAAAWG1kaGQAAAAA23uj+9t7o/sAAAcIAAAH0ABVUAABAAAAAAAhYm5oZAAAAAAAAAAAAAAAAC0AAAAA"
    
    print("\nTesting INGEST_VIDEO_DYAD...")
    # This should trigger the new consolidated logic
    result = engine.process_text(
        text="INGEST_VIDEO_DYAD: [PARSE] | A stable test video",
        video_dyad_b64=video_b64
    )
    
    print(f"Response: {result.get('response')}")
    print(f"Metrics: {result.get('metrics')}")
    
    # Check if multimodal support is reported correctly
    if result.get('metrics', {}).get('multimodal_fingerprint_support'):
        print("SUCCESS: Multimodal support correctly reported.")
    else:
        print("FAILURE: Multimodal support not reported.")

    # Check if Betti numbers or chiral score changed from 1.0 (manual command bypass usually sets to 1.0)
    # But for INGEST commands, it should be derived.
    print(f"Chiral Score: {result.get('metrics', {}).get('chiral_score')}")

if __name__ == "__main__":
    test_stabilized_ingestion()
