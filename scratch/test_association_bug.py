import torch
import sys
import os

# Add local project to path
sys.path.append(os.getcwd())

from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer, ResidueFusion

def test_fossilizer_none_fingerprint():
    print("[TEST] Testing Fossilizer with None fingerprint...")
    
    # Initialize components
    fusion = ResidueFusion(feature_dim=512)
    fossilizer = DyadFossilizer(storage_dir="scratch/test_fossils", fusion_layer=fusion)
    
    # Create a dyad with None fingerprint
    dyad = KnowledgeDyad(
        linguistic_description="Test Description",
        image_fingerprint=None 
    )
    
    # Mock text embedding
    text_embedding = torch.randn(1, 512)
    
    try:
        # This used to fail with "Could not infer dtype of NoneType"
        fossil_path = fossilizer.fossilize(dyad, text_embedding)
        print(f"[OK] Fossilized successfully at: {fossil_path}")
        
        # Verify the fossil file exists
        if os.path.exists(fossil_path):
            print("[OK] Fossil file exists.")
            # Load and check residue
            data = torch.load(fossil_path)
            print(f"[DATA] Residue vector shape: {data['residue_vector'].shape}")
            os.remove(fossil_path)
        else:
            print("[FAIL] Fossil file not found!")
            sys.exit(1)
            
    except Exception as e:
        print(f"[FAIL] Fossilization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_fossilizer_none_fingerprint()
    print("\n[VERIFICATION COMPLETE] Association NoneType bug is resolved.")
