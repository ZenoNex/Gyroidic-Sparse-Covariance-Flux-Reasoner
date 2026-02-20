#!/usr/bin/env python3
"""
Quick test to verify the PIL compatibility and pas_h type handling fixes.
"""

import torch
import sys
import os
from PIL import Image
import numpy as np

# Add src to path
sys.path.append('src')
sys.path.append('examples')

def test_pil_compatibility():
    """Test PIL compatibility fix in image_extension.py"""
    print("🖼️ Testing PIL Compatibility Fix")
    print("=" * 40)
    
    try:
        from image_extension import ImageProcessor
        
        # Create a simple test image
        test_img = Image.new('RGB', (32, 32), color=(128, 64, 192))
        test_img.save('temp_test_image.png')
        
        # Initialize processor
        processor = ImageProcessor()
        
        # Try to extract fingerprint (this will test the PIL resize fix)
        fingerprint = processor.extract_image_fingerprint('temp_test_image.png')
        
        if fingerprint is not None:
            print("✅ PIL compatibility fix working")
            print(f"   Fingerprint shape: {fingerprint.shape}")
            print(f"   Fingerprint range: [{fingerprint.min():.3f}, {fingerprint.max():.3f}]")
            
            # Test embedding projection
            embedding = processor.fingerprint_to_embedding_space(fingerprint)
            print(f"   Embedding shape: {embedding.shape}")
            
            # Clean up
            os.remove('temp_test_image.png')
            return True
        else:
            print("❌ PIL compatibility fix failed")
            return False
            
    except Exception as e:
        print(f"❌ PIL test failed: {e}")
        return False

def test_pas_h_type_handling():
    """Test pas_h type handling fix in love_invariant_protector.py"""
    print("\n🔧 Testing pas_h Type Handling Fix")
    print("=" * 40)
    
    try:
        from src.core.love_invariant_protector import SoftSaturatedGates
        
        # Initialize gates with correct parameters
        gates = SoftSaturatedGates(num_functionals=5, poly_degree=4)
        
        # Test with tensor pas_h
        test_signal = torch.randn(1, 5, 13)  # [batch, K, residue_dim]
        pas_h_tensor = torch.tensor(0.5)
        performance_scores = torch.rand(5)  # [K]
        
        result1 = gates.apply_soft_saturation(test_signal, pas_h_tensor, performance_scores)
        print("✅ Tensor pas_h handling working")
        print(f"   Input shape: {test_signal.shape}")
        print(f"   Output shape: {result1.shape}")
        
        # Test with float pas_h (this was causing the error)
        pas_h_float = 0.7
        
        result2 = gates.apply_soft_saturation(test_signal, pas_h_float, performance_scores)
        print("✅ Float pas_h handling working")
        print(f"   Float pas_h: {pas_h_float}")
        print(f"   Output shape: {result2.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ pas_h type handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_integration():
    """Test simple integration without complex dependencies"""
    print("\n🧠 Testing Simple Integration")
    print("=" * 40)
    
    try:
        from image_extension import ImageProcessor
        
        # Create processor
        processor = ImageProcessor()
        
        # Create synthetic fingerprint data (avoiding complex model dependencies)
        synthetic_fingerprint = torch.rand(137) * 0.5 + 0.25  # Range [0.25, 0.75]
        
        print(f"✅ Synthetic fingerprint created: {synthetic_fingerprint.shape}")
        
        # Test embedding projection
        embedding = processor.fingerprint_to_embedding_space(synthetic_fingerprint)
        print(f"✅ Embedding projection: {embedding.shape}")
        
        # Test reconstruction
        reconstructed = processor.embedding_to_fingerprint_space(embedding)
        print(f"✅ Fingerprint reconstruction: {reconstructed.shape}")
        
        # Test image generation
        test_image = processor.fingerprint_to_image(synthetic_fingerprint)
        print(f"✅ Image generation: {test_image.size}")
        
        # Calculate reconstruction error
        error = torch.mean((synthetic_fingerprint - reconstructed.squeeze(0))**2).item()
        print(f"✅ Reconstruction error: {error:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Verifying Bug Fixes")
    print("Testing PIL compatibility and pas_h type handling")
    print("=" * 60)
    
    results = []
    
    # Test PIL compatibility
    results.append(test_pil_compatibility())
    
    # Test pas_h type handling
    results.append(test_pas_h_type_handling())
    
    # Test simple integration
    results.append(test_simple_integration())
    
    print(f"\n🎯 Test Results Summary")
    print("=" * 30)
    print(f"PIL Compatibility: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"pas_h Type Handling: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"Simple Integration: {'✅ PASS' if results[2] else '❌ FAIL'}")
    
    if all(results):
        print(f"\n🚀 All fixes verified! Ready to proceed with image integration.")
    else:
        print(f"\n⚠️  Some tests failed. Check the error messages above.")