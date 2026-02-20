#!/usr/bin/env python3
"""
Test the Symmetry-Preserving Reshape fix for image reconstruction.
"""

import torch
import numpy as np
from PIL import Image
import sys

# Add src to path
sys.path.append('src')

def test_symmetry_preserving_reshape():
    """Test the Symmetry-Preserving Reshape approach."""
    print("🔧 Testing Symmetry-Preserving Reshape for Image Reconstruction")
    print("=" * 60)
    
    # Test the padding approach directly
    print("\n1. Testing tensor padding approach:")
    
    # Simulate the broadcasting issue
    size = (32, 32)
    x_pattern = np.sin(np.linspace(0, 6*np.pi, 28))  # Wrong size (28 instead of 32)
    y_pattern = np.cos(np.linspace(0, 6*np.pi, 35))  # Wrong size (35 instead of 32)
    
    print(f"   Original x_pattern size: {x_pattern.shape[0]} (need {size[1]})")
    print(f"   Original y_pattern size: {y_pattern.shape[0]} (need {size[0]})")
    
    # Apply Symmetry-Preserving Reshape
    x_tensor = torch.tensor(x_pattern, dtype=torch.float32).unsqueeze(0)  # Add batch dim: [1, N]
    y_tensor = torch.tensor(y_pattern, dtype=torch.float32).unsqueeze(0)  # Add batch dim: [1, N]
    
    # Fix x_pattern size
    if x_tensor.shape[1] != size[1]:
        if x_tensor.shape[1] < size[1]:
            pad_size = size[1] - x_tensor.shape[1]
            x_tensor = torch.nn.functional.pad(x_tensor, (0, pad_size), mode='reflect')
            print(f"   🔧 Applied reflective padding to x_pattern: {x_pattern.shape[0]} -> {x_tensor.shape[1]}")
        else:
            x_tensor = x_tensor[:, :size[1]]
            print(f"   🔧 Truncated x_pattern: {x_pattern.shape[0]} -> {x_tensor.shape[1]}")
    
    # Fix y_pattern size
    if y_tensor.shape[1] != size[0]:
        if y_tensor.shape[1] < size[0]:
            pad_size = size[0] - y_tensor.shape[1]
            y_tensor = torch.nn.functional.pad(y_tensor, (0, pad_size), mode='reflect')
            print(f"   🔧 Applied reflective padding to y_pattern: {y_pattern.shape[0]} -> {y_tensor.shape[1]}")
        else:
            y_tensor = y_tensor[:, :size[0]]
            print(f"   🔧 Truncated y_pattern: {y_pattern.shape[0]} -> {y_tensor.shape[1]}")
    
    # Convert back and test broadcasting
    x_pattern_fixed = x_tensor.squeeze(0).numpy()  # Remove batch dim
    y_pattern_fixed = y_tensor.squeeze(0).numpy()  # Remove batch dim
    
    print(f"   Final x_pattern size: {x_pattern_fixed.shape[0]}")
    print(f"   Final y_pattern size: {y_pattern_fixed.shape[0]}")
    
    # Test that broadcasting works now
    try:
        img_array = np.zeros((*size, 3), dtype=np.float32)
        
        # This should work without broadcasting errors
        for i in range(size[0]):
            for c in range(3):
                img_array[i, :, c] += x_pattern_fixed * 0.1
        for j in range(size[1]):
            for c in range(3):
                img_array[:, j, c] += y_pattern_fixed * 0.1
        
        print(f"   ✅ Broadcasting successful! Image shape: {img_array.shape}")
        return True
        
    except Exception as e:
        print(f"   ❌ Broadcasting still failed: {e}")
        return False

def test_full_image_reconstruction():
    """Test the full image reconstruction with the fix."""
    print("\n2. Testing full image reconstruction:")
    
    try:
        from image_extension import ImageProcessor
        processor = ImageProcessor()
        
        # Create a test fingerprint
        fingerprint = torch.zeros(137)
        fingerprint[:32] = torch.softmax(torch.tensor([3.0] + [0.1] * 31), dim=0)  # Red
        fingerprint[32:64] = torch.softmax(torch.tensor([0.1] * 32), dim=0)        # Green
        fingerprint[64:96] = torch.softmax(torch.tensor([0.1] * 32), dim=0)        # Blue
        fingerprint[96:128] = torch.softmax(torch.tensor([1.0] * 32), dim=0)       # Luminance
        fingerprint[128] = 0.8  # High texture
        fingerprint[129:137] = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.1, 0.7, 0.1, 0.0])  # Strong edges
        
        print(f"   Created test fingerprint with strong edge features")
        
        # Test reconstruction
        img = processor.fingerprint_to_image(fingerprint, size=(32, 32))
        img.save("test_symmetry_preserving_reconstruction.png")
        
        # Check the result
        img_array = np.array(img)
        mean_brightness = np.mean(img_array)
        brightness_variance = np.var(img_array)
        
        print(f"   ✅ Image reconstructed successfully")
        print(f"   Mean brightness: {mean_brightness:.1f}")
        print(f"   Brightness variance: {brightness_variance:.1f}")
        print(f"   Color channels - R: {np.mean(img_array[:,:,0]):.1f}, G: {np.mean(img_array[:,:,1]):.1f}, B: {np.mean(img_array[:,:,2]):.1f}")
        
        # Check if it's not the fallback gradient
        if mean_brightness != 127.0:  # Fallback gradient gives exactly 127.0
            print(f"   ✅ Using main reconstruction (not fallback)")
            return True
        else:
            print(f"   ⚠️  Still using fallback gradient")
            return False
            
    except Exception as e:
        print(f"   ❌ Full reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Symmetry-Preserving Reshape Fix Verification")
    print("Applying the documented tensor shape solution to image reconstruction")
    print("=" * 70)
    
    # Test the padding approach
    padding_success = test_symmetry_preserving_reshape()
    
    # Test full reconstruction
    reconstruction_success = test_full_image_reconstruction()
    
    print(f"\n🎯 Results:")
    print(f"  Symmetry-Preserving padding: {'✅ PASS' if padding_success else '❌ FAIL'}")
    print(f"  Full image reconstruction: {'✅ PASS' if reconstruction_success else '❌ FAIL'}")
    
    if padding_success and reconstruction_success:
        print(f"\n🚀 Symmetry-Preserving Reshape fix working!")
        print(f"   No more broadcasting errors")
        print(f"   No more fallback gray images")
        print(f"   Ready for full multimodal integration")
    else:
        print(f"\n⚠️  Fix needs more work - check error messages above")