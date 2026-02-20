#!/usr/bin/env python3
"""
Test the color reconstruction fix.
"""

import torch
import numpy as np
from PIL import Image
import sys

# Add src to path
sys.path.append('src')

def test_color_reconstruction():
    """Test the improved color reconstruction."""
    print("🎨 Testing Color Reconstruction Fix")
    print("=" * 40)
    
    try:
        from image_extension import ImageProcessor
        processor = ImageProcessor()
        
        # Test red square
        print("\n🔴 Testing Red Square:")
        red_fingerprint = torch.zeros(137)
        red_fingerprint[:32] = torch.softmax(torch.tensor([3.0] + [0.1] * 31), dim=0)  # Strong red at position 0
        red_fingerprint[32:64] = torch.softmax(torch.tensor([0.1] * 32), dim=0)        # Weak green
        red_fingerprint[64:96] = torch.softmax(torch.tensor([0.1] * 32), dim=0)        # Weak blue
        red_fingerprint[96:128] = torch.softmax(torch.tensor([1.0] * 32), dim=0)
        red_fingerprint[128] = 0.5
        red_fingerprint[129:137] = torch.tensor([0.5] * 8)
        
        red_img = processor.fingerprint_to_image(red_fingerprint, size=(32, 32))
        red_img.save("test_red_fix.png")
        
        red_array = np.array(red_img)
        print(f"  Red image - R: {np.mean(red_array[:,:,0]):.1f}, G: {np.mean(red_array[:,:,1]):.1f}, B: {np.mean(red_array[:,:,2]):.1f}")
        
        # Test green circle
        print("\n🟢 Testing Green Circle:")
        green_fingerprint = torch.zeros(137)
        green_fingerprint[:32] = torch.softmax(torch.tensor([0.1] * 32), dim=0)        # Weak red
        green_fingerprint[32:64] = torch.softmax(torch.tensor([0.1] * 16 + [3.0] + [0.1] * 15), dim=0)  # Strong green at position 16
        green_fingerprint[64:96] = torch.softmax(torch.tensor([0.1] * 32), dim=0)      # Weak blue
        green_fingerprint[96:128] = torch.softmax(torch.tensor([1.0] * 32), dim=0)
        green_fingerprint[128] = 0.5
        green_fingerprint[129:137] = torch.tensor([0.5] * 8)
        
        green_img = processor.fingerprint_to_image(green_fingerprint, size=(32, 32))
        green_img.save("test_green_fix.png")
        
        green_array = np.array(green_img)
        print(f"  Green image - R: {np.mean(green_array[:,:,0]):.1f}, G: {np.mean(green_array[:,:,1]):.1f}, B: {np.mean(green_array[:,:,2]):.1f}")
        
        # Test blue gradient
        print("\n🔵 Testing Blue Gradient:")
        blue_fingerprint = torch.zeros(137)
        blue_fingerprint[:32] = torch.softmax(torch.tensor([0.1] * 32), dim=0)         # Weak red
        blue_fingerprint[32:64] = torch.softmax(torch.tensor([0.1] * 32), dim=0)       # Weak green
        blue_fingerprint[64:96] = torch.softmax(torch.tensor([0.1] * 24 + [3.0] + [0.1] * 7), dim=0)  # Strong blue at position 24
        blue_fingerprint[96:128] = torch.softmax(torch.tensor([1.0] * 32), dim=0)
        blue_fingerprint[128] = 0.5
        blue_fingerprint[129:137] = torch.tensor([0.5] * 8)
        
        blue_img = processor.fingerprint_to_image(blue_fingerprint, size=(32, 32))
        blue_img.save("test_blue_fix.png")
        
        blue_array = np.array(blue_img)
        print(f"  Blue image - R: {np.mean(blue_array[:,:,0]):.1f}, G: {np.mean(blue_array[:,:,1]):.1f}, B: {np.mean(blue_array[:,:,2]):.1f}")
        
        # Check if colors are working
        red_dominant = np.mean(red_array[:,:,0]) > max(np.mean(red_array[:,:,1]), np.mean(red_array[:,:,2]))
        green_dominant = np.mean(green_array[:,:,1]) > max(np.mean(green_array[:,:,0]), np.mean(green_array[:,:,2]))
        blue_dominant = np.mean(blue_array[:,:,2]) > max(np.mean(blue_array[:,:,0]), np.mean(blue_array[:,:,1]))
        
        print(f"\n🎯 Results:")
        print(f"  Red dominant: {'✅' if red_dominant else '❌'}")
        print(f"  Green dominant: {'✅' if green_dominant else '❌'}")
        print(f"  Blue dominant: {'✅' if blue_dominant else '❌'}")
        
        return red_dominant and green_dominant and blue_dominant
        
    except Exception as e:
        print(f"❌ Color reconstruction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_color_reconstruction()
    
    if success:
        print(f"\n🚀 Color reconstruction fix working!")
    else:
        print(f"\n⚠️  Color reconstruction still needs work.")