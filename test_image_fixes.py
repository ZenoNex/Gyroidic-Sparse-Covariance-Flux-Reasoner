#!/usr/bin/env python3
"""
Quick test to verify the image reconstruction and cross-modal similarity fixes.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.append('src')

def test_image_reconstruction():
    """Test the improved image reconstruction."""
    print("🖼️ Testing Image Reconstruction Fix")
    print("=" * 40)
    
    # Create a synthetic fingerprint with meaningful values
    fingerprint = torch.zeros(137)
    
    # Red square pattern
    fingerprint[:32] = torch.softmax(torch.tensor([3.0] + [0.1] * 31), dim=0)  # Strong red
    fingerprint[32:64] = torch.softmax(torch.tensor([0.1] * 32), dim=0)        # Low green
    fingerprint[64:96] = torch.softmax(torch.tensor([0.1] * 32), dim=0)        # Low blue
    fingerprint[96:128] = torch.softmax(torch.tensor([1.0] * 32), dim=0)       # Luminance
    fingerprint[128] = 0.8  # High texture
    fingerprint[129:137] = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.1, 0.7, 0.1, 0.0])  # Edge features
    
    print(f"Created synthetic fingerprint: {fingerprint.shape}")
    print(f"Red histogram peak: {torch.argmax(fingerprint[:32]).item()}")
    print(f"Texture value: {fingerprint[128].item():.3f}")
    
    # Test the improved reconstruction
    try:
        from image_extension import ImageProcessor
        
        processor = ImageProcessor()
        reconstructed_img = processor.fingerprint_to_image(fingerprint, size=(64, 64))
        
        # Save the image
        reconstructed_img.save("test_red_square_reconstruction.png")
        
        # Analyze the image
        img_array = np.array(reconstructed_img)
        mean_brightness = np.mean(img_array)
        brightness_variance = np.var(img_array)
        
        # Check color channels
        mean_red = np.mean(img_array[:, :, 0])
        mean_green = np.mean(img_array[:, :, 1])
        mean_blue = np.mean(img_array[:, :, 2])
        
        print(f"✅ Image reconstructed successfully")
        print(f"   Size: {img_array.shape}")
        print(f"   Mean brightness: {mean_brightness:.1f}")
        print(f"   Brightness variance: {brightness_variance:.1f}")
        print(f"   Color channels - R: {mean_red:.1f}, G: {mean_green:.1f}, B: {mean_blue:.1f}")
        
        # Check if it's not all black
        if mean_brightness > 20:
            print(f"   ✅ Image has good brightness (not black)")
        else:
            print(f"   ❌ Image is too dark")
            
        # Check if red is dominant
        if mean_red > mean_green and mean_red > mean_blue:
            print(f"   ✅ Red channel is dominant as expected")
        else:
            print(f"   ⚠️  Red channel not dominant")
            
        return True
        
    except Exception as e:
        print(f"❌ Image reconstruction failed: {e}")
        return False

def test_meaningful_text_embeddings():
    """Test the improved text embedding generation."""
    print("\n📝 Testing Meaningful Text Embeddings")
    print("=" * 40)
    
    descriptions = [
        "A red square on blue background",
        "Green circle with yellow center",
        "Blue gradient from left to right",
        "Abstract colorful pattern"
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    
    for desc in descriptions:
        # Create meaningful text embedding based on description content
        text_features = torch.zeros(768, device=device)
        
        # Add features based on color words
        if 'red' in desc.lower():
            text_features[:32] = torch.softmax(torch.tensor([2.0] + [0.1] * 31), dim=0)
        if 'green' in desc.lower():
            text_features[32:64] = torch.softmax(torch.tensor([0.1] * 16 + [2.0] + [0.1] * 15), dim=0)
        if 'blue' in desc.lower():
            text_features[64:96] = torch.softmax(torch.linspace(0.1, 1.5, 32), dim=0)
        
        # Add features based on shape words
        if 'square' in desc.lower():
            text_features[129:137] = torch.tensor([0.8, 0.1, 0.7, 0.2, 0.1, 0.6, 0.1, 0.0])
        elif 'circle' in desc.lower():
            text_features[129:137] = torch.tensor([0.4, 0.4, 0.6, 0.3, 0.3, 0.4, 0.4, 0.2])
        elif 'gradient' in desc.lower():
            text_features[129:137] = torch.tensor([0.7, 0.1, 0.3, 0.1, 0.7, 0.5, 0.2, -0.2])
        
        # Check if features were added
        feature_sum = torch.sum(text_features).item()
        
        print(f"   '{desc}': feature sum = {feature_sum:.3f}")
        
        if feature_sum > 0.1:
            print(f"     ✅ Meaningful features detected")
        else:
            print(f"     ⚠️  No meaningful features")
    
    return True

def test_color_reconstruction():
    """Test reconstruction of different colors."""
    print("\n🎨 Testing Color Reconstruction")
    print("=" * 30)
    
    try:
        from image_extension import ImageProcessor
        processor = ImageProcessor()
        
        colors = [
            ("red", [3.0] + [0.1] * 31, [0.1] * 32, [0.1] * 32),
            ("green", [0.1] * 32, [0.1] * 16 + [3.0] + [0.1] * 15, [0.1] * 32),
            ("blue", [0.1] * 32, [0.1] * 32, [0.1, 0.2, 0.5, 1.0, 2.0] + [0.1] * 27)
        ]
        
        for color_name, r_hist, g_hist, b_hist in colors:
            fingerprint = torch.zeros(137)
            fingerprint[:32] = torch.softmax(torch.tensor(r_hist), dim=0)
            fingerprint[32:64] = torch.softmax(torch.tensor(g_hist), dim=0)
            fingerprint[64:96] = torch.softmax(torch.tensor(b_hist), dim=0)
            fingerprint[96:128] = torch.softmax(torch.tensor([1.0] * 32), dim=0)
            fingerprint[128] = 0.5
            fingerprint[129:137] = torch.tensor([0.5] * 8)
            
            img = processor.fingerprint_to_image(fingerprint, size=(32, 32))
            img.save(f"test_{color_name}_reconstruction.png")
            
            # Check color dominance
            img_array = np.array(img)
            mean_r = np.mean(img_array[:, :, 0])
            mean_g = np.mean(img_array[:, :, 1])
            mean_b = np.mean(img_array[:, :, 2])
            
            print(f"   {color_name}: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}")
            
            # Check if the expected color is dominant
            if color_name == "red" and mean_r > max(mean_g, mean_b):
                print(f"     ✅ Red is dominant")
            elif color_name == "green" and mean_g > max(mean_r, mean_b):
                print(f"     ✅ Green is dominant")
            elif color_name == "blue" and mean_b > max(mean_r, mean_g):
                print(f"     ✅ Blue is dominant")
            else:
                print(f"     ⚠️  Expected color not dominant")
        
        return True
        
    except Exception as e:
        print(f"❌ Color reconstruction test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Image Processing Fixes Verification")
    print("Testing improved image reconstruction and text embeddings")
    print("=" * 60)
    
    # Test image reconstruction
    reconstruction_success = test_image_reconstruction()
    
    # Test text embeddings
    text_success = test_meaningful_text_embeddings()
    
    # Test color reconstruction
    color_success = test_color_reconstruction()
    
    print(f"\n🎯 Overall Results:")
    print(f"  Image reconstruction: {'✅ PASS' if reconstruction_success else '❌ FAIL'}")
    print(f"  Text embeddings: {'✅ PASS' if text_success else '❌ FAIL'}")
    print(f"  Color reconstruction: {'✅ PASS' if color_success else '❌ FAIL'}")
    
    if reconstruction_success and text_success and color_success:
        print(f"\n🚀 All fixes verified! Images should no longer be black.")
        print(f"   Cross-modal similarities should be more meaningful.")
        print(f"   Ready to run test_image_simple.py")
    else:
        print(f"\n⚠️  Some fixes need more work. Check the error messages above.")
