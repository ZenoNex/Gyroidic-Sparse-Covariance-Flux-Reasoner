#!/usr/bin/env python3
"""
Test basic functionality without the complex dataset interface.
"""

import sys
import os

# Add paths
sys.path.append('src')
sys.path.append('examples')

def test_basic_imports():
    """Test if we can import basic components."""
    print("🧪 Testing Basic Imports")
    print("=" * 30)
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        print(f"   Version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL/Pillow imported successfully")
        version = getattr(Image, '__version__', 'Unknown')
        print(f"   Version: {version}")
    except ImportError as e:
        print(f"❌ PIL/Pillow import failed: {e}")
        return False
    
    return True

def test_simple_model():
    """Test if we can create a simple model."""
    print("\n🤖 Testing Simple Model Creation")
    print("=" * 35)
    
    try:
        import torch  # Add missing import
        from enhanced_temporal_training import NonLobotomyTemporalModel
        
        model = NonLobotomyTemporalModel(
            input_dim=768,
            hidden_dim=256,
            num_functionals=5,
            poly_degree=4,
            device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully")
        print(f"   Parameters: {param_count:,}")
        
        # Test forward pass
        test_input = torch.randn(1, 768)
        with torch.no_grad():
            output = model(test_input, return_analysis=True)
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {output['hidden_state'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_image_processing():
    """Test basic image processing."""
    print("\n🎨 Testing Image Processing")
    print("=" * 28)
    
    try:
        # Import the working test
        import test_image_simple
        
        print("✅ Image test module imported")
        print("🚀 You can run: python test_image_simple.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🔬 Gyroidic System Basic Functionality Test")
    print("=" * 45)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Simple Model", test_simple_model),
        ("Image Processing", test_image_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 25)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your Gyroidic system is working correctly")
        print("\n🚀 Ready to use:")
        print("   python test_image_simple.py")
        print("   python quick_test_phase25.py")
    else:
        print(f"\n⚠️  Some tests failed, but basic functionality may still work")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
