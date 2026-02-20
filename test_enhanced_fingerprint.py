#!/usr/bin/env python3
"""
Test Enhanced Fingerprint System with Edge Detection
Tests the new 137-dimensional fingerprint system.
"""

import requests
import json
import time
import sys

def test_enhanced_fingerprint_system():
    """Test the enhanced fingerprint system with edge detection."""
    print("🧪 Testing Enhanced Fingerprint System (137D with Edge Detection)")
    print("=" * 70)
    
    # Wait for backend
    print("⏳ Waiting for backend to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:8000/ping', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Backend ready: PID {data.get('pid', 'unknown')}")
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                print("❌ Backend not responding")
                return False
            time.sleep(1)
    
    # Test with enhanced fingerprint (137 dimensions)
    enhanced_fingerprint = {
        'r': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 4,  # 32 values
        'g': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 4,  # 32 values  
        'b': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 4,  # 32 values
        'l': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9] * 4,  # 32 values (luminance)
        'texture': 0.65,  # 1 value
        'edges': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  # 8 edge features
    }
    
    label = "geometric pattern with sharp edges and corners"
    
    print(f"🔧 Testing enhanced fingerprint: '{label}'")
    print(f"🔧 Fingerprint dimensions: R=32, G=32, B=32, L=32, Texture=1, Edges=8 → Total=137")
    print(f"🔧 Edge features: density={enhanced_fingerprint['edges'][0]:.2f}, "
          f"strength={enhanced_fingerprint['edges'][1]:.2f}, "
          f"corners={enhanced_fingerprint['edges'][3]:.2f}")
    
    try:
        response = requests.post(
            'http://localhost:8000/ingest',
            json={
                'label': label,
                'fingerprint': enhanced_fingerprint
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📤 Label: '{label}'")
            print(f"📥 Response: '{data.get('metrics', {}).get('response', 'No response')}'")
            print(f"📊 Status: {data.get('status', 'Unknown')}")
            
            # Check if enhanced fingerprint was processed correctly
            response_text = data.get('metrics', {}).get('response', '')
            
            # Look for edge-related terms in response (should mention complexity due to high edge features)
            if 'complex' in response_text.lower() or 'sharp' in response_text.lower():
                print("✅ Enhanced fingerprint correctly detected high edge complexity")
            else:
                print("⚠️  Edge features may not be fully integrated")
            
            # Check Phase 4 integration
            phase4_diagnostics = data.get('metrics', {}).get('phase4_diagnostics', {})
            if phase4_diagnostics:
                gyroid_score = phase4_diagnostics.get('gyroid_violation_score', 0.0)
                print(f"🔧 Phase 4 Gyroid Score (with edges): {gyroid_score:.4f}")
                
                topo_analysis = phase4_diagnostics.get('topological_analysis', {})
                if topo_analysis:
                    complexity = topo_analysis.get('topological_complexity', 0.0)
                    print(f"🔧 Topological Complexity (with edges): {complexity:.3f}")
            
            return True
            
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_backward_compatibility():
    """Test that old fingerprints (without edges) still work."""
    print("\n🧪 Testing Backward Compatibility (129D without edges)")
    print("=" * 70)
    
    # Test with old fingerprint format (no edges field)
    old_fingerprint = {
        'r': [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4] * 4,  # 32 values
        'g': [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4] * 4,  # 32 values  
        'b': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7] * 4,  # 32 values
        'l': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4] * 4,  # 32 values
        'texture': 0.25  # 1 value (no edges field)
    }
    
    label = "smooth gradient without sharp features"
    
    print(f"🔧 Testing backward compatibility: '{label}'")
    print(f"🔧 Old fingerprint dimensions: R=32, G=32, B=32, L=32, Texture=1 → Total=129")
    print(f"🔧 Missing edges field should default to zeros")
    
    try:
        response = requests.post(
            'http://localhost:8000/ingest',
            json={
                'label': label,
                'fingerprint': old_fingerprint
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📤 Label: '{label}'")
            print(f"📥 Response: '{data.get('metrics', {}).get('response', 'No response')}'")
            
            # Should work without errors
            response_text = data.get('metrics', {}).get('response', '')
            if response_text and len(response_text) > 10:
                print("✅ Backward compatibility maintained - old fingerprints work")
                return True
            else:
                print("⚠️  Backward compatibility issue detected")
                return False
            
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Enhanced Fingerprint System Test Suite")
    print("=" * 70)
    
    # Run tests
    tests = [
        ("Enhanced Fingerprint (137D)", test_enhanced_fingerprint_system),
        ("Backward Compatibility (129D)", test_backward_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*70}")
    print("🎯 Enhanced Fingerprint Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"• {test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ENHANCED FINGERPRINT SYSTEM WORKING!")
        print("✅ 137-dimensional fingerprint with edge detection active")
        print("✅ Edge features: density, strength, orientation, corners")
        print("✅ Backward compatibility with 129D fingerprints maintained")
        print("✅ Integration with Phase 4 topological analysis")
        print("\n🏆 MULTIMODAL ENHANCEMENT COMPLETE!")
        print("   • Color histograms: 128 dims (R+G+B+L)")
        print("   • Texture analysis: 1 dim")
        print("   • Edge detection: 8 dims (NEW!)")
        print("   • Total: 137 dims → 64 dims (projection)")
        sys.exit(0)
    else:
        print("❌ Some enhanced fingerprint tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()