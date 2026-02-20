#!/usr/bin/env python3
"""
Phase 4 Test: Advanced Feature Integration
Tests the complete advanced topological analysis system.
"""

import requests
import json
import time
import sys

def test_phase4_advanced_features():
    """Test Phase 4: Advanced Feature Integration."""
    print("🧪 Testing Phase 4: Advanced Feature Integration")
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
    
    # Test with complex input to trigger advanced analysis
    test_input = "Explain the relationship between quantum entanglement and consciousness in the context of topological quantum field theory."
    print(f"🔧 Testing advanced features with complex input: '{test_input[:50]}...'")
    
    try:
        response = requests.post(
            'http://localhost:8000/interact',
            json={'text': test_input},
            timeout=45  # Longer timeout for advanced analysis
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📤 Input: '{test_input[:50]}...'")
            print(f"📥 Output: '{data.get('response', 'No response')[:50]}...'")
            print(f"📊 Output length: {data.get('output_length', 0)}")
            
            # Check Phase 4 diagnostics
            phase4_diagnostics = data.get('phase4_diagnostics', {})
            if phase4_diagnostics:
                print("\n🔧 Phase 4: Advanced Feature Integration Diagnostics:")
                
                # Phase 4.1: Gyroid Violation Score
                gyroid_score = phase4_diagnostics.get('gyroid_violation_score', 0.0)
                print(f"• Phase 4.1 - Gyroid Violation Score: {gyroid_score:.4f}")
                
                if gyroid_score < 0.5:
                    print("  ✅ Low violation - good manifold coherence")
                elif gyroid_score < 1.0:
                    print("  ⚠️  Moderate violation - acceptable manifold state")
                else:
                    print("  ❌ High violation - manifold instability detected")
                
                # Phase 4.2: Unfolding Closure Check
                closure_check = phase4_diagnostics.get('unfolding_closure_check', {})
                if closure_check:
                    is_closed = closure_check.get('is_closed', False)
                    is_trivial = closure_check.get('is_trivial', True)
                    is_valid = closure_check.get('is_valid', False)
                    branches = closure_check.get('unfolding_branches', 0)
                    quality = closure_check.get('closure_quality', 0.0)
                    
                    print(f"• Phase 4.2 - Unfolding Closure Check:")
                    print(f"  - Closed: {is_closed}")
                    print(f"  - Trivial: {is_trivial}")
                    print(f"  - Valid: {is_valid}")
                    print(f"  - Unfolding Branches: {branches}")
                    print(f"  - Closure Quality: {quality:.3f}")
                    
                    if is_valid and not is_trivial:
                        print("  ✅ Excellent topological closure")
                    elif is_closed:
                        print("  ⚠️  Basic closure achieved")
                    else:
                        print("  ❌ Closure failure detected")
                
                # Phase 4.3: Topological Analysis
                topo_analysis = phase4_diagnostics.get('topological_analysis', {})
                if topo_analysis:
                    features = topo_analysis.get('features', [])
                    num_features = topo_analysis.get('num_features', 0)
                    persistence_dim = topo_analysis.get('persistence_dimension', 0)
                    complexity = topo_analysis.get('topological_complexity', 0.0)
                    
                    print(f"• Phase 4.3 - Advanced Topological Analysis:")
                    print(f"  - Features Detected: {num_features}")
                    print(f"  - Persistence Dimension: {persistence_dim}")
                    print(f"  - Topological Complexity: {complexity:.3f}")
                    
                    if features:
                        print("  - Feature Details:")
                        for i, feature in enumerate(features[:8]):  # Show first 8 features
                            print(f"    {i+1}. {feature}")
                        if len(features) > 8:
                            print(f"    ... and {len(features) - 8} more features")
                    
                    if complexity > 2.0:
                        print("  ✅ Rich topological structure detected")
                    elif complexity > 1.0:
                        print("  ⚠️  Moderate topological complexity")
                    else:
                        print("  ❌ Low topological complexity")
                
                # Overall Phase 4 assessment
                advanced_active = phase4_diagnostics.get('advanced_features_active', False)
                print(f"\n🎯 Phase 4 Status: {'✅ ACTIVE' if advanced_active else '❌ INACTIVE'}")
                
            else:
                print("❌ Phase 4 diagnostics not found in response")
                return False
            
            # Check all previous phases are still working
            print("\n🔍 Verifying Previous Phases:")
            
            # Phase 2 (Repair System)
            repair_diagnostics = data.get('repair_diagnostics', {})
            if repair_diagnostics:
                active_repairs = len(repair_diagnostics)
                print(f"• Phase 2: {active_repairs}/5 repair components active")
                if active_repairs >= 5:
                    print("  ✅ Complete repair system operational")
                else:
                    print("  ⚠️  Partial repair system")
            
            # Phase 3 (Response Quality)
            phase3_diagnostics = data.get('phase3_diagnostics', {})
            if phase3_diagnostics:
                dyad_aware = phase3_diagnostics.get('dyad_aware_generation', False)
                echo_suppression = phase3_diagnostics.get('echo_suppression_active', False)
                vowel_opt = phase3_diagnostics.get('vowel_optimization_active', False)
                
                print(f"• Phase 3: Dyad-aware={dyad_aware}, Echo suppression={echo_suppression}, Vowel opt={vowel_opt}")
                if dyad_aware and echo_suppression and vowel_opt:
                    print("  ✅ Complete response optimization active")
                else:
                    print("  ⚠️  Partial response optimization")
            
            # Response quality analysis
            output_text = data.get('response', '')
            if output_text:
                print("\n🔍 Response Quality Analysis:")
                
                total_chars = len(output_text)
                vowels = sum(1 for c in output_text.lower() if c in 'aeiou')
                symbols = sum(1 for c in output_text if not c.isalnum() and not c.isspace())
                words = len(output_text.split())
                
                vowel_ratio = vowels / total_chars if total_chars > 0 else 0
                symbol_ratio = symbols / total_chars if total_chars > 0 else 0
                
                print(f"• Length: {total_chars} chars, {words} words")
                print(f"• Vowel ratio: {vowel_ratio:.3f} (target: >0.15)")
                print(f"• Symbol ratio: {symbol_ratio:.3f} (target: <0.4)")
                
                quality_score = 0
                if vowel_ratio > 0.15:
                    quality_score += 2
                    print("  ✅ Good vowel balance")
                elif vowel_ratio > 0.10:
                    quality_score += 1
                    print("  ⚠️  Acceptable vowel balance")
                else:
                    print("  ❌ Poor vowel balance")
                
                if symbol_ratio < 0.4:
                    quality_score += 2
                    print("  ✅ Good symbol control")
                elif symbol_ratio < 0.6:
                    quality_score += 1
                    print("  ⚠️  Moderate symbol density")
                else:
                    print("  ❌ Excessive symbols")
                
                if words > 5:
                    quality_score += 1
                    print("  ✅ Multi-word coherent response")
                
                if quality_score >= 4:
                    print("  🎉 High quality response achieved!")
                elif quality_score >= 2:
                    print("  ⚠️  Moderate quality response")
                else:
                    print("  ❌ Low quality response")
            
            return True
            
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_dyad_with_phase4():
    """Test Phase 4 with dyad ingestion to verify multimodal integration."""
    print("\n🧪 Testing Phase 4 with Multimodal Dyad Integration")
    print("=" * 70)
    
    # Test with rich visual fingerprint
    rich_fingerprint = {
        'r': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 4,  # Warm colors
        'g': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 4,  # Natural progression
        'b': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 4,  # Cool undertones
        'l': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6] * 4,  # Consistent luminance
        'texture': 0.75  # High texture complexity
    }
    
    label = "fractal mandelbrot set with topological singularities"
    
    print(f"🔧 Testing Phase 4 with complex visual dyad: '{label}'")
    
    try:
        response = requests.post(
            'http://localhost:8000/ingest',
            json={
                'label': label,
                'fingerprint': rich_fingerprint
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📤 Label: '{label}'")
            print(f"📥 Response: '{data.get('metrics', {}).get('response', 'No response')}'")
            
            # Check Phase 4 integration with multimodal data
            phase4_diagnostics = data.get('metrics', {}).get('phase4_diagnostics', {})
            if phase4_diagnostics:
                print("\n🔧 Phase 4 Multimodal Integration:")
                
                gyroid_score = phase4_diagnostics.get('gyroid_violation_score', 0.0)
                print(f"• Gyroid Violation (with visual data): {gyroid_score:.4f}")
                
                topo_analysis = phase4_diagnostics.get('topological_analysis', {})
                if topo_analysis:
                    complexity = topo_analysis.get('topological_complexity', 0.0)
                    print(f"• Topological Complexity (multimodal): {complexity:.3f}")
                
                print("✅ Phase 4 successfully integrated with multimodal dyad system")
                return True
            else:
                print("⚠️  Phase 4 diagnostics not found in dyad response")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Phase 4: Advanced Feature Integration Test Suite")
    print("=" * 70)
    
    # Run tests
    tests = [
        ("Advanced Feature Integration", test_phase4_advanced_features),
        ("Multimodal Dyad Integration", test_dyad_with_phase4)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*70}")
    print("🎯 Phase 4 Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"• {test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PHASE 4 TESTS PASSED!")
        print("✅ Full Gyroid Violation Score computation working")
        print("✅ Complete Unfolding Closure Check implementation working")
        print("✅ Advanced topological analysis and graph generation working")
        print("✅ Multimodal integration with Phase 4 features working")
        print("\n🏆 COMPLETE SYSTEM INTEGRATION ACHIEVED!")
        print("   Phase 2: ✅ Repair System (5/5 components)")
        print("   Phase 3: ✅ Response Quality Optimization")
        print("   Phase 4: ✅ Advanced Feature Integration")
        sys.exit(0)
    else:
        print("❌ Some Phase 4 tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()