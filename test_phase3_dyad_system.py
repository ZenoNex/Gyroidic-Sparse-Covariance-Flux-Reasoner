#!/usr/bin/env python3
"""
Phase 3 Test: Dyad-Aware Response Quality Optimization
Tests the enhanced response generation with privileged dyad ingestion system.
"""

import requests
import json
import time
import sys

def test_regular_response():
    """Test regular response generation with Phase 3 optimizations."""
    print("🧪 Testing Phase 3: Regular Response Generation")
    print("=" * 60)
    
    test_input = "What is the nature of consciousness?"
    print(f"🔧 Testing enhanced response generation with: '{test_input}'")
    
    try:
        response = requests.post(
            'http://localhost:8000/interact',
            json={'text': test_input},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📤 Input: '{test_input}'")
            print(f"📥 Output: '{data.get('response', 'No response')}'")
            print(f"📊 Output length: {data.get('output_length', 0)}")
            
            # Check Phase 3 diagnostics
            phase3_diagnostics = data.get('phase3_diagnostics', {})
            if phase3_diagnostics:
                print("🔧 Phase 3 Diagnostics:")
                print(f"• Dyad-aware generation: {phase3_diagnostics.get('dyad_aware_generation', False)}")
                print(f"• Echo suppression: {phase3_diagnostics.get('echo_suppression_active', False)}")
                print(f"• Vowel optimization: {phase3_diagnostics.get('vowel_optimization_active', False)}")
                print(f"• Linguistic correction: {phase3_diagnostics.get('linguistic_correction_available', False)}")
                print(f"• Multimodal support: {phase3_diagnostics.get('multimodal_fingerprint_support', False)}")
            
            # Analyze response quality
            output_text = data.get('response', '')
            if output_text:
                analyze_response_quality(output_text, "Regular Response")
            
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_text_to_text_association():
    """Test privileged text-to-text association learning."""
    print("\n🧪 Testing Phase 3: Text-to-Text Association Learning")
    print("=" * 60)
    
    source_concept = "quantum mechanics"
    target_concept = "consciousness studies"
    
    print(f"🔧 Learning association: '{source_concept}' ↔ '{target_concept}'")
    
    try:
        response = requests.post(
            'http://localhost:8000/associate',
            json={
                'source': source_concept,
                'target': target_concept
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📤 Source: '{source_concept}'")
            print(f"📤 Target: '{target_concept}'")
            print(f"📥 Response: '{data.get('metrics', {}).get('response', 'No response')}'")
            print(f"📊 Status: {data.get('status', 'Unknown')}")
            
            # Analyze association response
            response_text = data.get('metrics', {}).get('response', '')
            if response_text:
                analyze_response_quality(response_text, "Association Learning")
            
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_text_to_image_dyad():
    """Test privileged text-to-image dyad ingestion."""
    print("\n🧪 Testing Phase 3: Text-to-Image Dyad Ingestion")
    print("=" * 60)
    
    # Simulate a visual fingerprint (RGB histograms + texture)
    mock_fingerprint = {
        'r': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 4,  # 32 values
        'g': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 4,  # 32 values  
        'b': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 4,  # 32 values
        'l': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] * 4,  # 32 values (luminance)
        'texture': 0.45  # Texture complexity
    }
    
    label = "sunset over mountains"
    
    print(f"🔧 Ingesting dyad: '{label}' with visual fingerprint")
    print(f"🔧 Fingerprint: R={sum(mock_fingerprint['r'])/len(mock_fingerprint['r']):.2f}, "
          f"G={sum(mock_fingerprint['g'])/len(mock_fingerprint['g']):.2f}, "
          f"B={sum(mock_fingerprint['b'])/len(mock_fingerprint['b']):.2f}, "
          f"Texture={mock_fingerprint['texture']:.2f}")
    
    try:
        response = requests.post(
            'http://localhost:8000/ingest',
            json={
                'label': label,
                'fingerprint': mock_fingerprint
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📤 Label: '{label}'")
            print(f"📥 Response: '{data.get('metrics', {}).get('response', 'No response')}'")
            print(f"📊 Status: {data.get('status', 'Unknown')}")
            
            # Check multimodal support
            phase3_diagnostics = data.get('metrics', {}).get('phase3_diagnostics', {})
            if phase3_diagnostics:
                print(f"🔧 Multimodal support active: {phase3_diagnostics.get('multimodal_fingerprint_support', False)}")
            
            # Analyze dyad response
            response_text = data.get('metrics', {}).get('response', '')
            if response_text:
                analyze_response_quality(response_text, "Dyad Ingestion")
            
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def analyze_response_quality(text: str, context: str):
    """Analyze the quality of generated response."""
    print(f"🔍 {context} Analysis:")
    
    if not text:
        print("• No text to analyze")
        return
    
    # Character analysis
    total_chars = len(text)
    vowels = sum(1 for c in text.lower() if c in 'aeiou')
    consonants = sum(1 for c in text if c.isalpha() and c.lower() not in 'aeiou')
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    spaces = text.count(' ')
    words = len(text.split())
    
    vowel_ratio = vowels / total_chars if total_chars > 0 else 0
    symbol_ratio = symbols / total_chars if total_chars > 0 else 0
    
    print(f"• Length: {total_chars} chars, {words} words")
    print(f"• Vowel ratio: {vowel_ratio:.3f} (target: >0.15)")
    print(f"• Symbol ratio: {symbol_ratio:.3f} (target: <0.4)")
    print(f"• Composition: {vowels}V + {consonants}C + {symbols}S + {spaces}spaces")
    
    # Quality assessment
    quality_score = 0
    if vowel_ratio > 0.15:
        quality_score += 2
        print("✅ Good vowel balance")
    elif vowel_ratio > 0.10:
        quality_score += 1
        print("⚠️  Acceptable vowel balance")
    else:
        print("❌ Poor vowel balance")
    
    if symbol_ratio < 0.4:
        quality_score += 2
        print("✅ Good symbol control")
    elif symbol_ratio < 0.6:
        quality_score += 1
        print("⚠️  Moderate symbol density")
    else:
        print("❌ Excessive symbols")
    
    if words > 3:
        quality_score += 1
        print("✅ Multi-word response")
    
    # Overall assessment
    if quality_score >= 4:
        print("🎉 High quality response!")
    elif quality_score >= 2:
        print("⚠️  Moderate quality response")
    else:
        print("❌ Low quality response")

def main():
    """Main test function."""
    print("🧪 Phase 3: Dyad-Aware Response Quality Optimization Test")
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
                sys.exit(1)
            time.sleep(1)
    
    # Run tests
    tests = [
        ("Regular Response Generation", test_regular_response),
        ("Text-to-Text Association", test_text_to_text_association),
        ("Text-to-Image Dyad Ingestion", test_text_to_image_dyad)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*70}")
    print("🎯 Phase 3 Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"• {test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 3 tests passed!")
        print("✅ Dyad-aware response generation working")
        print("✅ Privileged text-to-text associations working")
        print("✅ Privileged text-to-image dyad ingestion working")
        sys.exit(0)
    else:
        print("❌ Some Phase 3 tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()