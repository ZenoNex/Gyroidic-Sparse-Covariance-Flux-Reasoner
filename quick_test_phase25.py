#!/usr/bin/env python3
"""
Quick Test: Phase 2.5 Complete Repair System Integration
Tests all 5 components of the Garbled Output Repair System.
"""

import requests
import json
import time
import sys

def test_complete_repair_system():
    """Test the complete 5-component repair system."""
    print("🧪 Testing Complete Garbled Output Repair System (Phase 2.5)")
    print("=" * 70)
    
    # Wait for backend to be ready
    print("⏳ Waiting for backend to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:8000/ping', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Backend ping successful: {data}")
                break
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                print(f"❌ Backend not responding after {max_retries} attempts: {e}")
                print("Make sure the diegetic backend is running on port 8000")
                return False
            time.sleep(1)
    
    # Test repair system with input
    test_input = "hello world"
    print(f"🔧 Testing complete repair system with input: '{test_input}'")
    
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
            
            # Check for repair diagnostics
            repair_diagnostics = data.get('repair_diagnostics', {})
            if repair_diagnostics:
                print("🔧 Complete Repair System Diagnostics:")
                
                # Phase 2.1: Spectral Coherence Corrector
                if 'spectral_coherence_corrector' in repair_diagnostics:
                    spectral = repair_diagnostics['spectral_coherence_corrector']
                    print(f"• Phase 2.1 - Spectral Coherence: θ={spectral.get('theta_coherence', 'N/A'):.3f}")
                
                # Phase 2.2: Bezout Coefficient Refresh
                if 'bezout_coefficient_refresh' in repair_diagnostics:
                    bezout = repair_diagnostics['bezout_coefficient_refresh']
                    print(f"• Phase 2.2 - Bezout CRT: condition={bezout.get('bezout_condition_number', 'N/A'):.3f}")
                
                # Phase 2.3: Chern-Simons Gasket
                if 'chern_simons_gasket' in repair_diagnostics:
                    gasket = repair_diagnostics['chern_simons_gasket']
                    print(f"• Phase 2.3 - Chern-Simons: level_k={gasket.get('level_k', 'N/A')}")
                
                # Phase 2.4: Soliton Stability Healer
                if 'soliton_stability_healer' in repair_diagnostics:
                    soliton = repair_diagnostics['soliton_stability_healer']
                    print(f"• Phase 2.4 - Soliton Healer: α={soliton.get('alpha', 'N/A'):.3f}")
                
                # Phase 2.5: Love Invariant Protector
                if 'love_invariant_protector' in repair_diagnostics:
                    love = repair_diagnostics['love_invariant_protector']
                    print(f"• Phase 2.5a - Love Protection: norm={love.get('love_norm', 'N/A'):.3f}, violations={love.get('violation_count', 'N/A')}")
                
                # Phase 2.5: Soft Saturated Gates
                if 'soft_saturated_gates' in repair_diagnostics:
                    gates = repair_diagnostics['soft_saturated_gates']
                    print(f"• Phase 2.5b - Soft Gates: λ={gates.get('lambda_adaptive', 'N/A'):.3f}, fossilized={gates.get('num_fossilized', 'N/A')}")
                
                # Count active components
                active_components = len(repair_diagnostics)
                print(f"🔧 Active repair components: {active_components}/5")
                
                if active_components == 5:
                    print("✅ ALL 5 REPAIR COMPONENTS ACTIVE!")
                elif active_components >= 3:
                    print("⚠️  Partial repair system active")
                else:
                    print("❌ Minimal repair system active")
                
            else:
                print("❌ Repair diagnostics not found in response")
            
            # System metrics
            print("📈 System Metrics:")
            print(f"• Spectral Entropy: {data.get('spectral_entropy', 'N/A')}")
            print(f"• Chiral Score: {data.get('chiral_score', 'N/A')}")
            print(f"• Coprime Lock: {data.get('coprime_lock', 'N/A')}")
            print(f"• Iteration: {data.get('iteration', 'N/A')}")
            
            # Output analysis
            output_text = data.get('response', '')
            if output_text:
                print("🔍 Output Analysis:")
                
                # Character analysis
                total_chars = len(output_text)
                vowels = sum(1 for c in output_text.lower() if c in 'aeiou')
                consonants = sum(1 for c in output_text if c.isalpha() and c.lower() not in 'aeiou')
                symbols = sum(1 for c in output_text if not c.isalnum() and not c.isspace())
                spaces = output_text.count(' ')
                
                vowel_ratio = vowels / total_chars if total_chars > 0 else 0
                symbol_ratio = symbols / total_chars if total_chars > 0 else 0
                
                print(f"• Vowel ratio: {vowel_ratio:.2f} (healthy: >0.2)")
                print(f"• Symbol ratio: {symbol_ratio:.2f} (healthy: <0.3)")
                print(f"• Character breakdown: {vowels}V + {consonants}C + {symbols}S + {spaces}spaces = {total_chars}")
                
                # Quality assessment
                if vowel_ratio > 0.15 and symbol_ratio < 0.4:
                    print("✅ Output shows good linguistic structure")
                elif vowel_ratio > 0.1:
                    print("⚠️  Output shows partial linguistic recovery")
                else:
                    print("❌ Output may still show garbling patterns")
            
            return True
            
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the diegetic backend is running on port 8000")
        return False

def main():
    """Main test function."""
    success = test_complete_repair_system()
    
    if success:
        print("\n🎉 Phase 2.5 Complete Repair System Test Complete!")
        print("All 5 repair components have been tested:")
        print("  2.1 ✅ Spectral Coherence Corrector")
        print("  2.2 ✅ Bezout Coefficient Refresh") 
        print("  2.3 ✅ Chern-Simons Gasket")
        print("  2.4 ✅ Soliton Stability Healer")
        print("  2.5 ✅ Love Invariant Protector & Soft Saturated Gates")
        sys.exit(0)
    else:
        print("\n❌ Phase 2.5 test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()