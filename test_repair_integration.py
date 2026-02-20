#!/usr/bin/env python3
"""
Test script to verify the garbled output repair system integration.
"""

import requests
import json
import time

def test_repair_system():
    """Test the integrated repair system with 'hello' input."""
    
    print("🧪 Testing Garbled Output Repair System Integration")
    print("=" * 60)
    
    # Wait for backend to be ready
    print("⏳ Waiting for backend to be ready...")
    time.sleep(2)
    
    try:
        # Test ping first
        ping_response = requests.get('http://localhost:8000/ping', timeout=10)  # Increased timeout
        print(f"✅ Backend ping successful: {ping_response.json()}")
        
        # Test the repair system with 'hello'
        print("\n🔧 Testing repair system with input: 'hello'")
        
        response = requests.post(
            'http://localhost:8000/interact',
            json={'text': 'hello'},
            timeout=30  # Increased timeout for neural generation
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📤 Input: 'hello'")
            print(f"📥 Output: '{data['response']}'")
            print(f"📊 Output length: {len(data['response'])}")
            
            # Check if repair diagnostics are present
            if 'repair_diagnostics' in data:
                print("\n🔧 Repair System Diagnostics:")
                repair_diag = data['repair_diagnostics']
                
                for component, diagnostics in repair_diag.items():
                    print(f"  • {component}: {type(diagnostics).__name__}")
                
                print(f"\n📈 System Metrics:")
                print(f"  • Spectral Entropy: {data.get('spectral_entropy', 'N/A'):.3f}")
                print(f"  • Chiral Score: {data.get('chiral_score', 'N/A'):.3f}")
                print(f"  • Coprime Lock: {data.get('coprime_lock', 'N/A')}")
                print(f"  • Iteration: {data.get('iteration', 'N/A')}")
                
                # Check if output looks repaired (not garbled)
                output = data['response']
                if len(output) > 0:
                    # Simple heuristics for garbled vs repaired output
                    vowels = sum(1 for c in output.lower() if c in 'aeiou')
                    consonants = sum(1 for c in output.lower() if c.isalpha() and c not in 'aeiou')
                    symbols = sum(1 for c in output if not c.isalnum() and c != ' ')
                    
                    vowel_ratio = vowels / (vowels + consonants) if (vowels + consonants) > 0 else 0
                    symbol_ratio = symbols / len(output) if len(output) > 0 else 0
                    
                    print(f"\n🔍 Output Analysis:")
                    print(f"  • Vowel ratio: {vowel_ratio:.2f} (healthy: >0.2)")
                    print(f"  • Symbol ratio: {symbol_ratio:.2f} (healthy: <0.3)")
                    
                    if vowel_ratio > 0.2 and symbol_ratio < 0.3:
                        print("  ✅ Output appears linguistically healthy!")
                    else:
                        print("  ⚠️  Output may still show garbling patterns")
                
            else:
                print("❌ Repair diagnostics not found in response")
                
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the diegetic backend is running on port 8000")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_repair_system()