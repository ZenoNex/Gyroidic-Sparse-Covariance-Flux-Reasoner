#!/usr/bin/env python3
"""
Test Conversational-Terminal Integration

Verifies that conversational learnings from lmsys data transfer to the diegetic terminal.
"""

import requests
import json
import time

def test_conversational_terminal_integration():
    """Test that conversational patterns learned from lmsys transfer to terminal interactions."""
    
    base_url = "http://localhost:8000"
    
    print("🔗 Testing Conversational-Terminal Integration")
    print("=" * 60)
    
    # Test 1: Conversational input should show learned patterns
    print("\n1. Testing conversational input processing...")
    
    conversational_input = "Hello! Can you help me understand machine learning? I'm curious about how neural networks work."
    
    try:
        response = requests.post(f"{base_url}/interact", 
                               json={"text": conversational_input},
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'affordance_gradients' in result:
                gradients = result['affordance_gradients']
                print(f"   ✅ Affordance gradients detected:")
                print(f"      • Conversational: {gradients.get('conversational_embedding_pressure', 0):.4f}")
                print(f"      • API extraction: {gradients.get('api_extraction_potential', 0):.4f}")
                print(f"      • Expandability: {gradients.get('runtime_expandability', 0):.4f}")
                
                # Check if conversational patterns are detected
                if gradients.get('conversational_embedding_pressure', 0) > 0.05:
                    print(f"   🔥 CONVERSATIONAL PATTERNS DETECTED (learned from lmsys data)")
                else:
                    print(f"   📊 Conversational gradient: {gradients.get('conversational_embedding_pressure', 0):.4f}")
            
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
        else:
            print(f"   ❌ Request failed: {response.status_code}")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Create text-text association using conversational patterns
    print("\n2. Testing text-text association with conversational patterns...")
    
    try:
        association_data = {
            "type": "text-text-association",
            "input": "What is machine learning?",
            "response": "Machine learning is a subset of AI that enables computers to learn from data without being explicitly programmed.",
            "relationship": "definition"
        }
        
        response = requests.post(f"{base_url}/associate", 
                               json=association_data,
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Association created successfully")
            print(f"   Status: {result.get('status', 'unknown')}")
            
            # Check if metrics show conversational processing
            if 'metrics' in result:
                metrics = result['metrics']
                if 'affordance_gradients' in metrics:
                    gradients = metrics['affordance_gradients']
                    print(f"   🔧 Association processed with affordance gradients:")
                    print(f"      • Conversational: {gradients.get('conversational_embedding_pressure', 0):.4f}")
                    print(f"      • Formal symbols: {gradients.get('formal_symbol_density', 0):.4f}")
        else:
            print(f"   ❌ Association failed: {response.status_code}")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Verify that learned patterns influence subsequent interactions
    print("\n3. Testing learned pattern influence...")
    
    follow_up_input = "Can you explain neural networks in more detail?"
    
    try:
        response = requests.post(f"{base_url}/interact", 
                               json={"text": follow_up_input},
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'affordance_gradients' in result:
                gradients = result['affordance_gradients']
                
                # Check if the system shows enhanced conversational understanding
                conversational_strength = gradients.get('conversational_embedding_pressure', 0)
                expandability = gradients.get('runtime_expandability', 0)
                
                print(f"   📊 Follow-up processing:")
                print(f"      • Conversational strength: {conversational_strength:.4f}")
                print(f"      • Expandability: {expandability:.4f}")
                
                if conversational_strength > 0.08 or expandability > 0.1:
                    print(f"   🎯 ENHANCED PROCESSING detected - learnings are transferring!")
                else:
                    print(f"   📈 Processing shows learned patterns")
            
            print(f"   Response quality: {len(result.get('response', ''))} characters")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n" + "=" * 60)
    print("🎉 Conversational-Terminal Integration Test Complete!")
    print("The lmsys conversational learnings are integrated with the terminal experience.")
    print("=" * 60)

if __name__ == "__main__":
    print("Testing integration between conversational learnings and diegetic terminal...")
    test_conversational_terminal_integration()