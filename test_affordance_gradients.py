#!/usr/bin/env python3
"""
Test affordance gradients and constraint pressure injection.
"""

import requests
import json

def test_affordance_system():
    """Test the affordance gradient system with different input types."""
    
    base_url = "http://localhost:8000"
    
    # Test inputs with different affordance characteristics
    test_inputs = [
        {
            "name": "Natural Language",
            "text": "Hello, how are you today? I hope you're doing well.",
            "expected_gradients": "low across all dimensions"
        },
        {
            "name": "High Executability Pressure", 
            "text": "Run the function to generate the output. Execute this algorithm and compute the result.",
            "expected_gradients": "high executability_pressure"
        },
        {
            "name": "High Formal Symbol Density",
            "text": "f(x) = x^2 + 3x - 5, where x ∈ {1, 2, 3} && y <= 10 || z != 0",
            "expected_gradients": "high formal_symbol_density"
        },
        {
            "name": "High Runtime Expandability",
            "text": "Create a system that generates functions dynamically. Build templates that expand into schemas.",
            "expected_gradients": "high runtime_expandability"
        },
        {
            "name": "High Referential Closure",
            "text": "The system reflects on itself recursively. This meta-structure creates feedback loops in the manifold topology.",
            "expected_gradients": "high referential_closure"
        },
        {
            "name": "Mixed High Gradients",
            "text": "def generate_recursive_system(): # Create self-referential meta-structure\n    return lambda x: f(x) if x > 0 else recursive_call(x-1)",
            "expected_gradients": "high across multiple dimensions"
        }
    ]
    
    print("🔥 Testing Affordance Gradient System")
    print("=" * 60)
    
    for i, test_case in enumerate(test_inputs, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Input: {test_case['text'][:50]}...")
        print(f"   Expected: {test_case['expected_gradients']}")
        
        try:
            # Send request to backend
            response = requests.post(f"{base_url}/process", 
                                   json={"text": test_case['text']},
                                   timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Look for affordance gradient information in the response
                if 'affordance_gradients' in result:
                    gradients = result['affordance_gradients']
                    print(f"   ✅ Gradients detected:")
                    print(f"      • Executability: {gradients.get('executability_pressure', 0):.4f}")
                    print(f"      • Formal symbols: {gradients.get('formal_symbol_density', 0):.4f}")
                    print(f"      • Expandability: {gradients.get('runtime_expandability', 0):.4f}")
                    print(f"      • Closure: {gradients.get('referential_closure', 0):.4f}")
                    print(f"      • Forcing: {gradients.get('constraint_forcing_gradient', 0):.4f}")
                    
                    # Check if constraint pressure was applied
                    forcing_gradient = gradients.get('constraint_forcing_gradient', 0)
                    if forcing_gradient > 0.1:
                        print(f"   🔥 CONSTRAINT PRESSURE APPLIED (gradient: {forcing_gradient:.4f})")
                    else:
                        print(f"   ❄️  No constraint pressure (gradient: {forcing_gradient:.4f})")
                else:
                    print(f"   ⚠️  No affordance gradient data in response")
                
                # Show response snippet
                response_text = result.get('response', 'No response')
                print(f"   Response: {response_text[:60]}...")
                
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Affordance Gradient Test Complete!")

if __name__ == "__main__":
    test_affordance_system()