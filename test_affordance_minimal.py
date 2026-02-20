#!/usr/bin/env python3
"""
Minimal test of affordance gradient system.
"""

import sys
sys.path.append('.')

from src.ui.diegetic_backend import DiegeticPhysicsEngine

def test_affordance_gradients():
    """Test affordance gradient computation directly."""
    
    print("🔧 Initializing backend...")
    engine = DiegeticPhysicsEngine()
    
    test_inputs = [
        "Hello world",
        "def function(): return x + y",
        "Execute this algorithm and compute the result",
        "f(x) = x^2 + 3x - 5 where x ∈ {1,2,3}",
        "Create a recursive meta-structure that reflects on itself"
    ]
    
    print("\n🔥 Testing Affordance Gradients:")
    print("=" * 50)
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n{i}. Input: {text}")
        
        try:
            # Create input tensor
            input_tensor = engine._text_to_tensor(text)
            
            # Compute affordance gradients
            gradients = engine._compute_affordance_gradients(text, input_tensor)
            
            print(f"   Executability: {gradients['executability_pressure']:.4f}")
            print(f"   Formal symbols: {gradients['formal_symbol_density']:.4f}")
            print(f"   Expandability: {gradients['runtime_expandability']:.4f}")
            print(f"   Closure: {gradients['referential_closure']:.4f}")
            print(f"   Forcing: {gradients['constraint_forcing_gradient']:.4f}")
            
            if gradients['constraint_forcing_gradient'] > 0.1:
                print(f"   🔥 CONSTRAINT PRESSURE TRIGGERED")
            else:
                print(f"   ❄️  No constraint pressure")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Affordance gradient test complete!")

if __name__ == "__main__":
    test_affordance_gradients()