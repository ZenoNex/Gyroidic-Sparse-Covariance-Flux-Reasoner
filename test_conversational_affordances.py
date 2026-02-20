#!/usr/bin/env python3
"""
Test conversational affordance gradients and embedding extraction.
"""

import sys
import importlib

# Clear any cached modules
if 'src.ui.diegetic_backend' in sys.modules:
    del sys.modules['src.ui.diegetic_backend']

sys.path.append('.')

# Fresh import
from src.ui.diegetic_backend import DiegeticPhysicsEngine

def test_conversational_affordances():
    """Test enhanced affordance gradient system with conversational inputs."""
    
    print("🔧 Initializing backend for conversational affordance testing...")
    engine = DiegeticPhysicsEngine()
    
    # Test inputs covering different affordance types
    test_inputs = [
        # Code (should trigger executability + formal symbols)
        "def function(): return x + y",
        
        # Conversational (should trigger conversational embedding pressure)
        "What is the meaning of life? Can you explain how consciousness works?",
        
        # API extraction (should trigger API extraction potential)
        "Search Wikipedia for information about quantum physics and extract the latest research data",
        
        # Mixed conversational + API
        "Tell me about machine learning. Get the current information from online sources and explain it",
        
        # Knowledge seeking
        "I want to learn about neural networks. What are the key concepts I should understand?",
        
        # Pure conversation
        "Hello, how are you today? What do you think about artificial intelligence?",
        
        # Temporal/current info seeking
        "What are the latest developments in AI? Get me the most recent news and research"
    ]
    
    print("\n🔥 Testing Enhanced Affordance Gradients:")
    print("=" * 80)
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n{i}. Input: {text}")
        print("-" * 60)
        
        try:
            # Create input tensor
            input_tensor = engine._text_to_tensor(text)
            
            # Compute affordance gradients
            gradients = engine._compute_affordance_gradients(text, input_tensor)
            
            print(f"   Executability:     {gradients['executability_pressure']:.4f}")
            print(f"   Formal symbols:    {gradients['formal_symbol_density']:.4f}")
            print(f"   Expandability:     {gradients['runtime_expandability']:.4f}")
            print(f"   Closure:           {gradients['referential_closure']:.4f}")
            print(f"   Conversational:    {gradients['conversational_embedding_pressure']:.4f}")
            print(f"   API extraction:    {gradients['api_extraction_potential']:.4f}")
            print(f"   Constraint force:  {gradients['constraint_forcing_gradient']:.4f}")
            
            # Test conversational embedding extraction
            if gradients['conversational_embedding_pressure'] > 0.05 or gradients['api_extraction_potential'] > 0.05:
                print(f"\n   🔥 CONVERSATIONAL EXTRACTION TRIGGERED")
                conv_results = engine._extract_conversational_embeddings(text, gradients)
                print(f"   • Patterns detected: {len(conv_results.get('temporal_patterns_detected', []))}")
                print(f"   • Associations created: {conv_results.get('associations_created', 0)}")
                print(f"   • Constraint pressure: {conv_results.get('constraint_pressure_generated', 0.0):.4f}")
                
                if conv_results.get('api_content_extracted'):
                    print(f"   • API content extracted: {conv_results['api_content_extracted']['source']}")
            
            # Overall constraint forcing assessment
            if gradients['constraint_forcing_gradient'] > 0.1:
                print(f"   🔥 HIGH CONSTRAINT PRESSURE - System will inject constraints")
            elif gradients['constraint_forcing_gradient'] > 0.05:
                print(f"   ⚡ MEDIUM CONSTRAINT PRESSURE - Some constraint injection")
            else:
                print(f"   ❄️  LOW CONSTRAINT PRESSURE - Minimal constraint injection")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Conversational affordance test complete!")
    print("\n📊 Summary:")
    print("The enhanced affordance gradient system now detects:")
    print("• Code execution patterns (legacy)")
    print("• Conversational embedding opportunities (new)")
    print("• API extraction potential (new)")
    print("• Mixed constraint forcing scenarios (enhanced)")

def test_full_conversational_processing():
    """Test full processing pipeline with conversational input."""
    
    print("\n🔧 Testing Full Conversational Processing Pipeline...")
    engine = DiegeticPhysicsEngine()
    
    conversational_input = "What is machine learning? Can you explain the key concepts and get me some current research information?"
    
    print(f"\nProcessing: {conversational_input}")
    print("=" * 80)
    
    try:
        # Process through full pipeline
        result = engine.process_input(conversational_input)
        
        print(f"\n✅ Processing complete!")
        print(f"Response: {result.get('response', 'No response generated')}")
        print(f"Iteration: {result.get('iteration', 'Unknown')}")
        
        # Check for conversational processing indicators
        if 'phase3_diagnostics' in result:
            print(f"Phase 3 diagnostics available: {result['phase3_diagnostics']}")
        
        if 'phase4_diagnostics' in result:
            print(f"Phase 4 diagnostics available: {result['phase4_diagnostics']}")
            
    except Exception as e:
        print(f"❌ Full processing error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversational_affordances()
    test_full_conversational_processing()