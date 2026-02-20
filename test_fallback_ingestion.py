#!/usr/bin/env python3
"""
Test Fallback Ingestion System

Tests the synthetic data generation when datasets library is not available.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
from src.data.conversational_api_ingestor import ConversationalAPIIngestor

def test_fallback_ingestion():
    """Test the fallback ingestion system."""
    print("=" * 60)
    print("TESTING FALLBACK INGESTION SYSTEM")
    print("=" * 60)
    
    # Force CPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Get token from user or environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("\nNo HF_TOKEN found in environment")
        hf_token = input("Enter your HF token (or press Enter to use synthetic data only): ").strip()
        
        if hf_token:
            # Set in environment for this session
            os.environ['HF_TOKEN'] = hf_token
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            print(f"Token set: {hf_token[:10]}...{hf_token[-5:]}")
        else:
            print("No token provided - will use synthetic data generation")
            hf_token = "dummy_token"  # Will use synthetic data anyway
    
    try:
        # Create API ingestor
        print("\n1. Creating API ingestor...")
        api_ingestor = ConversationalAPIIngestor(device=device)
        
        # Set the token on the HF ingestor if we have one
        if hf_token and hf_token != "dummy_token":
            api_ingestor.hf_ingestor.hf_token = hf_token
        
        print("   ✓ API ingestor created successfully")
        
        # Test synthetic LMSYS data generation
        print("\n2. Testing synthetic LMSYS data generation...")
        lmsys_samples = api_ingestor.hf_ingestor._generate_synthetic_lmsys_data(5)
        
        if lmsys_samples:
            print(f"   ✓ Generated {len(lmsys_samples)} LMSYS samples")
            
            # Show first sample
            sample = lmsys_samples[0]
            print(f"   Sample conversation ID: {sample.get('conversation_id', 'N/A')}")
            print(f"   Sample turns: {len(sample.get('conversation', []))}")
            
            for i, turn in enumerate(sample.get('conversation', [])[:2]):
                print(f"     Turn {i+1} ({turn.get('role', 'unknown')}): {turn.get('content', '')[:50]}...")
        
        # Test full ingestion with fallback
        print("\n3. Testing full ingestion with fallback...")
        conversations = api_ingestor.ingest_huggingface_dataset('lmsys/lmsys-chat-1m', max_samples=3)
        
        if conversations:
            print(f"   ✓ Ingested {len(conversations)} conversations")
            
            # Get summary
            summary = api_ingestor.get_ingestion_summary(conversations)
            print(f"\n4. Ingestion Summary:")
            print(f"   Total conversations: {summary['total_conversations']}")
            print(f"   Total turns: {summary['total_turns']}")
            print(f"   Avg turns/conversation: {summary['avg_turns_per_conversation']:.2f}")
            print(f"   Avg text length/turn: {summary['avg_text_length_per_turn']:.1f}")
            
            # Check affordance gradients
            if 'affordance_gradient_stats' in summary:
                print(f"\n5. Affordance Gradient Statistics:")
                for gradient_type, stats in summary['affordance_gradient_stats'].items():
                    print(f"   {gradient_type}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            
            # Show first conversation details
            if len(conversations) > 0:
                conv = conversations[0]
                print(f"\n6. First Conversation Details:")
                print(f"   ID: {conv.conversation_id}")
                print(f"   Turns: {len(conv.turns)}")
                print(f"   Source: {conv.source}")
                
                if len(conv.turns) > 0:
                    turn = conv.turns[0]
                    print(f"   First turn: {turn.text[:100]}...")
            
            print(f"\n" + "=" * 60)
            print("SUCCESS: Fallback ingestion works perfectly!")
            print("=" * 60)
            return True
            
        else:
            print("   ✗ No conversations generated")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during fallback ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Testing fallback ingestion system...")
    print("This works WITHOUT the datasets library!")
    
    success = test_fallback_ingestion()
    
    if success:
        print("\n🎉 Fallback ingestion works!")
        print("You can now use the GUI even without the datasets library.")
        print("It will generate synthetic conversational data for training.")
    else:
        print("\n❌ Fallback ingestion failed")
        print("Check the error messages above.")

if __name__ == "__main__":
    main()
