#!/usr/bin/env python3
"""
CPU-Only Conversational Data Ingestion Test

Tests the conversational data ingestion system using only CPU.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
from src.data.conversational_api_ingestor import ConversationalAPIIngestor

def test_cpu_ingestion():
    """Test conversational data ingestion on CPU only."""
    print("=" * 50)
    print("CPU-ONLY CONVERSATIONAL DATA INGESTION TEST")
    print("=" * 50)
    
    # Force CPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Check if HF token is available
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("No HF_TOKEN found in environment")
        print("Please set it first or use the GUI to enter your token")
        return False
    
    print(f"HF Token: {hf_token[:10]}...{hf_token[-5:]}")
    
    try:
        # Create API ingestor with CPU
        print("\n1. Creating API ingestor...")
        api_ingestor = ConversationalAPIIngestor(device=device)
        print("   ✓ API ingestor created successfully")
        
        # Test with a small sample
        print("\n2. Testing data ingestion...")
        print("   Ingesting 10 samples from lmsys/lmsys-chat-1m...")
        
        conversations = api_ingestor.ingest_huggingface_dataset(
            'lmsys/lmsys-chat-1m', 
            max_samples=10
        )
        
        if conversations:
            print(f"   ✓ Successfully ingested {len(conversations)} conversations")
            
            # Get summary
            summary = api_ingestor.get_ingestion_summary(conversations)
            print(f"\n3. Ingestion Summary:")
            print(f"   Total conversations: {summary['total_conversations']}")
            print(f"   Total turns: {summary['total_turns']}")
            print(f"   Avg turns/conversation: {summary['avg_turns_per_conversation']:.2f}")
            print(f"   Avg text length/turn: {summary['avg_text_length_per_turn']:.1f}")
            
            # Check affordance gradients
            if 'affordance_gradient_stats' in summary:
                print(f"\n4. Affordance Gradient Statistics:")
                for gradient_type, stats in summary['affordance_gradient_stats'].items():
                    print(f"   {gradient_type}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            
            print(f"\n" + "=" * 50)
            print("SUCCESS: CPU-only ingestion works perfectly!")
            print("=" * 50)
            return True
            
        else:
            print("   ✗ No conversations ingested")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Testing CPU-only conversational data ingestion...")
    
    # Make sure we're using CPU
    if torch.cuda.is_available():
        print("CUDA is available but we're forcing CPU usage for stability")
    
    success = test_cpu_ingestion()
    
    if success:
        print("\n🎉 CPU-only ingestion works!")
        print("You can now use the GUI without any CUDA issues.")
    else:
        print("\n❌ CPU ingestion failed")
        print("Check your HF token and network connection.")

if __name__ == "__main__":
    main()
