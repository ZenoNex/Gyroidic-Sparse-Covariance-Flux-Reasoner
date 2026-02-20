#!/usr/bin/env python3
"""
Debug Ingestion Issue

Detailed debugging for the "No conversations ingested" error.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
import requests
from src.data.conversational_api_ingestor import ConversationalAPIIngestor

def debug_step_by_step():
    """Debug the ingestion process step by step."""
    print("=" * 60)
    print("DETAILED INGESTION DEBUGGING")
    print("=" * 60)
    
    # Step 1: Check environment
    print("1. Environment Check:")
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"   ✓ HF_TOKEN found: {hf_token[:10]}...{hf_token[-5:]}")
    else:
        print("   ✗ HF_TOKEN not found in environment")
        hf_token = input("   Enter your HF token: ").strip()
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        else:
            print("   ✗ No token provided, cannot continue")
            return
    
    # Step 2: Test token validity
    print("\n2. Token Validation:")
    try:
        headers = {'Authorization': f'Bearer {hf_token}'}
        response = requests.get('https://huggingface.co/api/whoami', 
                              headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Token valid, user: {data.get('name', 'Unknown')}")
        else:
            print(f"   ✗ Token invalid, status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return
    except Exception as e:
        print(f"   ✗ Token validation error: {e}")
        return
    
    # Step 3: Test dataset access
    print("\n3. Dataset Access Test:")
    try:
        dataset_response = requests.get('https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m', 
                                      headers=headers, timeout=15)
        
        print(f"   Dataset API status: {dataset_response.status_code}")
        
        if dataset_response.status_code == 200:
            dataset_info = dataset_response.json()
            print(f"   ✓ Dataset accessible")
            print(f"   Downloads: {dataset_info.get('downloads', 0):,}")
            print(f"   Private: {dataset_info.get('private', False)}")
            print(f"   Gated: {dataset_info.get('gated', False)}")
        elif dataset_response.status_code == 403:
            print("   ⚠ Dataset gated - need to accept agreement")
            print("   Go to: https://huggingface.co/datasets/lmsys/lmsys-chat-1m")
            return
        else:
            print(f"   ✗ Dataset access failed: {dataset_response.text[:200]}")
            return
            
    except Exception as e:
        print(f"   ✗ Dataset access error: {e}")
        return
    
    # Step 4: Test datasets library
    print("\n4. Datasets Library Test:")
    try:
        from datasets import load_dataset
        print("   ✓ datasets library imported")
        
        # Try to load just the config
        print("   Testing dataset loading...")
        
        # Use streaming to avoid downloading everything
        dataset = load_dataset('lmsys/lmsys-chat-1m', 
                             split='train', 
                             streaming=True,
                             use_auth_token=hf_token)
        
        print("   ✓ Dataset loaded in streaming mode")
        
        # Try to get first few examples
        print("   Getting first example...")
        iterator = iter(dataset)
        first_example = next(iterator)
        
        print(f"   ✓ First example keys: {list(first_example.keys())}")
        print(f"   Example conversation_id: {first_example.get('conversation_id', 'N/A')}")
        
    except ImportError:
        print("   ✗ datasets library not installed")
        print("   Run: pip install datasets")
        return
    except Exception as e:
        print(f"   ✗ Dataset loading error: {e}")
        print("   This might be the root cause!")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Test our ingestor
    print("\n5. Our Ingestor Test:")
    try:
        print("   Creating ConversationalAPIIngestor...")
        api_ingestor = ConversationalAPIIngestor(device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu')
        print("   ✓ Ingestor created")
        
        print("   Testing ingest_huggingface_dataset method...")
        
        # Try with very small sample
        conversations = api_ingestor.ingest_huggingface_dataset(
            'lmsys/lmsys-chat-1m', 
            max_samples=2  # Very small test
        )
        
        if conversations:
            print(f"   ✓ SUCCESS: Ingested {len(conversations)} conversations")
            
            # Show details of first conversation
            if len(conversations) > 0:
                conv = conversations[0]
                print(f"   First conversation:")
                print(f"     ID: {conv.conversation_id}")
                print(f"     Turns: {len(conv.turns)}")
                print(f"     Source: {conv.source}")
                
                if len(conv.turns) > 0:
                    turn = conv.turns[0]
                    print(f"     First turn text: {turn.text[:100]}...")
        else:
            print("   ✗ No conversations returned - this is the problem!")
            
    except Exception as e:
        print(f"   ✗ Ingestor error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n" + "=" * 60)
    print("DEBUGGING COMPLETE")
    print("=" * 60)

def main():
    """Main debugging function."""
    print("Debugging the 'No conversations ingested' error...")
    debug_step_by_step()

if __name__ == "__main__":
    main()
