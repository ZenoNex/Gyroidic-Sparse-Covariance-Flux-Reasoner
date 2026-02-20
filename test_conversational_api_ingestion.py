#!/usr/bin/env python3
"""
Test Conversational API Ingestion System

This script tests the conversational data ingestion from various APIs
and demonstrates integration with the affordance gradient system.

Author: William Matthew Bryant
Created: January 2026
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import json
from pathlib import Path
from typing import Dict, List, Any

# Import the conversational ingestor
from src.data.conversational_api_ingestor import (
    ConversationalAPIIngestor,
    HuggingFaceConversationalIngestor,
    ConversationalDataProcessor,
    Conversation,
    ConversationTurn
)


def test_basic_functionality():
    """Test basic functionality of the conversational ingestor."""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create ingestor
    ingestor = ConversationalAPIIngestor(device=device)
    print("✅ Created conversational API ingestor")
    
    # Test processor
    processor = ConversationalDataProcessor(device=device)
    print("✅ Created conversational data processor")
    
    # Test affordance gradient computation
    test_texts = [
        "Hello, how are you today?",
        "What is machine learning and how does it work?",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Search for the latest research on quantum computing",
        "Can you explain the difference between supervised and unsupervised learning?"
    ]
    
    print(f"\n🔍 Testing Affordance Gradient Computation:")
    for i, text in enumerate(test_texts):
        gradients = processor.compute_affordance_gradients(text)
        print(f"   Text {i+1}: '{text[:50]}...'")
        print(f"      Conversational: {gradients['conversational']:.3f}")
        print(f"      API extraction: {gradients['api_extraction']:.3f}")
        print(f"      Executability: {gradients['executability']:.3f}")
        print(f"      Formal symbols: {gradients['formal_symbols']:.3f}")
    
    return ingestor, processor


def test_synthetic_conversation_processing():
    """Test processing of synthetic conversations."""
    print("\n🧪 Testing Synthetic Conversation Processing")
    print("=" * 45)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    processor = ConversationalDataProcessor(device=device)
    
    # Create synthetic conversations
    synthetic_conversations = [
        Conversation(
            conversation_id="synthetic_1",
            turns=[
                ConversationTurn(
                    speaker_id="user",
                    text="Hello! Can you help me understand neural networks?"
                ),
                ConversationTurn(
                    speaker_id="assistant",
                    text="Of course! Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes called neurons."
                ),
                ConversationTurn(
                    speaker_id="user",
                    text="That's interesting. How do they learn from data?"
                ),
                ConversationTurn(
                    speaker_id="assistant",
                    text="Neural networks learn through a process called backpropagation, where they adjust their weights based on the error between predicted and actual outputs."
                )
            ],
            context={"topic": "machine_learning", "difficulty": "beginner"},
            source="synthetic",
            labels={"educational": True, "technical": True}
        ),
        Conversation(
            conversation_id="synthetic_2",
            turns=[
                ConversationTurn(
                    speaker_id="user",
                    text="def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])"
                ),
                ConversationTurn(
                    speaker_id="assistant",
                    text="That's a nice implementation of quicksort! It uses list comprehensions for a clean, functional style."
                ),
                ConversationTurn(
                    speaker_id="user",
                    text="Thanks! Can you explain the time complexity?"
                ),
                ConversationTurn(
                    speaker_id="assistant",
                    text="The average time complexity is O(n log n), but worst case is O(n²) when the pivot is always the smallest or largest element."
                )
            ],
            context={"topic": "algorithms", "language": "python"},
            source="synthetic",
            labels={"code": True, "educational": True}
        )
    ]
    
    # Process conversations
    processed = processor.process_conversations(synthetic_conversations)
    
    print(f"✅ Processed {len(processed)} synthetic conversations")
    
    # Analyze results
    for i, conv in enumerate(processed):
        print(f"\n📊 Conversation {i+1} Analysis:")
        print(f"   ID: {conv.conversation_id}")
        print(f"   Turns: {len(conv.turns)}")
        print(f"   Source: {conv.source}")
        print(f"   Pressure signature shape: {conv.pressure_signature.shape if conv.pressure_signature is not None else 'None'}")
        
        # Show turn analysis
        for j, turn in enumerate(conv.turns[:2]):  # First 2 turns
            print(f"   Turn {j+1} ({turn.speaker_id}):")
            print(f"      Text: '{turn.text[:60]}...'")
            print(f"      Embedding shape: {turn.embedding.shape if turn.embedding is not None else 'None'}")
            if turn.affordance_gradients:
                print(f"      Top gradients: {sorted(turn.affordance_gradients.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    return processed


def test_huggingface_integration():
    """Test Hugging Face dataset integration (if available)."""
    print("\n🧪 Testing Hugging Face Integration")
    print("=" * 35)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Try to import datasets library
        import datasets
        print("✅ Datasets library available")
        
        # Create HF ingestor
        hf_ingestor = HuggingFaceConversationalIngestor(device=device)
        
        # List some conversational datasets (this might fail without internet)
        try:
            datasets_list = hf_ingestor.list_conversational_datasets()
            print(f"✅ Found {len(datasets_list)} conversational datasets")
            
            # Show first few
            for i, dataset in enumerate(datasets_list[:3]):
                print(f"   {i+1}. {dataset.get('id', 'Unknown')}")
                
        except Exception as e:
            print(f"⚠️ Could not list datasets (offline?): {e}")
        
        # Test dataset info retrieval
        try:
            info = hf_ingestor.get_dataset_info('lmsys/lmsys-chat-1m')
            if info:
                print(f"✅ Retrieved info for lmsys/lmsys-chat-1m")
                print(f"   Downloads: {info.get('downloads', 'Unknown')}")
            else:
                print("⚠️ No info retrieved (might be offline)")
                
        except Exception as e:
            print(f"⚠️ Could not get dataset info: {e}")
        
    except ImportError:
        print("⚠️ Datasets library not available. Install with: pip install datasets")
        print("   Skipping Hugging Face integration test")


def test_affordance_integration():
    """Test integration with affordance gradient system."""
    print("\n🧪 Testing Affordance Integration")
    print("=" * 32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    
    # Create test conversations with different affordance patterns
    test_conversations = [
        # High conversational affordance
        Conversation(
            conversation_id="high_conversational",
            turns=[
                ConversationTurn("user", "What do you think about artificial intelligence? How do you feel about its impact on society?"),
                ConversationTurn("assistant", "I believe AI has tremendous potential to help solve complex problems, but we need to be thoughtful about its development.")
            ],
            context={}, source="test"
        ),
        
        # High API extraction affordance
        Conversation(
            conversation_id="high_api",
            turns=[
                ConversationTurn("user", "Search for the latest news about quantum computing research and get me current information"),
                ConversationTurn("assistant", "I'll help you find recent quantum computing developments.")
            ],
            context={}, source="test"
        ),
        
        # High code execution affordance
        Conversation(
            conversation_id="high_code",
            turns=[
                ConversationTurn("user", "def matrix_multiply(A, B): return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]"),
                ConversationTurn("assistant", "That's a clean implementation of matrix multiplication using nested list comprehensions.")
            ],
            context={}, source="test"
        )
    ]
    
    # Process conversations
    processor = ConversationalDataProcessor(device=device)
    processed = processor.process_conversations(test_conversations)
    
    print("📊 Affordance Analysis Results:")
    for conv in processed:
        print(f"\n   Conversation: {conv.conversation_id}")
        for turn in conv.turns:
            if turn.affordance_gradients:
                # Find dominant affordance
                dominant = max(turn.affordance_gradients.items(), key=lambda x: x[1])
                print(f"      {turn.speaker_id}: Dominant affordance = {dominant[0]} ({dominant[1]:.3f})")
                
                # Show all gradients
                gradients_str = ", ".join([f"{k}={v:.3f}" for k, v in turn.affordance_gradients.items()])
                print(f"         All: {gradients_str}")
    
    return processed


def test_pressure_signature_generation():
    """Test pressure signature generation."""
    print("\n🧪 Testing Pressure Signature Generation")
    print("=" * 37)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    processor = ConversationalDataProcessor(device=device)
    
    # Create conversations with different characteristics
    test_cases = [
        ("short_simple", [
            ConversationTurn("user", "Hi"),
            ConversationTurn("assistant", "Hello!")
        ]),
        ("long_complex", [
            ConversationTurn("user", "Can you explain the mathematical foundations of machine learning, including linear algebra, calculus, and probability theory?"),
            ConversationTurn("assistant", "Machine learning relies heavily on linear algebra for data representation and transformations. Vectors and matrices are fundamental..."),
            ConversationTurn("user", "That's very detailed. Can you also explain how gradient descent works in optimization?"),
            ConversationTurn("assistant", "Gradient descent is an iterative optimization algorithm that finds the minimum of a function by following the negative gradient...")
        ]),
        ("technical_code", [
            ConversationTurn("user", "class NeuralNetwork: def __init__(self, layers): self.layers = layers"),
            ConversationTurn("assistant", "Nice start! You'll want to add forward and backward propagation methods."),
            ConversationTurn("user", "def forward(self, x): for layer in self.layers: x = layer(x); return x")
        ])
    ]
    
    print("📊 Pressure Signature Analysis:")
    for case_name, turns in test_cases:
        conv = Conversation(
            conversation_id=case_name,
            turns=turns,
            context={},
            source="test"
        )
        
        # Generate pressure signature
        pressure_sig = processor.generate_pressure_signature(conv)
        
        print(f"\n   Case: {case_name}")
        print(f"      Turns: {len(turns)}")
        print(f"      Total text length: {sum(len(turn.text) for turn in turns)}")
        print(f"      Pressure signature: {pressure_sig.tolist()}")
        print(f"      Signature norm: {torch.norm(pressure_sig).item():.3f}")


def test_caching_system():
    """Test the caching system."""
    print("\n🧪 Testing Caching System")
    print("=" * 25)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    cache_dir = "./test_cache"
    
    # Create ingestor with test cache directory
    ingestor = ConversationalAPIIngestor(device=device, cache_dir=cache_dir)
    
    # Create test conversation
    test_conv = Conversation(
        conversation_id="cache_test",
        turns=[
            ConversationTurn("user", "Test caching system"),
            ConversationTurn("assistant", "This is a test conversation for caching")
        ],
        context={"test": True},
        source="test_cache"
    )
    
    # Process conversation
    processed = ingestor.processor.process_conversations([test_conv])
    
    # Test conversion to/from dict
    conv_dict = ingestor._conversation_to_dict(processed[0])
    restored_conv = ingestor._dict_to_conversation(conv_dict)
    
    print(f"✅ Conversation serialization test:")
    print(f"   Original ID: {processed[0].conversation_id}")
    print(f"   Restored ID: {restored_conv.conversation_id}")
    print(f"   Original turns: {len(processed[0].turns)}")
    print(f"   Restored turns: {len(restored_conv.turns)}")
    print(f"   Embeddings preserved: {restored_conv.turns[0].embedding is not None}")
    print(f"   Gradients preserved: {restored_conv.turns[0].affordance_gradients is not None}")
    
    # Clean up test cache
    import shutil
    if Path(cache_dir).exists():
        shutil.rmtree(cache_dir)
        print(f"🧹 Cleaned up test cache directory")


def run_comprehensive_test():
    """Run comprehensive test of the conversational API ingestion system."""
    print("🗣️ Conversational API Ingestion System - Comprehensive Test")
    print("=" * 65)
    
    try:
        # Test 1: Basic functionality
        ingestor, processor = test_basic_functionality()
        
        # Test 2: Synthetic conversation processing
        synthetic_processed = test_synthetic_conversation_processing()
        
        # Test 3: Hugging Face integration
        test_huggingface_integration()
        
        # Test 4: Affordance integration
        affordance_processed = test_affordance_integration()
        
        # Test 5: Pressure signature generation
        test_pressure_signature_generation()
        
        # Test 6: Caching system
        test_caching_system()
        
        # Summary
        print(f"\n🎯 Test Summary")
        print("=" * 15)
        print(f"✅ Basic functionality: PASSED")
        print(f"✅ Synthetic processing: PASSED ({len(synthetic_processed)} conversations)")
        print(f"✅ Affordance integration: PASSED ({len(affordance_processed)} conversations)")
        print(f"✅ Pressure signatures: PASSED")
        print(f"✅ Caching system: PASSED")
        
        # Get summary of all processed conversations
        all_conversations = synthetic_processed + affordance_processed
        summary = ingestor.get_ingestion_summary(all_conversations)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total conversations processed: {summary['total_conversations']}")
        print(f"   Total turns: {summary['total_turns']}")
        print(f"   Average turns per conversation: {summary['avg_turns_per_conversation']:.2f}")
        print(f"   Average text length per turn: {summary['avg_text_length_per_turn']:.1f}")
        
        if 'affordance_gradient_stats' in summary:
            print(f"\n🎯 Affordance Gradient Statistics:")
            for gradient_type, stats in summary['affordance_gradient_stats'].items():
                print(f"   {gradient_type}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        print(f"\n✅ All tests completed successfully!")
        print(f"\nNext steps:")
        print(f"• Set up API credentials for Reddit/Hugging Face")
        print(f"• Test with real conversational datasets")
        print(f"• Integrate with diegetic terminal backend")
        print(f"• Connect to temporal association training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
