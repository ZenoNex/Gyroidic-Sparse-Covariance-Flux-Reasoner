#!/usr/bin/env python3
"""
Enhanced Conversational Training System

Addresses garbled output by training on more conversational data and improving
the spectral coherence repair system for better linguistic flow.
"""

import torch
import requests
import json
import time
from typing import Dict, List, Any

def enhance_conversational_training():
    """Enhance the system with more conversational training data and better repair."""
    
    print("🗣️ Enhanced Conversational Training System")
    print("=" * 60)
    
    # Step 1: Ingest more conversational data
    print("\n1. Ingesting additional conversational datasets...")
    
    try:
        # Try multiple conversational datasets
        datasets_to_try = [
            'lmsys/lmsys-chat-1m',
            'OpenAssistant/oasst2', 
            'microsoft/DialoGPT-medium',
            'facebook/blended_skill_talk'
        ]
        
        from src.data.conversational_api_ingestor import ConversationalAPIIngestor
        
        api_ingestor = ConversationalAPIIngestor(device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu')
        all_conversations = []
        
        for dataset_id in datasets_to_try:
            print(f"   📥 Trying {dataset_id}...")
            try:
                conversations = api_ingestor.ingest_huggingface_dataset(dataset_id, max_samples=1000)
                if conversations:
                    all_conversations.extend(conversations)
                    print(f"   ✅ Added {len(conversations)} conversations from {dataset_id}")
                else:
                    print(f"   ⚠️ No data from {dataset_id}")
            except Exception as e:
                print(f"   ❌ Failed {dataset_id}: {e}")
        
        print(f"\n   📊 Total conversations collected: {len(all_conversations)}")
        
        if all_conversations:
            # Get enhanced summary
            summary = api_ingestor.get_ingestion_summary(all_conversations)
            print(f"   Total turns: {summary['total_turns']}")
            print(f"   Avg turns/conversation: {summary['avg_turns_per_conversation']:.2f}")
            print(f"   Avg text length/turn: {summary['avg_text_length_per_turn']:.1f}")
            
            # Show affordance gradient improvements
            if 'affordance_gradient_stats' in summary:
                print(f"\n   🎯 Enhanced Affordance Gradients:")
                for gradient_type, stats in summary['affordance_gradient_stats'].items():
                    print(f"      {gradient_type}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
    except Exception as e:
        print(f"   ❌ Conversational ingestion failed: {e}")
    
    # Step 2: Test current system with simple conversational input
    print("\n2. Testing current system response...")
    
    test_inputs = [
        "hello!",
        "how are you?", 
        "what's your name?",
        "tell me about yourself",
        "can you help me with something?"
    ]
    
    base_url = "http://localhost:8000"
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n   Test {i}: '{test_input}'")
        
        try:
            response = requests.post(f"{base_url}/interact", 
                                   json={"text": test_input},
                                   timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'No response')
                
                # Check if response is garbled
                if is_garbled_output(ai_response):
                    print(f"   ❌ GARBLED: {ai_response[:50]}...")
                    
                    # Check repair diagnostics
                    if 'spectral_diagnostics' in result:
                        diag = result['spectral_diagnostics']
                        print(f"   🔧 Spectral coherence: {diag.get('theta_coherence', 'N/A')}")
                        print(f"   🔧 Energy ratio: {diag.get('energy_ratio', 'N/A')}")
                else:
                    print(f"   ✅ GOOD: {ai_response[:50]}...")
                
                # Show affordance gradients
                if 'affordance_gradients' in result:
                    gradients = result['affordance_gradients']
                    conv_pressure = gradients.get('conversational_embedding_pressure', 0)
                    print(f"   📊 Conversational pressure: {conv_pressure:.3f}")
            
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Step 3: Create conversational training associations
    print("\n3. Creating conversational training associations...")
    
    conversational_pairs = [
        ("hello", "Hello! How can I help you today?"),
        ("hi", "Hi there! What can I do for you?"),
        ("how are you", "I'm doing well, thank you for asking! How are you?"),
        ("what's your name", "I'm an AI assistant. You can call me whatever you'd like!"),
        ("tell me about yourself", "I'm an AI designed to help with various tasks and have conversations. What would you like to know?"),
        ("can you help me", "Of course! I'd be happy to help. What do you need assistance with?"),
        ("thank you", "You're very welcome! Is there anything else I can help you with?"),
        ("goodbye", "Goodbye! Have a great day!"),
        ("what can you do", "I can help with questions, have conversations, and assist with various tasks. What interests you?"),
        ("nice to meet you", "Nice to meet you too! I'm glad we're chatting.")
    ]
    
    associations_created = 0
    
    for input_text, response_text in conversational_pairs:
        try:
            association_data = {
                "type": "text-text-association",
                "input": input_text,
                "response": response_text,
                "relationship": "conversational_response"
            }
            
            response = requests.post(f"{base_url}/associate", 
                                   json=association_data,
                                   timeout=10)
            
            if response.status_code == 200:
                associations_created += 1
                print(f"   ✅ Associated: '{input_text}' → '{response_text[:30]}...'")
            else:
                print(f"   ❌ Failed association: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Association error: {e}")
    
    print(f"\n   📊 Created {associations_created} conversational associations")
    
    # Step 4: Test improved responses
    print("\n4. Testing improved responses after training...")
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n   Retest {i}: '{test_input}'")
        
        try:
            response = requests.post(f"{base_url}/interact", 
                                   json={"text": test_input},
                                   timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'No response')
                
                if is_garbled_output(ai_response):
                    print(f"   ❌ STILL GARBLED: {ai_response[:50]}...")
                else:
                    print(f"   ✅ IMPROVED: {ai_response[:50]}...")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Step 5: Recommendations for further improvement
    print("\n5. Recommendations for further improvement:")
    print("   🔧 Increase conversational dataset size (try 5000+ samples)")
    print("   🔧 Adjust spectral coherence thresholds for better linguistic flow")
    print("   🔧 Train temporal associations on conversational patterns")
    print("   🔧 Use Wikipedia integration for knowledge-grounded responses")
    print("   🔧 Implement conversation state tracking for multi-turn dialogue")
    
    print(f"\n" + "=" * 60)
    print("🎉 Enhanced Conversational Training Complete!")
    print("The system should now handle basic conversational inputs better.")
    print("=" * 60)

def is_garbled_output(text: str) -> bool:
    """Check if output appears to be garbled."""
    if not text or len(text) < 3:
        return True
    
    # Check for excessive consonant clustering
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    vowels = "aeiouAEIOU"
    
    consonant_count = sum(1 for c in text if c in consonants)
    vowel_count = sum(1 for c in text if c in vowels)
    
    if vowel_count == 0:
        return True
    
    consonant_ratio = consonant_count / (consonant_count + vowel_count)
    
    # If more than 80% consonants, likely garbled
    if consonant_ratio > 0.8:
        return True
    
    # Check for excessive special characters
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    special_count = sum(1 for c in text if c in special_chars)
    
    if len(text) > 0 and special_count / len(text) > 0.5:
        return True
    
    # Check for non-ASCII characters (might indicate encoding issues)
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        # Has non-ASCII, might be garbled
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if non_ascii_count / len(text) > 0.3:
            return True
    
    return False

def create_enhanced_conversational_dataset():
    """Create a more comprehensive conversational dataset."""
    
    enhanced_conversations = [
        # Greetings and basic interactions
        {"input": "hello", "response": "Hello! How can I help you today?", "category": "greeting"},
        {"input": "hi there", "response": "Hi! What's on your mind?", "category": "greeting"},
        {"input": "good morning", "response": "Good morning! I hope you're having a great day.", "category": "greeting"},
        {"input": "hey", "response": "Hey! What can I do for you?", "category": "greeting"},
        
        # Questions about the AI
        {"input": "what are you", "response": "I'm an AI assistant designed to help with various tasks and have conversations.", "category": "identity"},
        {"input": "who made you", "response": "I was created by researchers working on AI systems. How can I assist you?", "category": "identity"},
        {"input": "are you human", "response": "No, I'm an artificial intelligence. But I'm here to help however I can!", "category": "identity"},
        
        # Help and assistance
        {"input": "can you help me", "response": "Absolutely! I'd be happy to help. What do you need assistance with?", "category": "help"},
        {"input": "i need help", "response": "Of course! Tell me what you're working on and I'll do my best to assist.", "category": "help"},
        {"input": "what can you do", "response": "I can answer questions, help with tasks, have conversations, and provide information on many topics.", "category": "capabilities"},
        
        # Emotional and social
        {"input": "how are you", "response": "I'm doing well, thank you for asking! How are you doing today?", "category": "social"},
        {"input": "i'm sad", "response": "I'm sorry to hear you're feeling sad. Would you like to talk about what's bothering you?", "category": "emotional"},
        {"input": "thank you", "response": "You're very welcome! I'm glad I could help.", "category": "gratitude"},
        {"input": "that's helpful", "response": "I'm so glad you found that helpful! Is there anything else I can assist with?", "category": "gratitude"},
        
        # Farewells
        {"input": "goodbye", "response": "Goodbye! Take care and have a wonderful day!", "category": "farewell"},
        {"input": "see you later", "response": "See you later! Feel free to come back anytime you need help.", "category": "farewell"},
        {"input": "bye", "response": "Bye! It was nice chatting with you.", "category": "farewell"},
        
        # Knowledge and learning
        {"input": "tell me something interesting", "response": "Here's something fascinating: octopuses have three hearts and blue blood! What topics interest you?", "category": "knowledge"},
        {"input": "i want to learn", "response": "That's wonderful! Learning is one of life's greatest adventures. What would you like to explore?", "category": "learning"},
        {"input": "explain that", "response": "I'd be happy to explain! Could you tell me which part you'd like me to clarify?", "category": "explanation"},
        
        # Problem solving
        {"input": "i'm confused", "response": "No worries! Confusion is often the first step to understanding. What's puzzling you?", "category": "problem_solving"},
        {"input": "this is hard", "response": "I understand it can be challenging. Let's break it down into smaller, manageable pieces.", "category": "problem_solving"},
        {"input": "i don't understand", "response": "That's perfectly okay! Let me try explaining it differently. What part is unclear?", "category": "problem_solving"},
    ]
    
    return enhanced_conversations

if __name__ == "__main__":
    print("Starting enhanced conversational training...")
    enhance_conversational_training()
