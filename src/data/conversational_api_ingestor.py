"""
Conversational API Data Ingestor

Implements programmatic ingestion of conversational data from various APIs
and sources, following the non-lobotomy architecture principles.

Key Features:
- Hugging Face Hub API integration
- Reddit API conversational thread extraction
- ConvoKit integration for labeled dialogues
- Structured conversation preprocessing
- Integration with affordance gradient system
- Pressure-based constraint generation

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import requests
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime
import hashlib
import asyncio

# Canonical projector for manifold-consistent embeddings
from src.data.canonical_projection import CanonicalProjector


def _stable_id(prefix: str, obj: Any) -> str:
    """Deterministic ID using Blake2s over canonical JSON (sorted keys)."""
    try:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(obj)
    hexdigest = hashlib.blake2s(payload.encode('utf-8'), digest_size=10).hexdigest()
    return f"{prefix}_{hexdigest}"

# Optional imports
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Core imports
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.data.pressure_ingestor import PressureIngestor


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    speaker_id: str
    text: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[torch.Tensor] = None
    affordance_gradients: Optional[Dict[str, float]] = None


@dataclass
class Conversation:
    """Complete conversation with multiple turns."""
    conversation_id: str
    turns: List[ConversationTurn]
    context: Dict[str, Any]
    source: str
    labels: Optional[Dict[str, Any]] = None
    pressure_signature: Optional[torch.Tensor] = None


class HuggingFaceConversationalIngestor:
    """
    Ingest conversational datasets from Hugging Face Hub API.
    
    Supports datasets like:
    - lmsys/lmsys-chat-1m
    - OpenAssistant/oasst2
    - UltraChat collections
    """
    
    def __init__(self, hf_token: Optional[str] = None, device: str = 'cpu'):
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.device = device
        self.base_url = "https://huggingface.co/api"
        
        # Headers for API requests
        self.headers = {}
        if self.hf_token:
            self.headers['Authorization'] = f'Bearer {self.hf_token}'
    
    def list_conversational_datasets(self) -> List[Dict[str, Any]]:
        """List available conversational datasets."""
        url = f"{self.base_url}/datasets"
        params = {
            'search': 'conversation OR dialog OR chat',
            'filter': 'task_categories:conversational',
            'limit': 100
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to list datasets: {e}")
            return []
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a dataset."""
        url = f"{self.base_url}/datasets/{dataset_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to get dataset info for {dataset_id}: {e}")
            return {}
    
    def download_dataset_sample(self, dataset_id: str, split: str = 'train', 
                              max_samples: int = 1000) -> List[Dict[str, Any]]:
        """Download a sample of conversational data."""
        try:
            # Try datasets library first
            from datasets import load_dataset
            
            print(f"   Loading dataset {dataset_id} with authentication...")
            
            # Load dataset with streaming for large datasets and authentication
            dataset = load_dataset(
                dataset_id, 
                split=split, 
                streaming=True,
                use_auth_token=self.hf_token  # Add authentication!
            )
            
            print(f"   Dataset loaded, collecting {max_samples} samples...")
            
            samples = []
            for i, sample in enumerate(dataset):
                if i >= max_samples:
                    break
                samples.append(sample)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   Collected {i + 1} samples...")
            
            print(f"   ✓ Collected {len(samples)} samples total")
            return samples
            
        except ImportError:
            print(f"   ⚠ datasets library not available, using direct API method...")
            return self._direct_api_download(dataset_id, max_samples)
            
        except Exception as e:
            print(f"   ❌ datasets library failed: {e}")
            print(f"   Trying direct API method...")
            return self._direct_api_download(dataset_id, max_samples)
    
    def _direct_api_download(self, dataset_id: str, max_samples: int) -> List[Dict[str, Any]]:
        """Direct API method to download real HF data when datasets library is not available."""
        print(f"   � Using direct HF API to download real data from {dataset_id}...")
        
        try:
            # Use HF Hub API to get dataset files
            from huggingface_hub import list_repo_files, hf_hub_download
            
            # List files in the dataset repo
            files = list_repo_files(dataset_id, repo_type="dataset", token=self.hf_token)
            print(f"   Found {len(files)} files in dataset")
            
            # Look for parquet or jsonl files
            data_files = [f for f in files if f.endswith(('.parquet', '.jsonl', '.json'))]
            
            if not data_files:
                print(f"   ❌ No data files found in {dataset_id}")
                return []
            
            # Download the first data file
            data_file = data_files[0]
            print(f"   📥 Downloading {data_file}...")
            
            file_path = hf_hub_download(
                repo_id=dataset_id,
                filename=data_file,
                repo_type="dataset",
                token=self.hf_token
            )
            
            # Parse the downloaded file
            samples = []
            if data_file.endswith('.parquet'):
                import pandas as pd
                df = pd.read_parquet(file_path)
                samples = df.head(max_samples).to_dict('records')
            elif data_file.endswith(('.jsonl', '.json')):
                import json
                with open(file_path, 'r') as f:
                    if data_file.endswith('.jsonl'):
                        for i, line in enumerate(f):
                            if i >= max_samples:
                                break
                            samples.append(json.loads(line))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            samples = data[:max_samples]
                        else:
                            samples = [data]
            
            print(f"   ✓ Downloaded {len(samples)} real samples via direct API")
            return samples
            
        except Exception as e:
            print(f"   ❌ Direct API download failed: {e}")
            print(f"   This means there's an issue with your HF token or dataset access")
            return []
    
    def _generate_synthetic_lmsys_data(self, max_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic LMSYS-style conversation data with diverse affordance patterns."""
        import random
        
        # Enhanced conversation templates with diverse affordance patterns
        conversation_templates = []
        
        # Basic conversational patterns
        basic_conversations = [
            {
                "conversation_id": f"synthetic_lmsys_{i}",
                "conversation": [
                    {"role": "human", "content": f"Hello! Can you help me understand {topic}?"},
                    {"role": "assistant", "content": f"Of course! I'd be happy to help you understand {topic}. {explanation}"},
                    {"role": "human", "content": "That's very helpful, thank you!"},
                    {"role": "assistant", "content": "You're welcome! Feel free to ask if you have any more questions about this topic."}
                ],
                "model": "synthetic-assistant",
                "timestamp": "2024-01-01T00:00:00Z",
                "language": "en"
            }
            for i, (topic, explanation) in enumerate([
                ("machine learning", "Machine learning is a subset of AI that enables computers to learn from data without being explicitly programmed."),
                ("Python programming", "Python is a versatile programming language known for its simplicity and readability."),
                ("data science", "Data science combines statistics, programming, and domain expertise to extract insights from data."),
                ("artificial intelligence", "AI refers to computer systems that can perform tasks typically requiring human intelligence."),
                ("web development", "Web development involves creating websites and web applications using various technologies.")
            ])
        ]
        
        # Code execution patterns (high executability + formal symbols)
        code_conversations = [
            {
                "conversation_id": f"synthetic_code_{i}",
                "conversation": [
                    {"role": "human", "content": f"Can you help me write a function to {task}?"},
                    {"role": "assistant", "content": f"Sure! Here's a Python function:\n\n```python\ndef {func_name}({params}):\n    {implementation}\n    return result\n```\n\nThis function {explanation}"},
                    {"role": "human", "content": "How do I run this code?"},
                    {"role": "assistant", "content": f"To execute this function, call it like this: `{func_name}({example_args})`. Make sure to import any required libraries first."}
                ],
                "model": "synthetic-coder",
                "timestamp": "2024-01-01T00:00:00Z",
                "language": "en"
            }
            for i, (task, func_name, params, implementation, explanation, example_args) in enumerate([
                ("calculate fibonacci numbers", "fibonacci", "n", "if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)", "calculates the nth Fibonacci number recursively", "10"),
                ("sort a list", "quicksort", "arr", "if len(arr) <= 1: return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])", "implements the quicksort algorithm", "[3, 1, 4, 1, 5]"),
                ("find prime numbers", "is_prime", "n", "if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True", "checks if a number is prime", "17")
            ])
        ]
        
        # API extraction patterns (high api_extraction + conversational)
        api_conversations = [
            {
                "conversation_id": f"synthetic_api_{i}",
                "conversation": [
                    {"role": "human", "content": f"Can you search for the latest information about {topic}? I need current data."},
                    {"role": "assistant", "content": f"I'll help you find the latest information about {topic}. Let me search for recent articles and data sources. You might want to check {sources} for the most up-to-date information."},
                    {"role": "human", "content": "Can you fetch the Wikipedia article about this topic?"},
                    {"role": "assistant", "content": f"I can help you access Wikipedia content about {topic}. You can also use APIs to extract structured data from various online sources and databases."}
                ],
                "model": "synthetic-researcher",
                "timestamp": "2024-01-01T00:00:00Z",
                "language": "en"
            }
            for i, (topic, sources) in enumerate([
                ("climate change", "NASA Climate Change, IPCC reports, and recent scientific publications"),
                ("cryptocurrency prices", "CoinGecko API, CoinMarketCap, and financial news websites"),
                ("stock market trends", "Yahoo Finance API, Bloomberg, and real-time market data feeds")
            ])
        ]
        
        # Formal symbols patterns (high formal_symbols + expandability)
        math_conversations = [
            {
                "conversation_id": f"synthetic_math_{i}",
                "conversation": [
                    {"role": "human", "content": f"Can you explain {concept}?"},
                    {"role": "assistant", "content": f"Certainly! {concept} can be expressed as: {formula}. This means {explanation}"},
                    {"role": "human", "content": "Can you show me how to solve this step by step?"},
                    {"role": "assistant", "content": f"Here's the step-by-step solution:\n1. Start with {formula}\n2. Apply the rule: {rule}\n3. Simplify: {simplified}\n4. Therefore: {conclusion}"}
                ],
                "model": "synthetic-mathematician",
                "timestamp": "2024-01-01T00:00:00Z",
                "language": "en"
            }
            for i, (concept, formula, explanation, rule, simplified, conclusion) in enumerate([
                ("quadratic formula", "x = (-b ± √(b² - 4ac)) / 2a", "this formula solves equations of the form ax² + bx + c = 0", "discriminant = b² - 4ac", "x = (-b ± √Δ) / 2a", "we get two solutions when Δ > 0"),
                ("derivatives", "f'(x) = lim(h→0) [f(x+h) - f(x)] / h", "this defines the instantaneous rate of change", "power rule: d/dx(x^n) = nx^(n-1)", "d/dx(x²) = 2x", "the derivative of x² is 2x"),
                ("set theory", "A ∪ B = {x | x ∈ A ∨ x ∈ B}", "this represents the union of sets A and B", "De Morgan's law: (A ∪ B)' = A' ∩ B'", "complement of union equals intersection of complements", "set operations follow logical rules")
            ])
        ]
        
        # Meta-recursive patterns (high referential_closure + expandability)
        meta_conversations = [
            {
                "conversation_id": f"synthetic_meta_{i}",
                "conversation": [
                    {"role": "human", "content": f"How does {system} work recursively?"},
                    {"role": "assistant", "content": f"{system} creates recursive feedback loops where {description}. The system reflects on itself, creating meta-structures that generate their own constraints and invariants."},
                    {"role": "human", "content": "Can this system modify itself?"},
                    {"role": "assistant", "content": f"Yes, {system} exhibits self-referential properties. It can analyze its own structure, create templates for expansion, and generate new patterns dynamically. This recursive self-modification creates emergent behaviors."}
                ],
                "model": "synthetic-philosopher",
                "timestamp": "2024-01-01T00:00:00Z",
                "language": "en"
            }
            for i, (system, description) in enumerate([
                ("neural networks", "each layer processes information and passes it to the next, while backpropagation creates feedback loops that modify the network's own weights"),
                ("language models", "they generate text that can describe their own generation process, creating recursive loops of self-description and meta-cognition"),
                ("fractal systems", "they contain copies of themselves at different scales, with each iteration generating new complexity through self-similar patterns")
            ])
        ]
        
        # Combine all conversation types
        all_conversations = basic_conversations + code_conversations + api_conversations + math_conversations + meta_conversations
        
        # Select up to max_samples conversations
        selected_conversations = all_conversations[:max_samples]
        
        print(f"   ✓ Generated {len(selected_conversations)} diverse synthetic LMSYS conversations")
        print(f"     - Basic: {min(len(basic_conversations), max_samples)}")
        print(f"     - Code: {min(len(code_conversations), max_samples - len(basic_conversations)) if max_samples > len(basic_conversations) else 0}")
        print(f"     - API: {min(len(api_conversations), max_samples - len(basic_conversations) - len(code_conversations)) if max_samples > len(basic_conversations) + len(code_conversations) else 0}")
        print(f"     - Math: {min(len(math_conversations), max_samples - len(basic_conversations) - len(code_conversations) - len(api_conversations)) if max_samples > len(basic_conversations) + len(code_conversations) + len(api_conversations) else 0}")
        print(f"     - Meta: {min(len(meta_conversations), max_samples - len(basic_conversations) - len(code_conversations) - len(api_conversations) - len(math_conversations)) if max_samples > len(basic_conversations) + len(code_conversations) + len(api_conversations) + len(math_conversations) else 0}")
        
        return selected_conversations
    
    def _generate_synthetic_oasst_data(self, max_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic OpenAssistant-style conversation data."""
        conversations = []
        
        for i in range(max_samples):
            conversations.append({
                "message_id": f"synthetic_oasst_{i}",
                "parent_id": None if i == 0 else f"synthetic_oasst_{i-1}",
                "text": f"This is a synthetic OpenAssistant conversation turn {i}. How can I help you today?",
                "role": "assistant" if i % 2 == 0 else "prompter",
                "lang": "en",
                "quality": {"rating": 4.5, "num_ratings": 10}
            })
        
        print(f"   ✓ Generated {len(conversations)} synthetic OpenAssistant conversations")
        return conversations
    
    def parse_lmsys_chat(self, samples: List[Dict[str, Any]]) -> List[Conversation]:
        """Parse LMSYS chat data into Conversation objects."""
        conversations = []
        
        for sample in samples:
            try:
                # LMSYS format: conversation with multiple turns
                conv_id = sample.get('conversation_id', _stable_id("lmsys", sample))
                
                turns = []
                if 'conversation' in sample:
                    for i, turn_data in enumerate(sample['conversation']):
                        turn = ConversationTurn(
                            speaker_id=turn_data.get('role', f'speaker_{i % 2}'),
                            text=turn_data.get('content', ''),
                            metadata={'turn_index': i}
                        )
                        turns.append(turn)
                
                conversation = Conversation(
                    conversation_id=conv_id,
                    turns=turns,
                    context={'model': sample.get('model', 'unknown')},
                    source='lmsys',
                    labels=sample.get('labels', {})
                )
                
                conversations.append(conversation)
                
            except Exception as e:
                print(f"⚠️ Failed to parse LMSYS sample: {e}")
                continue
        
        return conversations
    
    def parse_openassistant(self, samples: List[Dict[str, Any]]) -> List[Conversation]:
        """Parse OpenAssistant data into Conversation objects."""
        conversations = []
        
        for sample in samples:
            try:
                conv_id = sample.get('message_id', _stable_id("oasst", sample))
                
                # OpenAssistant has parent-child message structure
                turns = []
                if 'text' in sample:
                    turn = ConversationTurn(
                        speaker_id=sample.get('role', 'user'),
                        text=sample['text'],
                        metadata={
                            'parent_id': sample.get('parent_id'),
                            'rank': sample.get('rank', 0),
                            'lang': sample.get('lang', 'en')
                        }
                    )
                    turns.append(turn)
                
                conversation = Conversation(
                    conversation_id=conv_id,
                    turns=turns,
                    context={'language': sample.get('lang', 'en')},
                    source='openassistant',
                    labels={'quality': sample.get('quality', {})}
                )
                
                conversations.append(conversation)
                
            except Exception as e:
                print(f"⚠️ Failed to parse OpenAssistant sample: {e}")
                continue
        
        return conversations


class RedditConversationalIngestor:
    """
    Ingest conversational threads from Reddit API.
    
    Extracts threaded comment structures as multi-turn conversations.
    """
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str, device: str = 'cpu'):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.device = device
        self.access_token = None
        self.base_url = "https://oauth.reddit.com"
    
    def authenticate(self):
        """Authenticate with Reddit API."""
        auth_url = "https://www.reddit.com/api/v1/access_token"
        
        auth_data = {
            'grant_type': 'client_credentials'
        }
        
        auth = (self.client_id, self.client_secret)
        headers = {'User-Agent': self.user_agent}
        
        try:
            response = requests.post(auth_url, auth=auth, data=auth_data, headers=headers)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            print(f"✅ Reddit authentication successful")
            
        except Exception as e:
            print(f"❌ Reddit authentication failed: {e}")
    
    def get_subreddit_posts(self, subreddit: str, limit: int = 100, 
                           sort: str = 'hot') -> List[Dict[str, Any]]:
        """Get posts from a subreddit."""
        if not self.access_token:
            self.authenticate()
        
        url = f"{self.base_url}/r/{subreddit}/{sort}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        params = {'limit': limit}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data['data']['children']
            
        except Exception as e:
            print(f"❌ Failed to get posts from r/{subreddit}: {e}")
            return []
    
    def get_post_comments(self, subreddit: str, post_id: str, 
                         max_depth: int = 5) -> List[Dict[str, Any]]:
        """Get comments for a specific post."""
        if not self.access_token:
            self.authenticate()
        
        url = f"{self.base_url}/r/{subreddit}/comments/{post_id}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            # Second element contains comments
            if len(data) > 1:
                return data[1]['data']['children']
            return []
            
        except Exception as e:
            print(f"❌ Failed to get comments for post {post_id}: {e}")
            return []
    
    def parse_comment_thread(self, comments: List[Dict[str, Any]], 
                           post_data: Dict[str, Any]) -> List[Conversation]:
        """Parse Reddit comment thread into conversations."""
        conversations = []
        
        def extract_thread(comment_data, parent_text="", depth=0, max_depth=5):
            if depth > max_depth:
                return []
            
            comment = comment_data.get('data', {})
            if comment.get('body') in ['[deleted]', '[removed]', '']:
                return []
            
            turns = []
            
            # Add parent context if exists
            if parent_text and depth > 0:
                turns.append(ConversationTurn(
                    speaker_id='parent',
                    text=parent_text,
                    metadata={'depth': depth - 1}
                ))
            
            # Add current comment
            turns.append(ConversationTurn(
                speaker_id=comment.get('author', 'unknown'),
                text=comment.get('body', ''),
                timestamp=datetime.fromtimestamp(comment.get('created_utc', 0)),
                metadata={
                    'depth': depth,
                    'score': comment.get('score', 0),
                    'permalink': comment.get('permalink', '')
                }
            ))
            
            # Create conversation for this thread
            if len(turns) > 1:  # Only if we have a dialogue
                conv = Conversation(
                    conversation_id=f"reddit_{comment.get('id', 'unknown')}",
                    turns=turns,
                    context={
                        'subreddit': post_data.get('subreddit', ''),
                        'post_title': post_data.get('title', ''),
                        'thread_depth': depth
                    },
                    source='reddit'
                )
                conversations.append(conv)
            
            # Process replies recursively
            replies = comment.get('replies', {})
            if isinstance(replies, dict) and 'data' in replies:
                reply_children = replies['data'].get('children', [])
                for reply in reply_children:
                    conversations.extend(
                        extract_thread(reply, comment.get('body', ''), depth + 1, max_depth)
                    )
            
            return conversations
        
        # Process all top-level comments
        all_conversations = []
        for comment_data in comments:
            if comment_data.get('kind') == 't1':  # Comment type
                thread_convs = extract_thread(comment_data, post_data.get('title', ''))
                all_conversations.extend(thread_convs)
        
        return all_conversations


class ConvoKitIngestor:
    """
    Ingest labeled conversational data using ConvoKit library.
    
    Provides access to multiple labeled conversational corpora.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.available_corpora = [
            'conversations-gone-awry-corpus',
            'persuasionforgood-corpus',
            'movie-corpus',
            'supreme-corpus',
            'tennis-corpus',
            'wikipedia-corpus'
        ]
    
    def list_available_corpora(self) -> List[str]:
        """List available ConvoKit corpora."""
        try:
            import convokit
            return self.available_corpora
        except ImportError:
            print("❌ ConvoKit not installed. Install with: pip install convokit")
            return []
    
    def load_corpus(self, corpus_name: str) -> Optional[Any]:
        """Load a ConvoKit corpus."""
        try:
            import convokit
            corpus = convokit.download(corpus_name)
            print(f"✅ Loaded ConvoKit corpus: {corpus_name}")
            return corpus
        except Exception as e:
            print(f"❌ Failed to load corpus {corpus_name}: {e}")
            return None
    
    def parse_convokit_corpus(self, corpus, max_conversations: int = 1000) -> List[Conversation]:
        """Parse ConvoKit corpus into Conversation objects."""
        conversations = []
        
        try:
            conv_count = 0
            for convo_id in corpus.get_conversation_ids():
                if conv_count >= max_conversations:
                    break
                
                convo = corpus.get_conversation(convo_id)
                turns = []
                
                for utterance in convo.iter_utterances():
                    turn = ConversationTurn(
                        speaker_id=utterance.speaker.id,
                        text=utterance.text,
                        metadata={
                            'utterance_id': utterance.id,
                            'reply_to': utterance.reply_to,
                            'timestamp': utterance.timestamp,
                            **utterance.meta
                        }
                    )
                    turns.append(turn)
                
                conversation = Conversation(
                    conversation_id=convo_id,
                    turns=turns,
                    context=convo.meta,
                    source='convokit',
                    labels=convo.meta
                )
                
                conversations.append(conversation)
                conv_count += 1
            
            print(f"✅ Parsed {len(conversations)} conversations from ConvoKit corpus")
            return conversations
            
        except Exception as e:
            print(f"❌ Failed to parse ConvoKit corpus: {e}")
            return []


class ConversationalDataProcessor:
    """
    Process conversational data for integration with the Gyroidic system.
    
    Handles:
    - Text embedding generation
    - Affordance gradient computation
    - Pressure signature generation
    - Constraint geometry creation
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Initialize pressure ingestor for constraint generation
        self.pressure_ingestor = PressureIngestor(device=device)
        
        # Polynomial co-prime system for proper architecture
        self.polynomial_config = PolynomialCoprimeConfig(
            k=5, degree=4, basis_type='chebyshev',
            learnable=True, use_saturation=True, device=device
        )

        # Canonical projector aligned with topological pipeline
        self.projector = CanonicalProjector(dim=64, k=5, device=device)
    
    def compute_text_embedding(self, text: str) -> torch.Tensor:
        """Project text into manifold-consistent [1, dim] state using canonical projector."""
        proj = self.projector.project_text_to_state(text)
        # attach entropy to turn metadata upstream; here return just the vector
        state = proj['state']  # [1, 64]
        return state.squeeze(0)  # [64]
    
    def compute_affordance_gradients(self, text: str) -> Dict[str, float]:
        """Compute affordance gradients for conversational text."""
        # Conversational pattern detection
        conversational_signals = [
            '?', 'what', 'how', 'why', 'when', 'where', 'who',
            'tell me', 'explain', 'describe', 'think', 'feel',
            'opinion', 'believe', 'agree', 'disagree'
        ]
        
        # API extraction signals
        api_signals = [
            'search', 'find', 'lookup', 'get', 'fetch', 'download',
            'latest', 'current', 'recent', 'update', 'information'
        ]
        
        # Code execution signals
        code_signals = [
            'def ', 'class ', 'import ', 'return ', 'if ', 'for ',
            'while ', 'try:', 'except:', 'print(', 'execute', 'run'
        ]
        
        text_lower = text.lower()
        
        # Compute gradient strengths
        conversational = sum(1 for signal in conversational_signals if signal in text_lower) / len(text.split())
        api_extraction = sum(1 for signal in api_signals if signal in text_lower) / len(text.split())
        executability = sum(1 for signal in code_signals if signal in text_lower) / len(text.split())
        
        # Formal symbols (mathematical/logical content)
        formal_symbols = len([c for c in text if c in '()[]{}=+-*/^<>']) / len(text)
        
        # Expandability (complexity indicators)
        expandability = len([w for w in text.split() if len(w) > 8]) / len(text.split())
        
        # Closure (completeness indicators)
        closure_signals = ['.', '!', '?', ';', 'done', 'complete', 'finished']
        closure = sum(1 for signal in closure_signals if signal in text_lower) / len(text.split())
        
        return {
            'conversational': min(conversational, 1.0),
            'api_extraction': min(api_extraction, 1.0),
            'executability': min(executability, 1.0),
            'formal_symbols': min(formal_symbols, 1.0),
            'expandability': min(expandability, 1.0),
            'closure': min(closure, 1.0)
        }
    
    def generate_pressure_signature(self, conversation: Conversation) -> torch.Tensor:
        """Generate pressure signature for conversation using polynomial system."""
        # Combine all text in conversation
        full_text = " ".join([turn.text for turn in conversation.turns])
        
        # Compute text statistics
        text_length = len(full_text)
        turn_count = len(conversation.turns)
        avg_turn_length = text_length / max(turn_count, 1)
        
        # Create input tensor for polynomial evaluation
        stats_tensor = torch.tensor([
            text_length / 1000.0,  # Normalized length
            turn_count / 10.0,     # Normalized turn count
            avg_turn_length / 100.0  # Normalized avg turn length
        ], device=self.device).unsqueeze(0)  # [1, 3]
        
        # Evaluate polynomial functionals
        pressure_signature = self.polynomial_config.evaluate(stats_tensor)  # [1, 1, K]
        
        return pressure_signature.squeeze()  # [K]
    
    def process_conversation(self, conversation: Conversation) -> Conversation:
        """Process a single conversation with embeddings and gradients."""
        # Process each turn
        for turn in conversation.turns:
            # Compute text embedding via canonical projector and attach entropy
            proj = self.projector.project_text_to_state(turn.text)
            turn.embedding = proj['state'].squeeze(0)  # [64]
            if turn.metadata is None:
                turn.metadata = {}
            turn.metadata['gyroid_entropy'] = proj['entropy']
            
            # Compute affordance gradients
            turn.affordance_gradients = self.compute_affordance_gradients(turn.text)
        
        # Generate conversation-level pressure signature
        conversation.pressure_signature = self.generate_pressure_signature(conversation)

        # Attach fused manifold diagnostics at conversation level
        entropies = [t.metadata.get('gyroid_entropy', 0.0) for t in conversation.turns if t.metadata]
        if entropies:
            conversation.context = dict(conversation.context or {})
            conversation.context['avg_gyroid_entropy'] = float(np.mean(entropies))
        
        return conversation
    
    def process_conversations(self, conversations: List[Conversation]) -> List[Conversation]:
        """Process multiple conversations."""
        processed = []
        
        for i, conversation in enumerate(conversations):
            try:
                processed_conv = self.process_conversation(conversation)
                processed.append(processed_conv)
                
                if (i + 1) % 100 == 0:
                    print(f"✅ Processed {i + 1}/{len(conversations)} conversations")
                    
            except Exception as e:
                print(f"⚠️ Failed to process conversation {conversation.conversation_id}: {e}")
                continue
        
        return processed


class ConversationalAPIIngestor:
    """
    Main class for ingesting conversational data from multiple APIs.
    
    Coordinates different ingestors and provides unified interface.
    """
    
    def __init__(self, device: str = 'cpu', cache_dir: str = './data/conversational_cache'):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ingestors
        self.hf_ingestor = HuggingFaceConversationalIngestor(device=device)
        self.processor = ConversationalDataProcessor(device=device)
        
        # Reddit ingestor (requires credentials)
        self.reddit_ingestor = None
        
        # ConvoKit ingestor
        self.convokit_ingestor = ConvoKitIngestor(device=device)
    
    def setup_reddit(self, client_id: str, client_secret: str, user_agent: str):
        """Setup Reddit API access."""
        self.reddit_ingestor = RedditConversationalIngestor(
            client_id, client_secret, user_agent, self.device
        )
    
    def ingest_huggingface_dataset(self, dataset_id: str, max_samples: int = 1000) -> List[Conversation]:
        """Ingest conversational data from Hugging Face dataset."""
        print(f"🔄 Ingesting Hugging Face dataset: {dataset_id}")
        
        # Check cache first
        cache_file = self.cache_dir / f"hf_{dataset_id.replace('/', '_')}.json"
        if cache_file.exists():
            print(f"📁 Loading from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return [self._dict_to_conversation(conv_dict) for conv_dict in cached_data]
        
        # Download samples
        print(f"📥 Downloading {max_samples} samples from {dataset_id}...")
        samples = self.hf_ingestor.download_dataset_sample(dataset_id, max_samples=max_samples)
        
        if not samples:
            print(f"❌ No samples downloaded from {dataset_id}")
            return []
        
        print(f"✓ Downloaded {len(samples)} samples, parsing...")
        
        # Parse based on dataset type
        if 'lmsys' in dataset_id.lower():
            conversations = self.hf_ingestor.parse_lmsys_chat(samples)
        elif 'openassistant' in dataset_id.lower() or 'oasst' in dataset_id.lower():
            conversations = self.hf_ingestor.parse_openassistant(samples)
        else:
            # Generic parsing
            conversations = self.hf_ingestor.parse_lmsys_chat(samples)
        
        if not conversations:
            print(f"❌ No conversations parsed from {len(samples)} samples")
            return []
        
        print(f"✓ Parsed {len(conversations)} conversations, processing...")
        
        # Process conversations
        processed_conversations = self.processor.process_conversations(conversations)
        
        if not processed_conversations:
            print(f"❌ No conversations after processing")
            return []
        
        # Cache results
        try:
            with open(cache_file, 'w') as f:
                json.dump([self._conversation_to_dict(conv) for conv in processed_conversations], f)
            print(f"💾 Cached results to {cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to cache results: {e}")
        
        print(f"✅ Successfully ingested {len(processed_conversations)} conversations from {dataset_id}")
        return processed_conversations
    
    def ingest_reddit_subreddit(self, subreddit: str, max_posts: int = 50) -> List[Conversation]:
        """Ingest conversational threads from Reddit subreddit."""
        if not self.reddit_ingestor:
            print("❌ Reddit ingestor not configured. Use setup_reddit() first.")
            return []
        
        print(f"🔄 Ingesting Reddit subreddit: r/{subreddit}")
        
        # Get posts
        posts = self.reddit_ingestor.get_subreddit_posts(subreddit, limit=max_posts)
        
        all_conversations = []
        for post_data in posts:
            post = post_data['data']
            
            # Get comments for this post
            comments = self.reddit_ingestor.get_post_comments(subreddit, post['id'])
            
            # Parse comment threads
            conversations = self.reddit_ingestor.parse_comment_thread(comments, post)
            
            # Process conversations
            processed = self.processor.process_conversations(conversations)
            all_conversations.extend(processed)
        
        print(f"✅ Ingested {len(all_conversations)} conversations from r/{subreddit}")
        return all_conversations
    
    def ingest_convokit_corpus(self, corpus_name: str, max_conversations: int = 1000) -> List[Conversation]:
        """Ingest labeled conversational data from ConvoKit."""
        print(f"🔄 Ingesting ConvoKit corpus: {corpus_name}")
        
        # Load corpus
        corpus = self.convokit_ingestor.load_corpus(corpus_name)
        if not corpus:
            return []
        
        # Parse conversations
        conversations = self.convokit_ingestor.parse_convokit_corpus(corpus, max_conversations)
        
        # Process conversations
        processed_conversations = self.processor.process_conversations(conversations)
        
        print(f"✅ Ingested {len(processed_conversations)} conversations from {corpus_name}")
        return processed_conversations
    
    def _conversation_to_dict(self, conversation: Conversation) -> Dict[str, Any]:
        """Convert Conversation object to dictionary for caching."""
        return {
            'conversation_id': conversation.conversation_id,
            'turns': [
                {
                    'speaker_id': turn.speaker_id,
                    'text': turn.text,
                    'timestamp': turn.timestamp.isoformat() if turn.timestamp else None,
                    'metadata': turn.metadata,
                    'embedding': turn.embedding.tolist() if turn.embedding is not None else None,
                    'affordance_gradients': turn.affordance_gradients
                }
                for turn in conversation.turns
            ],
            'context': conversation.context,
            'source': conversation.source,
            'labels': conversation.labels,
            'pressure_signature': conversation.pressure_signature.tolist() if conversation.pressure_signature is not None else None
        }
    
    def _dict_to_conversation(self, conv_dict: Dict[str, Any]) -> Conversation:
        """Convert dictionary back to Conversation object."""
        turns = []
        for turn_dict in conv_dict['turns']:
            turn = ConversationTurn(
                speaker_id=turn_dict['speaker_id'],
                text=turn_dict['text'],
                timestamp=datetime.fromisoformat(turn_dict['timestamp']) if turn_dict['timestamp'] else None,
                metadata=turn_dict['metadata'],
                embedding=torch.tensor(turn_dict['embedding'], device=self.device) if turn_dict['embedding'] else None,
                affordance_gradients=turn_dict['affordance_gradients']
            )
            turns.append(turn)
        
        conversation = Conversation(
            conversation_id=conv_dict['conversation_id'],
            turns=turns,
            context=conv_dict['context'],
            source=conv_dict['source'],
            labels=conv_dict['labels'],
            pressure_signature=torch.tensor(conv_dict['pressure_signature'], device=self.device) if conv_dict['pressure_signature'] else None
        )
        
        return conversation
    
    def get_ingestion_summary(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Get summary statistics for ingested conversations."""
        if not conversations:
            return {}
        
        total_turns = sum(len(conv.turns) for conv in conversations)
        total_text_length = sum(len(turn.text) for conv in conversations for turn in conv.turns)
        
        sources = {}
        for conv in conversations:
            sources[conv.source] = sources.get(conv.source, 0) + 1
        
        # Affordance gradient statistics
        all_gradients = {}
        for conv in conversations:
            for turn in conv.turns:
                if turn.affordance_gradients:
                    for key, value in turn.affordance_gradients.items():
                        if key not in all_gradients:
                            all_gradients[key] = []
                        all_gradients[key].append(value)
        
        gradient_stats = {}
        for key, values in all_gradients.items():
            gradient_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values)
            }
        
        return {
            'total_conversations': len(conversations),
            'total_turns': total_turns,
            'total_text_length': total_text_length,
            'avg_turns_per_conversation': total_turns / len(conversations),
            'avg_text_length_per_turn': total_text_length / total_turns,
            'sources': sources,
            'affordance_gradient_stats': gradient_stats
        }


def create_conversational_ingestor(device: str = 'cpu') -> ConversationalAPIIngestor:
    """Create and configure conversational API ingestor."""
    return ConversationalAPIIngestor(device=device)


# Example usage functions
def demo_huggingface_ingestion():
    """Demonstrate Hugging Face dataset ingestion."""
    ingestor = create_conversational_ingestor()
    
    # Try to ingest a small sample from LMSYS chat dataset
    conversations = ingestor.ingest_huggingface_dataset('lmsys/lmsys-chat-1m', max_samples=100)
    
    if conversations:
        summary = ingestor.get_ingestion_summary(conversations)
        print(f"📊 Ingestion Summary:")
        print(f"   Conversations: {summary['total_conversations']}")
        print(f"   Total turns: {summary['total_turns']}")
        print(f"   Avg turns per conversation: {summary['avg_turns_per_conversation']:.2f}")
        print(f"   Sources: {summary['sources']}")
        
        # Show sample conversation
        if conversations:
            sample = conversations[0]
            print(f"\n💬 Sample Conversation ({sample.conversation_id}):")
            for i, turn in enumerate(sample.turns[:3]):  # First 3 turns
                print(f"   Turn {i+1} ({turn.speaker_id}): {turn.text[:100]}...")
                if turn.affordance_gradients:
                    print(f"      Gradients: {turn.affordance_gradients}")


def demo_convokit_ingestion():
    """Demonstrate ConvoKit corpus ingestion."""
    ingestor = create_conversational_ingestor()
    
    # Try to ingest conversations-gone-awry corpus
    conversations = ingestor.ingest_convokit_corpus('conversations-gone-awry-corpus', max_conversations=50)
    
    if conversations:
        summary = ingestor.get_ingestion_summary(conversations)
        print(f"📊 ConvoKit Ingestion Summary:")
        print(f"   Conversations: {summary['total_conversations']}")
        print(f"   Affordance gradients: {list(summary['affordance_gradient_stats'].keys())}")


if __name__ == "__main__":
    print("🗣️ Conversational API Data Ingestor")
    print("Demonstrates programmatic ingestion of conversational data")
    print("=" * 60)
    
    # Demo Hugging Face ingestion
    print("\n🤗 Hugging Face Dataset Ingestion:")
    demo_huggingface_ingestion()
    
    # Demo ConvoKit ingestion
    print("\n📚 ConvoKit Corpus Ingestion:")
    demo_convokit_ingestion()