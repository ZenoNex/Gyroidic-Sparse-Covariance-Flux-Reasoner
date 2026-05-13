"""
Sovereign Ingestion Suite

Implements zero-auth, platform-independent data ingestion from open APIs
and local snapshots, prioritizing "Nutrients over Poison" (Option D).

Mandates:
- Fuzzy Decoding for IRC logs (preserving topological friction).
- Local Snapshot priority for Reddit data (MADOC).
- Configurable REPOSITORY_ROOT for external capacity management.
- Hacker News entropy focus (Recent/Ask HN).

Author: William Matthew Bryant
Created: April 2026
"""

import os
import json
import requests
import time
from typing import List, Dict, Optional, Any, Generator
from pathlib import Path
from datetime import datetime

# Core ingestion types from neutral module
from src.data.conversational_types import Conversation, ConversationTurn, _stable_id

class SovereignIngestor:
    """
    Orchestrator for zero-auth "Sovereign" data sources.
    
    Bypasses centralized platform constraints in favor of direct
    API access and local repository snapshots.
    """
    
    def __init__(self, repository_root: Optional[str] = None):
        """
        Args:
            repository_root: Base path for local datasets (MADOC, IRC).
                             If None, defaults to current working directory/data/sovereign.
        """
        if repository_root is None:
            # Default to a sovereign data directory in the workspace
            self.root = Path(os.getcwd()) / 'data' / 'sovereign'
        else:
            self.root = Path(repository_root)
            
        print(f"[*] Sovereign Ingestor initialized with REPOSITORY_ROOT: {self.root}")
        
    def _fuzzy_decode(self, bytes_data: bytes) -> str:
        """
        Perform fuzzy decoding on raw bytes to preserve "topological friction".
        
        Mandated by Option D: Errors are not failures; they are valid 
        structural noise in the Mischief Band.
        """
        try:
            # Try UTF-8 first
            return bytes_data.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to ISO-8859-1 with replacement for truly unmapped bytes
            # This ensures the messy reality of IRC logs is preserved.
            return bytes_data.decode('latin-1', errors='replace')

    def ingest_hacker_news(self, limit: int = 100, mode: str = 'ask') -> List[Conversation]:
        """
        Ingest from Hacker News Firebase API.
        Focuses on 'ask' and 'recent' to maximize entropy per Option D.
        """
        print(f"Fetching HN {mode} stories for High-Entropy Ingestion...")
        base_url = "https://hacker-news.firebaseio.com/v0"
        
        endpoint = f"{mode}stories.json"
        try:
            response = requests.get(f"{base_url}/{endpoint}")
            response.raise_for_status()
            story_ids = response.json()[:limit]
        except Exception as e:
            print(f" HN API Fetch failed: {e}")
            return []

        conversations = []
        for story_id in story_ids:
            try:
                story_resp = requests.get(f"{base_url}/item/{story_id}.json")
                story = story_resp.json()
                
                if not story or not story.get('text') and not story.get('title'):
                    continue
                    
                turns = []
                text_content = story.get('title', '') + "\n" + story.get('text', '')
                words = text_content.split()
                avg_word_len = sum(len(w) for w in words) / max(1, len(words))
                
                turns.append(ConversationTurn(
                    speaker_id=story.get('by', 'anon'),
                    text=text_content,
                    timestamp=datetime.fromtimestamp(story.get('time', 0)),
                    metadata={'hn_id': story_id, 'type': story.get('type')}
                ))
                
                # Fetch top-level comments if they exist
                kids = story.get('kids', [])[:5] # Limit depth for initial ingestion
                for kid_id in kids:
                    kid_resp = requests.get(f"{base_url}/item/{kid_id}.json")
                    kid = kid_resp.json()
                    if kid and kid.get('text'):
                        turns.append(ConversationTurn(
                            speaker_id=kid.get('by', 'anon'),
                            text=kid.get('text', ''),
                            timestamp=datetime.fromtimestamp(kid.get('time', 0)),
                            metadata={'hn_id': kid_id, 'parent_hn_id': story_id}
                        ))
                
                conversations.append(Conversation(
                    conversation_id=f"hn_{story_id}",
                    turns=turns,
                    context={
                        'hn_url': f"https://news.ycombinator.com/item?id={story_id}",
                        'hn_score': story.get('score', 0),
                        'hn_descendants': story.get('descendants', 0),
                        'text_complexity': avg_word_len
                    },
                    source='hacker_news'
                ))
                
            except Exception:
                continue
                
        return conversations

    def ingest_stack_exchange(self, site: str = 'stackoverflow', limit: int = 50) -> List[Conversation]:
        """
        Ingest from Stack Exchange API (zero-auth).
        """
        print(f" Ingesting Sovereign Logic from {site}...")
        url = f"https://api.stackexchange.com/2.3/questions"
        params = {
            'order': 'desc',
            'sort': 'activity',
            'site': site,
            'pagesize': limit,
            'filter': 'withbody' # Requires body for turn text
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f" Stack Exchange Fetch failed: {e}")
            return []

        conversations = []
        for item in data.get('items', []):
            try:
                turns = []
                # Question turn
                turns.append(ConversationTurn(
                    speaker_id=item['owner'].get('display_name', 'anon'),
                    text=item['title'] + "\n" + item['body'],
                    timestamp=datetime.fromtimestamp(item['creation_date']),
                    metadata={'se_id': item['question_id'], 'tags': item.get('tags', [])}
                ))
                
                # Fetch answers if present (requires second hop or expanded filter)
                # For zero-auth simplicity, we'll keep it to the top question turn 
                # unless a separate answers fetch is implemented.
                
                conversations.append(Conversation(
                    conversation_id=f"se_{item['question_id']}",
                    turns=turns,
                    context={
                        'site': site,
                        'link': item['link'],
                        'se_score': item.get('score', 0),
                        'se_answer_count': item.get('answer_count', 0),
                        'se_view_count': item.get('view_count', 0)
                    },
                    source=f'stack_exchange/{site}'
                ))
            except Exception:
                continue
                
        return conversations

    def ingest_irc_logs(self, sub_path: str = 'irc_logs') -> List[Conversation]:
        """
        Ingest from local IRC snapshot with Fuzzy Decoding.
        """
        irc_dir = self.root / sub_path
        if not irc_dir.exists():
            print(f" IRC Directory not found at {irc_dir}")
            return []

        print(f" Ingesting IRC Logs with Fuzzy Decoding for Topological Friction...")
        conversations = []
        
        for log_file in irc_dir.glob('*.log'):
            try:
                with open(log_file, 'rb') as f:
                    content_bytes = f.read()
                    
                text = self._fuzzy_decode(content_bytes)
                lines = text.splitlines()
                
                turns = []
                for line in lines:
                    if not line.strip(): continue
                    # Simple IRC format parser [timestamp] <user> message
                    # or user: message
                    if '<' in line and '>' in line:
                        parts = line.split('>', 1)
                        speaker = parts[0].split('<')[-1]
                        msg = parts[1].strip()
                    elif ':' in line:
                        parts = line.split(':', 1)
                        speaker = parts[0].strip()
                        msg = parts[1].strip()
                    else:
                        speaker = 'system'
                        msg = line.strip()
                        
                    turns.append(ConversationTurn(
                        speaker_id=speaker,
                        text=msg,
                        metadata={'raw_line': line}
                    ))
                
                conversations.append(Conversation(
                    conversation_id=_stable_id("irc", log_file.name),
                    turns=turns,
                    context={'filename': log_file.name},
                    source='irc_snapshot'
                ))
            except Exception as e:
                print(f" Failed to parse IRC log {log_file.name}: {e}")
                
        return conversations

    def ingest_madoc_snapshot(self, sub_path: str = 'madoc') -> List[Conversation]:
        """
        Ingest from local MADOC snapshot (stable alternative to Reddit API).
        """
        madoc_dir = self.root / sub_path
        if not madoc_dir.exists():
            print(f" MADOC Snapshot not found at {madoc_dir}")
            return []

        print(f" Ingesting Immutable MADOC Snapshot for Silicon Sovereignty...")
        # Implementation depends on MADOC structure (usually JSON/JSONL)
        # Placeholder for directory-based scan
        return [] # MADOC logic would go here
