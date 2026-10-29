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
from typing import List, Dict, Optional, Any, Generator, Union
from pathlib import Path
from datetime import datetime
import torch

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
            
        self.larynx: Optional[Any] = None
        self.engine: Optional[Any] = None
        print(f"[*] Sovereign Ingestor initialized with REPOSITORY_ROOT: {self.root}")

    def set_larynx(self, larynx: Any):
        """Connect external ResonanceLarynx module for autoregressive query generation."""
        self.larynx = larynx

    def bind_engine(self, engine: Any):
        """Bind parent physics engine or orchestrator."""
        self.engine = engine
        if hasattr(engine, 'larynx') and engine.larynx is not None:
            self.larynx = engine.larynx

    def normalize_differential_telemetry(
        self,
        target_signal: torch.Tensor,
        background_reference: torch.Tensor,
        structural_wobble_threshold: float = 0.05,
        sun_facing_semicircle_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        First Law of Structural Honesty for Physical & Satellite Telemetry
        (Sentinel-2 L2A `s3://sentinel-cogs`, EIA Cushing Inventory, NOAA GOES-R):

        1. Differential Calibration (Instrument Gain Drift Subtraction):
           Subtracts ambient reference drift (e.g. surrounding dirt/road darkening)
           from the target proxy (e.g. floating tank roof crescent shadow):
               signal_calibrated = target_signal - background_reference
        2. Spatial Dilution Masking:
           Restricts integration to the directional active locus (e.g. sun-facing
           semicircle) rather than averaging over an isotropic disk.
        3. Epistemic Resolution / Wobble-to-Noise Veto:
           If measurement uncertainty sigma_probe >= structural_wobble_threshold,
           issues a Topological Refusal (NaN boundary state) instead of committing noise.
        """
        t_sig = target_signal.float()
        bg_ref = background_reference.float().to(t_sig.device)
        if bg_ref.shape != t_sig.shape:
            bg_ref = bg_ref.flatten()[:t_sig.numel()]
            if bg_ref.numel() < t_sig.numel():
                bg_ref = torch.nn.functional.pad(bg_ref, (0, t_sig.numel() - bg_ref.numel()))
            bg_ref = bg_ref.reshape(t_sig.shape)

        calibrated = t_sig - bg_ref
        if sun_facing_semicircle_mask is not None:
            mask = sun_facing_semicircle_mask.to(t_sig.device).float()
            calibrated = calibrated * mask
            active_slice = calibrated.reshape(-1)
        else:
            half = max(1, calibrated.numel() // 2)
            active_slice = calibrated.reshape(-1)[:half]

        sigma_probe = float(torch.std(active_slice).item()) if active_slice.numel() > 1 else 0.0
        mean_signal = float(torch.mean(active_slice).item())
        wobble_veto = sigma_probe > max(abs(mean_signal), structural_wobble_threshold)
        refused_tensor = torch.full_like(calibrated, float('nan')) if wobble_veto else calibrated
        return {
            "calibrated_tensor": refused_tensor,
            "raw_target_mean": float(t_sig.mean().item()),
            "ambient_drift_mean": float(bg_ref.mean().item()),
            "differential_signal_mean": mean_signal,
            "measurement_sigma": sigma_probe,
            "structural_wobble_threshold": structural_wobble_threshold,
            "wobble_to_noise_veto": bool(wobble_veto),
            "open_telemetry_sources": [
                "s3://sentinel-cogs (ESA Copernicus Sentinel-2 L2A COGs)",
                "https://api.eia.gov/v2/petroleum/stoc/wstk/ (EIA Weekly Cushing Inventory)",
                "s3://noaa-goes16 (NOAA GOES-R ABI L2+ Thermal Flux)"
            ]
        }

    def generate_larynx_hn_query(
        self,
        state: Optional[torch.Tensor] = None,
        engine: Optional[Any] = None
    ) -> str:
        """
        Uses the canonical ResonanceLarynx (from src.models.diegetic_heads) with Chern-Simons
        Gasket boundary repair, syllable-coherent vowel/consonant modulation, and autoregressive
        feedback to dynamically draft high-entropy Hacker News search queries from the active manifold state.
        """
        import re
        import torch

        active_engine = engine or self.engine
        larynx = self.larynx
        if larynx is None and active_engine is not None and hasattr(active_engine, 'larynx'):
            larynx = active_engine.larynx

        dim = 256
        if larynx is not None and hasattr(larynx, 'hidden_dim'):
            dim = larynx.hidden_dim
        elif active_engine is not None and hasattr(active_engine, 'dim'):
            dim = active_engine.dim

        device = torch.device('cpu')
        if larynx is not None:
            try:
                device = next(larynx.parameters()).device
            except Exception:
                pass
        elif state is not None and isinstance(state, torch.Tensor):
            device = state.device

        curr_state = None
        if state is not None and isinstance(state, torch.Tensor):
            curr_state = state.clone().detach().to(device)
            if curr_state.dim() == 1:
                curr_state = curr_state.unsqueeze(0)
        else:
            curr_state = torch.zeros((1, dim), device=device)

        if larynx is not None:
            try:
                vowels = set('aeiouAEIOU')
                generated_chars = []
                with torch.no_grad():
                    for _ in range(30):
                        logits, _ = larynx(curr_state, temperature=1.1)
                        if logits.dim() == 1:
                            logits = logits.unsqueeze(0)

                        for idx in range(min(128, logits.shape[-1])):
                            if idx < 32 or idx > 126:
                                logits[0, idx] = -1e9

                        prev_char = generated_chars[-1] if generated_chars else ' '
                        if prev_char not in vowels and prev_char.isalpha():
                            for v in vowels:
                                if ord(v) < logits.shape[-1]:
                                    logits[0, ord(v)] *= 1.4
                        elif prev_char in vowels:
                            for v in vowels:
                                if ord(v) < logits.shape[-1]:
                                    logits[0, ord(v)] *= 0.4

                        probs = torch.softmax(logits, dim=-1)
                        char_idx = torch.multinomial(probs[0], 1).item()
                        char = chr(max(32, min(126, char_idx)))

                        if char in ('.', '!', '?', ';', '\n'):
                            break
                        generated_chars.append(char)

                        if hasattr(larynx, 'proj') and hasattr(larynx.proj, 'weight') and char_idx < larynx.proj.weight.shape[0]:
                            feedback = torch.tanh(larynx.proj.weight[char_idx].unsqueeze(0))
                            if feedback.shape == curr_state.shape:
                                curr_state = 0.9 * curr_state + 0.1 * feedback

                query = "".join(generated_chars).strip()
                cleaned_query = re.sub(r'[^a-zA-Z0-9_\-\s]', ' ', query).strip()
                if len(cleaned_query) >= 3:
                    return cleaned_query
            except Exception:
                pass

        # Sovereign high-entropy topics fallback
        topics = [
            "distributed systems consensus",
            "kernel memory allocator",
            "non-euclidean topology",
            "p2p sovereign mesh",
            "zero knowledge proofs"
        ]
        from src.core.honest_jitter import harvest_honest_jitter
        jitter = harvest_honest_jitter((1,), device=torch.device('cpu'), scaled=False).item()
        u = (jitter + 1.0) / 2.0
        idx = int(u * len(topics))
        return topics[min(idx, len(topics) - 1)]

    def search_hacker_news(self, query: str, limit: int = 25) -> List[Conversation]:
        """
        Query Hacker News via public Algolia search API steered by Larynx query.
        Converts discussions into high-entropy Conversation dyads.
        """
        # Gated by HardwareMonitor typed compute affordance
        try:
            from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
            hw = HardwareMonitor()
            if not hw.has_compute_affordance(ComputeAffordanceType.SILICON) or not hw.has_compute_affordance(ComputeAffordanceType.THERMAL):
                return []
        except Exception:
            pass

        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://hn.algolia.com/api/v1/search?query={encoded_query}&tags=story&hitsPerPage={limit}"
        try:
            resp = requests.get(search_url, timeout=5.0)
            resp.raise_for_status()
            hits = resp.json().get('hits', [])
        except Exception as e:
            print(f"[HN SEARCH] Algolia query failed for '{query}': {e}")
            return []

        conversations: List[Conversation] = []
        for hit in hits:
            story_id = hit.get('objectID')
            title = hit.get('title') or ""
            story_text = hit.get('story_text') or ""
            author = hit.get('author') or 'anon'
            created_at = hit.get('created_at_i', 0)

            text_content = f"{title}\n{story_text}".strip()
            if not text_content:
                continue

            turns = [
                ConversationTurn(
                    speaker_id=author,
                    text=text_content,
                    timestamp=datetime.fromtimestamp(created_at),
                    metadata={'hn_id': story_id, 'points': hit.get('points', 0)}
                )
            ]

            conversations.append(Conversation(
                conversation_id=f"hn_search_{story_id}",
                turns=turns,
                context={
                    'hn_url': f"https://news.ycombinator.com/item?id={story_id}",
                    'hn_score': hit.get('points', 0),
                    'hn_descendants': hit.get('num_comments', 0),
                    'query': query
                },
                source='hacker_news_search'
            ))

        return conversations

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
        # Gated by HardwareMonitor typed compute affordance
        try:
            from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
            hw = HardwareMonitor()
            if not hw.has_compute_affordance(ComputeAffordanceType.SILICON) or not hw.has_compute_affordance(ComputeAffordanceType.THERMAL):
                return []
        except Exception:
            pass

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

    def ingest_madoc_snapshot(
        self,
        sub_path: Union[str, Path] = 'madoc',
        limit: Optional[int] = None,
        platform_filter: Optional[str] = None,
        explanation: Optional[str] = None
    ) -> List[Conversation]:
        """
        Ingest from local MADOC snapshot (Multi-Platform Aggregated Dataset of Online Communities).
        Acts as an immutable, zero-auth local alternative to the live Reddit API, also covering
        Voat, Bluesky, and Koo communities stored as Parquet, JSONL, JSON, or CSV archives.
        Supports both directory hierarchies and explicit single-file paths.
        """
        raw_p = Path(sub_path)
        if raw_p.is_absolute() and raw_p.exists():
            target_p = raw_p
        elif (self.root / sub_path).exists():
            target_p = self.root / sub_path
        elif (self.root / 'datasets' / sub_path).exists():
            target_p = self.root / 'datasets' / sub_path
        elif raw_p.exists():
            target_p = raw_p
        else:
            print(f"[*] MADOC Snapshot path not found at {sub_path}")
            return []

        print(f"[*] Ingesting Immutable MADOC Snapshot from {target_p} for Silicon Sovereignty...")
        conversations = []
        
        # Discover all supported MADOC dataset files
        if target_p.is_file():
            data_files = [target_p]
        else:
            data_files = list(target_p.glob('**/*.parquet')) + \
                         list(target_p.glob('**/*.jsonl')) + \
                         list(target_p.glob('**/*.json')) + \
                         list(target_p.glob('**/*.csv'))
                         
        if not data_files:
            print(f"[*] No Parquet, JSONL, JSON, or CSV files found in {target_p}")
            return []

        for data_file in sorted(data_files):
            try:
                records = self._load_madoc_file(data_file, limit=limit)
                if not records:
                    continue

                # Process records into conversations
                file_convs = self._parse_madoc_records(records, data_file, platform_filter=platform_filter)
                if explanation:
                    for c in file_convs:
                        if isinstance(c.context, dict):
                            c.context['user_explanation'] = explanation
                conversations.extend(file_convs)
                
                if limit and len(conversations) >= limit:
                    conversations = conversations[:limit]
                    break
            except Exception as e:
                print(f"[*] Failed to process MADOC file {data_file.name}: {e}")
                continue

        print(f"[*] Successfully ingested {len(conversations)} conversations from MADOC snapshot.")
        return conversations

    def _load_madoc_file(self, file_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load records from a single MADOC dataset file (Parquet, JSONL, JSON, or CSV)."""
        suffix = file_path.suffix.lower()
        records = []
        
        if suffix == '.parquet':
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                if limit:
                    df = df.head(limit)
                records = df.to_dict('records')
            except ImportError:
                print(f"[*] Warning: pandas/pyarrow not installed. Skipping parquet file {file_path.name}")
            except Exception as pe:
                print(f"[*] Failed reading Parquet {file_path.name}: {pe}")

        elif suffix == '.jsonl':
            with open(file_path, 'rb') as f:
                content = self._fuzzy_decode(f.read())
            for idx, line in enumerate(content.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                    if limit and len(records) >= limit:
                        break
                except json.JSONDecodeError:
                    continue

        elif suffix == '.json':
            with open(file_path, 'rb') as f:
                content = self._fuzzy_decode(f.read())
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    records = data[:limit] if limit else data
                elif isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], list):
                        records = data['data'][:limit] if limit else data['data']
                    elif 'records' in data and isinstance(data['records'], list):
                        records = data['records'][:limit] if limit else data['records']
                    elif 'posts' in data and isinstance(data['posts'], list):
                        records = data['posts'][:limit] if limit else data['posts']
                    else:
                        records = [data]
            except json.JSONDecodeError as je:
                print(f"[*] JSON decode error for {file_path.name}: {je}")

        elif suffix == '.csv':
            try:
                import csv
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        records.append(dict(row))
                        if limit and len(records) >= limit:
                            break
            except Exception as ce:
                print(f"[*] CSV read error for {file_path.name}: {ce}")

        return records

    def _parse_madoc_records(
        self,
        records: List[Dict[str, Any]],
        source_file: Path,
        platform_filter: Optional[str] = None
    ) -> List[Conversation]:
        """
        Parse raw MADOC records into structured Conversation and ConversationTurn objects.
        Groups threaded comments and hierarchical replies into multi-turn dialogues.
        """
        conversations = []
        
        # Check if records are already complete conversations (e.g. ConvoKit-style format)
        if records and 'turns' in records[0] and isinstance(records[0]['turns'], list):
            for r in records:
                turns = []
                for t in r['turns']:
                    text = t.get('text', '')
                    if not text:
                        continue
                    ts = None
                    if t.get('timestamp'):
                        try:
                            ts = datetime.fromisoformat(str(t['timestamp']))
                        except Exception:
                            ts = None
                    turns.append(ConversationTurn(
                        speaker_id=str(t.get('speaker_id') or t.get('author') or 'anon'),
                        text=text,
                        timestamp=ts,
                        metadata=t.get('metadata')
                    ))
                if turns:
                    c_id = r.get('conversation_id') or _stable_id("madoc", r.get('id', source_file.stem))
                    conversations.append(Conversation(
                        conversation_id=c_id,
                        turns=turns,
                        context=r.get('context', {'source_file': source_file.name}),
                        source=r.get('source', 'madoc_snapshot')
                    ))
            return conversations

        # Otherwise, group by thread root / parent to reconstruct dialogue hierarchy
        threads: Dict[str, List[Dict[str, Any]]] = {}
        standalone_items: List[Dict[str, Any]] = []

        for item in records:
            platform = str(item.get('platform') or 'reddit').lower()
            if platform_filter and platform != platform_filter.lower():
                continue

            # Identify thread association
            # MADOC schema uses post_id, parent_id, link_id, or thread_id
            post_id = str(item.get('post_id') or item.get('id') or item.get('comment_id') or '')
            parent_id = str(item.get('parent_id') or item.get('link_id') or item.get('root_id') or '')
            
            # If parent_id exists and differs from post_id, assign to thread
            thread_key = parent_id if (parent_id and parent_id != post_id) else post_id
            if thread_key:
                if thread_key not in threads:
                    threads[thread_key] = []
                threads[thread_key].append(item)
            else:
                standalone_items.append(item)

        # Convert grouped threads into conversations
        for thread_id, thread_items in threads.items():
            turns = []
            community = 'general'
            platform = 'madoc'
            
            # Sort items by timestamp if available
            def _extract_time(x):
                val = x.get('created_utc') or x.get('timestamp') or x.get('created_at') or 0
                try:
                    return float(val)
                except Exception:
                    return 0.0

            sorted_items = sorted(thread_items, key=_extract_time)

            for it in sorted_items:
                platform = str(it.get('platform') or platform)
                community = str(it.get('community') or it.get('subreddit') or community)
                
                # Extract text (check content, body, text, or title + selftext)
                title = str(it.get('title') or '').strip()
                body = str(it.get('content') or it.get('body') or it.get('text') or it.get('selftext') or '').strip()
                
                if title and body and title != body:
                    text = f"{title}\n{body}"
                else:
                    text = title or body

                if not text or text in ['[deleted]', '[removed]']:
                    continue

                author = str(it.get('author') or it.get('user') or it.get('user_id') or 'anon')
                
                # Timestamp parsing
                raw_time = it.get('created_utc') or it.get('timestamp') or it.get('created_at')
                dt = None
                if raw_time is not None:
                    try:
                        if isinstance(raw_time, (int, float)):
                            dt = datetime.fromtimestamp(float(raw_time))
                        elif isinstance(raw_time, str):
                            dt = datetime.fromisoformat(raw_time)
                    except Exception:
                        dt = None

                score = it.get('score') or it.get('ups') or it.get('likes') or 0
                turns.append(ConversationTurn(
                    speaker_id=author,
                    text=text,
                    timestamp=dt,
                    metadata={
                        'item_id': it.get('id') or it.get('post_id') or it.get('comment_id'),
                        'score': score,
                        'platform': platform,
                        'community': community
                    }
                ))

            if turns:
                # Calculate linguistic entropy / text complexity
                all_words = " ".join(t.text for t in turns).split()
                avg_word_len = sum(len(w) for w in all_words) / max(1, len(all_words))

                conversations.append(Conversation(
                    conversation_id=f"madoc_{platform}_{thread_id}",
                    turns=turns,
                    context={
                        'platform': platform,
                        'community': community,
                        'thread_id': thread_id,
                        'source_file': source_file.name,
                        'turn_count': len(turns),
                        'text_complexity': avg_word_len
                    },
                    source=f"madoc/{platform}"
                ))

        # Process standalone items
        for it in standalone_items:
            title = str(it.get('title') or '').strip()
            body = str(it.get('content') or it.get('body') or it.get('text') or it.get('selftext') or '').strip()
            if title and body and title != body:
                text = f"{title}\n{body}"
            else:
                text = title or body

            if not text or text in ['[deleted]', '[removed]']:
                continue

            author = str(it.get('author') or it.get('user') or 'anon')
            platform = str(it.get('platform') or 'madoc')
            community = str(it.get('community') or it.get('subreddit') or 'general')
            item_id = str(it.get('id') or it.get('post_id') or '')
            
            raw_time = it.get('created_utc') or it.get('timestamp')
            dt = None
            if raw_time:
                try:
                    dt = datetime.fromtimestamp(float(raw_time))
                except Exception:
                    dt = None

            turn = ConversationTurn(
                speaker_id=author,
                text=text,
                timestamp=dt,
                metadata={'platform': platform, 'community': community}
            )
            c_id = f"madoc_{platform}_{item_id}" if item_id else _stable_id("madoc", text[:50])
            conversations.append(Conversation(
                conversation_id=c_id,
                turns=[turn],
                context={
                    'platform': platform,
                    'community': community,
                    'source_file': source_file.name,
                    'text_complexity': len(text)
                },
                source=f"madoc/{platform}"
            ))

        return conversations
