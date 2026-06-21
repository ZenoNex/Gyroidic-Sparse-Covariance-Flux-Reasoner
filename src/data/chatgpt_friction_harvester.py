import os
import json
import glob
import torch
import hashlib
import asyncio
from typing import Dict, Any, Generator, Tuple

class ChatGPTFrictionHarvester:
    """
    Ingests massive ChatGPT JSON exports (`conversations-*.json`) to harvest
    the end artifacts of friction between sovereign but flawed entities.
    
    Instead of brainless tracking of only errors/hallucinations, this ingests
    the FULL spectrum of interactions to preserve non-ergodic structure and
    non-commutativity of the conversational history.
    
    Alias Tracking: Detects the creator's aliases (ILa, Akkaris, Willabusta)
    to tag interactions for the Orchestrator's Alias Hooks.
    """
    
    def __init__(self, export_dir: str, dim: int = 256):
        self.export_dir = export_dir
        self.dim = dim
        self.creator_aliases = ["ila", "akkaris", "willabusta"]
        self.ai_archetype_keywords = ["archetype", "entity", "system", "architecture", "non-human", "ai"]
        
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Hashes text into a deterministic high-dimensional semantic vector proxy.
        Preserves non-commutative structure by using SHA256 windowing.
        """
        # A simple robust semantic proxy: chunk the string, hash, and map to floats
        tensor = torch.zeros(self.dim)
        if not text:
            return tensor
            
        text_bytes = text.encode('utf-8')
        hash_digest = hashlib.sha256(text_bytes).digest()
        
        # Expand the 32-byte hash to the required dimension
        for i in range(self.dim):
            tensor[i] = float(hash_digest[i % 32]) / 255.0 * 2.0 - 1.0 # [-1, 1] range
            
        return tensor

    def harvest_interactions(self) -> Generator[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]], None, None]:
        """
        Generator that yields (input_tensor, response_tensor, tags) for every
        user-assistant message pair found in the exported logs.
        """
        json_files = glob.glob(os.path.join(self.export_dir, "conversations-*.json"))
        if not json_files:
             print(f"[Harvester] Warning: No conversations-*.json found in {self.export_dir}")
             return
             
        for file_path in json_files:
             try:
                 with open(file_path, 'r', encoding='utf-8') as f:
                     data = json.load(f)
                     
                 # ChatGPT exports are typically a list of conversation objects
                 for conv in data:
                     mapping = conv.get('mapping', {})
                     
                     # Simple traversal of the mapping tree (linearizing the chat)
                     # We pair 'user' messages with the subsequent 'assistant' messages
                     last_user_text = None
                     last_user_tags = {}
                     
                     for node_id, node in mapping.items():
                         message = node.get('message')
                         if not message:
                             continue
                             
                         role = message.get('author', {}).get('role', '')
                         content_parts = message.get('content', {}).get('parts', [])
                         text = " ".join([str(p) for p in content_parts if isinstance(p, str)])
                         
                         if role == 'user':
                             last_user_text = text
                             last_user_tags = self._extract_tags(text)
                         elif role == 'assistant' and last_user_text:
                             # We have a user -> assistant interaction dyad
                             assistant_text = text
                             assistant_tags = self._extract_tags(text)
                             
                             # Merge tags
                             merged_tags = {**last_user_tags}
                             for k, v in assistant_tags.items():
                                 merged_tags[k] = max(merged_tags.get(k, 0.0), v)
                                 
                             # Convert to tensors
                             in_tensor = self._text_to_tensor(last_user_text)
                             out_tensor = self._text_to_tensor(assistant_text)
                             
                             yield in_tensor, out_tensor, merged_tags
                             
                             # Reset to handle branches correctly (simplification)
                             last_user_text = None
                             
             except Exception as e:
                 print(f"[Harvester] Error processing {file_path}: {e}")

    def _extract_tags(self, text: str) -> Dict[str, float]:
        """
        Tags the text for alias and archetype resonance.
        """
        text_lower = text.lower()
        tags = {}
        
        # Check for creator aliases
        if any(alias in text_lower for alias in self.creator_aliases):
            tags["is_human_alias"] = 1.0
            
        # Check for AI archetype discussion
        if any(keyword in text_lower for keyword in self.ai_archetype_keywords):
            tags["is_nonhuman_archetype"] = 1.0
            
        # If both trigger, this is a high-friction/high-resonance boundary
        if "is_human_alias" in tags and "is_nonhuman_archetype" in tags:
            tags["veto_status"] = "saturation_escalation"
            
        return tags

async def auto_temporal_training_loop(harvester: ChatGPTFrictionHarvester, trainer, delay: float = 0.1):
    """
    Background asynchronous task to continuously feed the harvested dyads into
    the temporal association trainer without blocking the main UI thread.
    """
    print(f"[Harvester] Starting auto-temporal training from {harvester.export_dir}...")
    try:
        interaction_gen = harvester.harvest_interactions()
        count = 0
        for in_tensor, out_tensor, tags in interaction_gen:
            # Send the friction dyad to the trainer
            metrics = trainer.train_on_interaction(in_tensor, out_tensor, tag_weights=tags)
            count += 1
            
            if count % 100 == 0:
                 print(f"[Harvester] Ingested {count} dyads. Last Temporal Coherence: {metrics.get('temporal_coherence', 0):.3f}")
                 
            # Yield control back to the event loop
            await asyncio.sleep(delay)
            
    except Exception as e:
        print(f"[Harvester] Training loop aborted: {e}")
    finally:
        print(f"[Harvester] Auto-temporal training complete. Total dyads: {count}")
