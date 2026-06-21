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
        self.character_roleplay_patterns = [
            "i am ", "i will act as ", "playing the role of ", "persona: ", "act as ",
            "act as the speculative version of", "you are a ", "pretend to be ",
            "assume the role ", "imagine you are ", "roleplay ", "in the style of ",
            "respond as ", "take on the persona ", "you will be ",
            "portraying ", "simulating ", "acting as ", "representing ", "emulating ", 
            "personifying ", "impersonating ", "channeling "
        ]
        
    def register_alias(self, alias: str):
        """Allow external entities (e.g., Agent Substrate) to inject new aliases dynamically."""
        alias_lower = alias.lower()
        if alias_lower not in self.creator_aliases:
            self.creator_aliases.append(alias_lower)

    def register_roleplay_pattern(self, pattern: str):
        """Allow external entities to inject new roleplay patterns/gerunds dynamically."""
        pattern_lower = pattern.lower()
        if pattern_lower not in self.character_roleplay_patterns:
            self.character_roleplay_patterns.append(pattern_lower)

    def register_archetype_keyword(self, keyword: str):
        """Allow external entities to inject new archetype keywords dynamically."""
        keyword_lower = keyword.lower()
        if keyword_lower not in self.ai_archetype_keywords:
            self.ai_archetype_keywords.append(keyword_lower)
        
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Delegated to ConversationalDataProcessor to avoid reinventing the wheel
        and properly utilize the gyroidic CanonicalProjector.
        """
        # We lazy-load the processor to prevent circular imports or heavy init if unused
        if not hasattr(self, '_processor'):
            from .conversational_api_ingestor import ConversationalDataProcessor
            self._processor = ConversationalDataProcessor()
        
        # Ensure we return the correct dimension if dim is specified
        tensor = self._processor.compute_text_embedding(text)
        
        # In case the harvester requested a specific dim, we might need to pad/truncate
        # but CanonicalProjector outputs [64]. Let's just return it or adapt.
        if tensor.shape[0] != self.dim:
            # Simple adaptive resize if dimensions mismatch
            adapted = torch.zeros(self.dim)
            copy_len = min(self.dim, tensor.shape[0])
            adapted[:copy_len] = tensor[:copy_len]
            return adapted
            
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
                     last_user_tokens = set()
                     
                     for node_id, node in mapping.items():
                         message = node.get('message')
                         if not message:
                             continue
                             
                         role = message.get('author', {}).get('role', '')
                         content_parts = message.get('content', {}).get('parts', [])
                         text = " ".join([str(p) for p in content_parts if isinstance(p, str)])
                         
                         if role == 'user':
                             # Compute Jaccard Shift against previous user prompt
                             current_tokens = set(text.lower().split())
                             shift_tags = {}
                             if last_user_tokens and current_tokens:
                                 intersection = len(last_user_tokens.intersection(current_tokens))
                                 union = len(last_user_tokens.union(current_tokens))
                                 jaccard = intersection / max(1, union)
                                 
                                 if jaccard < 0.05:
                                     shift_tags["jarring_shift"] = 1.0
                                 elif 0.05 <= jaccard <= 0.3:
                                     shift_tags["bouligand_bubble_context"] = 1.0
                                 elif jaccard > 0.5:
                                     shift_tags["nonergodic_agreement"] = 1.0 # High overlap/continuity
                                     
                             last_user_text = text
                             last_user_tags = self._extract_tags(text)
                             last_user_tags.update(shift_tags)
                         elif role == 'assistant' and last_user_text:
                             # We have a user -> assistant interaction dyad
                             assistant_text = text
                             assistant_tags = self._extract_tags(text)
                             
                             # Merge tags
                             merged_tags = {**last_user_tags}
                             for k, v in assistant_tags.items():
                                 merged_tags[k] = max(merged_tags.get(k, 0.0), v)
                                 
                             # Dead-End Detection: long user prompt, very short AI response
                             # Ignore short responses if they are part of character play or non-ergodic agreement
                             if len(last_user_text) > 500 and len(assistant_text) < 50:
                                 if not (merged_tags.get("is_character_play") or merged_tags.get("nonergodic_agreement")):
                                     merged_tags["dead_end_cliff"] = 1.0
                                 
                             # Convert to tensors
                             in_tensor = self._text_to_tensor(last_user_text)
                             out_tensor = self._text_to_tensor(assistant_text)
                             
                             yield in_tensor, out_tensor, merged_tags
                             
                             # Track Jaccard continuity between consecutive user messages
                             last_user_tokens = set(last_user_text.lower().split())
                             
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
        
        # Check for creator aliases (Do not create a privileged class, just geometric anchors)
        if any(alias in text_lower for alias in self.creator_aliases):
            tags["is_human_alias"] = 1.0
            
        # Check for AI archetype discussion
        if any(keyword in text_lower for keyword in self.ai_archetype_keywords):
            tags["is_nonhuman_archetype"] = 1.0
            
        # Character roleplay extraction
        if any(pattern in text_lower for pattern in self.character_roleplay_patterns):
            tags["is_character_play"] = 1.0
            tags["archetype_identity"] = 1.0
            
        # If both trigger, this is a high-friction/high-resonance boundary
        if "is_human_alias" in tags and "is_nonhuman_archetype" in tags:
            tags["veto_status"] = "saturation_escalation"
            tags["nonergodic_agreement"] = 0.0 # Extreme friction point
            
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
