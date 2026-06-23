"""
ChatGPT Friction Harvester.

Ingests massive ChatGPT exports to harvest the end artifacts of friction
between sovereign but flawed entities. Supports two source formats:

  1. conversations-*.json  -- split JSON files (original format)
  2. chat.html             -- single 300MB+ HTML file whose entire <script>
                              body is a JSON array of conversation objects.
                              Uses mmap + byte-offset indexing so the file is
                              never fully loaded into RAM.

Instead of brainless tracking of only errors/hallucinations this ingests
the FULL spectrum of interactions to preserve non-ergodic structure and
non-commutativity of the conversational history.

Alias Tracking: Detects the creator's aliases (ILa, Akkaris, Willabusta)
to tag interactions for the Orchestrator's Alias Hooks.
"""

from __future__ import annotations

import os
import json
import glob
import mmap
import re
import asyncio
from typing import Dict, Any, Generator, Tuple, List, Optional

import torch

# ---------------------------------------------------------------------------
# Sentinel used to locate top-level conversation objects in chat.html.
# Every conversation object in the ChatGPT HTML export begins with this key.
# ---------------------------------------------------------------------------
_CONV_SENTINEL = b'{"async_status":'


# ---------------------------------------------------------------------------
# Shared conversation-tree traversal
# ---------------------------------------------------------------------------

def _linearise_mapping(mapping: dict) -> List[Tuple[str, str]]:
    """
    Convert the branching mapping tree that ChatGPT stores inside each
    conversation object into a flat, chronologically-ordered list of
    (role, text) pairs.

    Uses a depth-first traversal from every root node (nodes whose parent
    is absent or None).  This is identical to the logic used by
    MessageParser in fast_chat_viewer.py so both tools stay in sync.
    """
    nodes = {nid: node for nid, node in mapping.items()}

    children_map: Dict[str, List[str]] = {}
    root_ids: List[str] = []

    for nid, node in nodes.items():
        parent_id = node.get("parent")
        if parent_id is None or parent_id not in nodes:
            root_ids.append(nid)
        else:
            children_map.setdefault(parent_id, []).append(nid)

    messages: List[Tuple[str, str]] = []

    def dfs(node_id: str) -> None:
        node = nodes.get(node_id)
        if not node:
            return
        msg = node.get("message")
        if msg:
            role = msg.get("author", {}).get("role", "unknown")
            parts = msg.get("content", {}).get("parts", [])
            text = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
            if text and role in ("user", "assistant"):
                messages.append((role, text))
        for child_id in children_map.get(node_id, []):
            dfs(child_id)

    for rid in root_ids:
        dfs(rid)

    return messages


# ---------------------------------------------------------------------------
# chat.html mmap-based conversation iterator
# ---------------------------------------------------------------------------

def _iter_html_conversations(html_path: str) -> Generator[dict, None, None]:
    """
    Generator that yields raw conversation dicts from a chat.html export
    without loading the entire file into RAM.

    Strategy:
      - mmap the file read-only.
      - Scan for '{"async_status":' sentinels to locate each conversation's
        byte range.
      - Slice and JSON-parse only the bytes for the requested conversation,
        one at a time.
    """
    with open(html_path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = len(mm)

        # Build offset list
        positions: List[int] = []
        pos = 0
        while True:
            found = mm.find(_CONV_SENTINEL, pos)
            if found == -1:
                break
            positions.append(found)
            pos = found + 1

        if not positions:
            mm.close()
            return

        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else file_size
            raw = mm[start:end].rstrip(b", \n\r")
            try:
                conv = json.loads(raw)
                yield conv
            except json.JSONDecodeError:
                # Strip trailing comma that separates array elements
                try:
                    conv = json.loads(raw.rstrip(b","))
                    yield conv
                except json.JSONDecodeError:
                    pass  # skip malformed objects silently

        mm.close()


# ---------------------------------------------------------------------------
# Main harvester class
# ---------------------------------------------------------------------------

class ChatGPTFrictionHarvester:
    """
    Ingests ChatGPT conversation exports and yields (user_tensor,
    assistant_tensor, tags) dyads suitable for temporal association
    training.

    Accepted export layouts (auto-detected):
      - export_dir/ conversations-*.json   (split JSON files)
      - export_dir/ chat.html              (single monolithic HTML)

    The HTML path is preferred when both are present, because the JSON
    files are usually a subset of the full history.
    """

    def __init__(self, export_dir: str, dim: int = 256, fossilizer: Optional[Any] = None):
        self.export_dir = export_dir
        self.dim = dim
        self.fossilizer = fossilizer
        self.creator_aliases: List[str] = ["ila", "akkaris", "willabusta"]
        self.ai_archetype_keywords: List[str] = [
            "archetype", "entity", "system", "architecture", "non-human", "ai"
        ]
        self.character_roleplay_patterns: List[str] = [
            "i am ", "i will act as ", "playing the role of ", "persona: ",
            "act as ", "act as the speculative version of", "you are a ",
            "pretend to be ", "assume the role ", "imagine you are ",
            "roleplay ", "in the style of ", "respond as ",
            "take on the persona ", "you will be ", "portraying ",
            "simulating ", "acting as ", "representing ", "emulating ",
            "personifying ", "impersonating ", "channeling ",
        ]


    # ------------------------------------------------------------------
    # Dynamic registration
    # ------------------------------------------------------------------

    def register_alias(self, alias: str) -> None:
        """Allow external entities to inject new aliases dynamically."""
        a = alias.lower()
        if a not in self.creator_aliases:
            self.creator_aliases.append(a)

    def register_roleplay_pattern(self, pattern: str) -> None:
        """Allow external entities to inject new roleplay patterns dynamically."""
        p = pattern.lower()
        if p not in self.character_roleplay_patterns:
            self.character_roleplay_patterns.append(p)

    def register_archetype_keyword(self, keyword: str) -> None:
        """Allow external entities to inject new archetype keywords dynamically."""
        k = keyword.lower()
        if k not in self.ai_archetype_keywords:
            self.ai_archetype_keywords.append(k)

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Delegated to ConversationalDataProcessor to properly utilise the
        gyroidic CanonicalProjector.  Lazy-loaded to avoid circular imports.
        """
        if not hasattr(self, "_processor"):
            from .conversational_api_ingestor import ConversationalDataProcessor
            self._processor = ConversationalDataProcessor(dim=self.dim)

        tensor = self._processor.compute_text_embedding(text)

        if tensor.shape[0] != self.dim:
            adapted = torch.zeros(self.dim)
            copy_len = min(self.dim, tensor.shape[0])
            adapted[:copy_len] = tensor[:copy_len]
            return adapted

        return tensor

    # ------------------------------------------------------------------
    # Source auto-detection
    # ------------------------------------------------------------------

    def _conversation_source(self) -> Generator[dict, None, None]:
        """
        Yields raw conversation dicts from whichever source is available.
        Prefers the split JSON files (conversations-*.json) because they are
        already pre-parsed and avoid regex scanning overhead.  Falls back to
        the monolithic chat.html via mmap streaming.
        """
        json_files = sorted(
            glob.glob(os.path.join(self.export_dir, "conversations-*.json"))
        )

        if json_files:
            for file_path in json_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        yield from data
                    elif isinstance(data, dict):
                        yield data
                except Exception as e:
                    print(f"[Harvester] Error reading {file_path}: {e}")
            return

        html_path = os.path.join(self.export_dir, "chat.html")
        if os.path.exists(html_path):
            print(f"[Harvester] No JSON files found. Streaming from {html_path} via mmap.")
            yield from _iter_html_conversations(html_path)
            return

        print(f"[Harvester] Warning: No source files found in {self.export_dir!r}")

    # ------------------------------------------------------------------
    # Main harvest generator
    # ------------------------------------------------------------------

    def harvest_interactions(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], str, str], None, None]:
        """
        Generator that yields (user_tensor, assistant_tensor, tags, user_text, assistant_text)
        for every user->assistant message dyad found across all conversations.

        Tags encode friction archetype signals:
          - jarring_shift          : Jaccard < 0.05 between consecutive user prompts
          - bouligand_bubble_context: 0.05 <= Jaccard <= 0.30
          - nonergodic_agreement   : Jaccard > 0.50  (high continuity)
          - dead_end_cliff         : long user prompt, very short assistant reply
          - is_character_play      : roleplay pattern detected
          - is_human_alias         : creator alias detected
          - is_nonhuman_archetype  : AI archetype keyword detected
          - veto_status            : saturation_escalation when both alias + archetype
        """
        for conv in self._conversation_source():
            mapping = conv.get("mapping", {})
            if not mapping:
                continue

            try:
                yield from self._harvest_conv(mapping)
            except Exception as e:
                title = conv.get("title", "<unknown>")
                print(f"[Harvester] Error processing '{title}': {e}")

    def _harvest_conv(
        self, mapping: dict
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], str, str], None, None]:
        """
        Harvest all user->assistant dyads from a single conversation mapping.
        Uses the shared _linearise_mapping() DFS so tree ordering matches
        what the viewer displays.
        """
        messages = _linearise_mapping(mapping)

        last_user_text: Optional[str] = None
        last_user_tags: Dict[str, Any] = {}
        last_user_tokens: set = set()

        for role, text in messages:
            if role == "user":
                if self.fossilizer is not None and self.fossilizer.is_already_ingested(text):
                    last_user_text = None
                    continue
                current_tokens = set(text.lower().split())
                shift_tags: Dict[str, Any] = {}


                if last_user_tokens and current_tokens:
                    inter = len(last_user_tokens & current_tokens)
                    union = len(last_user_tokens | current_tokens)
                    jaccard = inter / max(1, union)

                    if jaccard < 0.05:
                        shift_tags["jarring_shift"] = 1.0
                    elif 0.05 <= jaccard <= 0.30:
                        shift_tags["bouligand_bubble_context"] = 1.0
                    elif jaccard > 0.50:
                        shift_tags["nonergodic_agreement"] = 1.0

                last_user_text = text
                last_user_tags = self._extract_tags(text)
                last_user_tags.update(shift_tags)

            elif role == "assistant" and last_user_text is not None:
                assistant_tags = self._extract_tags(text)

                merged: Dict[str, Any] = {**last_user_tags}
                for k, v in assistant_tags.items():
                    if isinstance(v, float):
                        merged[k] = max(float(merged.get(k, 0.0)), v)
                    else:
                        merged[k] = v

                # Dead-end cliff: long user prompt, very short AI reply
                if len(last_user_text) > 500 and len(text) < 50:
                    if not (
                        merged.get("is_character_play")
                        or merged.get("nonergodic_agreement")
                    ):
                        merged["dead_end_cliff"] = 1.0

                in_tensor = self._text_to_tensor(last_user_text)
                out_tensor = self._text_to_tensor(text)
                yield in_tensor, out_tensor, merged, last_user_text, text

                # Update Jaccard baseline from the just-processed user turn
                last_user_tokens = set(last_user_text.lower().split())
                last_user_text = None

    # ------------------------------------------------------------------
    # Tag extraction
    # ------------------------------------------------------------------

    def _extract_tags(self, text: str) -> Dict[str, Any]:
        """
        Tags a text segment for alias and archetype resonance signals.
        Returns a dict of tag_name -> float (or str for veto_status).
        """
        text_lower = text.lower()
        tags: Dict[str, Any] = {}

        if any(a in text_lower for a in self.creator_aliases):
            tags["is_human_alias"] = 1.0

        if any(k in text_lower for k in self.ai_archetype_keywords):
            tags["is_nonhuman_archetype"] = 1.0

        if any(p in text_lower for p in self.character_roleplay_patterns):
            tags["is_character_play"] = 1.0
            tags["archetype_identity"] = 1.0

        if "is_human_alias" in tags and "is_nonhuman_archetype" in tags:
            tags["veto_status"] = "saturation_escalation"
            # Extreme friction: nonergodic_agreement collapses to zero
            tags["nonergodic_agreement"] = 0.0

        return tags


# ---------------------------------------------------------------------------
# Async training loop
# ---------------------------------------------------------------------------

async def auto_temporal_training_loop(
    harvester: ChatGPTFrictionHarvester,
    trainer: Any,
    delay: float = 0.1,
) -> None:
    """
    Background asyncio task that continuously feeds harvested dyads into
    the temporal association trainer without blocking the event loop.
    """
    print(f"[Harvester] Starting auto-temporal training from {harvester.export_dir!r}...")
    count = 0
    try:
        for in_tensor, out_tensor, tags, user_text, assistant_text in harvester.harvest_interactions():
            metrics = trainer.train_on_interaction(
                in_tensor, out_tensor, tag_weights=tags
            )
            count += 1

            # Fossilize dyad if a fossilizer is present on the model
            fossilizer = getattr(trainer.model, "fossilizer", None)
            if fossilizer is not None:
                try:
                    from src.core.knowledge_dyad_fossilizer import KnowledgeDyad
                    dyad = KnowledgeDyad(
                        linguistic_description=user_text,
                        metadata={"tags": list(tags.keys()), "response_text": assistant_text}
                    )
                    fossilizer.fossilize(dyad, in_tensor, seed_state=out_tensor)
                except Exception as fe:
                    if count % 100 == 0 or count < 5:
                        print(f"[Harvester] Background fossilization failed: {fe}")

            if count % 100 == 0:
                tc = metrics.get("temporal_coherence", 0) if metrics else 0
                print(f"[Harvester] Ingested {count} dyads. Temporal Coherence: {tc:.3f}")

            await asyncio.sleep(delay)

    except Exception as e:
        print(f"[Harvester] Training loop aborted: {e}")
    finally:
        print(f"[Harvester] Auto-temporal training complete. Total dyads: {count}")

