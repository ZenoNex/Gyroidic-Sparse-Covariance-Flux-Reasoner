"""
Conversational Data Types

Shared dataclasses and utilities for the Gyroidic Reasoner's 
ingestion pipeline. Separated to prevent circular imports.

Author: William Matthew Bryant
Created: April 2026
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import hashlib
import torch

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

def _stable_id(prefix: str, obj: Any) -> str:
    """Deterministic ID using Blake2s over canonical JSON (sorted keys)."""
    try:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(obj)
    hexdigest = hashlib.blake2s(payload.encode('utf-8'), digest_size=10).hexdigest()
    return f"{prefix}_{hexdigest}"
