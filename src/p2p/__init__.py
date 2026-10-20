"""
P2P Decentralized Consensus and Zero-Knowledge Proof Package.

Includes:
- Bonfire Nomadic Ring consensus and Fractal Discontinuity Embeddings
- Freenet WebSocket client
- ZK Aggregator for Chern-Simons and Leontief invariants
"""

from .bonfire_consensus import (
    BonfireNomadicRing,
    FractalDiscontinuityEmbedding
)
from .freenet_ws_client import FreenetClient
from .zk_aggregator import ZKAggregator

__all__ = [
    'BonfireNomadicRing',
    'FractalDiscontinuityEmbedding',
    'FreenetClient',
    'ZKAggregator',
]
