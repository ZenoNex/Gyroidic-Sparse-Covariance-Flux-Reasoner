"""
P2P Decentralized Consensus and Zero-Knowledge Proof Package.

Includes:
- Bonfire Nomadic Ring consensus and Fractal Discontinuity Embeddings
- Freenet WebSocket client
- ZK Aggregator for Chern-Simons and Leontief invariants
"""

from .bonfire_consensus import (
    BonfireNomadicRing,
    FractalDiscontinuityEmbedding,
    ShamirThresholdSharing,
)
from .freenet_ws_client import FreenetClient, RoguePredictorDefense
from .zk_aggregator import ZKAggregator

__all__ = [
    'BonfireNomadicRing',
    'FractalDiscontinuityEmbedding',
    'ShamirThresholdSharing',
    'FreenetClient',
    'RoguePredictorDefense',
    'ZKAggregator',
]
