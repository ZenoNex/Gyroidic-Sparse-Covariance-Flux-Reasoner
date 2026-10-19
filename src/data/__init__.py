"""Data Ingestion Package for Gyroidic Sparse Covariance Flux Reasoner.

Contains Sovereign Ingestion (ArXiv, HN, Wikipedia), GLTF Gaussian Splats,
Local Datasets, and WebP metadata extractors.
"""

from .gltf_splat_ingestor import GltfSplatIngestionPipeline
from .webp_prompt_extractor import WebPPromptExtractor
from .conversational_types import ConversationTurn, Conversation

__all__ = [
    'GltfSplatIngestionPipeline',
    'WebPPromptExtractor',
    'ConversationTurn',
    'Conversation',
]
