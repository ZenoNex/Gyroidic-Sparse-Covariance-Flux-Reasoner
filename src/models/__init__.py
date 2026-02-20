"""Neural network models for Gyroidic Sparse Covariance Flux Reasoner.

Includes:
- Main reasoner model (GyroidicFluxReasoner)
- Modular attention with Birkhoff projection
- Polynomial embeddings
- Resonance cavity memory
- Geometric introspection heads

Author: William Matthew Bryant
Created: January 2026
"""

from .gyroid_reasoner import GyroidicFluxReasoner
from .modular_attention import ModularAttention, ModularTransformerLayer
from .polynomial_embeddings import PolynomialFunctionalEmbedder
from .resonance_cavity import ResonanceCavity, GyroidicFluxAlignment
from .introspection_head import GeometricSelfModelProbe, AggregateGeometricSelfModel

__all__ = [
    'GyroidicFluxReasoner',
    'ModularAttention',
    'ModularTransformerLayer',
    'PolynomialFunctionalEmbedder',
    'ResonanceCavity',
    'GyroidicFluxAlignment',
    'GeometricSelfModelProbe',
    'AggregateGeometricSelfModel',
]
