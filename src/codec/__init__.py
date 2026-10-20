"""
Gyroidic Image Codec Package.

Provides non-Abelian text-image CRT encoding, conformal log-polar projection,
and vision surgery operators (Intercosamination and MirrorTestProbe).
"""

from .gyroidic_codec import (
    GyroidicCodec,
    CodecConfig,
    EncodingResult,
    GyroidImageProjector
)
from .conformal_log_polar import ConformalLogPolarProjector
from .vision_surgery import IntercosaminationOperator, MirrorTestProbe

__all__ = [
    'GyroidicCodec',
    'CodecConfig',
    'EncodingResult',
    'GyroidImageProjector',
    'ConformalLogPolarProjector',
    'IntercosaminationOperator',
    'MirrorTestProbe',
]
