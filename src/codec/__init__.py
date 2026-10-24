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
from .vision_utilities import (
    get_russian_doll_projection,
    project_chromata_filaments,
    project_geometrize_primitives,
    project_multimodal_fingerprint_to_96
)

__all__ = [
    'GyroidicCodec',
    'CodecConfig',
    'EncodingResult',
    'GyroidImageProjector',
    'ConformalLogPolarProjector',
    'IntercosaminationOperator',
    'MirrorTestProbe',
    'get_russian_doll_projection',
    'project_chromata_filaments',
    'project_geometrize_primitives',
    'project_multimodal_fingerprint_to_96',
]

