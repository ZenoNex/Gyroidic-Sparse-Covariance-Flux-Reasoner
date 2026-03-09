"""
Unknowledge Flux: Backward-compatibility re-exports.

The canonical implementations now live in src/topology/unknowledge_domain.py.
This module re-exports them so existing imports continue to work.
"""

from src.topology.unknowledge_domain import NostalgicLeakFunctional, EntropicMischiefProbe

__all__ = ['NostalgicLeakFunctional', 'EntropicMischiefProbe']
