"""
Diegetic UI, Voxelboxter Engine, and Multi-Interface Server Package.

Exposes:
- DiegeticPhysicsEngine and HybridPhysicsBackend
- VoxelboxterEngine and PointerlessOctree simulation primitives
- WikipediaIntegration and conversational bridges
"""

try:
    from .diegetic_backend import DiegeticPhysicsEngine, HybridPhysicsBackend
except ImportError:
    DiegeticPhysicsEngine = None
    HybridPhysicsBackend = None

try:
    from .voxelboxter_simulation import VoxelboxterEngine, PointerlessOctree
except ImportError:
    VoxelboxterEngine = None
    PointerlessOctree = None

try:
    from .wikipedia_integration import WikipediaIntegration
except ImportError:
    WikipediaIntegration = None

__all__ = [
    'DiegeticPhysicsEngine',
    'HybridPhysicsBackend',
    'VoxelboxterEngine',
    'PointerlessOctree',
    'WikipediaIntegration',
]
