"""Topology and pressure evaluators."""

from .gyroid_covariance import SparseGyroidCovarianceProbe, SparseExplorerRouting, PalindromicRoutingCheck
from .homology_pressure import HomologyPressure, WeightedBettiNumber, ResidueHomologyDrift

# Hyper-Ring and Toroidal Arithmetic
try:
    from .hyper_ring import DiscreteHyperRingCirculation, HyperRing, RecurrentHyperRingConnectivity
except ImportError:
    DiscreteHyperRingCirculation = None
    HyperRing = None
    RecurrentHyperRingConnectivity = None

# Triadic Reciprocity
try:
    from .triadic_reciprocity import TriadicReciprocityChecker
except ImportError:
    TriadicReciprocityChecker = None

# Hyper-Ring Closure & Homological Rings
try:
    from .hyper_ring_closure import HyperRingOperator, HyperRingClosureChecker
except ImportError:
    HyperRingOperator = None
    HyperRingClosureChecker = None

# Unknowledge Domain (Tripartite Metaphysical Entropy)
try:
    from .unknowledge_domain import UnknowledgeDomain, NostalgicLeakFunctional, EntropicMischiefProbe
except ImportError:
    UnknowledgeDomain = None
    NostalgicLeakFunctional = None
    EntropicMischiefProbe = None

# Soliton Stability & Approximate PH
try:
    from .soliton_stability import SolitonStability
except ImportError:
    SolitonStability = None

try:
    from .approximate_ph import ApproximatePHProbe
except ImportError:
    ApproximatePHProbe = None

try:
    from .gyroid_covariance import LeyLineGeodesicMetric, MoebiusFiberBundle
except ImportError:
    LeyLineGeodesicMetric = None
    MoebiusFiberBundle = None

try:
    from .persistence_obstruction import ResidueFiltration, PersistentHomologyComputer, ResidueObstructionGraph
except ImportError:
    ResidueFiltration = None
    PersistentHomologyComputer = None
    ResidueObstructionGraph = None

# Alexander 1924 Wild Embedding & Fractal Cube Stacking Operator
try:
    from .alexander_wild_embedding import AlexanderWildEmbeddingOperator, AlexanderTransversalityReport
except ImportError:
    AlexanderWildEmbeddingOperator = None
    AlexanderTransversalityReport = None

__all__ = [
    'SparseGyroidCovarianceProbe',
    'SparseExplorerRouting',
    'PalindromicRoutingCheck',
    'LeyLineGeodesicMetric',
    'MoebiusFiberBundle',
    'HomologyPressure',
    'WeightedBettiNumber',
    'ResidueHomologyDrift',
    'DiscreteHyperRingCirculation',
    'HyperRing',
    'RecurrentHyperRingConnectivity',
    'TriadicReciprocityChecker',
    'HyperRingOperator',
    'HyperRingClosureChecker',
    'UnknowledgeDomain',
    'NostalgicLeakFunctional',
    'EntropicMischiefProbe',
    'SolitonStability',
    'ApproximatePHProbe',
    'ResidueFiltration',
    'PersistentHomologyComputer',
    'ResidueObstructionGraph',
    'AlexanderWildEmbeddingOperator',
    'AlexanderTransversalityReport',
]


