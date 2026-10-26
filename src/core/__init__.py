"""Core modules for polynomial co-prime functional system.

Includes Phase 1-3 implementations: Failure tokens, structural irreducibility,
continuous co-primality, and meta-invariant.
"""

from .device_utils import DEVICE
from .polynomial_coprime import (
    PolynomialBasis,
    BirkhoffPolytopeSampler,
    PolynomialCoprimeConfig
)
from .polynomial_crt import (
    PolynomialCRT,
    PolynomialCRTKernelDetector
)
from .decoupled_polynomial_crt import DecoupledPolynomialCRT
from .birkhoff_projection import sinkhorn_knopp, project_to_birkhoff
from .gdpo_normalization import SignalSovereignty, LearnableWeights
from .invariants import PhaseAlignmentInvariant, APAS_Zeta, compute_chirality
from .pressure_typing import StructuralPressure

# Phase 1: Failure Token System
from .failure_token import FailureToken, FailureTokenType, RuptureFunctional

# Phase 3: Advanced Constraints
from .structural_irreducibility import (
    EvidenceModuleProjection,
    StructuralIrreducibilityChecker
)
from .continuous_coprimality import (
    DiscreteEntropyComputer,
    ContinuousCoprimality
)
from .meta_invariant import MetaInvariant

# Ternary & Tri-State Components
try:
    from .love_invariant_protector import SoftSaturatedGates, LoveInvariantProtector
except ImportError:
    SoftSaturatedGates = None
    LoveInvariantProtector = None

try:
    from .evidence_modules import TripartiteEvidenceModule
except ImportError:
    TripartiteEvidenceModule = None

try:
    from .five_gate_pipeline import KnowledgeState, SearchGate, ConfabulationDetector, FiveGatePipeline
except ImportError:
    KnowledgeState = None
    SearchGate = None
    ConfabulationDetector = None
    FiveGatePipeline = None

try:
    from .conjugate_moment_transport import ConjugateMomentTransport
except ImportError:
    ConjugateMomentTransport = None

try:
    from .non_dual_coin import TripsodicLedger
except ImportError:
    TripsodicLedger = None

try:
    from .hessian_8x8_curvature_ledger import (
        Hessian8x8CurvatureLedger,
        Trajectory8x8Decombiner
    )
except ImportError:
    Hessian8x8CurvatureLedger = None
    Trajectory8x8Decombiner = None

try:
    from .texture_dsp import TextureDSPEngine, AdaptiveGaborWindowController
except ImportError:
    TextureDSPEngine = None
    AdaptiveGaborWindowController = None

try:
    from .hardware_monitor import (
        has_compute_affordance,
        get_typed_compute_affordance,
        get_affordance_gauges,
        ComputeAffordanceType,
        TypedComputeAffordance
    )
except ImportError:
    has_compute_affordance = None
    get_typed_compute_affordance = None
    get_affordance_gauges = None
    ComputeAffordanceType = None
    TypedComputeAffordance = None

try:
    from .adaptive_skeleton_harness import AdaptiveSkeletonHarness
except ImportError:
    AdaptiveSkeletonHarness = None

try:
    from .enhanced_bezout_crt import EnhancedBezoutCRT, CrossbarIKSolver
except ImportError:
    EnhancedBezoutCRT = None
    CrossbarIKSolver = None

try:
    from .quantum_tda import QuantumBettiApproximator
    from src.tda.chebyshev_filtration import MinimaxPolynomialApproximation
    from .polynomial_scaffold import PolynomialCoefficientFunctional
    from src.topology.speculative_homology import SpeculativeHomologyEngine
except ImportError:
    QuantumBettiApproximator = None
    MinimaxPolynomialApproximation = None
    PolynomialCoefficientFunctional = None
    SpeculativeHomologyEngine = None

try:
    from .euler_tour_lca import EulerTourSparseTableLCA
except ImportError:
    EulerTourSparseTableLCA = None

__all__ = [
    # Core polynomial system
    'PolynomialBasis',
    'BirkhoffPolytopeSampler',
    'PolynomialCoprimeConfig',
    'PolynomialCRT',
    'PolynomialCRTKernelDetector',
    'DecoupledPolynomialCRT',
    'sinkhorn_knopp',
    'project_to_birkhoff',
    'SignalSovereignty',
    'LearnableWeights',
    # Invariants
    'PhaseAlignmentInvariant',
    'APAS_Zeta',
    'compute_chirality',
    # Pressure typing
    'StructuralPressure',
    # Phase 1: Failure Token System
    'FailureToken',
    'FailureTokenType',
    'RuptureFunctional',
    # Phase 3: Advanced Constraints
    'EvidenceModuleProjection',
    'StructuralIrreducibilityChecker',
    'DiscreteEntropyComputer',
    'ContinuousCoprimality',
    'MetaInvariant',
    'DEVICE',
    # Ternary & Tri-State Components
    'SoftSaturatedGates',
    'LoveInvariantProtector',
    'TripartiteEvidenceModule',
    'KnowledgeState',
    'SearchGate',
    'ConfabulationDetector',
    'FiveGatePipeline',
    'ConjugateMomentTransport',
    'TripsodicLedger',
    'Hessian8x8CurvatureLedger',
    'Trajectory8x8Decombiner',
    # Rehybridized Core Primitives
    'TextureDSPEngine',
    'has_compute_affordance',
    'get_typed_compute_affordance',
    'get_affordance_gauges',
    'ComputeAffordanceType',
    'TypedComputeAffordance',
    'AdaptiveSkeletonHarness',
    'EnhancedBezoutCRT',
    'CrossbarIKSolver',
    'QuantumBettiApproximator',
    'MinimaxPolynomialApproximation',
    'PolynomialCoefficientFunctional',
    'SpeculativeHomologyEngine',
    'EulerTourSparseTableLCA',
]

