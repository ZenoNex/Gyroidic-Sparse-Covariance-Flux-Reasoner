"""Core modules for polynomial co-prime functional system.

Includes Phase 1-3 implementations: Failure tokens, structural irreducibility,
continuous co-primality, and meta-invariant.
"""

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
]
