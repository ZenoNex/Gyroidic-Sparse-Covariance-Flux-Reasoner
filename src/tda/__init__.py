"""
TDA (Topological Data Analysis) Package.

Exposes Chebyshev minimax polynomial approximations and filtration functions
for speculative Betti decoding and combinatorial Laplacian spectral estimation.
"""

from .chebyshev_filtration import MinimaxPolynomialApproximation

__all__ = [
    'MinimaxPolynomialApproximation',
]
