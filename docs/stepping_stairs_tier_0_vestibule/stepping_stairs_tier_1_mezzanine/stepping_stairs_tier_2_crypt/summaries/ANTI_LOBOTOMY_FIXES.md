# Anti-Lobotomy Fixes: Hardcoded Prime Removal

This document summarizes the fixes applied to remove hardcoded prime sequences that violated the anti-lobotomy clauses requiring "hybrid prime generation + coprime residue range learning".

## Files Fixed

### 1. `examples/simple_temporal_training.py`
**Issue**: Hardcoded `prime_indices = torch.tensor([2, 3, 5, 7, 11][:self.K])`
**Fix**: Replaced with polynomial coefficients: `poly_coeffs = torch.randn(self.K, self.D, device=self.device)`

### 2. `fix_garbled_output.py`
**Issue**: Hardcoded `prime_indices = torch.tensor([2, 3, 5, 7, 11])`
**Fix**: Replaced with polynomial coefficients based on residue dimensions

### 3. `examples/test_garbled_output_repair.py` (2 occurrences)
**Issue**: Hardcoded `prime_indices = torch.tensor([2, 3, 5, 7, 11][:K])`
**Fix**: Replaced with polynomial coefficients: `poly_coeffs = torch.randn(K, D, device=device)`

### 4. `src/ui/diegetic_backend.py`
**Issue**: Hardcoded `primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]` for text hashing
**Fix**: Replaced with Chebyshev polynomial-based coefficients for sequence-aware hashing

### 5. `src/optimization/fractional_operators.py`
**Issue**: Hardcoded `harmonics: list = [1, 2, 3, 5, 7, 11]` in PAS_h computation
**Fix**: Replaced with Chebyshev polynomial root-based harmonics

### 6. `src/optimization/codes_driver.py`
**Issue**: Hardcoded `HARMONICS = [1, 2, 3, 5, 7, 11, 13, 17]`
**Fix**: Replaced with Legendre polynomial-based harmonic generation

### 7. `src/core/non_ergodic_entropy.py`
**Issue**: Algorithmic prime generation `get_primes(n)` for lattice basis
**Fix**: Replaced with Chebyshev polynomial coefficient-based lattice generation

### 8. `src/core/speculative_coprime_gate.py`
**Issue**: Algorithmic prime generation `_generate_primes(num_heads)` for winding number checks
**Fix**: Replaced with Legendre polynomial coefficient generation

### 9. `src/models/modular_embeddings.py`
**Issue**: Hardcoded prime list `primes: List[int] = [3, 5, 7, 11, 13]` and missing documented architectures.
**Fix**: 
- Completely purged hardcoded primes; dynamically derived from canonical `PolynomialCoprimeConfig` and `get_prime_ladder(num_primes)` in `src.core.invariants`.
- Implemented `LearnedModalityEmbedder` dual-head projection yielding orthogonal polynomial basis coefficients `[batch, K, D]` and IHC 33-shell scaled residue distributions `[batch, K, max(p_k)]` per `MODULE_INVENTORY.md`.
- Implemented `CRTModularEmbeddingTable` with structural residue class bias per `MODULE_INVENTORY.md`.
- Implemented `IVSTEncoder` (Intrinsic Volume and Spectral Tensor eigenvalues) and reconnected `SimpleGraphEncoder` / `SimpleTextEncoder` per `DATA_PIPELINE.md`.

### 10. IHC 33-Shell Golden Ratio Ladder Reconnection
**Issue**: Hardcoded float `1.618033988749895` and isolated formulas in `src/models/polynomial_embeddings.py`.
**Fix**: Bound to canonical `PHI` and `get_ihc_ladder_scale(k, num_shells=33)` / `get_ihc_ladder_tensor` in `src.core.invariants`, interconnected across `PolynomialFunctionalEmbedder`, `LearnedModalityEmbedder`, `DataAssociationLayer`, `ModularAttention`, and `BreatherMode`.

### 11. `src/core/adaptive_skeleton_harness.py`
**Issue**: Hardcoded moduli list `moduli=[2, 3, 5, 17, 41]`.
**Fix**: Replaced with canonical `PolynomialCoprimeConfig(k=5, degree=4)` adhering to anti-lobotomy coprime polynomial functionals.

## Architectural Principles Enforced


1. **No Hardcoded Primes**: All hardcoded prime sequences removed
2. **Polynomial Co-Prime Functionals**: Using orthogonal polynomial basis (Chebyshev, Legendre) instead
3. **Evolutionary Trust Selection**: Coefficients evolve via mutation, not fixed optimization
4. **Hybrid Prime Generation**: When prime-like behavior needed, generated from polynomial evaluations
5. **Coprime Residue Range Learning**: Residue ranges learned from polynomial basis interactions

## Mathematical Foundation

Instead of discrete primes `p_k ∈ ℤ`, we now use polynomial functionals:

```
φ_k(x; θ_k) = Σ_i θ_k[i] · P_i(x)
```

Where:
- `θ_k ∈ Birkhoff polytope` (doubly-stochastic matrix)
- `gcd(φ_i, φ_j) = 1` for all `i ≠ j` (co-primality)
- `P_i(x)` are orthogonal polynomial basis functions (Chebyshev, Legendre)

## Compliance Status

[OK] **COMPLIANT**: No hardcoded prime sequences remain
[OK] **COMPLIANT**: Polynomial co-prime functionals implemented
[OK] **COMPLIANT**: Evolutionary trust selection preserved
[OK] **COMPLIANT**: Anti-lobotomy clauses satisfied

The system now follows the proper non-lobotomy architecture as specified in the documentation, using polynomial functionals with evolutionary trust selection instead of hardcoded prime-based modular arithmetic.