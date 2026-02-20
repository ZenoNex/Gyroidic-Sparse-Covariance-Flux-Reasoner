# Deprecation Tombstone

The following files were deleted on January 11, 2026, as part of the **Polynomial Co-Prime Migration**.

## Deleted Files

### 1. `src/core/crt_reconstruction.py`
*   **Original Purpose**: Implemented discrete Chinese Remainder Theorem for integer primes.
*   **Superseded By**: `src/core/polynomial_crt.py`
    *   Replaces discrete residues with **polynomial coefficient distributions**.
    *   Uses **Generalized Polynomial CRT** math.

### 2. `src/core/decoupled_crt.py`
*   **Original Purpose**: Implemented GDPO normalization for integer prime fields.
*   **Superseded By**: `src/core/decoupled_polynomial_crt.py`
    *   Applies GDPO to **orthogonal polynomial functionals**.
    *   Normalizes coefficient vectors instead of scalar residues.

### 3. `src/core/prime_utils.py`
*   **Original Purpose**: Utilities for generating discrete prime numbers and checking primality.
*   **Superseded By**: `src/core/polynomial_coprime.py`
    *   Polynomial functionals are defined by orthogonality and co-primality in the polynomial ring, removing the need for integer prime search.

### 4. `examples/example_gdpo_demo.py`
*   **Original Purpose**: Demonstrated GDPO collapse prevention with prime moduli.
*   **Superseded By**: `examples/example_runner.py` (Main runner now handles GDPO) and `src/core/decoupled_polynomial_crt.py`.
### 5. `docs/GDPO_INTEGRATION.md`
*   **Original Purpose**: Detailed the integration of GDPO into the system.
*   **Superseded By**: `docs/MATHEMATICAL_DETAILS.md`

### 6. `docs/RESONANCE_CAVITY_GDPO.md`
*   **Original Purpose**: Explained GDPO in the context of resonance cavities.
*   **Superseded By**: `docs/MATHEMATICAL_DETAILS.md`

### 7. `docs/DOCUMENTATION_UPDATE_SUMMARY.md`
*   **Original Purpose**: Temporary artifact for tracking documentation updates.
*   **Superseded By**: No direct replacement; information integrated into other documentation.

## Reason for Deletion
The architecture has fundamentally shifted from discrete modular arithmetic (which is non-differentiable or requires hard approximations) to continuous polynomial functional arithmetic (which is naturally differentiable and topologically richer). Sticking to the old prime-based files would cause confusion and import errors.
