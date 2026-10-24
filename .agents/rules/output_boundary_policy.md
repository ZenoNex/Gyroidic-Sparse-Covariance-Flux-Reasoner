# Output Boundary Policy

Ensure numerical stability, deterministic behavior, and machine-friendly outputs across all interface boundaries.

## 1. Finite-Only Outputs
**POLICY**: No `NaN` or `Inf` values are allowed at the boundary.
**IMPLEMENTATION**: 
- Floats: Replace any `NaN`/`Inf` with `0.0`.
- Tensors: Apply elementwise replacement of non-finite values with zeros (preserving shape and dtype).

## 2. Deterministic Repairs
**POLICY**: Post-load non-finite repairs map `NaN`/`Inf` in parameters/buffers to zeros deterministically and log a repair count.

## 3. Numeric-First Reporting
**POLICY**: Unfolding Closures and similar checks must report explicit numeric metrics (scores, thresholds, margins). Any boolean `closed` flags must be derived explicitly:
`closed = (float(score) <= float(threshold))`
Never rely on implicit truthiness of tensors.
