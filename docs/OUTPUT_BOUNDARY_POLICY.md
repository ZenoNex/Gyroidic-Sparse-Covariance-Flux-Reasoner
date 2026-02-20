Output Boundary Policy

Purpose
- Define guarantees at the interface boundary to downstream consumers (CLI, APIs, encoding records). The goal is to ensure numerical stability, deterministic behavior, and machine-friendly outputs.

1) Finite-Only Outputs
- Policy: No NaN or Inf values are emitted at the boundary.
- Implementation:
  - Floats: any NaN/Inf is replaced by 0.0
  - Tensors: elementwise replacement of non-finite values with zeros; shape and dtype preserved
  - Dicts/lists/tuples: recursively sanitized

2) Deterministic Repairs and Reductions
- Post-load non-finite repairs map NaN/Inf in parameters/buffers to zeros deterministically and log a repair count.
- Any tensor-based boolean conditions used for presentation are derived from explicit numeric reductions with thresholds, never from implicit truthiness.

3) Numeric-First Reporting
- Unfolding Closure and similar checks report numeric metrics (scores, thresholds, margins). Boolean closed flags (if shown) are derived explicitly at the boundary:
  - closed = (float(score) <= float(threshold))

4) Compatibility Notes
- When historical checkpoints include shape-mismatched parameters (e.g., residue-shaped tensors), non-strict loading is used and phases adapt at runtime via padding/view/flatten/truncate steps. This preserves forward behavior while logging missing/unexpected keys.

5) Logging Discipline
- All boundary values are sanitized prior to return.
- Phase-specific diagnostics (spectral, bezout, chern-simons, soliton, love, soft gates) are included for transparency and are sanitized to be finite.

Verification
- Run: python -m examples.verify_persistence
  - Expect stable, finite metrics with no ambiguous boolean tensor warnings.
  - Persistence verification should pass and logs should include non-strict load information if applicable.
