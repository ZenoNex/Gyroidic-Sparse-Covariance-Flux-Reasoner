Determinism and Persistence Policy

Scope
- This document defines the determinism, persistence, and interface-boundary policies governing ingestion, state loading, and result reporting in the Gyroidic Sparse Covariance Flux Reasoner.

1) Deterministic IDs (Data Ingestion)
- Policy: All ingestion-generated identifiers are deterministic across runs and hosts.
- Method: IDs are derived from a Blake2s digest of canonical JSON:
  - json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
  - hashlib.blake2s(payload.encode('utf-8'), digest_size=10).hexdigest()
  - ID format: "<prefix>_<hexdigest>"
- Rationale: Removes nondeterminism from Python builtin hash() and ensures cross-machine reproducibility of keys in caches and encodings.

2) Non-Strict State Loading (Forward Compatibility)
- Policy: Engine state is loaded with strict=False to remain forward-compatible as components evolve.
- Behavior:
  - torch.load(..., map_location='cpu')
  - self.load_state_dict(checkpoint, strict=False)
  - Log counts and previews of missing and unexpected keys
  - Deterministic post-load non-finite repair (see below)
- Rationale: Systems evolve (e.g., k, residue shapes, module names). Non-strict loading prevents brittle failures while maintaining transparency in logs.

3) Deterministic Non-Finite Repair (Post-Load and Boundary)
- Policy: All parameters and buffers must be finite. Any non-finite values are deterministically mapped to zero.
- Post-Load Repair:
  - After state_dict loading, scan all Tensors:
    - If any NaN/Inf present, replace with zeros in-place
    - Log number of repaired tensors
- Boundary Sanitation:
  - Before returning metrics/payloads to clients:
    - Floats: NaN/Inf -> 0.0
    - Tensors: elementwise !isfinite -> 0.0 (preserve dtype/shape)
- Rationale: Guarantees stability for downstream consumers and preserves debuggability by constraining the space of possible emitted values.

4) Numeric-Only Unfolding Closure Reporting
- Policy: Internal logic uses numeric metrics instead of boolean tensors. Any boolean display/export values are derived explicitly from numeric metrics at the boundary.
- Recommended Metrics:
  - closure_score: float in [0, +∞), lower is better (e.g., normalized residual norm, 1 - cosine similarity)
  - closure_threshold: float > 0
  - closure_margin = closure_threshold - closure_score
  - components: dict of auxiliary scalar metrics (finite)
- Exported boolean (optional):
  - is_closed = (float(closure_score) <= float(closure_threshold))
- Rationale: Avoids ambiguous truth values for multi-element tensors and keeps all internal control flows numeric-first and robust.

5) Logging and Transparency
- State loads log missing/unexpected keys (counts + preview lists)
- Non-finite repairs log repair counts
- Phase diagnostics (Spectral, Bezout, Chern-Simons, Soliton, Love, Soft Gates) are attached to metrics and sanitized to finite before return.

Verification Checklist
- python -m examples.verify_persistence
  - No ambiguous boolean warnings
  - Non-strict load messages appear as information (no abort)
  - Persistence Verification PASSES
- ingest paths produce stable IDs on repeated runs
