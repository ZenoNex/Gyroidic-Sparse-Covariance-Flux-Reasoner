Changelog (Append) — 2026-02-08

Scope
- Summarizes recent stability, determinism, ingestion, and dimensional-handling changes integrated into the system.

1) Deterministic IDs and persistence policy
- Ingestion identifiers now use Blake2s on canonical JSON (sorted keys, ensure_ascii=False), replacing nondeterministic builtin hash().
- State loading updated to non-strict behavior with explicit logging of missing/unexpected keys. Post-load deterministic repair replaces non-finite values (NaN/Inf) with zeros in-place and logs repair counts.
- Output boundary sanitation: all outgoing floats are sanitized to finite (NaN/Inf -> 0.0); tensors are sanitized elementwise; nested structures are recursively sanitized.
- Unfolding closure is now numeric-only internally (closure_score, closure_threshold, closure_margin). Any boolean presented is derived explicitly from these numeric values at the boundary.

2) Backend stability fixes (diegetic backend)
- Metrics assembly moved to occur after Phase 4 computations, eliminating undefined-variable errors.
- CALM scalar coercion corrected: no .item() calls on floats; robust scalar conversion in place.
- Unfolding Closure: integration layer derives a Python bool from numeric metrics for display; the internal check returns numeric-only metrics to avoid ambiguous tensor truth values.

3) Canonical topological ingestion (new module)
- Added src/data/canonical_projection.py with CanonicalProjector:
  - Text/topics projection: project_text_to_state/topic_to_state returns a [1, dim] manifold-aligned state with gyroid entropy.
  - Image projection (adaptive): project_image_path_to_state supports extreme sizes/aspect ratios with pixel caps and aspect-preserving resizes. Multi-scale (L0 baseline, gated L1) with entropy-weighted fusion.
  - Feature-level projection: project_multiscale_image fuses flat/2D feature levels to [1, dim], backward-compatible with old encodings.
  - Constraint fields via Asymptotic Coefficient Windowing (ACW): edge and group-coherence fields are mapped to smooth coefficient windows and applied in residue space before flatten/truncate.
  - All outputs are deterministic and finite; residue alignment matches the core pad → view [B,K,residue_dim] → transform → flatten → truncate loop, ensuring forward commutativity.

4) Conversational ingestion alignment
- src/data/conversational_api_ingestor.py now uses CanonicalProjector for text embeddings, returning [64]-length vectors (from [1,64] states). Per-turn metadata includes gyroid_entropy, and conversation.context aggregates avg_gyroid_entropy.
- Stable IDs and previous fixes preserved.

5) Documentation added
- docs/DETERMINISM_AND_PERSISTENCE.md — determinism policy, non-strict loading + repair, boundary sanitation, numeric-only closure.
- docs/RESIDUE_SHAPE_COMPATIBILITY.md — residue pipeline, coefficient adaptation, forward commutativity for historical checkpoints.
- docs/OUTPUT_BOUNDARY_POLICY.md — finite-only guarantees, numeric-first reporting, compatibility notes.

Backwards compatibility notes
- Existing frontends/backends that consume embedding vectors remain compatible (flat JSON-serializable lists). New diagnostics (entropy, edge density) are optional.
- CanonicalProjector adds APIs but does not remove any; legacy paths continue to run. All projections return [1, dim] states (default dim=64), aligned with the repair stack.

Known optional follow-ups
- Topic/wiki ingestion adapters via CanonicalProjector
- Knowledge dyad builder for resonance/closure/coprime alignment artifacts
- ROI-based L2 selective refinement (capped) for localized complexity
