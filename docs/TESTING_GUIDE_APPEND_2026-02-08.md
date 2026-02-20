Testing Guide (Append) — 2026-02-08

Overview
- This guide complements existing examples and tests, highlighting commands to validate recent determinism, persistence, ingestion alignment, and dimensional-handling changes.

Environment
- From the project root: D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner
- Use `python -m` to respect package layout.

Core persistence and stability
1) Verify persistence and advanced phases
- Command:
  - `python -m examples.verify_persistence`
- Validates:
  - Non-strict state loading logs (informational), deterministic non-finite repair
  - Phase 2.x repairs (Spectral, Bezout, Chern-Simons, Soliton, Love, Soft Gates)
  - Phase 4 metrics and Unfolding Closure reporting (numeric-only internally)
  - Clean return of metrics without ambiguous boolean warnings

2) Additional integration/feature demos (subset from examples/)
- `python -m examples.verify_invariants` — invariant checks
- `python -m examples.yield_flow_demo` — yield flow behavior
- `python -m examples.example_runner` — general runner
- `python -m examples.example_hybrid_admm` — hybrid ADMM path
- `python -m examples.run_low_intensity` — low-intensity validation

Frontends/backends reachability
- Server/terminal checks (subset):
  - `python test_minimal_server.py`
  - `python test_backend_startup.py`
  - `python test_backend_status.py`
  - `python test_backend_simple.py`
  - `python test_terminal_interface.py`
- These validate startup, status probing, and terminal UI connections.

Ingestion tests
- Conversational ingestion:
  - Demonstration entry points are provided in the module main:
    - `python -m src.data.conversational_api_ingestor` (runs demo functions if invoked directly)
  - Programmatic usage:
    - `from src.data.conversational_api_ingestor import create_conversational_ingestor`
    - `ing = create_conversational_ingestor()`
    - `convs = ing.ingest_huggingface_dataset('lmsys/lmsys-chat-1m', max_samples=100)`
  - Validate that each ConversationTurn now contains a [64]-length embedding and metadata['gyroid_entropy'].

- Canonical projector quick checks:
  - Text:
    - `from src.data.canonical_projection import CanonicalProjector`
    - `proj = CanonicalProjector(dim=64, k=5)`
    - `out = proj.project_text_to_state('hello world')`
    - Inspect `out['state'].shape == (1,64)` and `out['entropy']` is finite
  - Image path (requires Pillow):
    - `out = proj.project_image_path_to_state('path/to/image.png')`
    - Inspect `out['state'].shape == (1,64)`, `out['entropy']`, and `out['edge_density_l0']`
  - Feature levels (back-compat):
    - `out = proj.project_multiscale_image([torch.randn(32,32), torch.randn(64,64)])`
    - Inspect fused state shape and entropy

Additional test scripts (selected)
- Dataset/ingestion workflows:
  - `python dataset_ingestion_system.py`
  - `python simple_dataset_interface.py`
  - `python dataset_command_interface.py`
  - `python run_dataset.py`
- Stability and fixes verification:
  - `python test_comprehensive_fixes.py`
  - `python test_fixes_verification.py`
  - `python test_basic_functionality.py`
  - `python test_graph_visualization.py`
  - `python test_advanced_extensions.py`

Notes
- If tests previously relied on 768-length text embeddings for display, they will now receive 64-length states from the canonical projector; this is intended and topologically aligned.
- All outputs should be finite; any NaN/Inf at the boundary is sanitized.
- Non-strict state loading may log missing/unexpected keys when loading checkpoints from older versions; this is informational and expected.
