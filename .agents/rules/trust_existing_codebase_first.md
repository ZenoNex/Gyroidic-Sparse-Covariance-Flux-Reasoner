---
trigger: always_on
description: Always prioritize existing codebase implementations over creating new or redundant code. Trust that user references refer to existing mechanisms even if names vary.
---

# Trust Existing Codebase First & Prohibit Redundant Re-engineering

Always adhere strictly to these principles before writing, refactoring, or generating logic:

## 1. Prioritize Existing Implementations
- Trust that when the user mentions a mechanism, capability, algorithm, or loop (e.g., MAML adaptation, Larynx autoregression, category expansion, CRT coprimality, fossil graph traversal), **the code is already implemented elsewhere in the codebase**.
- Even if the user's phrasing does not match exact class, method, or variable names verbatim, search the existing modules and documentation to locate the canonical implementation.

## 2. Exhaustive Search Before Modifying
- Perform safe, targeted searches (using ripgrep / Select-String while excluding `.venv` and `data` directories) across `src/` and `docs/` to find where the capability already lives.
- Check existing architectural anchors:
  - Meta-learning / MAML online adaptation loops: `CALM.adapt` (`src/surrogates/calm_predictor.py`), `KAGHBlock.adapt_online` (`src/surrogates/kagh_networks.py`), `ResidueFusion.adapt_online` (`src/core/knowledge_dyad_fossilizer.py`), and `PolynomialFunctionalEmbedder.adapt_online` (`src/models/polynomial_embeddings.py`).
  - Larynx & linguistic resonance: `ResonanceLarynx` / `LazarusSoftmax` (`src/data/knowledge_ingestor.py`, `src/ui/diegetic_backend.py`).
  - Preprints & scientific feeds: `ArXivSovereignIngestor` (`src/data/knowledge_ingestor.py`).
  - Topological manifolds & invariants: `DiegeticPhysicsEngine`, `CODESConstraintFramework`, `CarnotMobiusLedger`.

## 3. Strict Ban on Ad-Hoc Re-engineering
- **FORBIDDEN**: Writing scratch, duplicate, or ad-hoc replacements for existing algorithms (such as manual `clone_module` autograd loops, hand-rolled multinomial token generators, or mock heuristic formulas).
- **REQUIRED**: Import, bind, connect to, and delegate to the existing canonical functions and components.

## 4. Only Create When Explicitly Commanded
- ONLY create new classes, novel algorithms, or new standalone modules from scratch when the user explicitly specifies that it is a new thing to create.
- In every other scenario, the task is integration, reconnection, cleanup, or delegation to existing code.
