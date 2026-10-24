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

### 8. `TriadicReciprocityCheck` & Continuous Reversal Matrices
*   **Original Purpose**: Continuously verified $M_{ca} M_{bc} M_{ab} \approx I$ and dynamically learned the Gyroid boundary spatial reversal matrix.
*   **Superseded By**: `PalindromicRouter` fast-paths and strict geometric anti-symmetry (Anti-Palindromic Gluing).
    *   Topological rigor replaces empirical continuous bounds, avoiding $O(N^3)$ triadic tracking overhead and nullifying Chern-Simons gasket penalties gracefully.


### 9. Scalar `morton_encode` Bit Interleaving (`src/ui/voxelboxter_simulation.py`)
*   **Original Purpose**: Redundant scalar 3D coordinate bit-interleaving inside simulation routines.
*   **Superseded By**: Canonical tensorized `morton_encode_3d` and `expand_bits` in `src/core/fgrt_primitives.py`.
    *   Consolidates spatial hashing to streaming GPU/CPU memory alignment without parallel reinvention.

### 10. `wikipedia_trainer.html`
*   **Original Purpose**: Visual prototype for ingestion.
*   **Superseded By**: Direct High-Throughput Ingestion API in `diegetic_backend.py`.
    *   Fossilized as it lacked active fetch logic to trigger the backbone ingestion cleanly.

### 11. Hardcoded Prime Sequences (`src/p2p/bonfire_consensus.py`)
*   **Original Purpose**: Static array `[2, 3, 5, 7, 11, 13, 17, 19]` for high-dimensional harmonic phases.
*   **Superseded By**: Dynamic `PrimeResonanceLadder` (`src/core/fgrt_primitives.py`).
    *   Adheres to the anti-lobotomy specification (`ANTI_LOBOTOMY_FIXES.md`). Generates incommensurate frequencies $f_{p_n} = 2\pi\ln(p_n)$ with Lazarus-prime prioritization and TailSlayer Gray-code interleaving, scaling dynamically beyond 14 dimensions without frequency collisions.

### 12. Identity Finite Differences in Gyroid Differentiation (`src/topology/gyroid_differentiation.py`)
*   **Original Purpose**: Trivial difference identity $((x+\delta)-x)/\delta \equiv 1$ that failed to differentiate actual manifold state.
*   **Superseded By**: Real functional stencils across directional perturbations $\Phi(r \pm \delta \hat{u})$, trajectory velocity tangents, and Hamiltonian skew-symmetric flow $v = J \nabla G$, dynamically scaled near boundary fractures via L'Hôpital limits $s(c) = \frac{\log(c+1)}{c}$.

### 13. Stranded Root Test `src/test_codec_invariants_resumption.py`
*   **Original Purpose**: Ad-hoc root script for testing non-Abelian commutativity gaps and GL(n) determinants.
*   **Superseded By**: Relocated to `tests/test_codec_invariants_resumption.py` under the standardized test harness.

### 14. Inline Sinkhorn-Knopp Projection (`src/core/polynomial_coprime.py`)
*   **Original Purpose**: Inline iterative normalization loop for doubly-stochastic matrix balancing.
*   **Superseded By**: Canonical `BirkhoffPolytopeProjection` in `src/core/birkhoff_projection.py`.

### 15. Raw Epsilon Division Approximations
*   **Original Purpose**: Division by raw `denom + 1e-8` causing unbounded derivative explosion near boundaries.
*   **Superseded By**: L'Hôpital boundary regularizer $s(c) = \frac{\log(c+1)}{c}$ ($\lim_{c \to 0} s(c) = 1.0$) across continuous scale factors, octree depths, and gradient stencils.

### 16. Stranded Backend Directory (`src/backend/hybrid_backend.py`)
*   **Original Purpose**: Redundant top-level backend copy outside standard architectural tiers.
*   **Superseded By**: Complete removal of `src/backend/` folder; unified dataflow routed through `src/core/` and `src/ui/diegetic_backend.py`.

### 17. Monolithic HybridPhysicsBackend (`src/core/hybrid_backend.py`)
*   **Original Purpose**: Monolithic data ingestion module attempting to handle spatial OpenStreetMap metrics, financial feeds, literature parsing, and octree voxelization in a single file inside `src/core/`.
*   **Superseded By**: Physically deleted and removed from `src/core/`. All operational capabilities redistributed to their canonical domains:
    *   **Financial Market Feeds, Larynx Connections & MAML Adaptation**: Integrated natively into [`InvestorNewsIngestor`](file:///d:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/investor_news_ingestor.py) (`fetch_financial_feeds`, `generate_larynx_market_query`, `adapt_maml`, `adapt_online`, `steer_state_from_payloads`).
    *   **Scientific Literature & Category Expansion**: Integrated natively into [`ArXivSovereignIngestor`](file:///d:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/knowledge_ingestor.py) (`fetch_scientific_feeds`, `expand_categories`, `_generate_larynx_query`).
    *   **Spatial Metric & Land-Rent Curvature**: Integrated natively into [`DiegeticPhysicsEngine`](file:///d:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/diegetic_backend.py) and [`HybridPhysicsBackend`](file:///d:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/diegetic_backend.py) (`fetch_spatial_land_value`, `multiplex_feed_chain`, `map_to_pointerless_octree`).
    *   **Octree Projection**: Integrated natively into [`PointerlessOctree`](file:///d:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/voxelboxter_simulation.py) (`ingest_spatial_land_value`).

## Reason for Deletion
The architecture has fundamentally shifted from discrete modular arithmetic (which is non-differentiable or requires hard approximations) to continuous polynomial functional arithmetic (which is naturally differentiable and topologically richer). Furthermore, with Phase 19 Topological Extensions and the L'Hôpital fractal unification, we moved from heuristic continuous metrics (like arbitrary triadic checks, unconstrained reversal matrices, and hardcoded prime arrays) to formal topological invariants (Palindromic symmetries, Repunit-CRT probes, dynamic Prime Resonance Ladders, and L'Hôpital-regularized boundary stencils), while eliminating redundant, monolithic ingestors in favor of domain-coherent subsystems.
