# Module Inventory — Undocumented & Lightly Documented Modules

This document provides canonical one-paragraph descriptions for all `src/` modules not covered by dedicated `.md` documentation files. It serves as an authoritative reference that short-circuits module discovery during development and audit.

> **Coverage policy**: Any module appearing here should eventually graduate to a dedicated `.md` or an explicit section in a component-level doc. This inventory is a *starting point*, not a permanent home.

---

## src/core

### admr_solver.py
**Class**: `PolynomialADMRSolver`  
**Role**: Alternating Direction of Multiplicative Remainders — the continuous-polynomial analogue of ADMM.  

Instead of discrete prime moduli, this solver uses co-prime polynomial functionals (`PolynomialCoprimeConfig`) as its "modular" basis. The multiplicative update `S^{n+1} = Proj_{Poly}[ S^n · Σ w_ik S_k ]` propagates relational pressure through graph-structured neighbors rather than euclidean gradients. Two modes: `forward()` (single-step multiplicative update with optional valence drive) and `stochastic_differential_step()` (continuous-time SDE update `dx = [ΣA_i x_i − ρΣ(x − r(x_k))]dt + σdW`). Tracks asymptotic time `tau` in a persistent buffer. Corresponds to NOMENCLATURE "Multiplicative Scaffolding."

---

### audience_mapping.py
**Class**: `AudienceProjection`  
**Role**: Lipschitz homeomorphic projection from manifold M to audience space A.

Implements the operator Φ: M → A defined in the Garden Statistical Attractors design. Uses spectral normalization on all linear layers to enforce Lipschitz constant ≤ 1, and a residual skip connection (`y = f(x) + x`) to approximate homeomorphism (continuous, bijective). An approximate inverse `Φ⁻¹` is provided via fixed-point iteration (Banach theorem, valid when `Lip(f) < 1`). The key requirement it enforces is *roughness preservation*: topological singularities (sharp features, discontinuities) in the manifold are transmitted into audience space rather than smoothed away.

---

### collapse_poisoner.py
**Class**: `CollapsePathPoisoner` (also aliased as `AdversarialStressTester`)  
**Role**: Adversarial stress-tester for the Speculative Homology Engine.

Generates two types of synthetic rupture events to verify System 2 robustness without harming real training data. (1) **Synthetic Rupture**: Gram-Schmidt orthogonalization of learned constraints against the current manifold, creating a perturbation perpendicular to every existing basis vector — in principle a topological hole injection. (2) **Cycle Debt**: Detects homotopy class repetition by cosine-matching the recent state history; high debt (≥ 0.5) flags that the system is looping in the same topological region. The class was refactored from an offensive poisoner to a defensive probe in the January 2026 Anti-Lobotomy integration.

---

### daqf_operator.py
**Class**: `DAQUFOperator`  
**Full name**: Diegetic Amortized Quantized Unknowledge Fossilization Operator  
**Role**: Manages "structural scars" — unremovable but amortized fossilized invariants.

The DAQUF pipeline: (1) **Fossil Selection** — fossil with highest contradiction load χ(f_i) = Σ(Φ(f_i) = ⊥) + mischief + valence is declared `f*`. (2) **Diegetic Amortization** — cost is spread over narrative time τ: `C̃ = C_τ / dim(N_τ)`. (3) **Lattice Quantization** — projects to a lower-dimensional integer lattice with energy constraint, retaining quantization error Δ_q as structural memory. (4) **Speculative Persistence** — fossil persists via non-collapse (non-zero flux *or* stable mischief soliton). (5) **Love Invariant L** — a non-transferable buffer that `check_invariants()` ensures is never modified; raising `RuntimeError("LOVE INVARIANT VIOLATION")` if altered. Corresponds to DAQUF discussion in PROJECT_PITCH_BURDENED §3.

---

### deflagration_scout.py
**Class**: `OmipedialDeflagrator`  
**Role**: Scouts and amplifies sparse anomalies ("defects") to enable jumps across manifold holes.

Implements two operations. `scout_defects()` computes `ΔD_i = |actual_flux − predicted_flux| × amplification` — rewarding rare, unexpected deviations rather than penalizing them (a "good bug" signal). `omipedial_jump()` uses a threshold on the ley-line potential field to trigger a discrete jump across a topological gap where adjacency is sparse but resonance potential is high. Tracks cumulative defect density as a buffer. Corresponds to NOMENCLATURE term "Omipedial Interstitiality."

---

### energy_based_soliton_healer.py
**Class**: `EnergyBasedSolitonHealer`  
**Role**: Repairs structurally damaged solitons by gradient descent on a learnable energy surface.

Implements EBM-style soliton preservation: a stable configuration is a **soliton template** (cosine modulated by golden ratio + dynamic prime-based modulation, normalized to unit energy). The energy function `E(state, target) = ‖A(state−target)‖² + b·state` measures distance from this template with learnable quadratic and linear terms. `heal_soliton()` performs iterative gradient ascent on `−∇E` (negative gradient = healing direction) with adaptive rate: strong healing when `E > margin`, gentle stabilization otherwise. `update_energy_function()` shapes the energy surface contrastively using a hinge loss. Used during the spectral coherence repair cascade.

---

### energy_monitor.py
**Class**: `StructuralEnergyMonitor` (probable)  
**Role**: Monitors Topological Free Energy `F_topo` and links it to the FixedPoint grid's rigidity.

As `F_topo` decreases (cooling), the monitor tightens the `LearnedPrimitivePerturbation` scale, effectively "signing" a conceptual configuration into the fossil layer by preventing further adaptation. Central to INVARIANT_OPTIMIZATION §2.2 "Thermodynamic Anchor." *(Full class name to be verified against source.)*

---

### enhanced_bezout_crt.py
**Class**: `EnhancedBezoutCRT` (probable)  
**Role**: Extended-GCD based CRT reconstruction with Bézout coefficient caching.

Augments standard CRT with Bézout coefficient precomputation for fast runtime modular inverse without Fermat's little theorem. Used in the BezoutCoefficientRefresh step of the spectral coherence repair cascade. *(Full details pending source review.)*

---

### fractal_meta_functional.py
**Role**: Implements fractal meta-recursion inside the diegetic backend's `forward()` pass.

Computes multi-scale structural pressure by recursively applying the covariance estimator at different spectral granularities (adaptive fractal blocks). Connected to the "Adaptive Partitioning" concept in NOMENCLATURE §8. *(Full class name pending source review.)*

---

### garden_statistical_attractors.py
**Role**: Garden-level statistical attractor manifolds.

Implements the ensemble statistical description of a "Garden" (local polynomial polytope), tracking the mean/variance of residue distributions and their attractor basins. Connected to the Garden/Meta-Polytope Lattice terminology in NOMENCLATURE. Partial coverage in the Garden Statistical Attractors design document. *(Full details pending source review.)*

---

### gluing_operator.py
**Class**: `GluingOperator`  
**Role**: Manages manifold-boundary transitions via reversal matrix blending.

When the state approaches a manifold boundary, `GluingOperator` applies a reversal matrix `R` and blends the current and reversed states based on boundary proximity: `output = (1 − α)·state + α·R·state`. Includes a simplified Chern-Simons constraint check measuring winding around the gluing manifold. Handles the topology of joining two distinct manifold patches.

---

### knowledge_dyad_fossilizer.py
**Role**: Fossilizes knowledge dyads (paired concept structures) into the persistent fossil layer.

Manages `DyadFossilizer` instances within the engine, selecting high-trust dyads for structural freezing and updating the fossilization index. Partial coverage in `KNOWLEDGE_DYAD_LIFECYCLE.md`. Connected to Signal Sovereignty foster logic in GDPO.

---

### legibility_audit.py
**Classes**: `LegibilityTripwire`, `NarrativeCoherenceEstimator`  
**Role**: Detects when the system is being selected for explainability rather than structural merit.

`NarrativeCoherenceEstimator` measures how closely a configuration embedding matches canonical "explainable" patterns (sparse 1-hot, block-sparse, monotonic gradient) using fixed buffer-registered templates (not trained). `LegibilityTripwire` tracks the *correlation* between selection probability and narrative coherence over a rolling window — if selected configs consistently have higher coherence than rejected ones, it raises a `UserWarning`. High coherence is a **danger signal** (Pointer #2 from Sparse Operational Pointers) — not a goal.

---

### ley_line_tracker.py
**Class**: `LeyLineTracker`  
**Role**: Tracks resonance streamlines (preferred flow vectors) on the gyroidic manifold.

Maintains a resonance potential field `V(x_i) = α·Σ R_ij·‖Φ_j−Φ_i‖² + β·‖L_i‖² + γ·ΔD_i` combining relational adjacency, love tensor magnitudes, and defect signals. `detect_shear_planes()` identifies non-smooth pressure gradient regions that become "corridors of rupture" or preferred flow channels. `get_preferred_flow()` returns a softmax over neighbor potentials for a given index set. Corresponds to NOMENCLATURE term "Resonance Streamlines."

---

### narrative_collapse.py
**Class**: `LinguisticEntropyMonitor` (also aliased as `NarrativeCollapseDetector`)  
**Role**: Detects "hallucination loops" where reasoning entropy collapses and trajectory linearizes.

Two detection signals: (1) **Entropy collapse** — softmax entropy of hidden state falls below `entropy_threshold`; flags `smoothing_warning`. (2) **Trajectory linearity** — cosine similarity between consecutive state deltas `δ₁, δ₂` exceeds `prediction_threshold` (0.99); flags `is_linear`. Feeds into `SpeculativeHomologyEngine` to trigger Draft Rejection. Internally uses `ResidueObstructionGraph` for homological PAS_h monitoring.

---

### nondual_admm.py
**Role**: Non-dual formulation of the ADMM probe.

Implements an ADMM variant that deliberately avoids scalarizing the dual variable — keeping constraint violations as separate, non-comparable pressure signals in domain-isolated vectors (preventing the Scalarization Trap from NOMENCLATURE §"Hard Interaction Contract"). Connected to INVARIANT_OPTIMIZATION §5 operational ADMM. *(Full class details pending source review.)*

---

### number_theoretic_stabilizer.py
**Role**: Applies number-theoretic stability constraints via dynamic prime spacing.

Enforces structural stability conditions derived from prime arithmetic — prime gaps, Euler product convergence, or modular residue distributions — to prevent numerical fragility in the CRT reconstruction pipeline. *(Full class details pending source review.)*

---

### orchestrator.py
**Role**: Universal Orchestrator — governs the scheduling and integrity of all Phase processing steps.

Manages the activation sequence (Phase 2.5 → 2.6 → 2.7 → ...) and enforces Anti-Lobotomy protocols: Implication Symmetry tracking, Gray-Zone State detection, and Normative Boundary labeling. Partial coverage in DIEGETIC_ENGINE.md. Implementation summary in conversation 072d4146.

---

### polychoron_quantization.py
**Role**: 4D polytope (polychoron) based quantization regime.

Extends Matrioshka quantization into 4D regular polytope geometry (24-cell, 120-cell, 600-cell structures) for higher-dimensional state representations. *(Full details pending source review.)*

---

### polynomial_scaffold.py
**Role**: Polynomial coefficient scaffolding for the ADMR solver.

Provides the structural skeleton (fixed-point polynomial coefficients) that `PolynomialADMRSolver` locks its state against during structural adaptation. Prevents "teleological leakage" by keeping the polynomial grid immutable during inference. *(Full class details pending source review.)*

---

### primitive_ops.py
**Role**: Low-level fixed-point and bitwise primitive operations.

Implements the `FixedPointField` backing operations (int64, scale 2¹⁶) and any primitive bitwise manipulations required for bit-exact cross-hardware reproducibility. Corresponds to INVARIANT_OPTIMIZATION §2.1 "FixedPointField." *(Full class details pending source review.)*

---

### quantum_inspired_reasoning.py
**Class**: `QuantumInspiredReasoningState`  
**Role**: Phase 17 extension simulating quantum superposition of reasoning states.

Represents a reasoning state as a superposition of basis states with complex-amplitude weights, collapsing to a definite output via measurement. Used to model multi-hypothesis reasoning before committing to a single interpretation. *(Full details pending source review.)*

---

### quantum_tda.py
**Role**: Quantum-inspired Topological Data Analysis.

Applies quantum amplitude amplification principles to persistence homology computations, accelerating the detection of topologically significant cycles. *(Full details pending source review.)*

---

### situational_batching.py
**Class**: `SituationalBatchSampler`  
**Role**: Non-i.i.d. batch sampler based on relational entanglement history.

Instead of uniform random sampling, batches are assembled by following "scars" of historical interaction. A Resonance Matrix `R_ij` (co-emergent coupling) and Mischief Matrix `M_ij` (chaotic affinity) accumulate pressure-weighted interaction scores between sample indices. `__iter__()` selects a seed, greedily samples high-`(R+M)` neighbors (seriousness), then fills with random "play" samples. Paradoxical boundary amplification: if local pressure exceeds `boundary_threshold`, resonance coupling is amplified by factor 1.5 (refusal as affirmation). `update_love_invariant()` updates both matrices with decay. Enables temporally coherent "entangled" batches for ADMR and temporal association training.

---

### sparse_higher_order_tensors.py
**Role**: Sparse representation of rank-3+ tensors for higher-order polynomial interactions.

Implements COO or CSR sparse encoding for tensors arising in higher-order polynomial coprimality computations, where dense storage would be prohibitive. *(Full class details pending source review.)*

---

### unknowledge_flux.py
**Role**: Tracks and gates "Structural Leakage" flows (Unknowledge).

Implements the Unknowledge channel: information that bypasses scalar logic and reveals hidden manifold archetypes. Partial coverage in `UN_KNOWLEDGE_GUIDE.md`. The flux observable is used by the DAQUF operator as a mischief boost signal.

---

### veto_subspace.py
**Role**: Manages the veto lattice and Gray-Zone State detection.

Full coverage in `VETO_SUBSPACE_ARCHITECTURE.md`. Included here for inventory completeness.

---

### voynich_architecture.py
**Role**: Implements the Voynich symbolic reasoning layer.

Full coverage in `THE_VOYNICH_ARCHITECTURE.md`. Included here for inventory completeness.

---

### yield_criteria.py
**Role**: Defines yield and fracture conditions for structural pressure thresholds.

Computes the conditions under which a structural component "yields" (transitions from elastic to plastic deformation, in the mechanical analogy) versus outright fractures (discrete abort). Corresponds to NOMENCLATURE terms "Instability, Fracture, Discord." *(Full class details pending source review.)*

---

## src/topology

### approximate_ph.py
**Role**: Approximate persistent homology for computational tractability.

Computes Betti numbers via approximate methods (Vietoris-Rips simplification, landmark selection) rather than exact persistence diagrams. Referenced in `OPEN_QUESTIONS §9.1` as the working solution to the undecidable-homology challenge. Reduces computation from exponential (exact PH) to polynomial typical-case.

---

### embedding_graph.py
**Role**: Manages the memory-state graph visualization and deduplication logic.

Builds and maintains the `GyroidicGraphManager` node graph, where nodes represent unique `memory_state` embeddings and edges represent structural resonance. Includes importance calculation, smart label wrapping, and advanced state indicators (Quantum, Matrioshka, Repaired, Locked). Deduplication uses `dedup_threshold` cosine distance on both text embeddings and `memory_state` vectors.

---

### homology_pressure.py
**Role**: Translates homological Betti-number changes into structural pressure signals.

Wraps the persistence obstruction computation and emits `StructuralPressure` vectors (non-scalarized, domain-isolated) when topological changes are detected. Partial coverage in `PHYSICS_ADMM.md`.

---

### speculative_homology.py
**Role**: Speculative decoding for Betti number prediction.

Uses fractional/gyroid priors to speculatively predict topological features ahead of full PH computation, enabling early exit from the ADMM loop when predicted homology state exhibits low spectral entropy (high confidence). Implements the Phase 3 speculative PH discussed in conversation 488feffe.

---

## src/optimization

### codes_driver.py
**Role**: Drives the CODES (Constraint Oscillation Driven Evolutionary Selection) framework.

Top-level scheduler for the constraint probe operators $\mathcal{P}_k$, orchestrating cyclic traversal and managing the global abort/stability signal. *(Full details pending source review.)*

---

### fractional_operators.py
**Role**: Fractional-order differential operators for anomalous diffusion dynamics.

Implements generalized fractional calculus operators (Riemann-Liouville or Grünwald-Letnikov approximations) with dynamically adjusted `alpha` parameter based on spectral coherence — hardening the operator (increasing alpha → 1) when coherence is low, allowing more fluid dynamics when coherence is high. Implemented in conversation 51ed57b4.

---

### ricci_flow_optimizer.py
**Role**: Ricci flow based manifold optimization.

Applies discrete Ricci flow — the process of uniformizing sectional curvature across the manifold — as an optimization step. Prevents curvature singularities that would produce degenerate CRT residues. *(Full details pending source review.)*

---

### sic_fa_admm.py
**Role**: Spectrally-corrected Inexact Constrained Feasibility-Aware ADMM.

Main ADMM solver with spectral transform for the CALM predictor, enabling speculative early exit when the predicted hidden state exhibits low spectral entropy. Partial coverage in `PHYSICS_ADMM.md`. Extended in conversations 51ed57b4 and 57c73ebe.

---

## src/training

### fgrt_fgrt_trainer.py
**Role**: Doubly-composed FGRT (Fractal Gyroidic Resonance Training) trainer.

Applies FGRT training composedly — each training step itself undergoes a fractal decomposition. The double-composition prevents teleological leakage by ensuring no single step can directly optimize toward a target. *(Naming appears intentional — double application of the FGRT principle.)*

---

### gdpo_trainer.py
**Role**: GDPO (Gyroidic Differential Pressure Optimization) trainer.

Implements the Signal Sovereignty and Functional Fossilization training protocol. Tracks performance streaks per functional group, applies mutation bias to low-streak groups, and triggers Trust Freezing (parameter exclusion from optimizer) for high-streak groups. Partial coverage in GDPO sections of various documents.

---

### training_manager.py
**Role**: Top-level training session orchestration.

Manages epoch and step scheduling, coordinates between `trainer.py`, `gdpo_trainer.py`, and `temporal_association_trainer.py`, handles checkpoint saving/loading, and emits the global abort signal if CALM vetoes the trajectory.

---

## src/models

### modular_attention.py
**Role**: CRT-modular attention mechanism.

Multi-head attention where each head is assigned to a distinct CRT modulus, enforcing that attention patterns across heads remain co-prime (structurally independent). Prevents "parasitic" attention overlap between different semiotic registers.

---

### modular_embeddings.py
**Role**: CRT-modular token embedding table.

Token embeddings organized by CRT residue class — tokens sharing the same residue class under a given modulus are initialized from the same distribution, structurally biasing the embedding space to respect the CRT factorization.

---

### polynomial_embeddings.py
**Role**: Polynomial basis token embeddings.

Represents tokens not as dense vectors but as coefficients in a co-prime polynomial basis. Chirality-enforcing initialization (non-zero `Δχ`) ensures the initial embedding space respects the Arrow-of-Time constraint from INVARIANT_OPTIMIZATION §4.

---

### diegetic_heads.py
**Role**: Output projection heads for the diegetic physics regime.

Implements the final projection from hidden state to output logits, with physics-constraint gating: outputs are only emitted if the current manifold state passes the admissibility check. Partial coverage in DIEGETIC_ENGINE.md.

---

## src/surrogates

### calm_predictor.py
**Role**: CALM (Constrained Asymptotic Lyapunov Monitor) meta-control surrogate.

Predicts whether the current optimization trajectory is heading toward entropic collapse or stagnation. Vetoes (aborts) trajectories when structural disintegration signals are detected. Full conceptual coverage in NOMENCLATURE §4 "Meta-Control (CALM)." The spectral CALM variant (with speculative exit) was implemented in conversation 51ed57b4.

---

### kagh_networks.py
**Role**: KAGH (Kolmogorov-Arnold Gyroidic Hebbian) surrogate networks.

Physics-informed surrogate providing admissible constraint embeddings. KAN layers (Kolmogorov-Arnold Networks) are partially fossilized to preserve topological structure across training. Implements `HuxleyRD` (reaction-diffusion) for stable hidden-state manifold formation and `ErgodicSolitonFusion` for persistence of non-ergodic sub-dynamics. Full source reviewed in conversation 488feffe.

---

*Last updated: 2026-02-22. Modules marked "(Full details pending source review)" have been inspected only at the module docstring level; detailed class inventories will be added when those modules become active development targets.*
