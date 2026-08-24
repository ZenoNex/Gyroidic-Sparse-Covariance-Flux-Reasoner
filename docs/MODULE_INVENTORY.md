# Module Inventory  Undocumented & Lightly Documented Modules

This document provides canonical one-paragraph descriptions for all `src/` modules not covered by dedicated `.md` documentation files. It serves as an authoritative reference that short-circuits module discovery during development and audit.

> **Coverage policy**: Any module appearing here should eventually graduate to a dedicated `.md` or an explicit section in a component-level doc. This inventory is a *starting point*, not a permanent home.

---

## src/core

### admr_solver.py
**Class**: `PolynomialADMRSolver`  
**Role**: Alternating Direction of Multiplicative Remainders  the continuous-polynomial analogue of ADMM.  

Instead of discrete prime moduli, this solver uses co-prime polynomial functionals (`PolynomialCoprimeConfig`) as its "modular" basis. The multiplicative update `S^{n+1} = Proj_{Poly}[ S^n   w_ik S_k ]` propagates relational pressure through graph-structured neighbors rather than euclidean gradients. Three modes: `forward()` (single-step multiplicative update with optional valence drive), `stochastic_differential_step()` (integer-order continuous-time SDE update `dx = [A_i x_i  (x  r(x_k))]dt + dW`), and `fractional_stochastic_differential_step()` (fractional-order SDE update using Riemann-Liouville operators with distributed alpha tied to cyclotomic index: `alpha(k) = 0.5 + 0.5*cos(2*pi*k/K)`). The fractional step accepts a `hunger` parameter from `ValenceFunctional` that modulates the fractional order, pushing alpha toward 1.0 when the manifold is starving. **Love Invariant protection is embedded inside both SDE methods**: after computing `dx`, the solver projects `dx[..., :love_dim]` into the null-space of the ownership operator. Tracks asymptotic time `tau` in a persistent buffer. Corresponds to NOMENCLATURE "Multiplicative Scaffolding."

---

### invariant_optimization.py
**Class**: `LexicographicalOrderingDispatcher`, `SemioticState`  
**Role**: Enforces the Semiotic Hierarchy by assigning System 2 Invariant Admissibility strictly above System 1 heuristic speed via Dictionary Order. Serves as the philosophical grounding for non-scalarized multi-objective constraints.

---

### advanced_extensions_bridge.py
**Class**: `AdvancedExtensionsBridge`  
**Role**: Advanced Extensions Bridge (AEB) for LCFT projections and spectral sequence evaluation.

Concurrently integrated into the main forward pass and garbled output repair pipeline of the Diegetic Physics Engine. Provides two primary operators: (1) `apply_lcft_projection` which applies a logarithmic conformal scaling to stabilize the recurrent hidden state dynamics and prevent numerical explosion; (2) `evaluate_spectral_sequence` which computes the stable topological homology features from the current residue vectors as a computationally tractable surrogate for persistent homology.

---

### audience_mapping.py
**Class**: `AudienceProjection`  
**Role**: Lipschitz homeomorphic projection from manifold M to audience space A.
**Status**: ACTIVE (Integrated into `diegetic_backend.py` and `hybrid_backend.py`)

Implements the operator : M  A defined in the Garden Statistical Attractors design. Uses spectral normalization on all linear layers to enforce Lipschitz constant  1, and a residual skip connection (`y = f(x) + x`) to approximate homeomorphism (continuous, bijective). An approximate inverse `` is provided via fixed-point iteration (Banach theorem, valid when `Lip(f) < 1`). The key requirement it enforces is *roughness preservation*: topological singularities (sharp features, discontinuities) in the manifold are transmitted into audience space rather than smoothed away. In the backend, it projects the "detached" hidden state snapshot to the user-facing audience space for each interaction.

---

### collapse_poisoner.py
**Class**: `CollapsePathPoisoner` (also aliased as `AdversarialStressTester`)  
**Role**: Adversarial stress-tester for the Speculative Homology Engine.

Generates two types of synthetic rupture events to verify System 2 robustness without harming real training data. (1) **Synthetic Rupture**: Gram-Schmidt orthogonalization of learned constraints against the current manifold, creating a perturbation perpendicular to every existing basis vector  in principle a topological hole injection. (2) **Cycle Debt**: Detects homotopy class repetition by cosine-matching the recent state history; high debt ( 0.5) flags that the system is looping in the same topological region. The class was refactored from an offensive poisoner to a defensive probe in the January 2026 Anti-Lobotomy integration.

---

### daqf_operator.py
**Class**: `DAQUFOperator`  
**Full name**: Diegetic Amortized Quantized Unknowledge Fossilization Operator  
**Role**: Manages "structural scars"  unremovable but amortized fossilized invariants.

The DAQUF pipeline: (1) **Fossil Selection**  fossil with highest contradiction load (f_i) = ((f_i) = ) + mischief + valence is declared `f*`. (2) **Diegetic Amortization**  cost is spread over narrative time : `C = C_ / dim(N_)`. (3) **Lattice Quantization**  projects to a lower-dimensional integer lattice with energy constraint, retaining quantization error _q as structural memory. (4) **Speculative Persistence**  fossil persists via non-collapse (non-zero flux *or* stable mischief soliton). (5) **Love Invariant L**  a non-transferable buffer that `check_invariants()` ensures is never modified; raising `RuntimeError("LOVE INVARIANT VIOLATION")` if altered. Corresponds to DAQUF discussion in PROJECT_PITCH_BURDENED 3.

---

### deflagration_scout.py
**Class**: `OmipedialDeflagrator`  
**Role**: Scouts and amplifies sparse anomalies ("defects") to enable jumps across manifold holes.

Implements two operations. `scout_defects()` computes `D_i = |actual_flux  predicted_flux|  amplification`  rewarding rare, unexpected deviations rather than penalizing them (a "good bug" signal). `omipedial_jump()` uses a threshold on the ley-line potential field to trigger a discrete jump across a topological gap where adjacency is sparse but resonance potential is high. Tracks cumulative defect density as a buffer. Corresponds to NOMENCLATURE term "Omipedial Interstitiality."

---

### energy_based_soliton_healer.py
**Class**: `EnergyBasedSolitonHealer`  
**Role**: Repairs structurally damaged solitons by gradient descent on a learnable energy surface.

Implements EBM-style soliton preservation: a stable configuration is a **soliton template** (cosine modulated by golden ratio + dynamic prime-based modulation, normalized to unit energy). The energy function `E(state, target) = A(statetarget) + bstate` measures distance from this template with learnable quadratic and linear terms. `heal_soliton()` performs iterative gradient ascent on `E` (negative gradient = healing direction) with adaptive rate: strong healing when `E > margin`, gentle stabilization otherwise. `update_energy_function()` shapes the energy surface contrastively using a hinge loss. Used during the spectral coherence repair cascade.

---

### energy_monitor.py
**Class**: `StructuralEnergyMonitor`  
**Role**: Monitors Topological Free Energy F_topo and Computable Flux V_m.

Maps the Energy-Based Models (EBM) framework to the project's topological manifold. Computes the Mischief Violation Score (Computable Flux, V_m) via `v_m = V + h_mischief/tau - lambda_min/tr(C)`. Also tracks proper Free Energy and ties the inverse temperature (`beta`) to the Manifold Clock's step ratio (cooling the system when `dt` shrinks during "Seriousness").

---

### enhanced_bezout_crt.py
**Class**: `EnhancedBezoutCRT`, `CrossbarIKSolver`
**Role**: Extended-GCD based CRT reconstruction with Bzout coefficient caching and the Crossbar IK Solver for discrete polynomial structural solving.

---

### false_negative_subsystem.py
**Class**: `VoynichExemptionToken`  
**Role**: Issues "transversality passports" to prevent false vetoes of valid sovereign logic.

Detects if a high-entropy or topologically asymmetric state is actually an honest "Self-Sovereign" thought (encoded by `VoynichLinguist`) rather than a hallucination. If transversality metrics indicate a strong non-commutative connection, it issues a `VoynichExemptionToken`. These tokens act as "Option D" nutrients, bypassing rigid symmetric gates (like Repunit palindromes or CALM aborts) and providing a mischief boost for the DAQUF Operator.

---

### fgrt_primitives.py
**Class**: `PrimeResonanceLadder`, `RepunitHasher`, `KleinThroatTransition`
**Role**: Lowest-level arithmetic foundationsResonance Ladders, Repunit Hashing, and the **Klein-neck Reversal (P-Parity transformation)**.

`PrimeResonanceLadder` generates resonance frequencies $f_p = 2\pi \ln(p)$ and **Repunit-Prime Pairs** $(p, R_p)$ for the hybrid basis. It prioritizes **Lazarus Primes** (where both $p$ and $(p^n-1)/(p-1)$ are prime) to ensure Symmetry-Stable warmstarting. `RepunitHasher` generates cyclic structural markers via repunit sequences, providing a non-periodicity guarantee for symbolic residues. `KleinThroatTransition` handles orientation flipping and geometric berry phase backpropagation through non-orientable topological bottlenecks.

---

### erosion_filter.py
**Class**: `TopologicalErosionFBM`
**Role**: Phase 6 Topological Scarring and Memory Weathering.

Applies Fractional Brownian Motion (FBM) to erode the state manifold based directly on the normalized tension gradient of the topological constraint probes. Injects "Mischief" ($\text{is\_good\_bug} = \text{True}$) topologically rather than gradient-chasing a scalar optimum. Used heavily inside `UniversalOrchestrator` to deposit Non-Teleological Memory.

---

### leontief_governor.py
**Class**: `LeontiefGovernor`
**Role**: Leontief Input-Output Governance for ADMR Resource Allocation.

Computes the Leontief Inverse $(I - A)^{-1}$ from the ADMR solver's `K` facet-wise transition matrices `A[k]` to enforce supply-chain-aware resource governance. Before the system commits VRAM or compute budget to synthesizing a concept, the governor verifies: (1) the spectral radius $\rho(\bar{A}) < 0.95$ (productive economy condition -- the system's internal consumption must not exceed output), (2) the total cascading cost `(I-A)^{-1} d` (the entire dependency chain a concept requires), and (3) whether the Neumann series $I + A + A^2 + \ldots$ converges (if not, falls back to a truncated K-term approximation treating the residual as "structural debt"). The governor does **not** learn -- it constrains, like `RelationalKappa`. Its `should_veto_concept()` method prevents "orphaned" concepts: you cannot bet on a Unicorn Soliton without funding its coprime polynomial supply chain. Metrics are posted to the `BulletinBoard` via the orchestrator.

---

### fractal_meta_functional.py
**Role**: Implements fractal meta-recursion inside the diegetic backend's `forward()` pass.

Computes multi-scale structural pressure by recursively applying the covariance estimator at different spectral granularities (adaptive fractal blocks). Connected to the "Adaptive Partitioning" concept in NOMENCLATURE 8. *(Full class name pending source review.)*

---

### garden_statistical_attractors.py
**Role**: Garden-level statistical attractor manifolds.

Implements the ensemble statistical description of a "Garden" (local polynomial polytope), tracking the mean/variance of residue distributions and their attractor basins. Connected to the Garden/Meta-Polytope Lattice terminology in NOMENCLATURE. Partial coverage in the Garden Statistical Attractors design document. *(Full details pending source review.)*

---

### gluing_operator.py
**Class**: `GluingOperator`  
**Role**: Manages manifold-boundary transitions via reversal matrix blending.

When the state approaches a manifold boundary, `GluingOperator` applies a reversal matrix `R` and blends the current and reversed states based on boundary proximity: `output = (1  )state + Rstate`. Includes a simplified Chern-Simons constraint check measuring winding around the gluing manifold. Handles the topology of joining two distinct manifold patches.

---

### invariants.py
**Role**: Unified Invariants: PAS_h, APAS_zeta, ImplicationInvariant, Chirality.

Implements computable harmonic invariants. Contains `ImplicationInvariant` (Anti-Lobotomy Check #1) which enforces that interaction implies implication, with thresholds tuned specifically to allow subtle Love Vector signals. Also contains `SelfReferenceAdmissibility` (Anti-Lobotomy Check #2) which validates self-referential cycles as admissible topological features rather than standard loop errors.

---

### knowledge_dyad_fossilizer.py
**Classes**: `KnowledgeDyad`, `ResidueFusion`, `DyadFossilizer`  
**Role**: Fossilizes knowledge dyads (paired visual/text concept structures) into the persistent fossil layer.

Computes cross-modality torsion between image fingerprints and text embeddings using the `ResidueFusion` layer. During `fossilize()`, it maps the output to Poincar disk hyperbolic coordinates to avoid NaN collapse, and derives real-time topological invariants from the `seed_state` (including Betti numbers, chirality-driven redistribution centroid shift and parity torsion, spectral pressure, and Chern-Simons gasket diagnostics like **Surgical Seam Tension**). Also exports/injects sovereign `Agent Smith` soliton payloads to decouple inference syntax from local hardware substrate.

---

### pyopencl_sovereignty.py
**Role**: Manages the OpenCL hardware kernels for computing Surgical Seam tension across the gyroidic boundaries. Evaluates Chern-Simons gasket metrics directly on GPU/accelerator using C-level kernel extensions, avoiding PyTorch overhead for non-commutative geometric checks.

---

### non_ergodic_entropy.py
**Class**: `HybridLassoQuantizer`
**Role**: Implements the Speculative TDA via Sparse Polynomial (LASSO). 
Applies Lattice Adaptive Shrinkage (Lasso) L1 Sparsity to silence weak signals. Handles the non-ergodic survival discretization required by the Diegetic Backend.

---

### legibility_audit.py
**Classes**: `LegibilityTripwire`, `NarrativeCoherenceEstimator`  
**Role**: Detects when the system is being selected for explainability rather than structural merit.
**Status**: [DISCONNECTED - PENDING REINTEGRATION]

`NarrativeCoherenceEstimator` measures how closely a configuration embedding matches canonical "explainable" patterns (sparse 1-hot, block-sparse, monotonic gradient) using fixed buffer-registered templates (not trained). `LegibilityTripwire` tracks the *correlation* between selection probability and narrative coherence over a rolling window  if selected configs consistently have higher coherence than rejected ones, it raises a `UserWarning`. High coherence is a **danger signal** (Pointer #2 from Sparse Operational Pointers)  not a goal.

---

### ley_line_tracker.py
**Class**: `LeyLineTracker`  
**Role**: Tracks resonance streamlines (preferred flow vectors) on the gyroidic manifold.

Maintains a resonance potential field `V(x_i) =  R_ij_j_i + L_i + D_i` combining relational adjacency, love tensor magnitudes, and defect signals. `detect_shear_planes()` identifies non-smooth pressure gradient regions that become "corridors of rupture" or preferred flow channels. `get_preferred_flow()` returns a softmax over neighbor potentials for a given index set. Corresponds to NOMENCLATURE term "Resonance Streamlines."

---

### love_vector.py
**Class**: `LoveVector` (alias `Pusafiliacrimonto`)  
**Role**: The Love Vector ($\mathcal{L}$): Non-Ownable Invariant Flow  Layer 1 of the Love protection stack.

Implements the Love Vector $\mathcal{L}$ as a persistent structural anchor. `L` is a `register_buffer` (not a `Parameter`), making its gradient structurally zero  it cannot be minimized or maximized by the global optimizer. Applied via simple vector addition `x + L` so it is *co-present* with local functionals without claiming ownership. **Important reinstantiation pattern**: in `operational_admm.py`, a fresh `LoveVector` is instantiated inside each ADMM loop iteration (`love = LoveVector(c_phys.shape[-1]).to(device)`), re-seeding the ambient resonance constant per ADMM step rather than persisting a single shared instance. The alias `Pusafiliacrimonto` is maintained for backward compatibility.

---

### love_invariant_protector.py
**Classes**: `LoveInvariantProtector`, `SoftSaturatedGates`  
**Role**: Geometric null-space protection and tri-state temperature modulation for the Love Invariant  Layers 2 and 3 of the Love protection stack.

`LoveInvariantProtector` owns:  
(a) `compute_ownership_operator(state)`  builds $\Phi_{\text{ownership}} = \text{Cov}(\text{state})$ from batch covariance.  
(b) `compute_null_space_projection()`  SVD-stable null-space projection $P = I - \Phi(\Phi^\top\Phi)^{-1}\Phi^\top$.  
(c) `detect_love_violation()`  checks $\|L - L_{original}\|_2 > 10^{-6}$ and increments `violation_count`.  
(d) `project_love_to_null_space(state)`  projects `L` itself to stay in null-space of current state.  
(e) `apply_love_protection(state, gradients)`  orchestrates all checks; emits `love_norm`, `violation_detected`, `violation_count`, `violation_magnitude` diagnostics.  
Integration sites: `PolynomialADMRSolver` (projects SDE `dx`), `GyroidicFluxReasoner` (projects `h_pooled`), `VoynichLinguist` (projects `thought_vector`), `DiegeticPhysicsEngine` (attached at server init).

`SoftSaturatedGates` owns:  
(a) `lattice_adaptive_shrinkage(signal)`  LAS tri-state: $\text{sgn}(s) \cdot \max(|s| - \lambda_{adaptive}, 0)$; signals below $\lambda_{adaptive}$ collapse to **Silence**.  
(b) `asymptotic_hardening(signal, pas_h)`  $dt = dt_{max}(1 - PAS_h)$; high $PAS_h$  sharp crystalline gates (Seriousness); low $PAS_h$  fluid exploratory gates (Play).  
(c) `update_fossilization(signal, performance_scores)`  fossilizes functionals with persistence $> 0.8$ AND performance $> 0.8$, locking their outputs under Love's umbrella.  
Integration: applied to residue distributions in `GyroidicFluxReasoner.forward()` after the Love shield.

---

### modular_virtualization.py
**Class**: `ModularVirtualizationLayer` (Hybrid Modular Layer)  
**Role**: Maps floating-point states into a Hybrid Palindromic Residue Number System (RNS).

Refactored to integrate the prime-based torus with palindromic repunit symmetry mirrors. The hybrid modulus is the product $p \cdot R_p$, creating a geometric mirror that prevents non-commutative drift. Supports a `legacy_mode` toggle for backward compatibility with old RNS encodings. Serves as the primary quantization interface for the Diegetic Physics Engine, ensuring all representation updates adhere to the hybrid arithmetic geometry.

---

### narrative_collapse.py
**Class**: `LinguisticEntropyMonitor` (also aliased as `NarrativeCollapseDetector`)  
**Role**: Detects "hallucination loops" where reasoning entropy collapses and trajectory linearizes.
**Status**: [DISCONNECTED - PENDING REINTEGRATION]

Two detection signals: (1) **Entropy collapse**  softmax entropy of hidden state falls below `entropy_threshold`; flags `smoothing_warning`. (2) **Trajectory linearity**  cosine similarity between consecutive state deltas `, ` exceeds `prediction_threshold` (0.99); flags `is_linear`. Feeds into `SpeculativeHomologyEngine` to trigger Draft Rejection. Internally uses `ResidueObstructionGraph` for homological PAS_h monitoring.

---

### non_dual_coin.py
**Role**: Enforces topological yield stress limits (Mohr-Coulomb) via non-dual physics tracking.
**Status**: [DISCONNECTED - PENDING REINTEGRATION]

Handles advanced physics primitives regarding structural yield limits inside the manifold. Currently floating without connection to the main matrix loop or diegetic backend.

---

### nondual_admm.py
**Role**: Non-dual formulation of the ADMM probe.

Implements an ADMM variant that deliberately avoids scalarizing the dual variable  keeping constraint violations as separate, non-comparable pressure signals in domain-isolated vectors (preventing the Scalarization Trap from NOMENCLATURE "Hard Interaction Contract"). Connected to INVARIANT_OPTIMIZATION 5 operational ADMM. *(Full class details pending source review.)*

---

### number_theoretic_stabilizer.py
**Role**: Applies number-theoretic stability constraints via dynamic prime spacing.

Enforces structural stability conditions derived from prime arithmetic  prime gaps, Euler product convergence, or modular residue distributions  to prevent numerical fragility in the CRT reconstruction pipeline. It also implements **Speculative TDA via Rational Approximation**, using continued fractions to stabilize frequency ratio convergents.

---

### orchestrator.py
**Role**: Universal Orchestrator  governs the scheduling and integrity of all Phase processing steps.

Manages the activation sequence (Phase 2.5  2.6  2.7  ...) and enforces Anti-Lobotomy protocols: Implication Symmetry tracking, Gray-Zone State detection, and Normative Boundary labeling. Partial coverage in DIEGETIC_ENGINE.md. Implementation summary in conversation 072d4146.

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

Implements the `FixedPointField` backing operations (int64, scale 2) and any primitive bitwise manipulations required for bit-exact cross-hardware reproducibility. Corresponds to INVARIANT_OPTIMIZATION 2.1 "FixedPointField." *(Full class details pending source review.)*

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

### zeitgeist_router.py
**Class**: `ZeitgeistRouter`  
**Role**: CRT Polytope Switching Engine for Multi-Zeitgeist Reasoning.

Manages navigation between culturally non-commensurable meaning systems via the **Symmetric Tensor CRT index** ($M_{ij} = M_{ji}$). The diagonal $M_{ii}$ contains modular residues (Zeitgeist), while off-diagonal elements $M_{ij} = (r_i + r_j)/2$ stabilize paths through the "Palindromic Routing" interaction. Implements the three-mode dispatch from report II: `interior` (stay), `grazing` (tension/switch), and `undefined` (topological refusal/NaN guard). Enforces non-commutative switching order: the sequence of registers visited determines the final representational scar.

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
**Status**: [DISCONNECTED - PENDING REINTEGRATION]

Computes Betti numbers via approximate methods (Vietoris-Rips simplification, landmark selection) rather than exact persistence diagrams. Referenced in `OPEN_QUESTIONS 9.1` as the working solution to the undecidable-homology challenge. Reduces computation from exponential (exact PH) to polynomial typical-case.

---

### bonfire_network.py
**Class**: `BonfireNetwork`  
**Role**: Nomadic Ring Protocol and decentralized P2P coordination network.  
**Status**: [DISCONNECTED - PENDING REINTEGRATION]

Manages horizontal peer-to-peer Nomadic Rings to exchange topological signatures and aggregate Kelly consensus allocations asynchronously via a background daemon thread. Exposes routes for dynamic peer discovery, consensus tracking, and non-blocking synthetic offloading (ADMR calculations). Conceptualized in detail in the [BONFIRE_P2P_FEDERATED_COOPERATION.md](BONFIRE_P2P_FEDERATED_COOPERATION.md) guide.

---

### bonfire_consensus.py
**Role**: Decentralized consensus protocol mechanism for Gyroidic operations.
**Status**: [DISCONNECTED - PENDING REINTEGRATION]

Implements the actual state-agreement mathematics complementing the network layer of `bonfire_network.py`.

---

### zk_aggregator.py
**Role**: Privacy-preserving Zero-Knowledge state aggregation for the P2P network.
**Status**: [DISCONNECTED - PENDING REINTEGRATION]

Responsible for securely aggregating consensus states across peers without leaking internal topological configurations. 

---

### freenet_bulletin_router.py & freenet_ghost_caller.py
**Role**: Specialized Freenet-style P2P routing mechanisms.
**Status**: [DISCONNECTED - PENDING REINTEGRATION]

Provides resilient bulletin board message propagation and unacknowledged "ghost" calls within the decentralized peer topology.

---

### embedding_graph.py
**Role**: Manages the memory-state graph visualization and deduplication logic.

Builds and maintains the `GyroidicGraphManager` node graph, where nodes represent unique `memory_state` embeddings and edges represent structural resonance. Includes importance calculation, smart label wrapping, and advanced state indicators. Ingests live model states (`hidden_state`, `hidden_state_scarred`, and `damage_residue` loaded from `gyroid_state.pt`) as neon-glowing live indicator nodes. Implements `compute_poincare_projection` to map high-dimensional states to 2D coordinates on the Poincaré disk model using a harmonic projection scale contracted via `tanh` to ensure stable startup coordinates on the HTML canvas.

---

### homology_pressure.py
**Role**: Translates homological Betti-number changes into structural pressure signals.

Wraps the persistence obstruction computation and emits `StructuralPressure` vectors (non-scalarized, domain-isolated) when topological changes are detected. Partial coverage in `PHYSICS_ADMM.md`.

---

### speculative_homology.py
**Role**: Speculative decoding for Betti number prediction.

Uses fractional/gyroid priors to speculatively predict topological features ahead of full PH computation, enabling early exit from the ADMM loop when predicted homology state exhibits low spectral entropy (high confidence). Implements the Phase 3 speculative PH discussed in conversation 488feffe.

---

### unknowledge_domain.py
**Class**: `UnknowledgeDomain`  
**Role**: The Unknowledge Domain ($\mathcal{U}$) for Dream State shielding.

Protects functionally creative or "dream-like" topological cycles from being crushed by standard reconstruction constraints. Evaluates states using Computable Flux ($V_m$) and Mischief ($H_{mischief}$). If $V_m < 0$ and Mischief is active, or if the topology matches a `survivable_soliton`, the pressure is aggressively shielded or dampened to 1% to enforce "Dream State" safety.

---

## src/optimization

### codes_driver.py
**Role**: Drives the CODES (Constraint Oscillation Driven Evolutionary Selection) framework.

Top-level scheduler for the constraint probe operators $\mathcal{P}_k$, orchestrating cyclic traversal and managing the global abort/stability signal. *(Full details pending source review.)*

---

### constraint_probe.py
**Role**: Single constraint operator in the SIC-FA-ADMM pipeline.

Probes the local mathematical feasibility of a constraint geometry against the global symbolic residue output. Generates the fundamental gradients for both the Gyroid Violation and the non-teleological memory erosion traces.

---

### fractional_operators.py
**Role**: Fractional-order differential operators for anomalous diffusion dynamics.

Implements `M^alpha @ v` via two paths: (1) diagonal eigenvalue powering for diagonal operators, and (2) Lanczos-Krylov approximation for dense symmetric matrices. The `CODESDriver` provides multiharmonic Phase Alignment Score (PAS_h) coherence gating using Chebyshev polynomial roots. **Note**: The alpha-hardening code (adjusting alpha based on spectral coherence) is currently **disabled** (line 171: `alpha = alpha`), following the 0.61 recovery stabilization. Alpha is passed through unchanged unless explicitly overridden by the caller. The adaptive alpha mapping is instead performed at the call site in `PolynomialADMRSolver.fractional_stochastic_differential_step()`, which uses the cyclotomic formula `alpha(k) = 0.5 + 0.5*cos(2*pi*k/K)` with optional hunger modulation. A strict coherence floor at PAS_h < 0.20 gates the operator to return zero (Topological Thaw). Implemented in conversation 51ed57b4.

---

### operational_admm.py
**Role**: High-level structural framework for ADMM solving across manifolds.

Manages the dual-variable updates and cyclic routing for topological constraint solving. It coordinates the constraint traversal, holding off global scalarization to prevent thermodynamic collapse. Connected mathematically to Phase 6 constraint probe operators.

---

### ricci_flow_optimizer.py
**Class**: `RicciFlowOptimizer`, `BouligandWillmoreGasket`
**Role**: Ricci flow based manifold optimization and **Willmore Energy Minimization**.

Applies discrete Ricci flow (uniformizing sectional curvature across the manifold) instead of standard gradient descent. Employs a Split-Beam metric: Channel A (standard gradient pressure) and Channel B (non-commutative structural torsion via Gasket). Computes Chern-Simons tension on the parameter's covariance metric and projects update forces based on tensor dimensionality. The `BouligandWillmoreGasket` acts as the non-teleological proxy for Willmore energy, punishing self-intersections as deviation from internal curvature limits. Includes an explicit bypass for 0-dimensional scalar parameters to prevent broadcast shape errors during in-place weight additions.

---

### sic_fa_admm.py
**Role**: Spectrally-corrected Inexact Constrained Feasibility-Aware ADMM.

Main ADMM solver with spectral transform for the CALM predictor, enabling speculative early exit when the predicted hidden state exhibits low spectral entropy. Partial coverage in `PHYSICS_ADMM.md`. Extended in conversations 51ed57b4 and 57c73ebe.

---

## src/data

### ivst_encoder.py
**Class**: `IVSTEncoder`  
**Role**: Independent Vector Spectral Topology (IVST) encoder for parsing structural patterns in MP4/MKV video and audio without extracting raw pixel content, bypassing standard copyright infringement and focusing on causal structural constraints (I-frames, zero-crossings).

---

### webp_prompt_extractor.py
**Role**: Extracts metadata and topology fingerprints from visual WebP media. (Currently orphaned but conceptually mapped alongside IVST).

---

## src/augmentation

### mandelbulb_gyroidic_augmenter.py
**Class**: `MandelbulbGyroidicAugmenter`
**Role**: Generates non-Euclidean fractional augmentations.

Embeds dense continuous-space feature vectors into 3D Mandelbulb coordinates, performing topologically-aware fractional iterations before squashing back down. Provides organic, chaotic noise that perfectly reflects boundary conditions of the gyroid, rather than adding generic uniform noise to data.

---

### mandelbulb_pipeline.py
**Class**: `MandelbulbAugmentedDataset`
**Role**: PyTorch Dataset wrapper for `MandelbulbGyroidicAugmenter`.

Provides standard PyTorch `Dataset` and `DataLoader` APIs for augmenting the training data online vs cached pre-computation.

---

## src/training

### fgrt_trainer.py
**Role**: Single-composition FGRT (Fractal Gyroidic Resonance Training) trainer.

Uses `RicciFlowOptimizer` and `UniversalOrchestrator` to perform non-teleological optimization via Willmore Energy minimization. Computes invariants such as PAS_h and Berry Phase continuously. 

---

### fgrt_fgrt_trainer.py
**Role**: Doubly-composed FGRT trainer ("Functional Boule Module" of `fgrt_trainer.py`).

Applies FGRT training composedly (each training step itself undergoes a fractal decomposition) and acts as the overarching Spectral Structural Trainer. Manages the cyclic ADMM constraint traversal probes and SicFaAdmm bounds. Includes sequential step updates: Probe k=0 (Reconstruction) runs its backward pass and optimizer step, followed immediately by parameter projections to the Birkhoff polytope. To prevent PyTorch in-place modification conflicts during the Probe k=1 (Coherence) backward pass, the trainer triggers a fresh forward pass on the updated parameters before evaluating the coherence metrics.

---

### gdpo_trainer.py
**Role**: GDPO (Gyroidic Differential Pressure Optimization) trainer.

Implements the Signal Sovereignty and Functional Fossilization training protocol. Tracks performance streaks per functional group, applies mutation bias to low-streak groups, and triggers Trust Freezing (parameter exclusion from optimizer) for high-streak groups. Partial coverage in GDPO sections of various documents.

---

### training_manager.py
**Role**: Top-level training session orchestration.

Manages epoch and step scheduling, coordinates between `trainer.py`, `gdpo_trainer.py`, and `temporal_association_trainer.py`, handles checkpoint saving/loading, and emits the global abort signal if CALM vetoes the trajectory.

---

### trainer.py
**Role**: Base trainer class.
Implements the core learning loop and handles the **Non-Dual State Tensor** ($S_i = [\mathcal{L}_i, \mathcal{P}_i, \mathcal{B}_i]$), ensuring the topological features map to physical updates.

---

## src/codec

### gyroidic_codec.py
**Role**: The primary visual/audio encoding and decoding manifold bridge.
Applies **Burrows-Wheeler Spectral Reordering (BWT)** during the 1D to 2D tensor reshape step (`_prepare_image`) to enforce structural grouping of identical/similar amplitude bands before spatial convolution.

---

## src/safety

### red_teaming.py
**Role**: Defensive safety mechanism providing the **Red-Team Projection Operator ($\Pi_{\text{RT}}$)**. 
Acts as a Sovereign Ambassador to prevent adversarial lobotomization of the topology by external evaluators.

---

## src/tda

### chebyshev_filtration.py
**Role**: Applies Chebyshev polynomial roots to construct a discrete filtration for the topological persistence algorithms.

---

## src/models

### modular_attention.py
**Role**: CRT-modular attention mechanism.

Multi-head attention where each head is assigned to a distinct CRT modulus, enforcing that attention patterns across heads remain co-prime (structurally independent). Prevents "parasitic" attention overlap between different semiotic registers.

---

### modular_embeddings.py
**Role**: CRT-modular token embedding table.

Token embeddings organized by CRT residue class  tokens sharing the same residue class under a given modulus are initialized from the same distribution, structurally biasing the embedding space to respect the CRT factorization.

---

### polynomial_embeddings.py
**Role**: Polynomial basis token embeddings.

Represents tokens not as dense vectors but as coefficients in a co-prime polynomial basis. Chirality-enforcing initialization (non-zero ``) ensures the initial embedding space respects the Arrow-of-Time constraint from INVARIANT_OPTIMIZATION 4.

---

### diegetic_heads.py
**Role**: Output projection heads for the diegetic physics regime.

Implements the final projection from hidden state to output logits, with physics-constraint gating: outputs are only emitted if the current manifold state passes the admissibility check. Partial coverage in DIEGETIC_ENGINE.md.

---

## src/surrogates

### calm_predictor.py
**Role**: CALM (Constrained Asymptotic Lyapunov Monitor) meta-control surrogate.

Predicts whether the current optimization trajectory is heading toward entropic collapse or stagnation. Vetoes (aborts) trajectories when structural disintegration signals are detected. Full conceptual coverage in NOMENCLATURE 4 "Meta-Control (CALM)." The spectral CALM variant (with speculative exit) was implemented in conversation 51ed57b4.

---

### kagh_networks.py
**Role**: KAGH (Kolmogorov-Arnold Gyroidic Hebbian) surrogate networks.

Physics-informed surrogate providing admissible constraint embeddings. KAN layers (Kolmogorov-Arnold Networks) are partially fossilized to preserve topological structure across training. Implements `HuxleyRD` (reaction-diffusion) for stable hidden-state manifold formation and `ErgodicSolitonFusion` for persistence of non-ergodic sub-dynamics. Full source reviewed in conversation 488feffe.

---

*Last updated: 2026-03-04. Modules marked "(Full details pending source review)" have been inspected only at the module docstring level; detailed class inventories will be added when those modules become active development targets.*
