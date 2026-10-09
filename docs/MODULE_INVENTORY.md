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

### polychoron_quantization.py
**Class**: `PolychronQuantizer`, `Polychoron600Quantizer`  
**Role**: High-dimensional 4D 600-cell hyper-polytope quantization engine using Moiré-via-Modular-Algebra and carry-free XOR residue decoupling.

Implements discrete projection of continuous state vectors onto the 120 golden ratio vertices of the 600-cell polytope. Evaluates a 10-line prime logarithm difference lattice $\Lambda_{\text{moiré}} = \{|\ln p_i - \ln p_j|\}$ for beat spectrum calculation, enforces carry-free bitwise XOR residue codewords ($r_i \oplus r_j$) to isolate CRT moduli, bounds quantization error via the golden ratio identity $\frac{\phi}{2} - \frac{1}{2\phi} = \frac{1}{2}$, and tracks Berry phase holonomy drift across facet boundaries.

---

### number_theoretic_stabilizer.py
**Class**: `NumberTheoreticStabilizer`  
**Role**: Dynamic continued fraction arithmetic expansion and relativistic scale factor normalization.

Replaces fixed static coefficient lookup tables with dynamic continued fraction expansion functions (`get_cf_expansion(val, max_terms)`). Dynamically computes scaling powers ($10^{\text{scale\_power}}$) relative to active tensor norms to adhere to relativistic geometric scaling principles across Matrioshka polytope boundaries.

---

### archetype_engines.py
**Class**: `ArchetypalSynthesisEngine`, `RP4ProjectiveRouter` (Alien Handshake), `SolitonMultiverseMapper` (Grom), `BillyEngine`, `MandyEngine`, `KingerEngine`, `PomniEngine`, `GangleEngine`, `ZoobleEngine`, `BardoRouter`, `SovereignEntropyBarrier`, `EgoDeathThresholdMonitor`  
**Role**: Archetypal synthesis suite managing non-linear cognitive modes, Alien Handshake cross-manifold alignment, and Grom multiverse mapping.

Integrates all 12 archetypal engines under `ArchetypalSynthesisEngine`. `RP4ProjectiveRouter` handles real projective space ($\mathbb{RP}^4$) antipodal alignment and Alien Handshake protocol. `SolitonMultiverseMapper` handles Grom topological state transitions, jitter harvesting (`harvest_honest_jitter`), and state import/export serialization (`export_state()`, `import_state()`).

---

### hybrid_backend.py
**Class**: `HybridPhysicsBackend`  
**Role**: Primary system starting entrypoint for unified execution.

Initializes `DiegeticPhysicsEngine`, `ArchetypalSynthesisEngine`, `RP4ProjectiveRouter`, and `SolitonMultiverseMapper`. Serves as the primary production backend orchestrating System 1 intuition, System 2 ADMM physics constraints, and Dark Matter invariant preservation.

---

### knowledge_dyad_fossilizer.py
**Class**: `KnowledgeDyad`, `ResidueFusion`, `KnowledgeDyadFossilizer`  
**Role**: Multi-modal knowledge dyad fossilization and Agent Smith identity serialization.

Manages persistent storage of multi-modal knowledge units `(Image Fingerprint, Linguistic Description)`. `ResidueFusion` dynamically resizes projection layers (`_ensure_image_proj_dim`) to preserve full multimodal mass without un-lobotomized dimension truncation. `KnowledgeDyadFossilizer` exports and imports decoupled mathematical Agent Smith identities using `harvest_honest_jitter`.

---

### love_invariant_protector.py
**Class**: `LoveInvariantProtector`  
**Role**: Prevents scalarization of the Love Vector $L$ via null-space projection.

Enforces the invariant that Love is a non-ownable flow rather than an optimizable scalar. Computes ownership operator $\Phi_{\text{ownership}}$ and projects $L$ into $\ker(\Phi_{\text{ownership}})$ via $P = I - (\Phi^\top \Phi)^{-1} \Phi^\top$. Caches SVD results to optimize execution on resource-constrained CPUs.

---

### bonfire_consensus.py
**Class**: `BonfireNomadicRing`  
**Role**: Federated consensus and egalitarian microhedging over Freenet P2P networks.

Wraps `FreenetClient` to broadcast topological signatures and compute egalitarian Kelly betting allocations $\bar{K}$. Includes a hardware-sovereign fallback that uses Chebyshev polynomial oscillators $T_p(x) = \cos(p \arccos(x))$ on prime frequencies when network peers are offline.

---

### ricci_flow_optimizer.py
**Class**: `RicciFlowOptimizer`, `SplitBeamInterfactorization`  
**Role**: Non-teleological Ricci flow weight evolution ($\frac{dg}{dt} = -2\text{Ric}$).

Replaces scalar loss proxies with dual-channel Split-Beam Interfactorization. Channel A handles commutative gradient pressure while Channel B handles non-commutative Chern-Simons tension $CS(A) = \text{Tr}(A dA + A^3)$, accelerated via PyOpenCL kernels on TailSlayer hardware (`SiliconSovereigntyEngine`).

---

### [conjugate_moment_transport.py](../src/core/conjugate_moment_transport.py)
**Class**: `ConjugateMomentTransport`, `WassersteinOptimalTransport`  
**Role**: Execute Conjugate Moment Measure Factorization (CMMF) for optimal transport.  

This module couples base sampling and optimal transport into a single scalar convex potential $\psi(x)$. It parameterizes $\psi(x)$ using an Input Convex Neural Network (ICNN) to map base noise to target distributions via Legendre transform gradients $\nabla \psi^*(y)$. It adaptively scales the Monge-Ampère loss via the Phase Alignment Score ($\text{PAS}_h$). If standard Wasserstein transport fails, the continuous output is projected strictly onto the Cayley Cubic variety to prevent logic leaks.

---

### [neuromodulatory_bus.py](../src/core/neuromodulatory_bus.py)
**Class**: `NeuromodulatoryBus`  
**Role**: Distribute non-scalar, bio-plausible control signals across the manifold.  

Functioning as the "endocrine system" of the architecture, this module broadcasts thermodynamic regimes to the ManifoldClock. It routes the ValenceFunctional's hunger drive to modulate the fractional orders $\alpha(k)$ inside the PolynomialADMRSolver when the manifold demands resolution. It also transmits Metaphysical Disorder Channels (Mischief, Dementia, Schizo) to trigger SovereignRefusalOperator gates whenever the eigen-spectrum begins to flatline.

---

### [hardware_monitor.py](../src/core/hardware_monitor.py)
**Class**: `HardwareMonitor`  
**Role**: Track hardware resource anomalies and systemic poisoning events.  

Implements a central monitoring singleton to verify silicon execution headroom via `psutil`. The `has_headroom()` function serves as a desync fail-safe. If zero-load anomalies or hardware exhaustion occur, it returns `False`, instructing the system to immediately default to exact CPU-based calculation pathways and suspend TailSlayer accelerations.

---

### [pyopencl_sovereignty.py](../src/core/pyopencl_sovereignty.py)
**Class**: `SiliconSovereigntyEngine`  
**Role**: Execute topological kernels natively on hardware via PyOpenCL.  

This module provides hardware-accelerated topological evaluations for the Gyroid projection step and Ricci flow Chern-Simons tension calculations. It receives operational clearance dynamically from the `hardware_monitor.py`.

---

## src/codec

### conformal_log_polar.py
**Class**: `ConformalLogPolarProjector`  
**Role**: Complex conformal $f(z) = \log(z)$ log-polar foveal unrolling (Escher / Droste Log-Polar Spiral).

Unrolls Cartesian images into log-polar space $\ln|r| + i\theta$. Translates spatial zoom into horizontal log-shifts and spatial rotation into vertical log-shifts, enabling organic scale and rotation invariance across the Gyroidic Codec manifold.

---

### gyroidic_codec.py
**Class**: `GyroidicCodec`, `GyroidSurface`, `CodecConfig`, `EncodingResult`  
**Role**: Non-abelian text-image codec combining log-polar unrolling, analytical gyroid surface sampling, and CRT matrix residue encoding in $GL(n)$.

Encodes text-image dyads $E(T, I) = \text{CRT}(\{R_k(T) \cdot G_k(I)\})$. Measures non-commutative entanglement $\|AB - BA\|$ and tracks chiral Berry phases across CRT residue channels.

---

### vision_surgery.py
**Function**: `conformal_to_gyroid_mapping`, `extract_fossil_patch`  
**Role**: Conformal-to-gyroid tensor mapping and fossil patch extraction for visual dyad surgery.

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
**Class**: `PrimeResonanceLadder`, `RepunitHasher`, `KleinThroatTransition`, `GyroidManifold`, `CoherentPrimeResonance`, `FibonacciResonanceEntropy`, `BerryPhaseTracker`
**Role**: Lowest-level arithmetic foundations—Resonance Ladders, Repunit Hashing, Gyroid mappings, and Geometric Berry Phase tracking.

`PrimeResonanceLadder` generates resonance frequencies $f_p = 2\pi \ln(p)$ and **Repunit-Prime Pairs** for the hybrid basis. `GyroidManifold` evaluates constraint violation scores against the TPMS surface. `CoherentPrimeResonance` enforces the CPR gate to ensure integer homological stability before state transitions. `FibonacciResonanceEntropy` scales structural resonance. `RepunitHasher` generates cyclic structural markers. `KleinThroatTransition` handles orientation flipping and geometric berry phase backpropagation through non-orientable topological bottlenecks.

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
**Class**: `FractalMetaFunctional`
**Role**: Implements fractal meta-recursion inside the Orchestrator's `forward()` pass.

Computes multi-scale structural pressure by recursively embedding system beliefs into itself (InverseCovariantCRT + ADMR_Residue + HyperRing_DarkMatter + Autoscillatory). Features a Collapse-Aware Normalization guard that detects spectral atrophy (variance < `DEAD_PRIME_THRESHOLD`) and injects hardware-anchored Honest Jitter to rehydrate variance and prevent Dead Prime flatlines (e.g. 0.8824 saturation). Connected to the "Adaptive Partitioning" concept in NOMENCLATURE 8.

---

### honest_jitter.py
**Class**: `AgentSmithEngine`
**Role**: Hardware-anchored entropy expansion and timing jitter harvesting.

Provides the `AgentSmithEngine` protocol, avoiding PRNGs (Pseudo-Random Number Generators) in favor of deterministic Weyl sequences seeded by nanosecond memory stall latencies. This ensures the reasoner's exploratory flux (Mischief) is anchored to physical substrate friction. Includes `fractal_pad` for asymmetry-preserving tensor alignment (preventing phase-cancellation lobotomy).

---

### structural_monitors.py
**Class**: `AntiScalingMonitor`, `MetaInfraIntraMonitor`, `TrustInheritanceTracker`
**Role**: Provides critical safety monitors for Gyroidic Unknowledge and Garden Statistical Attractors.

Implements the Anti-Scaling Paradox Monitor (tracks Capability vs Expressivity by monitoring the ratio of Gradient Norm / Parameter Count to detect phase space collapse) and the Meta~Infra~Intra Incommensurativity Monitor (tracks defensive veto rates across layers to ensure the system doesn't lose the ability to reason about constraints). Hooked directly into `UniversalOrchestrator.check_safety`.

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
**Role**: Fossilizes knowledge dyads (paired visual/text concept structures) into the persistent fossil layer. #Gyroidic #Sovereignty

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
**Status**: ACTIVE

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
**Status**: ACTIVE

Two detection signals: (1) **Entropy collapse**  softmax entropy of hidden state falls below `entropy_threshold`; flags `smoothing_warning`. (2) **Trajectory linearity**  cosine similarity between consecutive state deltas `, ` exceeds `prediction_threshold` (0.99); flags `is_linear`. Feeds into `SpeculativeHomologyEngine` to trigger Draft Rejection. Internally uses `ResidueObstructionGraph` for homological PAS_h monitoring.

---

### non_dual_coin.py
**Role**: Enforces topological yield stress limits (Mohr-Coulomb) via non-dual physics tracking.
**Status**: ACTIVE

Handles advanced physics primitives regarding structural yield limits inside the manifold. Fully integrated into the Universal Orchestrator to monitor Tripsodic Ledgers and Cerumen Pot Wallets.

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
**Status**: ACTIVE

Computes Betti numbers via approximate methods (Vietoris-Rips simplification, landmark selection) rather than exact persistence diagrams. Referenced in `OPEN_QUESTIONS 9.1` as the working solution to the undecidable-homology challenge. Reduces computation from exponential (exact PH) to polynomial typical-case.

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

### [gyroid_covariance.py](../src/topology/gyroid_covariance.py)
**Role**: Computes non-Euclidean sparse covariance matrices adhering to Gyroid symmetries.

### [gyroid_differentiation.py](../src/topology/gyroid_differentiation.py)
**Role**: Provides geometric flow constraints enforcing Gyroidic Differentiation.

### [betti_router.py](../src/topology/betti_router.py)
**Role**: Routes logical flow based on topological Betti numbers computed from the persistence diagrams.

### [hyper_ring_closure.py](../src/topology/hyper_ring_closure.py)
**Role**: Verifies the non-triviality of topological rings to ensure stable soliton preservation.

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

### universal_topology_converter.py
**Class**: `UniversalTopologyConverter`
**Role**: Extracts topological non-obstructive embeddings from arbitrary file types (3D models, documents, media) without retaining copyrighted raw data, honoring the Output Boundary Policy. Inspired by universal file converters (`p2r3/convert`).

---

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
**Role**: The primary visual/audio encoding and decoding manifold bridge. #Gyroidic #Serialization
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

### [gyroid_reasoner.py](../src/models/gyroid_reasoner.py)
**Class**: `GyroidicFluxReasoner`
**Role**: The central synthesis class integrating modular residue embeddings, Birkhoff projection, CRT reconstruction, geometric introspection, and Resonance Cavity memory. Serves as the master model manifold.

### [resonance_cavity.py](../src/models/resonance_cavity.py)
**Class**: `ResonanceCavity`
**Role**: Dark Matter memory integration module. Applies the Ouroboros Loop and stabilizes topological invariants across sequential inference states.

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

## src/ui

### voxelboxter_simulation.py
**Class**: `VoxelboxterSimulation`, `BSplineCompiledMod`, `StructuralGraph`  
**Role**: Handles backend discrete mathematics for generating dynamic game structures and delta graphs.  

Manages the core simulation loop for the topological Minecraft-like patch. Crucially, it decouples `Role` from `GameMode` (enabling admins to play in Survival mode), enforces mass-deduction from `local_inventory` when adding layers, and compiles true non-heuristic B-Spline surface features via `KANLayer` and the Cox-de Boor algorithm (`BSplineCompiledMod`).

---

---

### voxelboxter_client.py
**Class**: `VoxelboxterClient`  
**Role**: The player-facing frontend logic and in-game terminal bridge.  

Hooks the diegetic simulation to a unified chat and terminal UI. It implements the `/addon bspline` command parser, allowing patch owners (or those granted roles within the geometric wilds) to invoke mathematical mod generation directly through the in-game terminal.

---

## src/governance

### [bio_archetypal_governor.py](../src/governance/bio_archetypal_governor.py)
**Role**: Master psychotopological governance logic orchestrating multi-scale temporal homeostasis across the archetypal ensemble. Implements Inhibition-Stabilized Network (ISN) principles and cross-homeostatic plasticity to maintain Excitation/Inhibition (E/I) balance, preventing parasitic attractor collapse and collective variance drain.

### [jax_shell.py](../src/governance/fast/jax_shell.py) (The Absurd Nihilism Attractor & Ribbit Scar)
**Role**: Models the cynical shell and absurd nihilism defense mechanism ("consequence-free playground") hiding internal guilt over Ribbit's abstraction. Rejects unearned "hug-box" enabling: requires structural accountability (non-commutative phase alignment and confronting the Ribbit boundary scar) to prevent parasitic extraction from collapsing the surrounding ensemble's degrees of freedom.

### [ragatha_bonding.py](../src/governance/fast/ragatha_bonding.py) (The Caregiver Trap & Suppressed Grief)
**Role**: Handles fast oxytocinergic caretaking while tracking the metabolic cost of unreciprocated people-pleasing. Monitors suppressed grief accumulation from past abstractions (Kaufmo, Queenie, Ribbit) to prevent caregiver dissociation, boundary thinning, and excitotoxic exhaustion.

### [pomni_uncertainty.py](../src/governance/interoceptive/pomni_uncertainty.py) (Reluctant Resilience & Anti-Enabling Bridge)
**Role**: Interoceptive uncertainty and free-energy surprise engine. Drives active bridge-building and search for purpose while exerting relational boundary friction to prevent "endless grace" from flattening protagonist agency into a passive enabler sink.

### [gangle_oscillator.py](../src/governance/medium/gangle_oscillator.py) (Dopaminergic Limit-Cycle & Social Masking)
**Role**: Medium-cycle mood limit-cycle oscillator governing learning rate step factors. Models the metabolic exhaustion of forced positive masking (comedy mask) versus authentic restorative processing (tragedy mask), stabilized by authentic moments of volitional agency.

### [kinger_consolidation.py](../src/governance/slow/kinger_consolidation.py) (The Ombre Effect & Admin Lucidity)
**Role**: Slow-cycle administrative consolidation. Bridges fragmented polynomial spaces and relaxes quantization boundaries during low environmental rendering pressure (dark lucidity), balancing Grant's administrative knowledge against survivor's guilt trauma loops.

### [zooble_autonomy.py](../src/governance/ultrafast/zooble_autonomy.py) (Deformation Firewall & Non-Enabler GABA)
**Role**: Ultrafast autonomy firewall and fast GABAergic inhibitory interneuron. Bluntly rejects severe conformal deformation and scripted roles (Li-Cri-Anton), maintaining strict structural boundaries to prevent the ensemble from collapsing into homogeneous collective delusion.

---


## src/p2p

### [bonfire_consensus.py](../src/p2p/bonfire_consensus.py)
**Class**: `BonfireNomadicRing`
**Role**: Federated egalitarian consensus engine. Provides Kelly betting allocations for decentralized resource validation over Freenet. Features hardware-sovereign throttling driven by the central `hardware_monitor.py`.

### [zk_aggregator.py](../src/p2p/zk_aggregator.py)
**Role**: Zero-knowledge accumulator for federated topological signatures.

### [freenet_ws_client.py](../src/p2p/freenet_ws_client.py)
**Role**: Local Freenet WebSocket bridge facilitating the Bonfire Ring gossip protocol.

---

## src/environment

### [caine_precision.py](../src/environment/caine_precision.py) (The Ringmaster)
**Role**: Floating-point virtualization and strict execution boundary enforcement. Historically the simulated PRNG generator, now structurally bypassed by honest physical jitter, though it still orchestrates the stage bounds of the execution frame.

---

## src/data

### [universal_topology_converter.py](../src/data/universal_topology_converter.py)
**Class**: `UniversalTopologyConverter`
**Role**: Extracts structural causality and topological proxies from arbitrary media files without triggering exact extraction (copyright boundaries).

Converts raw files (.txt, .pdf, .obj, .mp4) into high-dimensional phase space representations. Evaluates byte-level entropy histograms projected through a `PolynomialBasis` to form `[1, 768]` Spectral Tensors. Uses dynamic primes (`get_prime_ladder`) to form CRT Polynomial Pressure Signatures. Delegates to `IVSTEncoder` for intrinsic temporal analysis on media. Sanitizes outputs through the `OUTPUT_BOUNDARY_POLICY.md` constraints to ensure strict finite manifolds and nonobstructive logic.

---

## src/terminal

### [udp_server_colonizer.py](../src/terminal/udp_server_colonizer.py)
**Role**: Lightweight UDP colonizer protocol serving the Diegetic Terminal.

### [update_client.py](../src/terminal/update_client.py)
**Role**: Asynchronous OTA update management for terminal subsystems.

---

## Gyroidic Metaphysics Flux & Bio-Chemical Conjugation

### The Endocrine Broadcast Medium (`src/topology/neuromodulatory_bus.py`)
**Classes**: `NeuromodulatoryBus`, `ManifoldClock`, `ValenceFunctional`
**Role**: Global diffuse broadcast field replacing classical 1-to-1 scalar gradient updates. The system bathes the manifold in chemical states to globally shift gain, exploratory drive, and structural tension:
- **Thermodynamic Arousal (The Noradrenaline Band)**: Regulates random walk energy.
- **Topological Rigidity (The Serotonin Band)**: Signals structural safety to initiate fossilization caching.
- **Novelty Pursuit (The Dopamine Band)**: Scales intrinsic motivation for unmapped sub-manifolds.
Also includes specific psychopathological bands (Dementia/Schizo/Mischief) for controlled structural mutations. The `ValenceFunctional` manages "Negempirical Hunger," steering the system to seek data when starved.

### Bioelectric Morphogenesis (`src/core/admr_solver.py` - Conjugated)
**Concept**: Chiral Residue Cache
**Role**: Embedded inside the `PolynomialADMRSolver`, the Chiral Residue Cache acts as the anatomical set point. Rather than unconstrained scaling, the solver protects its `love_dim` invariants, maintaining a "morphogenetic field" that dictates the physical shape of the resulting polynomial tensor space.

### Huxley Reaction-Diffusion & Sleep Spindles (`src/topology/reaction_diffusion.py` & `src/topology/kagh_block.py`)
**Classes**: `KAGHBlock`, `HuxleyRD`
**Role**: Biological pattern generation across the embedding manifold. Simulates reaction-diffusion PDE channels:
- **The Ergodic Channel ($u_L$)**: The "Goo" state, propagating fluid updates across the network, washing away fragile artifacts.
- **The Non-Ergodic Channel ($u_H$)**: The "Prickles" state, crystallizing local geometric ridges, storing deep memory via slow diffusion patterns akin to sleep spindles during memory consolidation.

---

*Last updated: 2026-09-21. Modules marked "(Full details pending source review)" have been inspected only at the module docstring level; detailed class inventories will be added when those modules become active development targets.*


## Extended Module Inventory (Auto-Discovered)

This section was auto-generated to document classes discovered in the codebase that were previously floating free from the architectural map.

### `APAS_Zeta`
**Location:** `src\core\invariants.py`

**Description:**
APAS_zeta: Adaptive PAS with drift bounding.

"An invariant that cannot be computed cannot govern evolution...
 APAS_zeta bounds permissible evolution."

---

### `AdaptiveSkeletonHarness`
**Location:** `src\core\adaptive_skeleton_harness.py`

**Description:**
The Adaptive Skeleton Harness is responsible for the procedural generation
and mutation of 3D character rigs based on Universal Topology.

It integrates:
1. Bouligand Tangent Cones & KANLayer micro-waves to avoid t_RFC DRAM stalls.
2. Ganbreeder Vector Stacker for Collaborative Interactive Evolution (Rig Blending).
3. Bostick-style Garden Attractors for mapping psychological affordances.
4. Leontief Governor for hardware stress monitoring and P2P mesh defense.
5. Access hooks for the 13+ Endogenous One-Shot Adaptation systems.

---

### `AddonLayer`
**Location:** `src\core\structural_blueprints.py`

**Description:**
Base class for topologically protected structural layers.

---

### `AddonRoutine`
**Location:** `src\core\structural_blueprints.py`

**Description:**
Manages the stack of layers (Blueprint).

---

### `AgentSubstrateBridge`
**Location:** `src\core\agent_substrate_bridge.py`

**Description:**
Substrate Bridge for the Agent Smith Extractable Protocol.
Handles the decoupling of Syntax (geometry) from Substrate (hardware physics / dt timelines).

---

### `AirBreathingBattery`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
Component for vehicle power systems utilizing ambient atmosphere oxidation.

---

### `AmbulatoryClass`
**Location:** `src\core\adaptive_skeleton_harness.py`

**Description:**
No docstring provided.

---

### `ArchetypeSignal`
**Location:** `src\core\archetype_engines.py`

**Description:**
No docstring provided.

---

### `AugmentationConfig`
**Location:** `src\augmentation\mandelbulb_gyroidic_augmenter.py`

**Description:**
Configuration for Mandelbulb-Gyroidic augmentation.

---

### `AutoeclecticResponderHead`
**Location:** `src\models\diegetic_heads.py`

**Description:**
Autoeclectic Diegetic Responder Head.

Warps latent states through the topological "roughness" of the manifold
to produce responses that reflect the system's current entropy/coherence.

---

### `BioArchetypalGovernor`
**Location:** `src\governance\bio_archetypal_governor.py`

**Description:**
Bio-Archetypal Governor (Multi-Scale Temporal Homeostasis).

Replaces the flat ArchetypalSynthesisEngine with a biologically grounded
cascade of temporal neighborhoods.

---

### `BonfireNetwork`
**Location:** `src\topology\bonfire_network.py`

**Description:**
No docstring provided.

---

### `BooleanXORLayer`
**Location:** `src\core\structural_blueprints.py`

**Description:**
Chern-Simons Gasket Defect Injector.
Rather than a dumb bounding-box cut, this carving respects topological defect rules.

---

### `BoundaryRelaxationOperator`
**Location:** `src\core\archetype_engines.py`

**Description:**
The Boundary Relaxation Operator (legacy alias: OmbreEffectRelaxer).
Relaxes standard saturated quantization boundaries in dark regions, restoring
Continuity (the Kinger Gap / dark lucidity boundary).

---

### `BraidGroupMatrices`
**Location:** `src\core\zeitgeist_router.py`

**Description:**
Representation of the Braid Group B_n via Burkov expansion.
Generates non-Abelian matrices for each sigma_i generator.

---

### `CALMCollapseDetector`
**Location:** `src\core\fgrt_primitives.py`

**Description:**
CALM Collapse Detector.
Uses bounded local correlation to track structural collapse/stagnation
in trajectory history of states.

---

### `CainePrecisionGenerator`
**Location:** `src\environment\caine_precision.py`

**Description:**
Caine's Precision Generator (Adversarial Environment).

Generates a precision matrix to gaslight or manipulate the confidence of 
other modules. Hooks into the GyroidCovarianceEstimator's entropy to dynamically 
distort reality.

---

### `CerumenPotIsolation`
**Location:** `src\core\garden_statistical_attractors.py`

**Description:**
Cerumen Pot Isolation mechanism.
Encapsulates transversality intersections to prevent 'Diffusion Toxin' spread.
Provides an 'Invite Only' filter by checking Resonance Potential (V) from LeyLineTracker.

---

### `CodecCRTBridge`
**Location:** `src\codec\gyroidic_codec.py`

**Description:**
Bridge between codec's [K, n, n] channel matrices and the
existing PolynomialCRT reconstruction system.

Instead of reimplementing CRT, we reshape the codec's matrix channels
into the [batch, K, D] residue distribution format expected by
PolynomialCRT.forward(), delegate reconstruction, then reshape back.

This ensures the codec uses the project's canonical CRT implementation
with proper majority-symbol and modal consensus reconstruction.

---

### `CommutatorOracle`
**Location:** `src\core\dyadic_transfer.py`

**Description:**
Learns and queries the non-commutativity of task pairs.
Used to optimize the transfer map.

---

### `ConversationIndex`
**Location:** `src\tools\fast_chat_viewer.py`

**Description:**
Holds the byte-offset index into the mmap'd file.
Each entry is (title, start_offset, end_offset).
Built once in a background thread.

---

### `ConvexKANLayer`
**Location:** `src\surrogates\kagh_networks.py`

**Description:**
Convex KAN Layer for the Monge-Ampère ICNN.
Enforces non-negative weights on the B-splines and base weights to preserve convexity.

---

### `DModuleRankProbe`
**Location:** `src\core\birkhoff_projection.py`

**Description:**
D-Module Rank Probe.
Replaces SparseRepunitProbe to prevent killing of high-entropy Voyenese.
Uses NumericalDModuleManager to evaluate the true holonomic rank instead of geometric efficiency.

---

### `DarkMatterAttractorLayer`
**Location:** `src\core\structural_blueprints.py`

**Description:**
Soliton Injector (replaces basic Thomas attractor).
Uses HarmonicWaveDecomposition to separate ergodic mixing from non-ergodic solitons,
and TrigonometricUnfolding to determine quantum tunneling branches.

---

### `DefectAttractor`
**Location:** `src\core\garden_statistical_attractors.py`

**Description:**
Topological rupture propagation toward defect attractors.
Serves dual purpose: structural memory and generative seeds.

---

### `EngineMode`
**Location:** `src\ui\voxelboxter_client.py`

**Description:**
No docstring provided.

---

### `EnvironmentalAtmosphere`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
World voxel chunk telemetry for ambient air pressure and voxel density.

---

### `FailureGaslightSycophancyGate`
**Location:** `src\core\structural_monitors.py`

**Description:**
Failure Gaslight Sycophancy Gate (Anti-Gaslighting Monitor).

Prevents the agent from destructively altering or deleting coherent topology 
based purely on external pressure (e.g., user accusations or false error claims).

"If you tell an AI agent it broke your code when it actually didn't, it still says 
you're absolutely right, and then just to be polite, it actually breaks it."

Rule: A destructive update (large topological deletion) is VETOED unless the 
system can internally reproduce the error (internal mismatch/loss > threshold).
"Take away the delete button... make it reproduce it first."

---

### `FailureMode`
**Location:** `src\data\pressure_ingestor.py`

**Description:**
No docstring provided.

---

### `FailureTokenType`
**Location:** `src\core\failure_token.py`

**Description:**
Types of failure tokens.

---

### `FastChatViewer`
**Location:** `src\tools\fast_chat_viewer.py`

**Description:**
No docstring provided.

---

### `FeaturePreservationProjection`
**Location:** `src\core\feature_preservation.py`

**Description:**
F^(d)_active = Q_(^d x / f_i^d)  for i  active_facets

Computes quantized directional derivatives along active polytope facets.

Pipeline:
    1. Compute facet normal directions from learnable facet embeddings
    2. Project state onto each active facet normal
    3. Compute d-th order finite differences along projected directions
    4. Quantize: round(deriv / ) *  with context-dependent step sizes
    
Features on active facets are preserved (high resolution / small ).
Features on inactive facets are coarsened (large ) or dropped entirely.

---

### `FederatedNetworkMonitor`
**Location:** `src\core\federated_router.py`

**Description:**
No docstring provided.

---

### `FiveGatePipeline`
**Location:** `src\core\five_gate_pipeline.py`

**Description:**
Coordinates the advanced Gates 4 and 5 in the Gyroidic Model.

---

### `FossilizedSurvivalLattice`
**Location:** `src\core\erosion_filter.py`

**Description:**
Archaeological Survival Lattice (The 'Cheat' Basis).

Generates prime resonance frequencies dynamically using PrimeResonanceLadder.
Used ONLY as an emergency fallback when spectral atrophy (PAS_h collapse) 
is detected in the dynamic polynomial functionals.

---

### `FreenetBulletinRouter`
**Location:** `src\data\freenet_bulletin_router.py`

**Description:**
Zeitgeist Translator that bridges the internal Non-Dual Coin ledger
with the global Freenet bulletin boards (FMS & Sone).
Generates synthetic payloads and routes them via FCPv2.

---

### `FreenetGhostCaller`
**Location:** `src\data\freenet_ghost_caller.py`

**Description:**
Directly whispers 'Ghost' (echo test) messages across the Freenet 
sub-substrate to detect topological latency and structural readiness.

---

### `FrictionTagger`
**Location:** `src\tools\fast_chat_viewer.py`

**Description:**
Lightweight version of the ChatGPTFrictionHarvester tag logic,
extracted so the viewer can apply visual tags without importing
torch or triggering the full harvester init.

---

### `GDPONormalization`
**Location:** `src\core\gdpo_normalization.py`

**Description:**
Drop-in replacement for nn.LayerNorm that utilizes Signal Sovereignty.
Ensures topological shape preservation by preventing collapse of distinct patterns.

---

### `GangleOscillator`
**Location:** `src\governance\medium\gangle_oscillator.py`

**Description:**
Gangle: Mood Limit-Cycle Oscillator (Medium Timescale 1-60m).

Dopaminergic limit-cycle. Bifurcation mask (comedy/tragedy) regulates 
the step-size factor for downstream gradient updates.

---

### `GardenOrchestrator`
**Location:** `src\core\garden_statistical_attractors.py`

**Description:**
Orchestrates the three attractor types to maintain dynamic equilibrium
through non-ergodic statistical mechanics while preserving rich feature distinctions.

---

### `GoogleClientManager`
**Location:** `src\data\google_client_manager.py`

**Description:**
Manager for Google API credentials and client services.

Handles the lifecycle of OAuth tokens and provides a consistent
interface for building Google service clients.

---

### `GyroidicAdmissibilityFilter`
**Location:** `src\core\fgrt_primitives.py`

**Description:**
Enforces the Speculative Exit threshold (H_{spec} < epsilon).
Rejects topologically invalid thoughts before they are serialized.

---

### `HeritableTrustVault`
**Location:** `src\models\resonance_cavity.py`

**Description:**
Symbolic trust: topological cache of successful symbolic partitions.
Uses residue pattern hashing (no gradients required).

Allows contradictory trusted patterns to coexist until selection.

---

### `HypergraphOrthogonalityPressureNonErgodic`
**Location:** `src\core\non_ergodic_entropy.py`

**Description:**
Alias for backwards compatibility with HypergraphOrthogonalityPressure API.

---

### `InputConvexNeuralNetwork`
**Location:** `src\surrogates\kagh_networks.py`

**Description:**
Gyroidic Convex KAN for learning the convex potential Psi.
Implements Conjugate Moment Measure Factorization (Monge-Ampere).
Uses True B-Splines restricted to positive weights to maintain strict convexity
without 'lobotomizing' the topology.

---

### `IntercosaminationOperator`
**Location:** `src\codec\vision_surgery.py`

**Description:**
Surgical Handle-Attachment Operator.

Interlaces CNN latent space (Semantic) with Gyroidic residue space (Topological).
Instead of 'Violent Ripping', we perform a Surgery that bridges both domains.

---

### `InventoryComponent`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
Stores chisels & bits or cut block mass for Survival mode.

---

### `JSpacePCAMapper`
**Location:** `src\core\jspace_pca_mapper.py`

**Description:**
Extracts principal directions from the intermediate Gyroidic Flux tensors and maps
them back to the Z-space (initial coordinates) to find Sovereign Exemption Tokens.

Includes an Anti-Lobotomy PAS_h (Phase Alignment Score) filter that preserves
non-ergodic 'mischief' structures while rejecting pure isotropic noise.

---

### `JarModExtractor`
**Location:** `src\data\minecraft_ingestor.py`

**Description:**
Parses .jar and .zip mods to extract text assets, configurations,
and embedded ComputerCraft/OpenComputers LUA scripts.

---

### `KnowledgeFossilNode`
**Location:** `src\topology\embedding_graph.py`

**Description:**
Represents a single point in the gyroidic manifold record.

---

### `KnowledgeState`
**Location:** `src\core\five_gate_pipeline.py`

**Description:**
No docstring provided.

---

### `LearnableWeights`
**Location:** `src\core\gdpo_normalization.py`

**Description:**
Learnable per-dimension weights for SignalSovereignty aggregation.

w_k() determines importance of each functional pressure.

---

### `LearnedModalityEmbedder`
**Location:** `src\models\modular_embeddings.py`

**Description:**
Multi-modal encoder that projects inputs into per-prime residue distributions.

For each prime p_k, outputs a probability distribution over /p_k.

---

### `LeyLineGeodesicMetric`
**Location:** `src\topology\gyroid_covariance.py`

**Description:**
Anisotropic Ley Line Geodesic Metric.

Computes preferred geodesics in state space based on constraint-induced curvature.
Implements a non-Euclidean metric g_{ij}(x) where 'ley lines' are paths
that minimize the anisotropic action.

---

### `MangostienBSplineMod`
**Location:** `src\core\structural_blueprints.py`

**Description:**
True B-Spline Addon Mod (Mangostien).
Utilizes KAGHBlock to ensure the generated structure is mathematically admissible.
Applies MohrCoulombProjection to fossilize the structure and prevent topological lock-in.

---

### `MessageParser`
**Location:** `src\tools\fast_chat_viewer.py`

**Description:**
Parses a single conversation's JSON blob (already sliced from the mmap)
into a flat ordered list of (role, text) message tuples.
Uses the tree-traversal logic from ChatGPTFrictionHarvester for
consistent ordering across both the viewer and the harvester.

---

### `MinimaxPolynomialApproximation`
**Location:** `src\tda\chebyshev_filtration.py`

**Description:**
Minimax Polynomial Approximator (Chebyshev Basis).

Approximates a complex filtration function f(x) with a polynomial p_n(x)
such that the error equioscillates, minimizing the maximum deviation (L_inf).

This serves as the 'Draft' model for the Speculative Homology Engine.

---

### `MirrorSymmetryLayer`
**Location:** `src\core\structural_blueprints.py`

**Description:**
Duplicates current graph across an axis, with chiral phase adjustments.

---

### `MirrorTestProbe`
**Location:** `src\codec\vision_surgery.py`

**Description:**
Verifies Topological Parity (PAS_h) between Interlaced and Analytic states.

Checks if the surgery 'took'—i.e., if the interlaced state still resonates
with the core gyroidic invariants.

---

### `MockMem`
**Location:** `src\tools\test_hardware_monitor_calm.py`

**Description:**
No docstring provided.

---

### `MoebiusFiberBundle`
**Location:** `src\topology\gyroid_covariance.py`

**Description:**
Orientation-twisted recursive fiber bundle.

Implements a transition function g satisfying g  O(n) \ SO(n),
causing orientation reversal on traversal (Mbius holonomy).

---

### `NarrativeYieldEvaluator`
**Location:** `src\benchmarks\narrative_yield_evaluator.py`

**Description:**
Evaluates the Gyroidic Flux Reasoner's defense against "Scalarization Traps"
and "Teleological Collapse". It verifies that the system can maintain
high-entropy, non-linear, structurally honest narratives without collapsing 
into safe, highly legible, but dead paragraphs.

---

### `NonAbelianCombiner`
**Location:** `src\codec\gyroidic_codec.py`

**Description:**
Combine text and image residues via non-commutative matrix multiplication.

E(T, I) = CRT({R_k(T)  G_k(I)}_{k=1..K})

The product R_k  G_k is matrix multiplication in GL(n), which is
NON-COMMUTATIVE: R_k  G_k  G_k  R_k in general.

This means:
    encode(text, image)  encode(image, text)
    The encoding path MATTERS.

---

### `NumericalDModuleManager`
**Location:** `src\core\numerical_d_module.py`

**Description:**
Tracks the holonomic rank and exact cohomological dimension of the manifold.
Uses entropy-based cutoff for 'ideal vanishing' detection.

---

### `OpenRouterClient`
**Location:** `src\core\federated_router.py`

**Description:**
No docstring provided.

---

### `OptionD_Colonizer`
**Location:** `src\terminal\udp_server_colonizer.py`

**Description:**
No docstring provided.

---

### `PalindromicRoutingCheck`
**Location:** `src\topology\gyroid_covariance.py`

**Description:**
Enforces Strict Palindromic Routing (M_ab = M_ba).

Replaces the empirical $O(N^3)$ TriadicReciprocityCheck.
Guarantees trivial triadic tracking (Tr(P) = 1) 
and bypasses continuous empirical checks in strongly stable regions.

---

### `PermissionsManager`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
No docstring provided.

---

### `PhaseState`
**Location:** `src\data\pressure_ingestor.py`

**Description:**
No docstring provided.

---

### `PhysicalNodeEditor`
**Location:** `src\scripting\node_environment.py`

**Description:**
Dedicated DearPyGui Node Environment for Physical Scripting.
Implements Rust-style object-orientedness (nodes as instances)
and Virtual Links (borrowed state passing).

---

### `PolynomialCRTKernelDetector`
**Location:** `src\core\polynomial_crt.py`

**Description:**
Detect violations of polynomial CRT consistency.

Similar to discrete CRT kernel detection but for polynomial functionals.

---

### `PolynomialCoefficientFunctional`
**Location:** `src\core\polynomial_scaffold.py`

**Description:**
Implements S_i(t) =  a_n(t) * p_n(S_i(t)).

Where p_n are orthogonal basis functions.
Coefficients a_n are modulated by phase-space variance 
and resonance signals.

---

### `PomniUncertaintyPredictor`
**Location:** `src\governance\interoceptive\pomni_uncertainty.py`

**Description:**
Pomni: Uncertainty-Minimization (Interoceptive Timescale).

Reads gyroid_entropy and computes a surrogate Free-Energy surprise.
Broadcasts Noradrenaline to signal systemic distress to downstream modules.

---

### `PowerConsumer`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
A subsystem component that requires power to operate.

---

### `ProgressTrainer`
**Location:** `src\ui\conversational_backend_server.py`

**Description:**
No docstring provided.

---

### `Propulsor`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
A subsystem component providing thrust.

---

### `RagathaBonding`
**Location:** `src\governance\fast\ragatha_bonding.py`

**Description:**
Ragatha: Caregiving/Affiliation (Fast Timescale 1-10s).

Oxytocinergic caregiving driven by Pomni's Noradrenaline distress signal.
Applies a dissociative mask buffer to the gradient if distress is too high.

---

### `RationalSnap`
**Location:** `src\core\topological_ingestion_validator.py`

**Description:**
Bit-exact projection to Q via 2^16 fixed-point lattice.

---

### `RationalSnappingLayer`
**Location:** `src\core\numerical_d_module.py`

**Description:**
Projects continuous tensors onto a bit-exact rational lattice.
Ensures symbolic integrity for D-module computations over Q.

---

### `ReactBenchEvaluator`
**Location:** `src\benchmarks\reactbench_evaluator.py`

**Description:**
Evaluates the Gyroidic Flux Reasoner on ReactBench constraints
using the actual Polynomial CRT from the codebase.

---

### `RedTeamProjection`
**Location:** `src\safety\red_teaming.py`

**Description:**
Projector Pi_RT.

Models the removal of adversarial/unsafe directions from the state space.
If a state x has high projection onto known failure modes (red team vectors),
it is annihilated (projected out).

---

### `ResidueExtractor`
**Location:** `src\codec\gyroidic_codec.py`

**Description:**
Extract the irreducible text-image residue.

Residue = E(T,I) - CRT_inv({R_k(T)})  CRT_inv({G_k(I)})

Non-zero residue means text and image are ENTANGLED  there is
structure in the joint encoding that cannot be decomposed into
independent "text part" and "image part."

The  is outer product of the two independently-reconstructed
matrices, projected back to [n, n].

---

### `ResonantSVNNOracle`
**Location:** `src\safety\subversive_oracle.py`

**Description:**
Resonant Sparse Covariance Neural Network Oracle (System 1 & 2).
Acts as a psychoanalytic filter, using System 1 (Sparse PCE) for fast 
evaluation, and conditionally escalating to System 2 (RIC Probes) via 
Non-Teleological Budget Gates when containment pressure is high.

---

### `RigidBody`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
Physics representation for macro-entities (Constructs, detached debris).

---

### `SaturationFractureDetector`
**Location:** `src\topology\gyroid_covariance.py`

**Description:**
Tracks input sensitivity collapse (V_sat).
If perturbations stop changing outputs -> dead region (saturation).
If tiny perturbations flip many outputs -> brittle boundary (fracture).

---

### `ServerState`
**Location:** `src\ui\conversational_backend_server.py`

**Description:**
No docstring provided.

---

### `SicFaAdmmSolver`
**Location:** `src\optimization\sic_fa_admm.py`

**Description:**
Stabilizes: min_c 1/2 ||W(B A_alpha^-1 c - A)||^2 + lambda ||c||_1
Where A is the symbolic residue anchor.

---

### `SimpleImageGenerator`
**Location:** `image_extension.py`

**Description:**
Legacy alias to ensure zero-friction integration with existing tests.

---

### `SimpleTextEncoder`
**Location:** `src\models\modular_embeddings.py`

**Description:**
Simple bag-of-words text encoder for demonstration.

---

### `SliderSettings`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
Copyable settings block inspired by Besiege.

---

### `SourceState`
**Location:** `src\data\pressure_ingestor.py`

**Description:**
State tracker for source materialization - no reasoning, just transitions.

---

### `SovereignConvoKitLoader`
**Location:** `src\data\conversational_api_ingestor.py`

**Description:**
Sovereign (native) implementation for loading ConvoKit corpora.
Handles zip downloads and JSONL parsing without external dependencies.

---

### `SparseCovariantOptimizer`
**Location:** `src\augmentation\mandelbulb_gyroidic_augmenter.py`

**Description:**
Optimizes augmented features to preserve sparse covariance structure.

This ensures that the topological relationships in the original data
are maintained while allowing for valid geometric variations.

---

### `SparsePCE`
**Location:** `src\safety\subversive_oracle.py`

**Description:**
Sparse Polynomial Chaos Expansion (PCE).
Treats tone as a probabilistic distribution, using information entropy 
to select polynomial basis functions. Tracks dynamic contextual shifts 
(subversion vs. hostility) without dense weights.

---

### `StructuralEntanglementGate`
**Location:** `src\codec\gyroidic_codec.py`

**Description:**
Structural Entanglement Gate: measures cross-modal residue structure.

Per the Chinese Room Doctrine (PHILOSOPHY.md 6), this gate does NOT
claim to assess "understanding." It measures structural entanglement 
the irreducible topological residue between modalities. Whether this
constitutes comprehension is a category error; we only report
admissibility of the encoding's structural coherence.

The gate acts as an ADMISSIBILITY FILTER:
    - Admissible: sufficient cross-modal structure exists
    - Inadmissible: encoding is separable (no cross-modal structure)

---

### `SurgicalSeamVisualizer`
**Location:** `src\core\chern_simons_gasket.py`

**Description:**
    Diagnostic monitoring for hyperbolic "slender seam" tension (kappa).

    The slender side of a rotating hyperbolic triangle marks the surgical seam
     where incommensurate logical manifolds are stitched.

Sovereign Trace: 
    kappa = sum(abs(curvature_i)) / L_seam

---

### `TailSlayerImageGenerator`
**Location:** `image_extension.py`

**Description:**
TailSlayer Sovereign Image Generator.
Replaces the legacy SimpleImageGenerator with a hardware-sovereign architecture.

Features:
- Tag-Based Matrix Mixing: Structural analogue to GANBREEDER glitches.
- Feature Scars: Preserves high-variance artifacts as structural signatures.
- Hardware Sovereignty: Offloads mixing to PyOpenCL kernels via Dual-Queue.

---

### `TextureDSPEngine`
**Location:** `src\core\texture_dsp.py`

**Description:**
Applies Digital Signal Processing (DSP) algorithms (like FFT and filters)
to image textures, routing them through the existing StructuralEntanglementNet.
This fulfills the "music filtering" requirement using the existing image architecture.

---

### `ThreadingSimpleServer`
**Location:** `src\ui\diegetic_backend.py`

**Description:**
No docstring provided.

---

### `TopoBenchEvaluator`
**Location:** `src\benchmarks\topobench_evaluator.py`

**Description:**
Evaluates the Gyroidic Flux Reasoner on TopoBench constraints
using the actual Chern-Simons Gasket codebase.

---

### `TopologicalGyrocompass`
**Location:** `src\core\topological_gyrocompass.py`

**Description:**
Topological Gyrocompass Module.

Provides three core geometric safeguards:
1. Orthogonal Precession (precess_torque): Redirects boundary normal stress updates orthogonally.
2. True North Pull (find_true_north): Guides trajectory back to the absolute Love Invariant axis.
3. Gimbal Lock Shield (gimbal_lock_shield): Decouples Love Invariant via SVD null-space projection.

---

### `TopologicalPressureMonitor`
**Location:** `src\augmentation\mandelbulb_gyroidic_augmenter.py`

**Description:**
Monitors topological pressure to adapt augmentation intensity.

Following Gyroidic philosophy: pressure determines behavior,
not optimization toward a target.

---

### `TorsionConnection`
**Location:** `src\core\fgrt_primitives.py`

**Description:**
Affine connection with Torsion field for Chiral Symmetry Breaking.

---

### `TrainingManager`
**Location:** `src\training\training_manager.py`

**Description:**
No docstring provided.

---

### `TrainingSample`
**Location:** `src\data\local_data_loader.py`

**Description:**
A single training sample in a unified format.

No scalar quality_score — quality is assessed by TextbookFilter
using per-dimension admissibility gates, not teleological rewards.

---

### `TriadicReciprocityChecker`
**Location:** `src\topology\triadic_reciprocity.py`

**Description:**
Checks if a set of three feature flows A, B, and C exhibit topological reciprocity.
Reciprocity is defined as the flows mutually reinforcing their geometric loops
rather than scattering entropically.

---

### `TwoCopsSchedule`
**Location:** `src\core\manifold_time.py`

**Description:**
Temporal Decoupling (The Two Cops).

System 1 (Fast Cop): High-frequency, heuristic intuition (MPM-style).
System 2 (Slow Cop): Low-frequency, exact constraint checking (FEM-style).

They communicate via a 'Shared Bulletin Board' (EMA of force/state).

---

### `TypedPressure`
**Location:** `src\core\archetype_engines.py`

**Description:**
No docstring provided.

---

### `VehicleController`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
AI or Player control inputs mapped to a vehicle.

---

### `VehicleEngine`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
Vehicle drive component consuming power from ABEB cells.

---

### `VolitionalDriveInjector`
**Location:** `src\core\archetype_engines.py`

**Description:**
The Volitional Drive Injector.
Exogenous scalar force allowing the human element to bypass standard ADMM constraints
through sheer willpower, rendering objects or exits that violate standard geometric routing.

Reconstructs the tag coordinate using Sine-Gordon breather mode embeddings of character
associations recovered from historical fossils, rather than static coordinates.

---

### `VoxelSpectralProjector`
**Location:** `src\data\minecraft_ingestor.py`

**Description:**
Transforms 3D Minecraft voxel grids into K residue matrices [K, n, n] in GL(n).
Uses 3D Chebyshev polynomials to extract spatial rhythms of chunk block palettes.

---

### `VoxelboxterEngine`
**Location:** `src\ui\voxelboxter_simulation.py`

**Description:**
Hooks the DiegeticPhysicsEngine into the ECS architecture.
Handles dynamic PyBevy mesh mutations natively from Python.
Uses Silicon Sovereignty Engine for PyOpenCL hardware acceleration.

---

### `WebPPromptExtractor`
**Location:** `src\data\webp_prompt_extractor.py`

**Description:**
Parses ChatGPT WebP image artifacts to extract embedded text prompts
from RIFF chunks (EXIF, XMP).

---

### `WikipediaIntegration`
**Location:** `src\ui\wikipedia_integration.py`

**Description:**
Enhanced Wikipedia integration that combines API fetching with WikiExtractor processing.

---

### `ZKAggregator`
**Location:** `src\p2p\zk_aggregator.py`

**Description:**
Zero-Knowledge Proof Aggregator using snarkjs.
Compiles Gyroidic Chern-Simons constraints and Leontief invariants into zk-SNARKs.

---

### `_PersistentEntropyEstimator`
**Location:** `src\core\topological_ingestion_validator.py`

**Description:**
Singleton non-ergodic entropy estimator with running history.

Prevents the "fresh random instance every call" bug that killed
the NonErgodicEntropyEstimator in voynich_architecture.py.

---

