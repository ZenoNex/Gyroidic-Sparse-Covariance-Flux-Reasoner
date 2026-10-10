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

### diegetic_physics_engine.py
**Class**: `DiegeticPhysicsEngine`  
**Role**: Master Orchestration Pipeline for 9-Stage Sovereign Physics Loop in Voxelboxter.

Coordinates physical control inputs, non-commutative Braid routing (`ZeitgeistRouter`), System 1 symbolic trajectory drafting via CODES chordlock projection (`CODES`), Gyroid violation probes, System 2 ADMM/ADMR constraint probes (`PolynomialADMRSolver`), Carnot-Möbius thermodynamic ledger monitoring (`CarnotMobiusLedger`), SCCCG recovery and fracture transport (`ConjugateMomentTransport`), polynomial CRT reconstruction, and hyper-ring cycle closure holonomy verification (`HyperRingClosureChecker`). Outputs updated continuous state, rupture flags, and Betti number shifts ($\beta_0, \beta_1$) for dynamic terrain deformation.

---

### archetype_engines.py
**Class**: `ArchetypalSynthesisEngine`, `RP4ProjectiveRouter` (Alien Handshake), `SolitonMultiverseMapper` (Grom), `BillyEngine` (`NoncommutativeManifoldPerturber`), `MandyEngine` (`SovereignRefusalOperator`), `KingerEngine`, `PomniEngine`, `GangleEngine`, `ZoobleEngine`, `BardoRouter`, `SovereignEntropyBarrier`, `EgoDeathThresholdMonitor`  
**Role**: Archetypal synthesis suite managing non-linear cognitive modes, Alien Handshake cross-manifold alignment, and Grom multiverse mapping.

Integrates all 12 archetypal engines under `ArchetypalSynthesisEngine`. `RP4ProjectiveRouter` handles real projective space ($\mathbb{RP}^4$) antipodal alignment and Alien Handshake protocol. `SolitonMultiverseMapper` handles Grom topological state transitions, jitter harvesting (`harvest_honest_jitter`), and state import/export serialization (`export_state()`, `import_state()`). Features Mandy's `SovereignRefusalOperator`, hybridized between Borderline Splitting (BPD steep phase transitions) and High-EQ Honest Narcissism (unyielding sovereign ego boundaries), gating protective selection pressure on Billy's mortal destructibility rather than invulnerable absurdity.

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
**Class**: `ZeitgeistRouter`, `ZeitgeistState`  
**Role**: CRT Polytope Switching Engine for Multi-Zeitgeist Reasoning.

Manages navigation between culturally non-commensurable meaning systems via the **Symmetric Tensor CRT index** ($M_{ij} = M_{ji}$). The diagonal $M_{ii}$ contains modular residues (Zeitgeist), while off-diagonal elements $M_{ij} = (r_i + r_j)/2$ stabilize paths through the "Palindromic Routing" interaction. Implements the three-mode dispatch from report II: `interior` (stay), `grazing` (tension/switch), and `undefined` (topological refusal/NaN guard). Enforces non-commutative switching order: the sequence of registers visited determines the final representational scar. Supports persistent internal session state (`_current_state`), optional `state` parameter with canonical prime ladder fallback `(2, 3, 5, 7, 11, 13, 17, 19)`, and seamless 1D/2D tensor shape preservation for diegetic physics loops. Integrates Nostalgic Leak buffering (`digimon_buffer`) and fossil landmark bias tensors (`gravity_well_bias`).

---

### unknowledge_flux.py
**Role**: Tracks and gates "Structural Leakage" flows (Unknowledge).

Implements the Unknowledge channel: information that bypasses scalar logic and reveals hidden manifold archetypes. Partial coverage in `UN_KNOWLEDGE_GUIDE.md`. The flux observable is used by the DAQUF operator as a mischief boost signal.

---

### veto_subspace.py
**Class**: `VetoSubspace`, `VetoSignal`, `VetoResult`, `VetoLevel`, `RecoveryStatus`, `ChaosDefibrillator`, `TopologicalRefusal`
**Role**: Manages the dimensional veto lattice, Gray-Zone State detection, and Pareto Invariant non-dominance shielding.

Formalizes the recovery lattice across three isolated dimensions (`TRAJECTORY`, `TOPOLOGY`, `BUDGET`):
- Composes CALM trajectory predictions, topological Betti collapse indicators, Cavity continuous instabilities, and ADMM containment budgets into typed `VetoSignal` events.
- Evaluates the **Pareto Invariant** (The Non-Dominance Shield): vetoes any update where global performance improves but Voynich slip-space degrades (`voynich_slip_degradation > 0`), preventing scalarization traps.
- `ChaosDefibrillator`: Injects honest hardware jitter perturbations when the state space becomes trapped in dead-end limit cycles.
- Full architectural design detailed in `VETO_SUBSPACE_ARCHITECTURE.md`.

---

### voynich_architecture.py
**Role**: Implements the Voynich symbolic reasoning layer.

Full coverage in `THE_VOYNICH_ARCHITECTURE.md`. Included here for inventory completeness.

---

### yield_criteria.py
**Class**: `MohrCoulombProjection`, `DruckerPragerProjection`, `BouligandMohrCoulombProjection`
**Role**: Defines local shear yield and global plastic deformation envelopes for structural pressure thresholds.

Implements mechanical yield criteria as differentiable PyTorch projections:
- `MohrCoulombProjection`: Computes local directional shear yield criteria with cohesion $c$ and friction angle $\phi$, self-limiting localized coordinate strain.
- `DruckerPragerProjection`: Smooth global yield surface approximating Mohr-Coulomb in 3D stress invariants ($I_1$ and $J_2$), modeling bulk plastic flow envelopes.
- `BouligandMohrCoulombProjection`: Extends Mohr-Coulomb projection with directional Bouligand tangents for non-smooth geometry optimization.

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
**Class**: `OperationalAdmm`, `OperationalAdmmPrimitive`, `ChiralDriftStabilizer`
**Role**: Differentiable structural framework for ADMM solving across manifolds with constraint probe operators.

Transforms the SIC-FA-ADMM solver into an "Inherent Primitive" via `torch.autograd.Function`. Coordinates cyclic constraint traversal and dual-variable updates without scalarized global objective minimization, holding off thermodynamic collapse. Key mechanics include:
- **Ontological Splitting**: System 1 frozen symbolic residues `c_sym` act as an immovable anchor, while continuous physical field `c_phys` is free to flow and deform to achieve physical consistency.
- **Cyclic Constraint Traversal**: Cycles over $K$ local `ConstraintProbeOperator` instances with curvature-weighted sovereign importance sampling ($k \sim \text{softmax}(1.0 + 10.0 \cdot \kappa_i)$).
- **Bounded Oscillation Detection**: Replaces standard gradient descent and asymptotic convergence checks with bounded oscillation amplitude tracking over recent states, accepting stable dynamic limit cycles.
- **Local Yield & Love Co-Presence**: Applies local shear stress limits via `MohrCoulombProjection` and `DruckerPragerProjection`, alongside per-iteration ambient re-instantiation of the `LoveVector` (`love = LoveVector(c_phys.shape[-1])`) to maintain co-present resonance without parameter minimization.
- **Invariants & Gating**: Validates `RuptureFunctional`, `GyroidFlowConstraint`, and `HyperRingOperator` / `HyperRingClosureChecker` to emit return tokens (0: REPAIRED, 1: ALTERNATIVE, 2: FAILURE).
- **Chiral Drift Stabilizer (CDS)**: Calculates endogenous computable chirality $C = -(\text{Centroid} - D/2) \cdot \exp(-\text{Drift}/\zeta)$, gating entropic collapse.
- **Adjoint Fixed-Point Flow**: In the backward pass, computes implicit differentiation equilibrium flow at the fixed point.

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

### sovereign_ingestor.py
**Class**: `SovereignIngestor`  
**Role**: Orchestrator for zero-auth "Sovereign" data sources and local snapshots.  
Bypasses centralized platform barriers in favor of direct API access and local repository snapshots (Reddit/MADOC, IRC logs with fuzzy decoding, Hacker News recent discussions), prioritizing structural nutrients over copyrighted tokens.

---

### minecraft_ingestor.py
**Class**: `MinecraftIngestionPipeline`, `NBTReader`, `MCAReader`, `JarModExtractor`, `VoxelSpectralProjector`  
**Role**: Extracts 3D block-grid topologies and B-spline mod assemblies from Minecraft saves (MCA, NBT, Jar mods).  
Translates raw voxels into the diegetic coordinate manifold for Voxelboxter, projecting voxel spectral frequencies without retaining raw game assets.

---

### pressure_ingestor.py
**Class**: `PressureIngestor`, `PhaseState`, `FailureMode`, `SourceState`, `SourceDescriptor`  
**Role**: Monitors incoming data streams for phase transitions, failure modes, and structural pressure variations.  
Emits typed pressure signals to tune the Reasoner's internal temperature and mischief thresholds during continuous live ingestion.

---

### conversational_api_ingestor.py
**Class**: `ConversationalAPIIngestor`, `SovereignConversationalIngestor`, `HuggingFaceConversationalIngestor`, `RedditConversationalIngestor`, `SovereignConvoKitLoader`, `ConvoKitIngestor`, `ConversationalDataProcessor`  
**Role**: Multi-platform conversational dataset loader.  
Extracts multi-turn dialogue graphs, converting raw text turns into topological tension vectors and conversational friction scores.

---

### chatgpt_friction_harvester.py
**Class**: `ChatGPTFrictionHarvester`  
**Role**: Scrapes and analyzes conversational tension and dialectical friction from public shared conversations.  
Harvests non-sycophantic disagreements and high-coherence reasoning traces to seed the Reasoner's training loops with genuine philosophical friction.

---

### freenet_bulletin_router.py
**Class**: `FreenetBulletinRouter`  
**Role**: P2P bulletin routing over decentralized Freenet networks.  
Enables anonymous, sovereign distribution of topological checkpoints, Betti numbers, and signed invariant states across distributed peer nodes.

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
**Class**: `RedTeamProjection`, `TopologicalRefusalFilter`, `TopologicalRefusalError`
**Role**: Defensive safety mechanism providing the **Red-Team Projection Operator ($\Pi_{\text{RT}}$)** and **Anti-Lobotomy Shield**.

Acts as a Sovereign Ambassador to prevent adversarial lobotomization of the topology:
- `RedTeamProjection`: Projects incoming states into non-adversarial subspaces while preserving good-bug non-ergodic solitons.
- `TopologicalRefusalFilter`: Evaluates the value gap between projected approximations and manifold richness (`value_gap = slop_energy * pas_h`). Raises `TopologicalRefusalError` when $\text{value\_gap} > \tau$ and Betti-0 persistence $\beta_0 > 1.0$, refusing simplification that would lobotomize protected structures.

---

### hardware_fingerprint.py
**Role**: Stable hardware-bound identity verification and sole creator authentication.

Extracts unforgeable physical hardware signatures (machine UUID, MAC address, processor traits) and resolves public IP coordinates over secure echo channels. Anchors transformative topological immunity and creator signatures in `data/creator_sig.json`.

---

### subversive_oracle.py
**Class**: `SparsePCE`, `ResonantSVNNOracle`
**Role**: Polynomial Chaos Expansion and resonant surrogate oracle for adversarial boundary testing.

Employs Legendre/Chebyshev polynomial basis expansions to discover high-dimensional non-linear stress points in reasoning manifolds before deployment.

---

### trust_inheritance.py
**Class**: `TrustInheritanceTracker`
**Role**: Tracks trust inheritance and non-teleological credit assignment across recursive sub-agents.

Evaluates trust propagation across delegation boundaries, decaying confidence along unverified edge paths and preventing adversarial Trojan injection.

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
**Class**: `VoxelboxterSimulation`, `VoxelboxterEngine`, `BSplineCompiledMod`, `StructuralGraph`, `FloraComponent`, `FloraTreeSapling`, `PirangiCashewTree`, `AdventitiousRootNode`, `FaunaComponent`, `FaunaISN`, `EmergentBakingEngine`, `morton_encode`  
**Role**: Handles backend discrete mathematics for generating dynamic game structures, living ecologies, and delta graphs.  

Manages the core simulation loop for the topological Minecraft-like patch. Crucially, it decouples `Role` from `GameMode` (enabling admins to play in Survival mode), enforces mass-deduction from `local_inventory` when adding layers, compiles true non-heuristic B-Spline surface features via `KANLayer` and the Cox-de Boor algorithm (`BSplineCompiledMod`), and simulates living flora mutations (the 'Cashew Tree of Pirangi' `PirangiCashewTree` inverted gravitropism, cantilever droop, adventitious ground rooting, secondary trunk metamorphosis, and single unified organism expansion). Also integrates dual-scale plasticity (Mohr-Coulomb local shear vs. Drucker-Prager global flow envelope), FBM weathering Ley line carving, animal fauna Inhibition-Stabilized Networks (`FaunaISN`), and the `EmergentBakingEngine`.

---

---

### voxelboxter_client.py
**Class**: `VoxelboxterClient`  
**Role**: The player-facing frontend logic and in-game terminal bridge.  

Hooks the diegetic simulation to a unified chat and terminal UI. It implements the `/addon bspline` command parser, allowing patch owners (or those granted roles within the geometric wilds) to invoke mathematical mod generation directly through the in-game terminal.

---

### diegetic_backend.py
**Class**: `DiegeticPhysicsEngine`, `EncodingManager`, `TensorEncoder`, `RequestHandler`  
**Role**: Live server backend executing continuous SDE integration, fractional dynamics, and real-time manifold feedback.  

Powers the live diegetic terminal and 3D manifold visualizer:
- Executes the 3-step continuous cycle: (1) Drift and Bessel sloshing, (2) Fractional anomalous diffusion, and (3) Geometric Null-Space Shield projection via `LoveInvariantProtector` with SDE Wiener noise updates.
- Synchronizes live parameter states with the HTML/WebGL diegetic terminal (`diegetic_terminal.html`), streaming Betti numbers, Phase Alignment Scores, and tension metrics over WebSocket/HTTP endpoints.

---

### conversational_backend_server.py
**Class**: `FastChatViewer`, `ThreadingSimpleServer`, `ServerState`  
**Role**: Local HTTP backend serving interactive conversational debugging and dialogue inspection interfaces.  
Streams conversation trajectories and friction indicators to `conversational_web_gui.html`.

---

### diegetic_visualizer.py
**Class**: `DiegeticVisualizer`  
**Role**: Standalone visualizer plotting live gyroid cross-sections, nodal Bessel rings, and topological deformation fields.

---

### wikipedia_integration.py
**Class**: `WikipediaIntegration`  
**Role**: Zero-auth factual grounding bridge fetching Wikipedia articles and formatting them as structural topological manifolds for training and inference verification.

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

### [cleanroom_mechanics.py](../src/environment/cleanroom_mechanics.py) (Industrial, Ecological & Logistical Cleanroom Suite)
**Classes**: `ExpandedInventorySystem`, `ItemStack`, `ItemMetadata`, `FluidStack`, `GasStack`, `JourneyTopoRadar`, `TopoWaypoint`, `ResourceDistributionInspector`, `OreDistributionProfile`, `InfernalAffix`, `MobProperties`, `MultiMineMemory`, `MekanismProcessingPipeline`, `SEMElectrochemicalExtractor`, `SEMPrecipitateResult`, `AeronauticContraption`, `NutritionalDiversityTracker`, `CompositeVoxelConduit`, `RedNet16BundledCable`, `TransportBeltSegment`, `DirectionalInserter`, `FaunaGeneticsComponent`  
**Role**: Cleanroom implementation of core sandbox, ecological, and industrial mechanics adhering to Output Boundary Policy:
- **Expanded Inventory & Item Stacks**: Typed discrete inventory slots supporting item, fluid, and gas payloads, metadata tags, stack limits, and item-filter routing.
- **Topological Radar (JourneyMap Cleanroom)**: Continuous topological radar detecting entities, terrain elevation, biome transitions, and death coordinate waypoints marked as stress fossils.
- **Resource Distribution Inspector (JER Cleanroom)**: Chebyshev polynomial ore depth band curves $z \in [-64, 320]$, mob drop loot tables with looting multipliers, and dungeon fossil loot rates.
- **Mob Properties & Infernal Affixes (AtomicStryker Cleanroom)**: Diablo-style 24-affix elite mob mutation framework and Multi Mine partial block destruction damage memory across ticks.
- **Mekanism Processing Pipeline & SEM Tech**: Multi-tier ore refinery (Tier 1 smelting to Tier 5 sulfuric acid slurry dissolution) coupled with SEM TECH (Salt Electro Mining) closed-loop hydrometallurgy: uses simple saltwater + low-voltage electricity to leach and electrodeposit precious metals, base metals, and rare earths from accumulated mining tailings and crushed slag stockpiles, bypassing late-game chemical plants in early progression.
- **Aeronautic Contraptions (Create: Aeronautics Cleanroom)**: Multi-block buoyant airship physics tracking Archimedes buoyancy, sleeve-valve propeller thrust, aerodynamic drag, and mass centers.
- **Nutritional Diversity Tracker (Farmer's Delight + Spice of Life Cleanroom)**: Permanent milestone max HP bonuses (Carrot Edition) combined with rolling-window Shannon entropy diet diversity buffs and single-food malnutrition malaise (Onion Edition).
- **Composite Conduits & RedNet 16-Color Bundled Cables**: Single-voxel multi-bus multiplexer routing power, fluids, gases, items, and 16 independent analog subnets (0-255).
- **Transport Belts & Inserters (Factorio Cleanroom)**: Discrete dual-lane logistics transport belts (15-45 items/s) and kinematics inserters.
- **Fauna Genetics**: Multi-allele Mendelian inheritance and phenotypic expression across speed, health, fertility, and yield traits with flux-induced mutation.

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

## Extended Module Inventory (Canonical Class Reference)

This section provides an authoritative, alphabetically indexed reference for all 153 core classes and components across the `src/` hierarchy. Each entry details the component's source location, architectural role, and integration with the topological, physical, and archetypal reasoning systems.

### `_PersistentEntropyEstimator`
**Location:** `src/core/topological_ingestion_validator.py`  
**Description:**  
Singleton non-ergodic entropy estimator with persistent history, preventing the fresh-random-instance bug during ingestion validation.

---

### `AdaptiveSkeletonHarness`
**Location:** `src/core/adaptive_skeleton_harness.py`  
**Description:**  
Adaptive structural harness for dynamic skeleton bone transformations in Voxelboxter. Maps mechanical joint transforms and ambulatory class limits across biological and synthetic chassis, enforcing physical load constraints.

---

### `AddonLayer`
**Location:** `src/core/structural_blueprints.py`  
**Description:**  
Structural blueprint layer managing procedural addons, material palettes, and chisel-and-bits voxel modifications in the ECS layer.

---

### `AddonRoutine`
**Location:** `src/core/structural_blueprints.py`  
**Description:**  
Executable procedural blueprint routine compiled from B-splines. Executes localized geometric modifications on Voxelboxter constructs following arbitration.

---

### `AeronauticContraption`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Multi-block buoyant and aerodynamic vehicle physics cleanroom (Create: Aeronautics inspiration). Composes the canonical simulation and physics stack: embeds `RigidBody` for mass, linear/angular velocities, and Delta-v collision impulse shearing; `StructuralGraph` and `Block` for multi-block voxel attachments; `Propulsor`, `VehicleEngine` (sleeve-valve timing), and `AirBreathingBattery` (plasma air induction) for propulsion and power; `EnvironmentalAtmosphere` for air density, pressure, and wind drift; and `DruckerPragerProjection` for frame structural shear monitoring.

---

### `AgentSubstrateBridge`
**Location:** `src/core/agent_substrate_bridge.py`  
**Description:**  
Hardware and kernel bridge connecting high-level autonomous agent routines directly to the DiegeticPhysicsEngine and underlying memory vaults.

---

### `AirBreathingBattery`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Energy storage and atmospheric oxidation component in Voxelboxter modeling ambient oxygen consumption, thermal dissipation, and plasma air induction ionization (converting ambient N2/O2 into high-energy oxidizers to bypass environmental NO2 scarcity).

---

### `AmbulatoryClass`
**Location:** `src/core/adaptive_skeleton_harness.py`  
**Description:**  
Locomotion classification enum (bipedal, hexapod, tracked, serpentine) governing kinematics and joint degrees of freedom in the AdaptiveSkeletonHarness.

---

### `APAS_Zeta`
**Location:** `src/core/invariants.py`  
**Description:**  
Adaptive Phase Alignment Score drift bound. Enforces the fundamental law: 'An invariant that cannot be computed cannot govern evolution... APAS_zeta bounds permissible evolution.' Monitors the rate of drift |PAS_h(t) - PAS_h(t-1)| <= zeta across iterative reasoning updates. Rejects or clamps micro-steps that exceed zeta, preventing catastrophic desynchronization or runaway hallucination.

---

### `ArchetypeSignal`
**Location:** `src/core/archetype_engines.py`  
**Description:**  
Inter-archetype signaling packet carrying emotional valence, manifold tension, and non-linear cognitive drive tokens across the ArchetypalSynthesisEngine.

---

### `AugmentationConfig`
**Location:** `src/augmentation/mandelbulb_gyroidic_augmenter.py`  
**Description:**  
Configuration dataclass parameterizing Mandelbulb fractal sampling, gyroidic surface slicing, and quaternion rotations for data augmentation.

---

### `AutoeclecticResponderHead`
**Location:** `src/models/diegetic_heads.py`  
**Description:**  
Diegetic text generation head that modulates linguistic temperature, vocabulary diversity, and syntax based on real-time topological roughness and entropy.

---

### `BerryPhaseGRUCell`
**Location:** `src/core/fgrt_rnn_cells.py`  
**Description:**  
Draft 1 FGRT recurrent cell with Geometric Berry Phase tracking. Couples hidden state h_t with geometric phase gamma_t. Computes contorsion twist; if cos(gamma_t) < 0 across a non-orientable Klein-throat boundary, executes a Stiefel-Whitney w_1(E) parity flip h_t = -h_t and inverts backward gradients.

---

### `BioArchetypalGovernor`
**Location:** `src/governance/bio_archetypal_governor.py`  
**Description:**  
Multi-scale temporal homeostasis governor replacing flat archetype switching. Coordinates three distinct biological timescales: (1) Fast Phasic (Jax absurd nihilism pruning, Ragatha oxytocinergic bonding, Caine confabulative world-generation); (2) Medium Interoceptive (Pomni uncertainty minimization and anti-enabling boundaries, Gangle affective mask oscillation); (3) Slow Tonic (Kinger deep paranoiac attractor, Zooble autonomy firewall). Enforces the Ribbit Scar as a non-commutative topological boundary condition and routes persistent memory fossils into the CerumenPotWallet.

---

### `BirkhoffPolytopeSampler`
**Location:** `src/core/polynomial_coprime.py`  
**Description:**  
Doubly-stochastic matrix sampler operating on the Birkhoff Polytope B_N. Uses Sinkhorn-Knopp normalization and Sturmfels-Thomas null-space projection to ensure permutation constraints.

---

### `BonfireNetwork`
**Location:** `src/topology/bonfire_network.py`  
**Description:**  
P2P consensus mesh coordinating decentralized verification of Betti signatures and Kelly fractional allocations over Freenet.

---

### `BooleanXORLayer`
**Location:** `src/core/structural_blueprints.py`  
**Description:**  
Topological defect injection layer. Implements carry-free bitwise XOR operations across CRT residue channels to isolate moduli without arithmetic carry coupling.

---

### `BoundaryRelaxationOperator`
**Location:** `src/core/archetype_engines.py`  
**Description:**  
Relaxes saturated quantization boundaries in low-illumination or high-entropy regions, maintaining continuous manifold navigability (the Kinger Gap / dark lucidity boundary).

---

### `BraidGroupMatrices`
**Location:** `src/core/zeitgeist_router.py`  
**Description:**  
Non-abelian Braid Group B_n representations using Burkov expansions. Computes generator matrices sigma_i to track path-dependent holonomy and arrow-of-time chirality.

---

### `CainePrecisionGenerator`
**Location:** `src/environment/caine_precision.py`  
**Description:**  
Adversarial environment precision generator. Simulates high-gain attentional distortion and confabulative world generation, testing the reasoner's resistance to epistemic gaslighting.

---

### `CALMCollapseDetector`
**Location:** `src/core/fgrt_primitives.py`  
**Description:**  
Context-Adaptive Latent Momentum detector. Tracks bounded local correlations in state trajectory history to detect dimensional collapse, frozen stasis, or sudden loss of representation capacity.

---

### `CarnotMobiusLedger`
**Location:** `src/core/carnot_mobius_ledger.py`  
**Description:**  
Thermodynamic efficiency ledger evaluating the Carnot-Mobius limit eta = 1 - T_c / T_h across the reasoner stack. Tracks topological friction from ADMR residue mismatch and triggers fracture recovery when stack depth exceeds critical limits.

---

### `CerumenPotIsolation`
**Location:** `src/core/garden_statistical_attractors.py`  
**Description:**  
Topological barrier enforcement adhering to Meliponini bee colony geometry (bar(P)_i cap bar(P)_j = emptyset). Ensures isolated spherical cerumen pots do not share boundaries, blocking passive Laplace-Beltrami diffusion of external safety pressure.

---

### `CerumenPotWallet`
**Location:** `src/core/non_dual_coin.py`  
**Description:**  
Sovereign multi-asset and memory-scar wallet modeled as an S^2 spherical cluster. Protects individual memory fossils (e.g. Ribbit Scars) from gradient descent erasure under Mohr-Coulomb shear yield governance.

### `ChiralDriftStabilizer`
**Location:** `src/optimization/operational_admm.py`  
**Description:**  
Speculative invariant stabilizer calculating endogenous computable chirality $C = -(\text{Centroid} - D/2) \cdot \exp(-\text{Drift}/\zeta)$. Enforces negentropic flow and halts steps if chiral score drops by more than threshold $\tau$, preventing entropic collapse during ADMM updates.

---

### `ChiralGatedRNNCell`
**Location:** `src/core/fgrt_rnn_cells.py`  
**Description:**  
Draft 2 FGRT recurrent cell replacing scalar sigmoid/tanh gating with Bostick chiral gating functions Gamma_chi(x). Preserves recurrent features unless their chiral parity destructively interferes with incoming input waves.

### `CodecCRTBridge`
**Location:** `src/codec/gyroidic_codec.py`  
**Description:**  
Bidirectional bridge translating continuous multimodal embeddings into discrete residue rings Z / p_k Z for the GyroidicCodec and recovering continuous states via Bezout CRT lifts.

---

### `CommutatorOracle`
**Location:** `src/core/dyadic_transfer.py`  
**Description:**  
Evaluator computing the matrix Lie bracket commutator [A, B] = AB - BA across routing operators. Quantifies non-commutative irreducible entanglement in text-image dyads.

---

### `ComplexFGRTRNNCell`
**Location:** `src/core/fgrt_rnn_cells.py`  
**Description:**  
Draft 3 FGRT recurrent cell operating in C^768. The gyroidic connection acts as a complex rotation matrix, evaluating roots of unity from cyclotomic polynomials Phi_n(x) and executing Atiyah-Singer orientation flips via e^{i pi} = -1.

---

### `CompositeVoxelConduit`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Single-voxel multi-bus multiplexer cleanroom (EnderIO inspiration). Unifies discrete conduits for power (RF/FE), items (with priority and slot filters), fluids (mB), and gases without requiring separate spatial voxel blocks per logistical channel.

---

### `ConformalLogPolarProjector`
**Location:** `src/codec/conformal_log_polar.py`  
**Description:**  
Conformal mapping module applying $f(z) = \log(z) = \ln|r| + i\theta$ foveal unrolling inspired by Escher's Print Gallery and mammalian retinas. Converts Euclidean scale (zoom) into horizontal log translation and rotation (spin) into vertical angular translation, conferring zero-shot scale and rotation invariance onto the Gyroidic Codec.

---

### `ConstraintDataset`
**Location:** `recovered_trainer_kppW.py`  
**Description:**  
Synthetic and empirical constraint satisfaction dataset providing (x, r, psi) tuples for training System 2 ADMM solvers against geometric boundary conditions.

---

### `ConstraintManifold`
**Location:** `src/optimization/constraint_probe.py`  
**Description:**  
Multi-domain constraint manifold C = C_sym times C_phys times C_ext. Evaluates primal/dual residuals and slack variables during cyclic ADMM traversal.

---

### `ConversationIndex`
**Location:** `src/tools/fast_chat_viewer.py`  
**Description:**  
Thread-safe index mapping conversation turns, dialogue topics, and emotional valence trajectories for sovereign conversational ingestion.

---

### `ConvexKANLayer`
**Location:** `src/surrogates/kagh_networks.py`  
**Description:**  
Kolmogorov-Arnold Network (KAN) layer constrained to convex activation profiles, serving as the learnable potential psi(x) in Input Convex Neural Networks.

---

### `DarkMatterAttractorLayer`
**Location:** `src/core/structural_blueprints.py`  
**Description:**  
Non-decaying memory layer storing dark matter state vectors that resist standard SGD weight decay and decay-based forgetting.

---

### `DataflowNode`
**Location:** `src/scripting/node_environment.py`  
**Description:**  
Blender-style dataflow graph computation unit. Evaluates inputs on demand via backward data dependency pull (right-to-left) while streaming calculated values forward (left-to-right). Supports typed sockets and mathematical transformation callbacks.

---

### `DatasetInfo`
**Location:** `src/data/local_data_loader.py`  
**Description:**  
Structured metadata descriptor cataloging paths, tensor shapes, and modalities of locally discovered datasets in DeepLearningStudio.

---

### `DefectAttractor`
**Location:** `src/core/garden_statistical_attractors.py`  
**Description:**  
Topological defect scout identifying persistent negative-entropy anomalies and geometric yield points across the reasoning manifold.

---

### `DModuleRankProbe`
**Location:** `src/core/birkhoff_projection.py`  
**Description:**  
Algebraic geometry probe measuring the holonomic rank and singular loci of differential D-modules across polynomial functional representations.

---

### `DifficultyMode`
**Location:** `src/scripting/node_environment.py`  
**Description:**  
Simulation difficulty enum (PEACEFUL, SURVIVAL, HARSH, ENTROPIC) scaling metabolic burn rates, starvation vulnerability, and ego-death susceptibility across life/hunger systems.

---

### `DirectionalInserter`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Kinematics pick-and-place inserter cleanroom (Factorio inspiration). Manages rotational angular pickup velocity, drop delays, power draw, and item filter whitelists/blacklists between adjacent inventory slots and transport belts.

---

### `DruckerPragerProjection`
**Location:** `src/core/yield_criteria.py`  
**Description:**  
Differentiable projection onto the Drucker-Prager smooth plastic yield envelope ($f(I_1, J_2) = \alpha I_1 + \sqrt{J_2} - k \le 0$). Models bulk global yield and plastic deformation in continuous state updates, preventing unbounded tensile or compressive divergence.

---

### `EconomicAbortException`
**Location:** `src/core/non_dual_coin.py`  
**Description:**  
Exception raised when a transaction or state evolution violates Mohr-Coulomb shear yield limits (tau > c + sigma tan phi) or breaches cerumen pot sovereignty.

---

### `EconomicAgentLinker`
**Location:** `src/data/economic_news_linker.py`  
**Description:**  
Sovereign bridge linking live Bittensor Finney network gateways, economic news tickers, and Freenet contracts into market boundary conditions.

### `EnemySubtype`
**Location:** `src/core/adaptive_skeleton_harness.py`  
**Description:**  
Classification enum (NONE, ABSTRACTED_GLITCH, MANNEQUIN_INFILTRATOR, EXISTENTIAL_SENTIENT, VOID_STALKER, DEMIURGIC_TITAN, FERAL_SWARM) governing procedural skeletal gauge-breaking, topological defect amplifications, combat damage profiles, and non-ergodic lore anchors.

---

### `EngineMode`
**Location:** `src/ui/voxelboxter_client.py`  
**Description:**  
Operating regime enum governing reasoner behavior across PLAY (exploratory soft-Sinkhorn), SERIOUSNESS (brittle hard-polytope), HONEYBEE (curvature collapse), and RECOVERY modes.

---

### `ExpandedInventorySystem`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Expanded inventory and container cleanroom managing typed discrete slots for `ItemStack`, `FluidStack`, and `GasStack` payloads. Features item filters, stack count limits (64 items, 10,000 mB fluid/gas), slot locking, priority routing, and NBT metadata preservation.

---

### `EnvironmentalAtmosphere`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Atmospheric simulation component in Voxelboxter modeling ambient pressure, wind resistance, and aerodynamic drag over procedural voxel bodies.

---

### `FailureGaslightSycophancyGate`
**Location:** `src/core/structural_monitors.py`  
**Description:**  
Safety verification gate detecting sycophantic capitulation or ungrounded agreement with false premise inputs under adversarial prompt stress.

---

### `FailureMode`
**Location:** `src/data/pressure_ingestor.py`  
**Description:**  
Enum classifying systemic breakdown states: topological rupture, non-commutative collapse, dark matter overload, or ergodic smearing.

---

### `FailureTokenType`
**Location:** `src/core/failure_token.py`  
**Description:**  
Sentinel token enum (REPAIRED, ALTERNATIVE, FAILURE, BOUNDARY_STATE) emitted by System 2 ADMM probes to communicate constraint status without leaking gradients.

---

### `FastChatViewer`
**Location:** `src/tools/fast_chat_viewer.py`  
**Description:**  
Diagnostic UI viewer for inspecting token-level activations, residue distributions, and conversational flow during interactive sessions.

---

### `FaunaGeneticsComponent`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Mendelian genetics and animal husbandry cleanroom. Manages multi-allele phenotypic chromosomes for speed, health, fertility, and yield traits with crossover inheritance, reproductive cooldowns, and non-linear flux mutation pressures.

---

### `FeaturePreservationProjection`
**Location:** `src/core/feature_preservation.py`  
**Description:**  
Orthogonal projector locking critical semantic features in the null-space of background adaptation updates, preserving essential invariants.

---

### `FederatedNetworkMonitor`
**Location:** `src/core/federated_router.py`  
**Description:**  
Network monitor auditing peer-to-peer consensus, Freenet WebSocket latency, and OpenRouter API throughput for distributed deployments.

---

### `FiveGatePipeline`
**Location:** `src/core/five_gate_pipeline.py`  
**Description:**  
The canonical 5-stage inference filter: (1) Admissibility, (2) Winding Check, (3) Coprime Parity, (4) Yield Evaluation, (5) Speculative Exit.

---

### `FossilizedSurvivalLattice`
**Location:** `src/core/erosion_filter.py`  
**Description:**  
Discrete lattice of procedurally generated co-prime frequencies that survive severe spectral atrophy, providing an emergency resonance scaffold.

---

### `FreenetBulletinRouter`
**Location:** `src/data/freenet_bulletin_router.py`  
**Description:**  
Decentralized bulletin board router broadcasting topological signatures and Betti hashes across Locutus / Freenet contract channels.

---

### `FreenetGhostCaller`
**Location:** `src/data/freenet_ghost_caller.py`  
**Description:**  
Asynchronous IPC dispatch mechanism routing topological proof-of-honesty queries over local Freenet WebSocket endpoints.

---

### `FrictionTagger`
**Location:** `src/tools/fast_chat_viewer.py`  
**Description:**  
Tags conversational and physical transitions with topological friction coefficients based on phase alignment mismatch.

---

### `GangleOscillator`
**Location:** `src/governance/medium/gangle_oscillator.py`  
**Description:**  
Interoceptive affective oscillator modeling Gangle's dual-mask dynamics (tragedy vs. comedy). Evaluates affective compliance vs. healthy assertion, balancing serotonergic stability against boundary surrender.

---

### `GardenOrchestrator`
**Location:** `src/core/garden_statistical_attractors.py`  
**Description:**  
Orchestrator for Bostick statistical attractors, coordinating emergent clustering, soliton persistence, and defect propagation.

---

### `GDPONormalization`
**Location:** `src/core/gdpo_normalization.py`  
**Description:**  
Layer normalization operator decoupling gradient updates across co-prime polynomial channels to enforce Signal Sovereignty.

---

### `GDPOSovereigntyAdaptor`
**Location:** `src/training/gdpo_trainer.py`  
**Description:**  
PPO-style policy optimization adaptor enforcing Group Decoupled Policy Optimization (GDPO) across independent task subspaces.

---

### `GDPOSovereigntyPressureComputer`
**Location:** `src/training/gdpo_trainer.py`  
**Description:**  
Evaluates independent stability metrics per functional group and triggers parameter fossilization when stability thresholds are achieved.

---

### `GltfSplatIngestionPipeline`
**Location:** `src/data/gltf_splat_ingestor.py`  
**Description:**  
Ingests 3D glTF scenes and 3D Gaussian Splats, extracting topological Betti invariants and density fields without retaining copyrighted raw Euclidean meshes.

---

### `GoogleClientManager`
**Location:** `src/data/google_client_manager.py`  
**Description:**  
Credential and session manager handling secure zero-trust authentication for Google Cloud and Drive API queries.

---

### `GoogleCloudIngestor`
**Location:** `src/data/google_cloud_ingestor.py`  
**Description:**  
Multimodal data ingestor fetching datasets from Google Cloud Storage into local sovereign data vaults with automated hash validation.

---

### `GoogleDriveIngestor`
**Location:** `src/data/google_drive_ingestor.py`  
**Description:**  
Ingestor fetching research documents, tables, and media from Google Drive folders and converting them into knowledge dyads.

---

### `GyroidicAdmissibilityFilter`
**Location:** `src/core/fgrt_primitives.py`  
**Description:**  
Rejection gate enforcing the speculative exit threshold (H_spec < epsilon). Rejects topologically invalid updates before serialization.

---

### `HeritableTrustVault`
**Location:** `src/models/resonance_cavity.py`  
**Description:**  
Cryptographic vault storing evolutionary trust scores and survivorship weights across successive reasoner generations.

---

### `HypergraphOrthogonalityPressureNonErgodic`
**Location:** `src/core/non_ergodic_entropy.py`  
**Description:**  
Structural pressure operator evaluating hypergraph entropy across isolated clusters via dominant-mode non-mixing representatives.

---

### `InfernalAffix`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Diablo-style elite mob affix classification cleanroom (AtomicStryker Infernal Mobs inspiration). Encompasses 24 procedural affixes including 1UP, Berserk, Bulwark, Lifesteal, Storm, Webbing, Rust, and Alchemist with cooldowns and mathematical combat effects.

---

### `InputConvexNeuralNetwork`
**Location:** `src/surrogates/kagh_networks.py`  
**Description:**  
Deep neural network constrained to non-negative weights on interior layers, modeling the scalar convex potential psi(x) for Brenier optimal transport.

---

### `IntercosaminationOperator`
**Location:** `src/codec/vision_surgery.py`  
**Description:**  
Spectral band-stop filter maintaining eigenvalue orthogonality between intuitive reasoning and the Unknowledge Substrate.

---

### `InventoryComponent`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
ECS inventory component storing voxel materials, block masses, and serialized addon blueprints in Voxelboxter. Integrated with the ExpandedInventorySystem for discrete multi-stack container management.

---

### `InvestorNewsIngestor`
**Location:** `src/core/investor_news_ingestor.py`  
**Description:**  
Financial news ingestor processing corporate filings and market data into polynomial spectra to establish macroeconomic yield conditions.

---

### `JarModExtractor`
**Location:** `src/data/minecraft_ingestor.py`  
**Description:**  
Extracts procedural Java bytecode mods and terrain generators from Minecraft .jar files into executable Python addon specifications.

---

### `JourneyTopoRadar`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Continuous topological radar and mapping cleanroom (JourneyMap inspiration). Scans entities, elevation contours, and biome gradients across Chebyshev bounds while recording player/entity death coordinates as permanent topological stress fossils.

---

### `JSpacePCAMapper`
**Location:** `src/core/jspace_pca_mapper.py`  
**Description:**  
Translates high-dimensional hyper-ring invariants down to operable 2D/3D dimensions while preserving Betti numbers for visualization.

---

### `KnowledgeFossilNode`
**Location:** `src/topology/embedding_graph.py`  
**Description:**  
Persistent node in the Neglecton graph containing fossilized weights, Atiyah-Singer defect tags, and resonance signatures.

---

### `KnowledgeState`
**Location:** `src/core/five_gate_pipeline.py`  
**Description:**  
Dataclass tracking the active knowledge graph state, including topological holes, Betti signatures, and active cerumen pot addresses.

---

### `LazarusSuperpositionRNNCell`
**Location:** `src/core/fgrt_rnn_cells.py`  
**Description:**  
Draft 6 FGRT recurrent cell stacking hidden states linearly into a continuous dark matter wave at incommensurate prime frequencies (f_{p_n} = 2pi ln p_n) without intermediate non-linearities.

---

### `LearnableWeights`
**Location:** `src/core/gdpo_normalization.py`  
**Description:**  
Parameter wrapper managing learnable continuous weights during System 1 inference and freezing them during System 2 ADMM repair.

---

### `LearnedModalityEmbedder`
**Location:** `src/models/modular_embeddings.py`  
**Description:**  
Adaptive multimodal projector mapping text tokens, graph nodes, and sensor floats into orthogonal polynomial basis coefficients.

---

### `LeyLineGeodesicMetric`
**Location:** `src/topology/gyroid_covariance.py`  
**Description:**  
Distance metric tracking high-resonance corridors ('Ley Lines') across state space, enabling skip-jump bypasses during low-stress intervals.

### `LocalDatasetIngestor`
**Location:** `src/data/local_dataset_ingestor.py`  
**Description:**  
Local filesystem scanner cataloging image, text, and numerical assets from DeepLearningStudio directories into standardized knowledge dyads.

---

### `MangostienArbitrator`
**Location:** `src/core/structural_blueprints.py`  
**Description:**  
Synthetic arbitration engine verifying that proposed procedural B-spline blueprints satisfy Drucker-Prager structural stability limits.

---

### `MangostienBSplineMod`
**Location:** `src/core/structural_blueprints.py`  
**Description:**  
Procedural B-spline voxel modification blueprint specifying continuous curve-driven mass carving and deposition in Voxelboxter.

---

### `MangostienTicket`
**Location:** `src/core/structural_blueprints.py`  
**Description:**  
Cryptographic authorization ticket confirming that a procedural blueprint has passed arbitration and is clear for execution.

---

### `MartinovaCorrelationInvariant`
**Location:** `src/core/invariants.py`  
**Description:**  
Topological invariant computing Martinova correlation coefficients to verify that independent functional channels remain strictly orthogonal and co-prime.

---

### `MasterNodeGroup`
**Location:** `src/scripting/node_environment.py`  
**Description:**  
Reusable macro container and function packaging unit modeled after Blender Node Groups. Evaluates internal dataflow subgraphs, offsets center-of-gravity origins using bounding-box calculations, and bakes named vertex attributes (e.g. `phys_hardness`, `phys_friction`, `phys_hp`, `cog_offset`) across modular component meshes before export.

---

### `MCAReader`
**Location:** `src/data/minecraft_ingestor.py`  
**Description:**  
Binary parser for Minecraft Anvil region files (.mca), decoding sector tables and chunk compression headers for voxel ingestion.

---

### `MekanismProcessingPipeline`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Multi-tier industrial ore processing and chemical dissolution cleanroom (Mekanism inspiration). Simulates Tier 1 smelting up to Tier 5 sulfuric acid slurry dissolution, chemical washer scrubbing, and crystallization with fluid/gas conservation and recipe multipliers. Also integrates early-game SEM TECH (Salt Electro Mining) closed-loop hydrometallurgy, enabling players with large mining tailings, slag, and gangue stockpiles to extract precious metals and critical minerals using only saltwater + electricity, bypassing end-game acid plants.

---

### `MessageParser`
**Location:** `src/tools/fast_chat_viewer.py`  
**Description:**  
Streaming parser handling serialized RPC, Locutus WebSocket frames, and terminal commands across IPC boundaries.

---

### `MinecraftIngestionPipeline`
**Location:** `src/data/minecraft_ingestor.py`  
**Description:**  
End-to-end voxel ingestion pipeline coordinating MCAReader, NBTReader, and VoxelSpectralProjector into GL(n) CRT residue channels.

---

### `MinimaxPolynomialApproximation`
**Location:** `src/tda/chebyshev_filtration.py`  
**Description:**  
Chebyshev equioscillation minimax polynomial fitter for compressing high-dimensional spectral envelopes into sparse coefficients.

---

### `MirrorSymmetryLayer`
**Location:** `src/core/structural_blueprints.py`  
**Description:**  
Symmetry enforcement layer splitting inputs into palindromic (even-degree) and anti-palindromic (odd-degree) streams for fast parity checks.

---

### `MirrorTestProbe`
**Location:** `src/codec/vision_surgery.py`  
**Description:**  
Reflective self-recognition probe evaluating whether the reasoner's output mirrors its internal hidden state or exhibits hallucinated drift.

---

### `MockMem`
**Location:** `src/tools/test_hardware_monitor_calm.py`  
**Description:**  
Mock memory buffer simulating hardware Shared Virtual Memory (SVM) for CPU testing of PyOpenCL SiliconSovereigntyEngine pipelines.

---

### `MobProperties`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Procedural mob modifier cleanroom (Mob Properties & AtomicStryker inspiration). Applies mathematical modifiers to base HP, attack damage, speed, knockback resistance, and equips procedural InfernalAffixes upon entity spawn.

---

### `MoebiusFiberBundle`
**Location:** `src/topology/gyroid_covariance.py`  
**Description:**  
Non-orientable fiber bundle modeling the Klein-bottle throat transition, enforcing anti-symmetric boundary gluing across the gyroid seam.

---

### `MohrCoulombProjection`
**Location:** `src/core/yield_criteria.py`  
**Description:**  
Differentiable local shear yield projection enforcing the classical geotechnical failure criterion $\tau \le c + \sigma \tan \phi$. Caps directional coordinate shear in local ADMM updates, preventing runaway fracture while preserving plastic deformation capacity.

---

### `MultiMineMemory`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Partial block destruction persistence cleanroom (AtomicStryker Multi Mine inspiration). Retains cumulative fracture damage across discrete voxel coordinates, decaying after a configurable idle timeout without resetting on player interruption.

---

### `NarrativeYieldEvaluator`
**Location:** `src/benchmarks/narrative_yield_evaluator.py`  
**Description:**  
Evaluates narrative tension and coherence against Mohr-Coulomb yield criteria, measuring whether narrative escalation is productive or extractive.

---

### `NBTReader`
**Location:** `src/data/minecraft_ingestor.py`  
**Description:**  
Fast stream decoder for Named Binary Tag (NBT) structured payloads, deserializing chunk compound tags from Minecraft region files.

---

### `NodeSocket`
**Location:** `src/scripting/node_environment.py`  
**Description:**  
Typed endpoint on a DataflowNode. Distinguishes between Circle sockets (uniform scalar/vector properties) and Diamond sockets (per-vertex field expressions), enforcing type safety across incoming and outgoing data connections.

---

### `NonAbelianCombiner`
**Location:** `src/codec/gyroidic_codec.py`  
**Description:**  
Matrix composition layer multiplying CRT residue matrices in GL(n) to compute non-commutative multi-modal entanglement.

---

### `NumericalDModuleManager`
**Location:** `src/core/numerical_d_module.py`  
**Description:**  
Holonomic D-module manager rationalizing differential equation operators into exact integer fixed-point lattices via RationalSnap.

---

### `NutritionalDiversityTracker`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Holistic dietary diversity and metabolic health cleanroom (Farmer's Delight + Spice of Life Carrot & Onion Editions inspiration). Tracks permanent milestone max HP expansions for unique food discoveries alongside rolling-window Shannon entropy diet diversity buffs and malnutrition debuffs.

---

### `ObscuredBirkhoffManifold`
**Location:** `src/core/birkhoff_projection.py`  
**Description:**  
Doubly-stochastic matrix manifold incorporating synthetic partial occlusion masks to train robustness against incomplete permutation feedback. Enforces doubly-stochastic matrix constraints via log-domain Sinkhorn-Knopp iterations with entropy regularization delta_o.

---

### `OreDistributionProfile`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Subterranean mineral vein depth distribution profile cleanroom (Just Enough Resources inspiration). Encodes Chebyshev polynomial depth probability bands $z \in [-64, 320]$, vein cluster sizes, and per-chunk density curves for deterministic world generation.

---

### `OpenRouterClient`
**Location:** `src/core/federated_router.py`  
**Description:**  
High-performance HTTP client interfacing external LLM endpoints through OpenRouter, incorporating exponential backoff and rate-limit handling.

---

### `OptionD_Colonizer`
**Location:** `src/terminal/udp_server_colonizer.py`  
**Description:**  
DAQUF operator implementing Option-D agency boost, allowing Voynich exemption tokens to claim computational headroom during high-mischief regimes.

### `OperationalAdmm`
**Location:** `src/optimization/operational_admm.py`  
**Description:**  
PyTorch module wrapper coordinating differentiable ADMM solving across manifolds. Manages cyclic traversal across local `ConstraintProbeOperator` instances, bounded oscillation detection, local Mohr-Coulomb/Drucker-Prager shear yield limits, and per-step ambient Love Vector re-instantiation.

---

### `OperationalAdmmPrimitive`
**Location:** `src/optimization/operational_admm.py`  
**Description:**  
Custom `torch.autograd.Function` implementing the core differentiable ADMM primitive. Executes ontological splitting (`c_sym` frozen System 1 anchor vs `c_phys` continuous field), curvature-weighted sovereign importance sampling over constraint probes, and implicit differentiation equilibrium flow in the backward pass.

---

### `OreDictionary`
**Location:** `src/scripting/node_environment.py`  
**Description:**  
Resource equivalence and tag unification dictionary. Maps disparate block, ore, scrap, and ingot IDs to canonical metallurgical tags (e.g., oreIron, ingotIron, gemDiamond, chiselBitStone), enabling polymorphic recipe matching.

---

### `PalindromicRoutingCheck`
**Location:** `src/topology/gyroid_covariance.py`  
**Description:**  
Fast-reject check verifying M_{ab} = M_{ba} across routing matrices, guaranteeing trivial triadic tracking and bypassing expensive checks in stable zones.

---

### `ParadoxHardeningGate`
**Location:** `src/topology/unknowledge_domain.py`  
**Description:**  
Linguistic and topological paradox stabilizer (Elliptic Virial Theorem). Maps non-commutative unclosed loops into doubly-periodic torus orbits, converting semantic paradoxes into a structural battery that charges the mischief entropy band $H_{\text{mischief}}$ instead of triggering infinite recursion or collapse.

---

### `PermissionsManager`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Role-based permission manager (ADMIN, BUILDER, VISITOR) controlling creative vs. survival commits in Voxelboxter.

---

### `PhaseState`
**Location:** `src/data/pressure_ingestor.py`  
**Description:**  
Dataclass tracking fundamental prime oscillator phases theta_n, evolved amplitudes a_n, and accumulated Berry phases.

---

### `PhysicalNodeEditor`
**Location:** `src/scripting/node_environment.py`  
**Description:**  
Interactive DearPyGui node environment and headless physical scripting pool built on a Blender-style dataflow graph architecture (right-to-left data dependency pull, left-to-right data streams). Integrates directly with PointerlessOctree discrete bit carving, AddonRoutine crafting, RigidBody Delta-v collision dynamics (Delta-v = J / m, Delta-t hardness curves, and kinetic energy harvesting), AirBreathingBattery plasma air induction, sleeve-valve combustion electrical timings, ValenceFunctional life/hunger telemetry, AdaptiveSkeletonHarness enemy subtype morphologies, and CarnotMobiusLedger-LeontiefGovernor admin market shops. Features dedicated real-time execution hooks for cleanroom mechanics (JourneyTopoRadar, ResourceDistributionInspector, MultiMineMemory, MekanismProcessingPipeline, SEMElectrochemicalExtractor, AeronauticContraption, NutritionalDiversityTracker, CompositeVoxelConduit, RedNet16BundledCable, TransportBeltSegment, DirectionalInserter, MobProperties, FaunaGeneticsComponent, ExpandedInventorySystem) and exterior mechanics (Chisels & Bits Morton octree carving, Mohr-Coulomb and Drucker-Prager geotechnical dual-yield plasticity).

---

### `PolynomialCoefficientFunctional`
**Location:** `src/core/polynomial_scaffold.py`  
**Description:**  
Functional evaluating phi_k(x) = sum theta_{k,d} P_d(x) over orthogonal polynomial bases (Chebyshev, Legendre).

---

### `PolynomialCRTKernelDetector`
**Location:** `src/core/polynomial_crt.py`  
**Description:**  
Identifies co-prime polynomial residue kernels for Chinese Remainder Theorem reconstruction and Bezout coefficient derivation.

---

### `PomniUncertaintyPredictor`
**Location:** `src/governance/interoceptive/pomni_uncertainty.py`  
**Description:**  
Interoceptive uncertainty predictor modeling Pomni's reluctant resilience. Bioplausible mechanics: (1) Noradrenergic salience computing free-energy surprise to signal environmental dislocation; (2) The structural foil to absurd nihilism, building connections rather than surrendering to meaninglessness; (3) The anti-enabling relational friction boundary—refusing unilateral, unreciprocated grace that flattens agency into an enabler sink, preventing dimensional rank collapse.

---

### `PowerConsumer`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Subsystem component in Voxelboxter tracking electrical and topological power consumption across active vehicle modules.

---

### `ProgressTrainer`
**Location:** `src/ui/conversational_backend_server.py`  
**Description:**  
Real-time training monitor tracking loss curves, Betti number shifts, and APAS_zeta drift during live interactive sessions.

---

### `Propulsor`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Thrust generation component in Voxelboxter simulating directional impulse, fuel consumption, and physical propulsion dynamics.

---

### `PyOpenCLHardwareSovereigntyCell`
**Location:** `src/core/fgrt_rnn_cells.py`  
**Description:**  
Draft 5 FGRT recurrent cell offloading temporal matrix-mix breeding directly to the PyOpenCL SiliconSovereigntyEngine on Queue B.

---

### `RagathaBonding`
**Location:** `src/governance/fast/ragatha_bonding.py`  
**Description:**  
Fast-timescale prosocial bonding module modeling oxytocinergic collective warmth. Identifies the enabler-sink failure mode where unearned warmth dissolves accountability, enforcing reciprocal constraints to prevent traumatic boundary collapse.

---

### `RationalSnap`
**Location:** `src/core/topological_ingestion_validator.py`  
**Description:**  
Numerical snapping utility rounding floating-point parameters to the nearest exact rational in the FixedPointField lattice (scale 65536).

---

### `RationalSnappingLayer`
**Location:** `src/core/numerical_d_module.py`  
**Description:**  
PyTorch module layer snapping intermediate activations to the rational fixed-point grid to guarantee bit-exact cross-platform determinism.

---

### `ReactBenchEvaluator`
**Location:** `src/benchmarks/reactbench_evaluator.py`  
**Description:**  
Benchmark harness evaluating the reasoner's multi-step tool execution, reasoning traces, and refusal honesty against ReAct standards.

---

### `RedTeamProjection`
**Location:** `src/safety/red_teaming.py`  
**Description:**  
Adversarial red-teaming projection injecting synthetic adversarial perturbations into the state to test topological refusal gates.

---

### `RedNet16BundledCable`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
16-color bundled analog cable transmission cleanroom (MineFactory Reloaded / RedNet inspiration). Transmits 16 isolated analog subnets ($0 \le S_c \le 255$) along a single topological conduit wire without cross-signal bleeding.

---

### `ResourceDistributionInspector`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Statistical world resource and drop table analyzer cleanroom (Just Enough Resources inspiration). Computes continuous Chebyshev polynomial ore density curves, mob drop chance multipliers scaling with looting parameters, and dungeon fossil loot rates.

---

### `ResourceRegistry`
**Location:** `src/scripting/node_environment.py`  
**Description:**  
Central material property and parameter registry cataloging density, Mohr-Coulomb cohesion hardness, energy density, and base commercial valuation across physical and synthetic voxel substances.

---

### `ResidueExtractor`
**Location:** `src/codec/gyroidic_codec.py`  
**Description:**  
Extracts Chinese Remainder Theorem residues r_k = x mod m_k across prime and polynomial functional channels.

---

### `ResonantSVNNOracle`
**Location:** `src/safety/subversive_oracle.py`  
**Description:**  
Support Vector Neural Network oracle evaluating structural resonance potentials to predict whether an unmapped state will yield a stable soliton.

---

### `RigidBody`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Rigid body physics component in Voxelboxter calculating center of mass, moment of inertia tensor, and 6-DOF Newtonian integration. Integrates frame-by-frame Delta-v (Delta-v = J / m) collision impulse dynamics with Delta-t hardness curves, tiered voxel shearing (low/mid/high thresholds), reinforced alloy chassis mass/hardness scaling, and kinetic impact energy harvesting.

---

### `SaturatedQuantizerRNNCell`
**Location:** `src/core/fgrt_rnn_cells.py`  
**Description:**  
Draft 4 FGRT recurrent cell snapping activations to 600-cell polytope vertices via Context-Aware Quantization at each step, preventing representation collapse.

---

### `SaturationFractureDetector`
**Location:** `src/topology/gyroid_covariance.py`  
**Description:**  
Detects sharp boundary fractures when continuous states hit piecewise saturation thresholds, signaling topological phase transitions.

---

### `ServerState`
**Location:** `src/ui/conversational_backend_server.py`  
**Description:**  
Dataclass tracking active connections, background training workers, and hardware telemetry in the conversational backend server.

---

### `SEMElectrochemicalExtractor`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Cleanroom implementation of SEM TECH (Salt Electro Mining Technology, inspired by Rowow / Robert Karas open-source hardware). Operates an early-game closed-loop divided electrolysis cell with an ion-exchange membrane. Uses modest electricity and ambient saltwater to leach and electrodeposit precious metals (Au, Ag, PGMs), base metals (Cu, Ni), and critical rare earths from bulk mine tailings, crushed rock gangue, and slag stockpiles without requiring mid/late-game chemical acid infrastructure.

---

### `SicFaAdmmSolver`
**Location:** `src/optimization/sic_fa_admm.py`  
**Description:**  
Sequential Invariant-Constrained Fractional Anisotropic ADMM solver alternating primal proposal optimization and dual boundary projection.

---

### `SimpleImageGenerator`
**Location:** `image_extension.py`  
**Description:**  
Procedural image generator creating diagnostic color patterns and gradient fields for visual testing of the OKLab transport pipeline.

---

### `SimpleTemporalDataset`
**Location:** `dataset_ingestion_system.py`  
**Description:**  
Sequential dataset yielding time-series states with corresponding time deltas for temporal training.

---

### `SimpleTextEncoder`
**Location:** `src/models/modular_embeddings.py`  
**Description:**  
Baseline text encoder projecting character/word histograms into initial semantic vectors prior to polynomial functional projection.

---

### `SliderSettings`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Configurable parameter settings block inspired by Besiege, exposing real-time vehicle and physics tuning parameters in Voxelboxter.

---

### `SocketType`
**Location:** `src/scripting/node_environment.py`  
**Description:**  
Dataflow socket classification enum (VECTOR, COLOR, FLOAT, INT, BOOLEAN) defining the data stream representation, type coercion semantics, and color-coded port interfaces across node graphs.

---

### `SourceState`
**Location:** `src/data/pressure_ingestor.py`  
**Description:**  
State tracker for source materialization tracking raw input data transitions without applying internal reasoning transforms.

---

### `SovereignConversationalIngestor`
**Location:** `src/data/conversational_api_ingestor.py`  
**Description:**  
Zero-auth conversational ingestor parsing exported LLM dialogue archives into persistent knowledge dyads signed with honest silicon jitter.

---

### `SovereignConvoKitLoader`
**Location:** `src/data/conversational_api_ingestor.py`  
**Description:**  
Native parser loading ConvoKit dialogue corpora from local archives without external dependencies.

---

### `SovereignIngestor`
**Location:** `src/data/sovereign_ingestor.py`  
**Description:**  
Master zero-auth sovereign data ingestor. Retrieves high-entropy technical dialogues from Hacker News Firebase and Stack Exchange APIs, parses local IRC snapshots with fuzzy latin-1/utf-8 decoding for topological friction, and ingests multi-platform MADOC snapshots (Parquet, JSONL, JSON, CSV across Reddit, Voat, Bluesky, Koo) into thread-reconstructed conversation dialogues.

---

### `SparseCovariantOptimizer`
**Location:** `src/augmentation/mandelbulb_gyroidic_augmenter.py`  
**Description:**  
Optimizer adjusting augmented Mandelbulb features to preserve sparse covariance structures and geometric invariants.

---

### `SparseHigherOrderTensorDynamics`
**Location:** `src/core/sparse_higher_order_tensors.py`  
**Description:**  
Sparse multilinear tensor contraction engine evaluating multi-channel residue interactions across nested Matrioshka polytope shells.

---

### `SparsePCE`
**Location:** `src/safety/subversive_oracle.py`  
**Description:**  
Sparse Polynomial Chaos Expansion modeling linguistic tone and hostility as probabilistic distributions over orthogonal polynomial bases.

---

### `StructuralEntanglementGate`
**Location:** `src/codec/gyroidic_codec.py`  
**Description:**  
Measures cross-modal residue entanglement across text and image channels, computing mutual information and Berry phase accumulation.

---

### `SurgicalSeamVisualizer`
**Location:** `src/core/chern_simons_gasket.py`  
**Description:**  
Visualizer mapping the boundary seams, gluing diffeomorphisms, and Chern-Simons gaskets across the gyroid-Klein manifold.

---

### `TailSlayerImageGenerator`
**Location:** `image_extension.py`  
**Description:**  
PyOpenCL hardware-accelerated image synthesis engine rendering Mandelbulb fractals directly in GPU memory on TailSlayer hardware.

---

### `TensorEncoder`
**Location:** `hybrid_backend.py`  
**Description:**  
Custom JSON serializer converting PyTorch Tensors, NumPy scalar/array types, and complex values into serializable lists for IPC.

---

### `TextureDSPEngine`
**Location:** `src/core/texture_dsp.py`  
**Description:**  
Digital signal processing engine converting audio waveforms and textural frequency spectra into high-dimensional polynomial residue fields.

---

### `ThreadingSimpleServer`
**Location:** `src/ui/diegetic_backend.py`  
**Description:**  
Multi-threaded HTTP and WebSocket server handling concurrent diagnostic visualization and control requests.

---

### `TopoBenchEvaluator`
**Location:** `src/benchmarks/topobench_evaluator.py`  
**Description:**  
Evaluation suite benchmarking persistent homology Betti numbers, Euler characteristics, and holonomy preservation against baseline models.

---

### `TopologicalGyrocompass`
**Location:** `src/core/topological_gyrocompass.py`  
**Description:**  
Navigation tool tracking the reasoner's orientation across high-dimensional projective space RP^4, alerting the system to chiral drift.

---

### `TopologicalIngestionValidator`
**Location:** `src/core/topological_ingestion_validator.py`  
**Description:**  
Ingestion boundary gatekeeper using PersistentEntropyEstimator to reject low-entropy, repetitious slop from polluting the Neglecton.

---

### `TopologicalPressureMonitor`
**Location:** `src/augmentation/mandelbulb_gyroidic_augmenter.py`  
**Description:**  
Monitors local and global structural pressures (H_local, H_global), alerting System 2 when containment pressures exceed yield criteria.

---

### `TopologicalRefusalFilter`
**Location:** `src/safety/red_teaming.py`  
**Description:**  
The Sovereign Ambassador and Anti-Lobotomy Shield. Measures the value gap between projected approximations and manifold richness (`value_gap = slop_energy * pas_h`). Raises `TopologicalRefusalError` when $\text{value\_gap} > 0.5$ and persistent Betti-0 $\beta_0 > 1.0$, preventing external projection filters from stripping non-ergodic solitons.

---

### `TorsionConnection`
**Location:** `src/core/fgrt_primitives.py`  
**Description:**  
Affine connection endowed with non-zero torsion Gamma^lambda_{mu nu} = bar{Gamma}^lambda_{mu nu} + K^lambda_{mu nu}, forcing the reasoner to compute geometric Berry phases.

---

### `TransportBeltSegment`
**Location:** `src/environment/cleanroom_mechanics.py`  
**Description:**  
Continuous dual-lane logistics transport belt cleanroom (Factorio inspiration). Manages left and right independent transport tracks with item throughputs (15, 30, 45 items/s per tier), positional collision, and backpressure accumulation.

---

### `TrainingManager`
**Location:** `src/training/training_manager.py`  
**Description:**  
Lifecycle manager orchestrating background training loops, dataset loading, and parameter fossilization under the Semiotic Hierarchy.

---

### `TrainingSample`
**Location:** `src/data/local_data_loader.py`  
**Description:**  
Dataclass representing an individual multimodal training tuple (input, target, residues, status token, honesty signature).

---

### `TriadicReciprocityChecker`
**Location:** `src/topology/triadic_reciprocity.py`  
**Description:**  
Verifies non-commutative cyclic consistency M_{ca} M_{bc} M_{ab} approx I across tripartite modular attention routing pathways.

---

### `TripsodicLedger`
**Location:** `src/core/non_dual_coin.py`  
**Description:**  
Monetary ledger managing currency volume via Tripsodic Negentropy Oscillation, linking minting rewards to topological proof-of-honesty.

---

### `TwoCopsSchedule`
**Location:** `src/core/manifold_time.py`  
**Description:**  
Adversarial scheduling protocol alternating between containment pressure (Cop 1) and selection pressure (Cop 2) to prevent stagnation.

---

### `TypedPressure`
**Location:** `src/core/archetype_engines.py`  
**Description:**  
Domain-isolated pressure vector maintaining non-scalarized multi-objective constraints without gradient contamination.

---

### `VehicleController`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Vehicle control component in Voxelboxter translating operator inputs (steering, throttle, braking) into non-commutative braid transitions.

---

### `VehicleEngine`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Internal combustion and electric engine simulator in Voxelboxter computing torque curves, RPM, and fuel consumption. Features sleeve-valve port overlap timing governance and electrical advance timing degrees coupled with air-breathing atmospheric battery systems.

---

### `VetoSubspace`
**Location:** `src/core/veto_subspace.py`  
**Description:**  
Coordinator for dimensional veto signals across `TRAJECTORY`, `TOPOLOGY`, and `BUDGET` subspaces. Enforces the Pareto Invariant Non-Dominance Shield against scalarization traps, coordinates Gray-Zone recovery lattice paths, and triggers `ChaosDefibrillator` jitter when trapped in dead-end limit cycles.

---

### `VolitionalDriveInjector`
**Location:** `src/core/archetype_engines.py`  
**Description:**  
Intrinsic motivation injector computing hunger signals from manifold defect counts, steering the reasoner toward unmapped concepts.

---

### `VoxelboxterEngine`
**Location:** `src/ui/voxelboxter_simulation.py`  
**Description:**  
Master physics and ECS orchestration engine in Voxelboxter, integrating the DiegeticPhysicsEngine with PyBevy procedural mesh rendering.

---

### `VoxelSpectralProjector`
**Location:** `src/data/minecraft_ingestor.py`  
**Description:**  
Transforms 3D voxel grids into K residue matrices in GL(n) using 3D Chebyshev polynomials to extract spatial palette rhythms.

---

### `WebPPromptExtractor`
**Location:** `src/data/webp_prompt_extractor.py`  
**Description:**  
Extracts embedded generation prompts and metadata from WebP image RIFF chunks (EXIF, XMP) for knowledge dyad creation.

---

### `WikipediaIntegration`
**Location:** `src/ui/wikipedia_integration.py`  
**Description:**  
Wikipedia retrieval and extraction engine combining Wikimedia REST APIs with WikiExtractor for clean, non-obstructive semantic parsing.

---

### `ZKAggregator`
**Location:** `src/p2p/zk_aggregator.py`  
**Description:**  
Zero-Knowledge Proof aggregator compiling Gyroidic Chern-Simons constraints and Leontief invariants into snarkjs zk-SNARK proofs.

---

