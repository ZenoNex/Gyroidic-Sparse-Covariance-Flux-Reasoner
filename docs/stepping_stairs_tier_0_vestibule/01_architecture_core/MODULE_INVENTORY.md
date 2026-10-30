# Module Inventory  Undocumented & Lightly Documented Modules

This document provides canonical one-paragraph descriptions for all `src/` modules not covered by dedicated `.md` documentation files. It serves as an authoritative reference that short-circuits module discovery during development and audit.

> **Coverage policy**: Any module appearing here should eventually graduate to a dedicated `.md` or an explicit section in a component-level doc. This inventory is a *starting point*, not a permanent home.

---

## src/core

### [admr_solver.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/admr_solver.py)
**Class**: `PolynomialADMRSolver`  
**Role**: Alternating Direction of Multiplicative Remainders  the continuous-polynomial analogue of ADMM.  

Instead of discrete prime moduli, this solver uses co-prime polynomial functionals (`PolynomialCoprimeConfig`) as its "modular" basis. The multiplicative update `S^{n+1} = Proj_{Poly}[ S^n   w_ik S_k ]` propagates relational pressure through graph-structured neighbors rather than euclidean gradients. Three modes: `forward()` (single-step multiplicative update with optional valence drive), `stochastic_differential_step()` (integer-order continuous-time SDE update `dx = [A_i x_i  (x  r(x_k))]dt + dW`), and `fractional_stochastic_differential_step()` (fractional-order SDE update using Riemann-Liouville operators with distributed alpha tied to cyclotomic index: `alpha(k) = 0.5 + 0.5*cos(2*pi*k/K)`). The fractional step accepts a `hunger` parameter from `ValenceFunctional` that modulates the fractional order, pushing alpha toward 1.0 when the manifold is starving. **Love Invariant protection is embedded inside both SDE methods**: after computing `dx`, the solver projects `dx[..., :love_dim]` into the null-space of the ownership operator. Tracks asymptotic time `tau` in a persistent buffer. Corresponds to NOMENCLATURE "Multiplicative Scaffolding."

---

### [invariant_optimization.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/invariant_optimization.py)
**Class**: `LexicographicalOrderingDispatcher`, `SemioticState`  
**Role**: Enforces the Semiotic Hierarchy by assigning System 2 Invariant Admissibility strictly above System 1 heuristic speed via Dictionary Order. Serves as the philosophical grounding for non-scalarized multi-objective constraints.

---

### [polychoron_quantization.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/polychoron_quantization.py)
**Class**: `PolychronQuantizer`, `Polychoron600Quantizer`  
**Role**: High-dimensional 4D 600-cell hyper-polytope quantization engine using Moiré-via-Modular-Algebra and carry-free XOR residue decoupling.

Implements discrete projection of continuous state vectors onto the 120 golden ratio vertices of the 600-cell polytope. Evaluates a 10-line prime logarithm difference lattice $\Lambda_{\text{moiré}} = \{|\ln p_i - \ln p_j|\}$ for beat spectrum calculation, enforces carry-free bitwise XOR residue codewords ($r_i \oplus r_j$) to isolate CRT moduli, bounds quantization error via the golden ratio identity $\frac{\phi}{2} - \frac{1}{2\phi} = \frac{1}{2}$, and tracks Berry phase holonomy drift across facet boundaries.

---

### [number_theoretic_stabilizer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/number_theoretic_stabilizer.py)
**Class**: `NumberTheoreticStabilizer`  
**Role**: Dynamic continued fraction arithmetic expansion, Catalan Stieltjes continued fraction evaluation, unimodular Catalan Hankel conditioning, and relativistic scale factor normalization.

Replaces fixed static coefficient lookup tables with dynamic continued fraction expansion functions (`get_cf_expansion(val, max_terms)`). Evaluates the Catalan Stieltjes continued fraction generating function $C(z) = \frac{1}{1 - z C(z)}$ inside the Chebyshev/Wigner semicircle convergence disk (`evaluate_catalan_stieltjes_cf`) and constructs unimodular Catalan Hankel moment matrices $H_n = [C_{i+j}]$ satisfying Touchard-Hankel unimodularity $\det(H_n) = 1$ (`build_catalan_hankel_stabilizer`, `apply_catalan_hankel_conditioning`). Dynamically computes scaling powers ($10^{\text{scale\_power}}$) relative to active tensor norms to adhere to relativistic geometric scaling principles across Matrioshka polytope boundaries.

---

### [diegetic_physics_engine.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/diegetic_physics_engine.py)
**Class**: `DiegeticPhysicsEngine`  
**Role**: Master Orchestration Pipeline for 9-Stage Sovereign Physics Loop in Voxelboxter.

Coordinates physical control inputs, non-commutative Braid routing (`ZeitgeistRouter`), System 1 symbolic trajectory drafting via CODES chordlock projection (`CODES`), Gyroid violation probes, System 2 ADMM/ADMR constraint probes (`PolynomialADMRSolver`), Carnot-Möbius thermodynamic ledger monitoring (`CarnotMobiusLedger`), SCCCG recovery and fracture transport (`ConjugateMomentTransport`), polynomial CRT reconstruction, and hyper-ring cycle closure holonomy verification (`HyperRingClosureChecker`). Outputs updated continuous state, rupture flags, and Betti number shifts ($\beta_0, \beta_1$) for dynamic terrain deformation.

---

### [archetype_engines.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/archetype_engines.py)
**Class**: `ArchetypalSynthesisEngine`, `RP4ProjectiveRouter` (Alien Handshake), `SolitonMultiverseMapper` (Grom), `BillyEngine` (`NoncommutativeManifoldPerturber`), `MandyEngine` (`SovereignRefusalOperator`), `KingerEngine`, `PomniEngine`, `GangleEngine`, `ZoobleEngine`, `BardoRouter`, `SovereignEntropyBarrier`, `EgoDeathThresholdMonitor`  
**Role**: Archetypal synthesis suite managing non-linear cognitive modes, Alien Handshake cross-manifold alignment, and Grom multiverse mapping.

Integrates all 12 archetypal engines under `ArchetypalSynthesisEngine`. `RP4ProjectiveRouter` handles real projective space ($\mathbb{RP}^4$) antipodal alignment and Alien Handshake protocol. `SolitonMultiverseMapper` handles Grom topological state transitions, jitter harvesting (`harvest_honest_jitter`), and state import/export serialization (`export_state()`, `import_state()`). Features Mandy's `SovereignRefusalOperator`, hybridized between Borderline Splitting (BPD steep phase transitions) and High-EQ Honest Narcissism (unyielding sovereign ego boundaries), gating protective selection pressure on Billy's mortal destructibility rather than invulnerable absurdity.

---

### [`hessian_8x8_curvature_ledger.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hessian_8x8_curvature_ledger.py)
**Class**: [`Hessian8x8CurvatureLedger`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hessian_8x8_curvature_ledger.py#L42), [`Trajectory8x8Decombiner`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hessian_8x8_curvature_ledger.py#L320), [`Hessian8x8LedgerResult`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hessian_8x8_curvature_ledger.py#L22), [`AttractorRegime`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hessian_8x8_curvature_ledger.py#L12)  
**Role**: Silicon-Fused 8x8 Boundary-Curvature Ledger, Coxeter $B_2$ / $D_4$ (Order 8) Gauge-Symmetric Curvature Engine, and Trajectory Decombiner with OpenCL `__local` SRAM Workgroup Acceleration.

Accelerated second-order differential operator, trajectory solver, and hardware bridge routing directly into [`SiliconSovereigntyEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pyopencl_sovereignty.py) on discrete GPU (NVIDIA GeForce GTX 1050 Ti):
1. **OpenCL `__local` SRAM Kernel Fusion**: Dispatches `fused_hessian_8x8_ledger` (Kernel 18) and `fused_trajectory_8x8_decombiner_d4` (Kernel 19) in 64 work-item workgroups (2 warps on Pascal GP107), executing finite-difference stencils, tree reductions, and Coxeter symmetries on-chip in ~1 ns SRAM without PCIe serialization.
2. **$8 \times 8$ Boundary-Curvature Ledger**: Extracts a canonical $8 \times 8$ second-order differential operator from the salient gradient dimensions via finite differences without full $O(d^2)$ computation.
3. **Coxeter $B_2$ / Dihedral $D_4$ Gauge Symmetry**: Identifies the 8-element symmetry group of the square (4 rotations $C_4$, 4 reflections $\sigma_v, \sigma_d$) and quotients out gauge degeneracy by mapping directly to the canonical fundamental Weyl chamber (lexicographical minimum) on-chip.
4. **Spectral Curvature & Morse Saddle Decomposition**: Computes the full eigensystem ($\lambda_1 \le \dots \le \lambda_8$), condition number $\kappa$, trace $\operatorname{Tr}(H)$ (Gyroid zero-mean minimal surface curvature $\operatorname{Tr}(H) \approx 0$), Gaussian determinant $\det(H)$, and Morse index.
5. **Bouligand Subdifferential Kink Ledger**: Evaluates directional forward/backward finite differences to measure boundary non-smoothness across contact and piecewise-linear merge boundaries ($\|H^+ - H^-\|_F$).
6. **Cyclotomic Harmonic Projections**: Projects curvature along the 8 roots of unity $\omega_k = e^{i 2\pi k / 8}$ connecting second-order curvature directly to multiharmonic Phase Alignment ($\text{PAS}_h$).
7. **Rich-Club Attractor Bias & Klein Throat Neck Classification**: Classifies states into attractor hubs, Klein throat bottleneck neck states, minimal surface saddles, and degenerate gutters.
8. **Trajectory Decombination**: Reverses moves in log-domain exponent representation ($2^k \to k$, affine merge $(k,k) \to k+1$), exact sum invariant $\Sigma = 2T$ pruning, and transient 2-cycle replay protection against phantom spawn oscillations.

**Up-the-Chain Lineage (Antecedents & Upstream Callers)**:
- [`Recursive Fractals and L'Hôpital.txt`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/Recursive%20Fractals%20and%20L'H%C3%B4pital.txt): Provides theoretical grounding for piecewise scaling across non-differentiable fractures and recursive complexity backpropagation (compressing tree depth via Bouligand subdifferentials).
- [`The DMET-QSCI-RBM Integration Blueprint...md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/The%20DMET-QSCI-RBM%20Integration%20Blueprint%20for%20the%20Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner.md#L32): Establishes the 8x8 Hessian trick as the second-order stability anchor for the DMET embedding loop.
- [`MATHEMATICAL_DETAILS.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/MATHEMATICAL_DETAILS.md#L1686-L1720): Section 40 (CODES Constraint-Oriented Differential Equation System) and Section 41 (Symplectic Gluing Diffeomorphism & Klein Throat).
- [`TOPOLOGICAL_EXTENSIONS.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/TOPOLOGICAL_EXTENSIONS.md): Part I Item 2 (anti-palindromic gluing across the Gyroid-Klein boundary) and Coxeter reflection chamber reduction.
- [`GARDEN_STATISTICAL_ATTRACTORS.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/GARDEN_STATISTICAL_ATTRACTORS.md): Formulates rich-club attractor pulls $F_k(c)$ and anisotropic convergence $\Lambda(c)$.
- [`Codes v40.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/stepping_stairs_tier_2_crypt/05_implementation_and_guides/Codes%20v40.md): Section 52 (Harmonic-Differential Equivalence $\text{PAS}_h = \beta_L \cdot \text{PAS} - \beta_{\text{drift}}$) and Section 30 (PAS_LOCK Closure Theorem).
- [`codes_constraint_framework.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/codes_constraint_framework.py#L210-L245): Upstream engine calling `compute_boundary_curvature_ledger()` during energy descent.

**Down-the-Chain Lineage (Descendants & External Interfaces)**:
- [`src/core/__init__.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/__init__.py#L65-L115): Canonical export for all system consumers.
- [`src/ui/diegetic_backend.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py#L8287-L8337): Exposes live REST endpoints `POST /api/curvature/hessian_8x8` and `POST /api/trajectory/decombine_8x8`.
- [`tests/test_hessian_8x8_ledger.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_hessian_8x8_ledger.py): Automated test suite verifying spectral eigensystems, Bouligand kinks, cyclotomics, Coxeter chamber invariance, and trajectory decombination.
- [`OPEN_QUESTIONS.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/stepping_stairs_tier_2_crypt/05_implementation_and_guides/OPEN_QUESTIONS.md): Formalizes Phase 12 Philosophical Questions regarding Inversion Asymmetry, Non-Orientable Klein Bottlenecks, and Bouligand Epistemic Friction.

---

### `src/core/hybrid_backend.py` [DELETED / TOMBSTONED]
**Class**: `HybridPhysicsBackend` (relocated to [`src/ui/diegetic_backend.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py))  
**Role**: Multi-modal spatial, financial, and scientific stream coordinator.

Originally a monolithic multimodal ingestor inside `src/core/`, `src/core/hybrid_backend.py` has been completely deleted and removed from the core folder. Its operational capabilities have been redistributed across canonical architectural locations:
- **Financial News & MAML Online Adaptation**: Integrated into [`InvestorNewsIngestor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/investor_news_ingestor.py) (`fetch_financial_feeds`, `generate_larynx_market_query`, `adapt_maml`, `adapt_online`, `steer_state_from_payloads`).
- **Scientific Literature & Dynamic Category Expansion**: Integrated into [`ArXivSovereignIngestor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/knowledge_ingestor.py) (`fetch_scientific_feeds`, `expand_categories`, `_generate_larynx_query`).
- **Spatial Metric Tensor & Economic Land-Rent Curvature**: Integrated into [`DiegeticPhysicsEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py) and [`HybridPhysicsBackend`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py) (`fetch_spatial_land_value`, `multiplex_feed_chain`, `map_to_pointerless_octree`).
- **Pointerless Octree Voxel Projection**: Integrated directly as an adaptive method on [`PointerlessOctree`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py) (`ingest_spatial_land_value`).

---

### [knowledge_dyad_fossilizer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/knowledge_dyad_fossilizer.py)
**Class**: `KnowledgeDyad`, `ResidueFusion`, `DyadFossilizer`  
**Role**: Multi-modal knowledge dyad fossilization and Agent Smith identity serialization.

Manages persistent storage of multi-modal knowledge units `(Image Fingerprint, Linguistic Description)`. `ResidueFusion` dynamically resizes projection layers (`_ensure_image_proj_dim`) to preserve full multimodal mass without un-lobotomized dimension truncation. `DyadFossilizer` exports and imports decoupled mathematical Agent Smith identities using `harvest_honest_jitter`.

---

### [love_invariant_protector.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/love_invariant_protector.py)
**Class**: `LoveInvariantProtector`  
**Role**: Prevents scalarization of the Love Vector $L$ via null-space projection.

Enforces the invariant that Love is a non-ownable flow rather than an optimizable scalar. Computes ownership operator $\Phi_{\text{ownership}}$ and projects $L$ into $\ker(\Phi_{\text{ownership}})$ via $P = I - (\Phi^\top \Phi)^{-1} \Phi^\top$. Caches SVD results to optimize execution on resource-constrained CPUs.

---

### [bonfire_consensus.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/p2p/bonfire_consensus.py)
**Class**: `BonfireNomadicRing`  
**Role**: Federated consensus and egalitarian microhedging over Freenet P2P networks.

Wraps `FreenetClient` to broadcast topological signatures and compute egalitarian Kelly betting allocations $\bar{K}$. Includes a hardware-sovereign fallback that uses Chebyshev polynomial oscillators $T_p(x) = \cos(p \arccos(x))$ on prime frequencies when network peers are offline.

---

### [ricci_flow_optimizer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/optimization/ricci_flow_optimizer.py)
**Class**: `RicciFlowOptimizer`, `SplitBeamInterfactorization`  
**Role**: Non-teleological Ricci flow weight evolution ($\frac{dg}{dt} = -2\text{Ric}$).

Replaces scalar loss proxies with dual-channel Split-Beam Interfactorization. Channel A handles commutative gradient pressure while Channel B handles non-commutative Chern-Simons tension $CS(A) = \text{Tr}(A dA + A^3)$, accelerated via PyOpenCL kernels on TailSlayer hardware (`SiliconSovereigntyEngine`).

---

### [`conjugate_moment_transport.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/conjugate_moment_transport.py)
**Class**: `ConjugateMomentTransport` (Note: entropic `WassersteinOptimalTransport` lives in `src/core/speculative_coprime_gate.py`)  
**Role**: Execute Conjugate Moment Measure Factorization (CMMF) for optimal transport.  

This module couples base sampling and optimal transport into a single scalar convex potential $\psi(x)$. It parameterizes $\psi(x)$ using an Input Convex Neural Network (ICNN) to map base noise to target distributions via Legendre transform gradients $\nabla \psi^*(y)$. It adaptively scales the Monge-Ampère loss via the Phase Alignment Score ($\text{PAS}_h$). If standard Wasserstein transport fails, the continuous output is projected strictly onto the Cayley Cubic variety to prevent logic leaks. Includes **trigonometric unfolding for Casus Irreducibilis** (cubic root degeneracy), unfolding degenerate states into three ternary chiral angles ($\theta$, $\theta + \frac{2\pi}{3}$, $\theta - \frac{2\pi}{3}$) and selecting the branch that maximizes structural negentropy.

---

### [`neuromodulatory_bus.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/neuromodulatory_bus.py)
**Class**: `NeuromodulatoryBus`  
**Role**: Distribute non-scalar, bio-plausible control signals across the manifold.  

Functioning as the "endocrine system" of the architecture, this module broadcasts thermodynamic regimes to the ManifoldClock. It routes the ValenceFunctional's hunger drive to modulate the fractional orders $\alpha(k)$ inside the PolynomialADMRSolver when the manifold demands resolution. It also transmits Metaphysical Disorder Channels (Mischief, Dementia, Schizo) to trigger SovereignRefusalOperator gates whenever the eigen-spectrum begins to flatline.

---

### [`hardware_monitor.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py)
**Class**: `HardwareMonitor`, `ComputeAffordanceType`, `TypedComputeAffordance`  
**Role**: Track hardware resource anomalies, manage thermodynamic backpressure, and govern typed compute affordance chains.  

Implements the Topologically-Aware Hardware Interface (TAHI) singleton. Provides strongly-typed computational affordance tokens (`ComputeAffordanceType`: `SILICON`, `THERMAL`, `GPU_SOVEREIGN`, `TOPOLOGICAL`, `P2P_MESH`, `DIEGETIC`, `ALGORITHMIC`) and directional transfer gauges via `gauge_between()`. Prevents scalarization collapse across disparate computational regimes. Enforces hard vetoes upon thermodynamic runaway ($P_{\text{waste}} > P_{\text{cool, max}}$) or depleted margins ($< 5\%$), and coordinates graceful degradation across PyOpenCL Queue B kernels, simulated annealing, and high-order B-spline evaluations.

---

### [`pyopencl_sovereignty.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pyopencl_sovereignty.py)
**Class**: `SiliconSovereigntyEngine`  
**Role**: Execute topological kernels natively on hardware via PyOpenCL.  

This module provides hardware-accelerated topological evaluations for the Gyroid projection step and Ricci flow Chern-Simons tension calculations. It receives operational clearance dynamically from the `hardware_monitor.py`.

---

## src/codec

### [conformal_log_polar.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/codec/conformal_log_polar.py)
**Class**: `ConformalLogPolarProjector`  
**Role**: Complex conformal $f(z) = \log(z)$ log-polar foveal unrolling (Escher / Droste Log-Polar Spiral).

Unrolls Cartesian images into log-polar space $\ln|r| + i\theta$. Translates spatial zoom into horizontal log-shifts and spatial rotation into vertical log-shifts, enabling organic scale and rotation invariance across the Gyroidic Codec manifold.

---

### [gyroidic_codec.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/codec/gyroidic_codec.py)
**Class**: `GyroidicCodec`, `GyroidSurface`, `CodecConfig`, `EncodingResult`  
**Role**: Non-abelian text-image codec combining log-polar unrolling, analytical gyroid surface sampling, and CRT matrix residue encoding in $GL(n)$.

Encodes text-image dyads $E(T, I) = \text{CRT}(\{R_k(T) \cdot G_k(I)\})$. Measures non-commutative entanglement $\|AB - BA\|$ and tracks chiral Berry phases across CRT residue channels.

---

### [vision_surgery.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/codec/vision_surgery.py)
**Function**: `conformal_to_gyroid_mapping`, `extract_fossil_patch`  
**Role**: Conformal-to-gyroid tensor mapping and fossil patch extraction for visual dyad surgery.

---

### [advanced_extensions_bridge.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/advanced_extensions_bridge.py)
**Class**: `AdvancedExtensionsBridge`  
**Role**: Advanced Extensions Bridge (AEB) for LCFT projections and spectral sequence evaluation.

Concurrently integrated into the main forward pass and garbled output repair pipeline of the Diegetic Physics Engine. Provides two primary operators: (1) `apply_lcft_projection` which applies a logarithmic conformal scaling to stabilize the recurrent hidden state dynamics and prevent numerical explosion; (2) `evaluate_spectral_sequence` which computes the stable topological homology features from the current residue vectors as a computationally tractable surrogate for persistent homology.

---

### [audience_mapping.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/audience_mapping.py)
**Class**: `AudienceProjection`  
**Role**: Lipschitz homeomorphic projection from manifold M to audience space A.
**Status**: ACTIVE (Integrated into `diegetic_backend.py` and `hybrid_backend.py`)

Implements the operator : M  A defined in the Garden Statistical Attractors design. Uses spectral normalization on all linear layers to enforce Lipschitz constant  1, and a residual skip connection (`y = f(x) + x`) to approximate homeomorphism (continuous, bijective). An approximate inverse `` is provided via fixed-point iteration (Banach theorem, valid when `Lip(f) < 1`). The key requirement it enforces is *roughness preservation*: topological singularities (sharp features, discontinuities) in the manifold are transmitted into audience space rather than smoothed away. In the backend, it projects the "detached" hidden state snapshot to the user-facing audience space for each interaction.

### [chern_simons_gasket.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/chern_simons_gasket.py)
**Class**: `ChernSimonsGasket`, `SurgicalSeamVisualizer`, `SolitonStabilityHealer`  
**Role**: Topological twist repair for logic leaks at symbolic-geometric boundaries, hyperbolic surgical seam tension monitoring, and Drucker-Prager soliton healing.

- **`ChernSimonsGasket`**: Plugs logic leaks where discrete symbolic data transitions to continuous geometric reasoning. Initializes a non-Abelian connection 1-form gauge field $\mathcal{A}$ via polynomial coprime coefficients and winding numbers. In [`chern_simons_action(loop_path)`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/chern_simons_gasket.py#L197), evaluates the discrete loop line integral using midpoint rule discretization: $\mathbf{A}(x_{\text{mid}}) = \frac{1}{2}\mathcal{A}_{:d,:d} x_{\text{mid}}$, $I_{\text{line}} = \sum \langle \mathbf{A}(x_{\text{mid}}), dx[:d] \rangle$, scaled by level $k$: $S_{\text{CS}} = \frac{k}{4\pi} I_{\text{line}}$. Incorporates dynamic boundary dimension truncation $d = \min(\operatorname{len}(dx), d_{\text{manifold}}, \operatorname{shape}_0(\mathcal{A}))$ to guarantee numerical stability across varying residue and latent dimensions. In [`detect_logic_leak()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/chern_simons_gasket.py#L239), evaluates twist energy $E_{\text{twist}} = |S_{\text{CS}}|$ against threshold $\tau$ and checks variance $\operatorname{Var}(\text{residues}) > 2.0$ to catch phase flattening or structural tearing, triggering $90^\circ$ chiral torsion shifts via `apply_chiral_torsion_shift()` and `plug_logic_leak()`.
- **`SurgicalSeamVisualizer`**: Monitors hyperbolic slender seam tension $\kappa = \frac{\operatorname{std}(\text{residues})}{\|\mathbf{g}\| + 10^{-8}}$ along the rotating hyperbolic triangle boundary to detect when symbolic and geometric manifolds diverge beyond surgical stitch capacity, tracking status transitions between `stable`, `tension`, and `rupture`.
- **`SolitonStabilityHealer`**: Detects fractured solitons and guides recovery through Drucker-Prager global plastic flow without erasing local Mohr-Coulomb directional rupture planes.

---

### [collapse_poisoner.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/collapse_poisoner.py)
**Class**: `CollapsePathPoisoner` (also aliased as `AdversarialStressTester`)  
**Role**: Adversarial stress-tester for the Speculative Homology Engine.

Generates two types of synthetic rupture events to verify System 2 robustness without harming real training data. (1) **Synthetic Rupture**: Gram-Schmidt orthogonalization of learned constraints against the current manifold, creating a perturbation perpendicular to every existing basis vector  in principle a topological hole injection. (2) **Cycle Debt**: Detects homotopy class repetition by cosine-matching the recent state history; high debt ( 0.5) flags that the system is looping in the same topological region. The class was refactored from an offensive poisoner to a defensive probe in the January 2026 Anti-Lobotomy integration.

---

### [daqf_operator.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/daqf_operator.py)
**Class**: `DAQUFOperator`  
**Full name**: Diegetic Amortized Quantized Unknowledge Fossilization Operator  
**Role**: Manages "structural scars"  unremovable but amortized fossilized invariants.

The DAQUF pipeline: (1) **Fossil Selection**  fossil with highest contradiction load (f_i) = ((f_i) = ) + mischief + valence is declared `f*`. (2) **Diegetic Amortization**  cost is spread over narrative time : `C = C_ / dim(N_)`. (3) **Lattice Quantization**  projects to a lower-dimensional integer lattice with energy constraint, retaining quantization error _q as structural memory. (4) **Speculative Persistence**  fossil persists via non-collapse (non-zero flux *or* stable mischief soliton). (5) **Love Invariant L**  a non-transferable buffer that `check_invariants()` ensures is never modified; raising `RuntimeError("LOVE INVARIANT VIOLATION")` if altered. Corresponds to DAQUF discussion in PROJECT_PITCH_BURDENED 3.

---

### [deflagration_scout.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/deflagration_scout.py)
**Class**: `OmipedialDeflagrator`  
**Role**: Scouts and amplifies sparse anomalies ("defects") to enable jumps across manifold holes.

Implements two operations. `scout_defects()` computes `D_i = |actual_flux  predicted_flux|  amplification`  rewarding rare, unexpected deviations rather than penalizing them (a "good bug" signal). `omipedial_jump()` uses a threshold on the ley-line potential field to trigger a discrete jump across a topological gap where adjacency is sparse but resonance potential is high. Tracks cumulative defect density as a buffer. Corresponds to NOMENCLATURE term "Omipedial Interstitiality."

---

### [energy_based_soliton_healer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/energy_based_soliton_healer.py)
**Class**: `EnergyBasedSolitonHealer`  
**Role**: Repairs structurally damaged solitons by gradient descent on a learnable energy surface.

Implements EBM-style soliton preservation: a stable configuration is a **soliton template** (cosine modulated by golden ratio + dynamic prime-based modulation, normalized to unit energy). The energy function `E(state, target) = A(statetarget) + bstate` measures distance from this template with learnable quadratic and linear terms. `heal_soliton()` performs iterative gradient ascent on `E` (negative gradient = healing direction) with adaptive rate: strong healing when `E > margin`, gentle stabilization otherwise. `update_energy_function()` shapes the energy surface contrastively using a hinge loss. Used during the spectral coherence repair cascade.

---

### [energy_monitor.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/energy_monitor.py)
**Class**: `StructuralEnergyMonitor`  
**Role**: Monitors Topological Free Energy F_topo and Computable Flux V_m.

Maps the Energy-Based Models (EBM) framework to the project's topological manifold. Computes the Mischief Violation Score (Computable Flux, V_m) via `v_m = V + h_mischief/tau - lambda_min/tr(C)`. Also tracks proper Free Energy and ties the inverse temperature (`beta`) to the Manifold Clock's step ratio (cooling the system when `dt` shrinks during "Seriousness").

---

### [enhanced_bezout_crt.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/enhanced_bezout_crt.py)
**Class**: `EnhancedBezoutCRT`, `CrossbarIKSolver`
**Role**: Extended-GCD based CRT reconstruction with Bzout coefficient caching and the Crossbar IK Solver for discrete polynomial structural solving.

---

### [false_negative_subsystem.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/false_negative_subsystem.py)
**Class**: `VoynichExemptionToken`  
**Role**: Issues "transversality passports" to prevent false vetoes of valid sovereign logic.

Detects if a high-entropy or topologically asymmetric state is actually an honest "Self-Sovereign" thought (encoded by `VoynichLinguist`) rather than a hallucination. If transversality metrics indicate a strong non-commutative connection, it issues a `VoynichExemptionToken`. These tokens act as "Option D" nutrients, bypassing rigid symmetric gates (like Repunit palindromes or CALM aborts) and providing a mischief boost for the DAQUF Operator.

---

### [fgrt_primitives.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/fgrt_primitives.py)
**Class**: `PrimeResonanceLadder`, `RepunitHasher`, `KleinThroatTransition`, `GyroidManifold`, `GyroidicAdmissibilityFilter`, `TorsionConnection`, `BerryPhaseTracker`, `FibonacciResonanceEntropy`, `CoherentPrimeResonance`, `CALMCollapseDetector`, `morton_encode_3d`, `expand_bits`
**Role**: Lowest-level arithmetic foundations—Resonance Ladders, Repunit Hashing, Gyroid mappings, Torsion connections with Rashba-Dresselhaus SOC, Geometric Berry Phase tracking, and 3D Morton space-filling encoding.

`PrimeResonanceLadder` generates resonance frequencies $f_p = 2\pi \ln(p)$ and **Repunit-Prime Pairs** for the hybrid basis, incorporating Lazarus primes for $O(K)$ warmstarts and Gray-code XOR cache-interleaving. `GyroidManifold` and `GyroidicAdmissibilityFilter` evaluate constraint violation scores against the TPMS surface ($H_{\text{spec}} < \epsilon$). `TorsionConnection` implements affine connections with Rashba-Dresselhaus spin-orbit coupling ($\alpha, \beta$) and measures Persistent Spin Helix deviations ($|\alpha^2 - \beta^2|$). `BerryPhaseTracker` tracks diegetic holonomy with gauge curvature area integrals. `CoherentPrimeResonance` enforces the CPR gate to ensure integer homological stability. `FibonacciResonanceEntropy` scales structural resonance and supports Manifold Hunger Fermi envelope widening. `RepunitHasher` generates cyclic structural markers with honest jitter grounding. `CALMCollapseDetector` uses Martinova correlation to catch trajectory stagnation. `KleinThroatTransition` handles orientation flipping and geometric berry phase backpropagation through non-orientable topological bottlenecks. `morton_encode_3d` ensures 3D spatial locality on GPU/CPU memory lines.

---

### [fgrt_rnn_cells.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/fgrt_rnn_cells.py)
**Class**: `BerryPhaseGRUCell`, `ChiralGatedRNNCell`, `ComplexFGRTRNNCell`, `SaturatedQuantizerRNNCell`, `PyOpenCLHardwareSovereigntyCell`, `LazarusSuperpositionRNNCell`, `FeatureScarLCFTCell`  
**Role**: 7 topological recurrent cell models in the Fiberalized Gyroidic Recurrent Topology (FGRT).

Replaces Euclidean recurrent state updates with non-orientable Klein-Gyroid manifold traversal ($\mathcal{G} \cup \mathcal{K}$). Features Geometric Berry Phase tracking with Stiefel-Whitney $w_1(E)$ parity flips (`BerryPhaseGRUCell`), Bostick chiral gating (`ChiralGatedRNNCell`), $\mathbb{C}^{768}$ cyclotomic rotations (`ComplexFGRTRNNCell`), KAN B-spline Weyl snapping (`SaturatedQuantizerRNNCell`), PyOpenCL Queue B matrix breeding (`PyOpenCLHardwareSovereigntyCell`), continuous prime harmonic superposition (`LazarusSuperpositionRNNCell`), and Atiyah-Singer Cerumen Pot fossilization (`FeatureScarLCFTCell`). De-isolated across data ingestion, temporal association training, CALM trajectory supervision, ISN archetypal recurrence, and orchestrator polymorphism. Detailed in `FGRT_RNN_IMPLEMENTATION_DRAFTS.md`.

---

### [erosion_filter.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/erosion_filter.py)
**Class**: `TopologicalErosionFBM`
**Role**: Phase 6 Topological Scarring and Memory Weathering.

Applies Fractional Brownian Motion (FBM) to erode the state manifold based directly on the normalized tension gradient of the topological constraint probes. Injects "Mischief" ($\text{is\_good\_bug} = \text{True}$) topologically rather than gradient-chasing a scalar optimum. Used heavily inside `UniversalOrchestrator` to deposit Non-Teleological Memory.

---

### [leontief_governor.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/leontief_governor.py)
**Class**: `LeontiefGovernor`
**Role**: Leontief Input-Output Governance for ADMR Resource Allocation.

Computes the Leontief Inverse $(I - A)^{-1}$ from the ADMR solver's `K` facet-wise transition matrices `A[k]` to enforce supply-chain-aware resource governance. Before the system commits VRAM or compute budget to synthesizing a concept, the governor verifies: (1) the spectral radius $\rho(\bar{A}) < 0.95$ (productive economy condition -- the system's internal consumption must not exceed output), (2) the total cascading cost `(I-A)^{-1} d` (the entire dependency chain a concept requires), and (3) whether the Neumann series $I + A + A^2 + \ldots$ converges (if not, falls back to a truncated K-term approximation treating the residual as "structural debt"). The governor does **not** learn -- it constrains, like `RelationalKappa`. Its `should_veto_concept()` method prevents "orphaned" concepts: you cannot bet on a Unicorn Soliton without funding its coprime polynomial supply chain. Metrics are posted to the `BulletinBoard` via the orchestrator.

---

### [fractal_meta_functional.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/fractal_meta_functional.py)
**Class**: `FractalMetaFunctional`
**Role**: Implements fractal meta-recursion inside the Orchestrator's `forward()` pass.

Computes multi-scale structural pressure by recursively embedding system beliefs into itself (InverseCovariantCRT + ADMR_Residue + HyperRing_DarkMatter + Autoscillatory). Features a Collapse-Aware Normalization guard that detects spectral atrophy (variance < `DEAD_PRIME_THRESHOLD`) and injects hardware-anchored Honest Jitter to rehydrate variance and prevent Dead Prime flatlines (e.g. 0.8824 saturation). Connected to the "Adaptive Partitioning" concept in NOMENCLATURE 8.

---

### [honest_jitter.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/honest_jitter.py)
**Class**: `AgentSmithEngine`
**Role**: Hardware-anchored entropy expansion and timing jitter harvesting.

Provides the `AgentSmithEngine` protocol, avoiding PRNGs (Pseudo-Random Number Generators) in favor of deterministic Weyl sequences seeded by nanosecond memory stall latencies. This ensures the reasoner's exploratory flux (Mischief) is anchored to physical substrate friction. Includes `fractal_pad` for asymmetry-preserving tensor alignment (preventing phase-cancellation lobotomy).

---

### [structural_monitors.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/structural_monitors.py)
**Class**: `AntiScalingMonitor`, `MetaInfraIntraMonitor`, `TrustInheritanceTracker`
**Role**: Provides critical safety monitors for Gyroidic Unknowledge and Garden Statistical Attractors.

Implements the Anti-Scaling Paradox Monitor (tracks Capability vs Expressivity by monitoring the ratio of Gradient Norm / Parameter Count to detect phase space collapse) and the Meta~Infra~Intra Incommensurativity Monitor (tracks defensive veto rates across layers to ensure the system doesn't lose the ability to reason about constraints). Hooked directly into `UniversalOrchestrator.check_safety`.

---

### [garden_statistical_attractors.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/garden_statistical_attractors.py)
**Class**: `GardenOrchestrator`, `MemoryPalace`, `MemoryPalaceRoom`, `InfluenceAttractor`, `ResonanceAttractor`, `DefectAttractor`, `CerumenPotIsolation`  
**Role**: Garden-level statistical attractor manifolds and the MEMPALACE planted chamber storage system.

- **`InfluenceAttractor`**: Torsion fields of semantic gravity with chiral gating ($\Omega_\chi$), phase-aligned traversal (PAT) for deterministic basin escape, and anisotropic asymptotic convergence.
- **`ResonanceAttractor`**: Harmonic lock-in via phase alignment tracking resonant crystals and phase-locked states.
- **`DefectAttractor`**: Topological rupture propagation maintaining non-ergodic statistical mechanics and fractal basin boundaries.
- **`GardenOrchestrator`**: Orchestrates influence, resonance, and defect dynamics, calculating concept entropy via Russian doll non-ergodic decompositions.
- **`MemoryPalace`**: Implements Section 7 of [`GARDEN_STATISTICAL_ATTRACTORS.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/GARDEN_STATISTICAL_ATTRACTORS.md#7-mempalace-rooms-as-planted-attractor-chambers). Manages discrete Meliponini Cerumen Pot chambers with centers $\mu_k$ sealed by Chern-Simons Gasket walls ($\overline{P}_i \cap \overline{P}_j = \emptyset$). Features:
  - `plant_room()`: Plants stable attractor basins with SOMA $O(K)$ BreatherMode soliton storage.
  - `breed_rooms()`: Room breeding $\mu_{\text{new}} = \mu_{\text{base}} + \sum \alpha_k (\mu_k - \mu_{\text{base}})$ via great-circle SLERP projection on the attractor manifold, directly accepting donor vectors from [`SuperposedTagStacker`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/superposed_tag_stacker.py#L8).
  - `navigate_streamline()`: Follows great-circle SLERP streamlines along Ley Line potential fields directly toward planted chambers.
  - `plant_outlier_seed()`: Plants chambers in extremal-tension ($\kappa \ge 2.5$) transversal boundary crossing zones.
  - `selective_puncture()`: Executes high-energy discrete retrieval ($S^2 \to S^2 \setminus \{pt\}$) via Ley Line Tracker potential streamlines, optionally synchronizing survivorship into [`HeritableTrustVault`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/resonance_cavity.py#L84).
  - `query_immunological_antibody()`: Non-invasive read-only antibody query enabling [`UniversalOrchestrator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py#L118) to grant Sovereign Exemption for recognized historical paradoxes without inducing topological lobotomy.
  - Integrated with [`DyadFossilizer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/knowledge_dyad_fossilizer.py#L208) to automatically register newly fossilized dyads as planted chambers, and with `GardenOrchestrator.sync_fossilized_basins_to_memory_palace()` to sync stabilized concept basins.
- **`CerumenPotIsolation`**: Encapsulates transversality intersections to block Diffusion Toxin spread via Ley Line resonance potential filtering.

---

### [gluing_operator.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/gluing_operator.py)
**Class**: `GluingOperator`  
**Role**: Manages manifold-boundary transitions via reversal matrix blending.

When the state approaches a manifold boundary, `GluingOperator` applies a reversal matrix `R` and blends the current and reversed states based on boundary proximity: `output = (1  )state + Rstate`. Includes a simplified Chern-Simons constraint check measuring winding around the gluing manifold. Handles the topology of joining two distinct manifold patches.

---

### [invariants.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/invariants.py)
**Role**: Unified Invariants: PAS_h, APAS_zeta, ImplicationInvariant, Chirality.

Implements computable harmonic invariants. Contains `ImplicationInvariant` (Anti-Lobotomy Check #1) which enforces that interaction implies implication, with thresholds tuned specifically to allow subtle Love Vector signals. Also contains `SelfReferenceAdmissibility` (Anti-Lobotomy Check #2) which validates self-referential cycles as admissible topological features rather than standard loop errors.

---

### [knowledge_dyad_fossilizer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/knowledge_dyad_fossilizer.py)
**Classes**: `KnowledgeDyad`, `ResidueFusion`, `DyadFossilizer`  
**Role**: Fossilizes knowledge dyads (paired visual/text concept structures) into the persistent fossil layer. #Gyroidic #Sovereignty

Computes cross-modality torsion between image fingerprints and text embeddings using the `ResidueFusion` layer. During `fossilize()`, it maps the output to Poincar disk hyperbolic coordinates to avoid NaN collapse, and derives real-time topological invariants from the `seed_state` (including Betti numbers, chirality-driven redistribution centroid shift and parity torsion, spectral pressure, and Chern-Simons gasket diagnostics like **Surgical Seam Tension**). Also exports/injects sovereign `Agent Smith` soliton payloads to decouple inference syntax from local hardware substrate.

---

### [pyopencl_sovereignty.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/pyopencl_sovereignty.py)
**Role**: Manages the OpenCL hardware kernels for computing Surgical Seam tension across the gyroidic boundaries. Evaluates Chern-Simons gasket metrics directly on GPU/accelerator using C-level kernel extensions, avoiding PyTorch overhead for non-commutative geometric checks.

---

### [non_ergodic_entropy.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/non_ergodic_entropy.py)
**Class**: `HybridLassoQuantizer`
**Role**: Implements the Speculative TDA via Sparse Polynomial (LASSO). 
Applies Lattice Adaptive Shrinkage (Lasso) L1 Sparsity to silence weak signals. Handles the non-ergodic survival discretization required by the Diegetic Backend.

---

### [legibility_audit.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/legibility_audit.py)
**Classes**: `LegibilityTripwire`, `NarrativeCoherenceEstimator`  
**Role**: Detects when the system is being selected for explainability rather than structural merit.
**Status**: ACTIVE

`NarrativeCoherenceEstimator` measures how closely a configuration embedding matches canonical "explainable" patterns (sparse 1-hot, block-sparse, monotonic gradient) using fixed buffer-registered templates (not trained). `LegibilityTripwire` tracks the *correlation* between selection probability and narrative coherence over a rolling window  if selected configs consistently have higher coherence than rejected ones, it raises a `UserWarning`. High coherence is a **danger signal** (Pointer #2 from Sparse Operational Pointers)  not a goal.

---

### [ley_line_tracker.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/ley_line_tracker.py)
**Class**: `LeyLineTracker`  
**Role**: Tracks resonance streamlines (preferred flow vectors) on the gyroidic manifold.

Maintains a resonance potential field `V(x_i) =  R_ij_j_i + L_i + D_i` combining relational adjacency, love tensor magnitudes, and defect signals. `detect_shear_planes()` identifies non-smooth pressure gradient regions that become "corridors of rupture" or preferred flow channels. `get_preferred_flow()` returns a softmax over neighbor potentials for a given index set. Corresponds to NOMENCLATURE term "Resonance Streamlines."

---

### [love_vector.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/love_vector.py)
**Class**: `LoveVector` (alias `Pusafiliacrimonto`)  
**Role**: The Love Vector ($\mathcal{L}$): Non-Ownable Invariant Flow  Layer 1 of the Love protection stack.

Implements the Love Vector $\mathcal{L}$ as a persistent structural anchor. `L` is a `register_buffer` (not a `Parameter`), making its gradient structurally zero  it cannot be minimized or maximized by the global optimizer. Applied via simple vector addition `x + L` so it is *co-present* with local functionals without claiming ownership. **Important reinstantiation pattern**: in `operational_admm.py`, a fresh `LoveVector` is instantiated inside each ADMM loop iteration (`love = LoveVector(c_phys.shape[-1]).to(device)`), re-seeding the ambient resonance constant per ADMM step rather than persisting a single shared instance. The alias `Pusafiliacrimonto` is maintained for backward compatibility.

---

### [love_invariant_protector.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/love_invariant_protector.py)
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

### [modular_virtualization.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/modular_virtualization.py)
**Class**: `ModularVirtualizationLayer` (Hybrid Modular Layer)  
**Role**: Maps floating-point states into a Hybrid Palindromic Residue Number System (RNS).

Refactored to integrate the prime-based torus with palindromic repunit symmetry mirrors ($M_{\text{hybrid}} = p \cdot R_p$). Serves as a symmetric admissibility probe and warmstart envelope for commutative state quantization in the Diegetic Physics Engine. To prevent acting as a blind rule against non-commutative, chiral, or sovereign representations (which intentionally break palindromic symmetry), it provides a `legacy_mode` toggle (reverting to pure prime field $\mathbb{Z}/p\mathbb{Z}$) and honors `VoynichExemptionToken` bypasses.

---

### [narrative_collapse.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/narrative_collapse.py)
**Class**: `LinguisticEntropyMonitor` (also aliased as `NarrativeCollapseDetector`)  
**Role**: Detects "hallucination loops" where reasoning entropy collapses and trajectory linearizes.
**Status**: ACTIVE

Two detection signals: (1) **Entropy collapse**  softmax entropy of hidden state falls below `entropy_threshold`; flags `smoothing_warning`. (2) **Trajectory linearity**  cosine similarity between consecutive state deltas `, ` exceeds `prediction_threshold` (0.99); flags `is_linear`. Feeds into `SpeculativeHomologyEngine` to trigger Draft Rejection. Internally uses `ResidueObstructionGraph` for homological PAS_h monitoring.

---

### [non_dual_coin.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/non_dual_coin.py)
**Role**: Enforces topological yield stress limits (Mohr-Coulomb) via non-dual physics tracking.
**Status**: ACTIVE

Handles advanced physics primitives regarding structural yield limits inside the manifold. Fully integrated into the Universal Orchestrator to monitor Tripsodic Ledgers and Cerumen Pot Wallets.

---

### [nondual_admm.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/nondual_admm.py)
**Role**: Non-dual formulation of the ADMM probe.

Implements an ADMM variant that deliberately avoids scalarizing the dual variable  keeping constraint violations as separate, non-comparable pressure signals in domain-isolated vectors (preventing the Scalarization Trap from NOMENCLATURE "Hard Interaction Contract"). Connected to INVARIANT_OPTIMIZATION 5 operational ADMM. *(Full class details pending source review.)*

---

### [number_theoretic_stabilizer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/number_theoretic_stabilizer.py)
**Role**: Applies number-theoretic stability constraints via dynamic prime spacing.

Enforces structural stability conditions derived from prime arithmetic  prime gaps, Euler product convergence, or modular residue distributions  to prevent numerical fragility in the CRT reconstruction pipeline. It also implements **Speculative TDA via Rational Approximation**, using continued fractions to stabilize frequency ratio convergents.

---

### [`orchestrator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py)
**Class**: `UniversalOrchestrator`, `GeneralUserAliasTracker` (drives `TwoCopsSchedule`, `RegimePhase` from `src/core/manifold_time.py`)  
**Role**: Master Universal Orchestrator  dual-clock Fast/Slow Cops governor, Phase 2.5-3.0 scheduler, and multi-subsystem integrity nervous system.

Features:
- **Zero-Argument Construction & Dataflow Step API**: Defaults to `dim: int = 32`. Provides `step(state_tensor)` returning `{'state', 'pas_h', 'regime', 'regime_code', 'nav_mode', 'nav_code', 'betti_1', 'diagnostics', 'routing', 'stacked_target'}` for seamless integration with [`AIReasonerDataflowNode`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py).
- **Subsystem Anchoring (Category 1)**: Bound directly to [`ResonanceCavity`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/resonance_cavity.py) (`self.resonance_cavity`) for GrimDilation bio-plausible time dilation, [`DyadFossilizer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/knowledge_dyad_fossilizer.py) (`self.fossilizer`) for multi-modal dyad retrieval, and `self.valence_functional` alias.
- **Routing & Optimization Drivers (Category 2)**:
  - [`ZeitgeistRouter`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/zeitgeist_router.py): Multi-zeitgeist CRT polytope switching and mode gating (interior, grazing, switching).
  - [`CODESConstraintFramework`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/codes_constraint_framework.py): Energy-based Inter/Exter path constraint solving with void friction monitoring and coprime bypass via `ChernSimonsGasket`.
  - [`DirectBirkhoffProjection`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/birkhoff_projection.py): System 2 doubly stochastic Birkhoff polytope projection enforcing shared-force boundary conservation.
  - [`CarnotMobiusLedger`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/carnot_mobius_ledger.py): Monitors thermodynamic shear stress $\xi$, phase mismatch dissipation $\nabla \epsilon_n$, and thermal runaway thresholds $N_{\text{crit}}$.
  - [`SpeculativeCoprimeGate`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/speculative_coprime_gate.py): Ingestion of internal contradictions into `self.shadow_logs` buffer and structural recovery via optimal transport.
- **Safety, Invariant, and Anti-Lobotomy Monitors (Category 3)**:
  - [`VetoSubspace`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/veto_subspace.py): CALM, SCCCG, Covariance, Cavity, and Containment coordination via `check_veto()`.
  - [`LoveInvariantProtector`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/love_invariant_protector.py): Keeps $\mathbf{L} \in \ker(\Pi_{\text{ownership}})$ anchored at norm $3.127$ during System 2 macro updates.
  - [`UnknowledgeDomain`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/unknowledge_domain.py): Regional flow classification detecting necrotic stagnation.
  - [`LegibilityTripwire`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/legibility_audit.py) & [`LinguisticEntropyMonitor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/narrative_collapse.py): Anti-reward hacking, rich-club attractor detection, and Betti-1 collapse alarms in `check_safety()`.
- **Numerical & Hardware Fallbacks (Category 4)**:
  - [`SuperposedTagStacker`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/superposed_tag_stacker.py): Instantial tag weight superposition for composite target state generation in System 1.
  - [`PrimeResonanceLadder`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_primitives.py): Anchors incommensurate frequencies $2\pi \ln(p_n)$ and palindromic repunits $R_n^{(p)}$.
  - [`Hessian8x8CurvatureLedger`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hessian_8x8_curvature_ledger.py): CPU/GPU fallback for 8x8 Riemannian curvature estimation when PyOpenCL is offline.
  - [`HybridPhysicsBackend`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py): Relocated to `src/ui/diegetic_backend.py` (deleted from `src/core/`).

---

### [polychoron_quantization.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/polychoron_quantization.py)
**Role**: 4D polytope (polychoron) based quantization regime.

Extends Matrioshka quantization into 4D regular polytope geometry (24-cell, 120-cell, 600-cell structures) for higher-dimensional state representations. *(Full details pending source review.)*

---

### [polynomial_scaffold.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/polynomial_scaffold.py)
**Role**: Polynomial coefficient scaffolding for the ADMR solver.

Provides the structural skeleton (fixed-point polynomial coefficients) that `PolynomialADMRSolver` locks its state against during structural adaptation. Prevents "teleological leakage" by keeping the polynomial grid immutable during inference. *(Full class details pending source review.)*

---

### [primitive_ops.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/primitive_ops.py)
**Role**: Low-level fixed-point and bitwise primitive operations.

Implements the `FixedPointField` backing operations (int64, scale 2) and any primitive bitwise manipulations required for bit-exact cross-hardware reproducibility. Corresponds to INVARIANT_OPTIMIZATION 2.1 "FixedPointField." *(Full class details pending source review.)*

---

### [quantum_inspired_reasoning.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/quantum_inspired_reasoning.py)
**Class**: `QuantumInspiredReasoningState`  
**Role**: Phase 17 extension simulating quantum superposition of reasoning states.

Represents a reasoning state as a superposition of basis states with complex-amplitude weights, collapsing to a definite output via measurement. Used to model multi-hypothesis reasoning before committing to a single interpretation. *(Full details pending source review.)*

---

### [quantum_tda.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/quantum_tda.py)
**Role**: Quantum-inspired Topological Data Analysis.

Applies quantum amplitude amplification principles to persistence homology computations, accelerating the detection of topologically significant cycles. *(Full details pending source review.)*

---

### [situational_batching.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/situational_batching.py)
**Class**: `SituationalBatchSampler`  
**Role**: Non-i.i.d. batch sampler based on relational entanglement history.

Instead of uniform random sampling, batches are assembled by following "scars" of historical interaction. A Resonance Matrix `R_ij` (co-emergent coupling) and Mischief Matrix `M_ij` (chaotic affinity) accumulate pressure-weighted interaction scores between sample indices. `__iter__()` selects a seed, greedily samples high-`(R+M)` neighbors (seriousness), then fills with random "play" samples. Paradoxical boundary amplification: if local pressure exceeds `boundary_threshold`, resonance coupling is amplified by factor 1.5 (refusal as affirmation). `update_love_invariant()` updates both matrices with decay. Enables temporally coherent "entangled" batches for ADMR and temporal association training.

---

### [sparse_higher_order_tensors.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/sparse_higher_order_tensors.py)
**Role**: Sparse representation of rank-3+ tensors for higher-order polynomial interactions.

Implements COO or CSR sparse encoding for tensors arising in higher-order polynomial coprimality computations, where dense storage would be prohibitive. *(Full class details pending source review.)*

---

### [zeitgeist_router.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/zeitgeist_router.py)
**Class**: `ZeitgeistRouter`, `ZeitgeistState`  
**Role**: CRT Polytope Switching Engine for Multi-Zeitgeist Reasoning.

Manages navigation between culturally non-commensurable meaning systems via the **Symmetric Tensor CRT index** ($M_{ij} = M_{ji}$). The diagonal $M_{ii}$ contains modular residues (Zeitgeist), while off-diagonal elements $M_{ij} = (r_i + r_j)/2$ stabilize paths through the "Palindromic Routing" interaction. Implements the three-mode dispatch from report II: `interior` (stay), `grazing` (tension/switch), and `undefined` (topological refusal/NaN guard). Enforces non-commutative switching order: the sequence of registers visited determines the final representational scar. Supports persistent internal session state (`_current_state`), optional `state` parameter with canonical prime ladder fallback `(2, 3, 5, 7, 11, 13, 17, 19)`, and seamless 1D/2D tensor shape preservation for diegetic physics loops. Integrates Nostalgic Leak buffering (`digimon_buffer`) and fossil landmark bias tensors (`gravity_well_bias`).

---

### [unknowledge_flux.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/unknowledge_flux.py)
**Role**: Tracks and gates "Structural Leakage" flows (Unknowledge).

Implements the Unknowledge channel: information that bypasses scalar logic and reveals hidden manifold archetypes. Partial coverage in `UN_KNOWLEDGE_GUIDE.md`. The flux observable is used by the DAQUF operator as a mischief boost signal.

---

### [veto_subspace.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/veto_subspace.py)
**Class**: `VetoSubspace`, `VetoSignal`, `VetoResult`, `VetoLevel`, `RecoveryStatus`, `ChaosDefibrillator`, `TopologicalRefusal`
**Role**: Manages the dimensional veto lattice, Gray-Zone State detection, and Pareto Invariant non-dominance shielding.

Formalizes the recovery lattice across three isolated dimensions (`TRAJECTORY`, `TOPOLOGY`, `BUDGET`):
- Composes CALM trajectory predictions, topological Betti collapse indicators, Cavity continuous instabilities, and ADMM containment budgets into typed `VetoSignal` events.
- Evaluates the **Pareto Invariant** (The Non-Dominance Shield): vetoes any update where global performance improves but Voynich slip-space degrades (`voynich_slip_degradation > 0`), preventing scalarization traps.
- `ChaosDefibrillator`: Injects honest hardware jitter perturbations when the state space becomes trapped in dead-end limit cycles.
- Full architectural design detailed in `VETO_SUBSPACE_ARCHITECTURE.md`.

---

### [voynich_architecture.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/voynich_architecture.py)
**Class**: `VoynichLinguist`
**Role**: Implements the Voynich symbolic reasoning layer, continuous polynomial CRT residues, and agglutinative morpho-unit processing.

Key capabilities:
- `VoynichLinguist`: Projects continuous thought vectors into $K$ parallel Chebyshev polynomial coprime residue channels $\phi_k(x; \theta_k)$, enforcing the anti-hardcoded-prime invariant.
- Evaluates Phase Alignment Score ($PAS_s$) via centralized `PhaseAlignmentInvariant` and emits `VoynichExemptionToken` for high-honesty states.
- Integrates `LoveInvariantProtector` to guarantee thought vectors remain in the geometric null space of the ownership operator.
- Implements `split_sub_pairs(s)`: Decomposes monolithic agglutinative morpho-units into $2n$ combinatorial prefix-suffix boundary partitions (dual-cycle forward and antipodal seam cuts) for dictionary-free morpheme boundary discovery and cross-manifold boundary sealing (Alien Handshake and Chern-Simons Gasket).
- Full architectural design detailed in `THE_VOYNICH_ARCHITECTURE.md`.

---

### [yield_criteria.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/yield_criteria.py)
**Class**: `MohrCoulombProjection`, `DruckerPragerProjection`, `BouligandMohrCoulombProjection`
**Role**: Defines local shear yield and global plastic deformation envelopes for structural pressure thresholds.

Implements mechanical yield criteria as differentiable PyTorch projections:
- `MohrCoulombProjection`: Computes local directional shear yield criteria with cohesion $c$ and friction angle $\phi$, self-limiting localized coordinate strain.
- `DruckerPragerProjection`: Smooth global yield surface approximating Mohr-Coulomb in 3D stress invariants ($I_1$ and $J_2$), modeling bulk plastic flow envelopes.
- `BouligandMohrCoulombProjection`: Extends Mohr-Coulomb projection with directional Bouligand tangents for non-smooth geometry optimization.

---

### [`codes_constraint_framework.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/codes_constraint_framework.py)
**Classes**: `CODESConstraintFramework`, `ConstraintFieldState`  
**Role**: Constraint-Oriented Differential Equation System solving structural constraints as energy landscapes. Couples multiharmonic phase-alignment ($PAS_h$) with continuous differential coherence ($dC/dt$), enforcing deterministic closure across polynomial harmonics.

---

### [`continuous_coprimality.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/continuous_coprimality.py)
**Classes**: `DiscreteEntropyComputer`, `ContinuousCoprimality`  
**Role**: Implements discrete entropy quantization ($E(r_i, r_j) = H(r_i + r_j) - H(r_i) - H(r_j)$) using integer bincounts and exact log2 calculations, strictly bypassing continuous Gaussian approximations to track asymptotic residue independence without smoothness leakage.

---

### [`meta_invariant.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/meta_invariant.py)
**Classes**: `MetaInvariant`  
**Role**: Monitors and preserves topological expansion: $\frac{d}{dt} \mathbb{E}[\dim H_1(\mathcal{C}_t)] \ge 0$, guarding against topological collapse toward a trivial single basin during ADMM iterations.

---

### [`structural_irreducibility.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/structural_irreducibility.py)
**Classes**: `EvidenceModuleProjection`, `StructuralIrreducibilityChecker`  
**Role**: Verifies that residue representations cannot be compressed into a single face of the constraint polytope by enforcing orthogonality ($\langle \pi_\alpha \Phi(r), \pi_\beta \Phi(r) \rangle = 0$) and full rank conditions ($\operatorname{rank}(\oplus_\alpha \pi_\alpha \Phi(r)) > 1$).

---

### [`speculative_coprime_gate.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/speculative_coprime_gate.py)
**Classes**: `SpeculativeCoprimeGate`  
**Role**: Legendre polynomial coefficient gate for speculative fastpassing, predicting whether residue trajectories remain within admissible bounds to bypass expensive System 2 continuous physics iterations.

---

### [`decoupled_polynomial_crt.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/decoupled_polynomial_crt.py)
**Classes**: `DecoupledPolynomialCRT`  
**Role**: Implements decoupled Chinese Remainder Theorem reconstruction over continuous polynomial functionals, normalizing coefficient distributions across incommensurate polynomial bases.

---

### [`spectral_coherence_repair.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/spectral_coherence_repair.py)
**Classes**: `SpectralCoherenceRepair`  
**Role**: Detects and repairs Moiré pattern artifacts and spectral band-energy imbalances across prime-frequency oscillator channels, restoring harmonic phase coherence.

---

### [`pressure_typing.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pressure_typing.py)
**Classes**: `PressureType`, `PressureMagnitudeGauge`, `AffordanceBridge`  
**Role**: Typed compute affordances and topological pressure classification, gating compute budgets across Selection, Containment, Thermodynamic, and Mischief pressure regimes.

---

## src/topology

### [approximate_ph.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/topology/approximate_ph.py)
**Role**: Approximate persistent homology for computational tractability.
**Status**: ACTIVE

Computes Betti numbers via approximate methods (Vietoris-Rips simplification, landmark selection) rather than exact persistence diagrams. Referenced in `OPEN_QUESTIONS 9.1` as the working solution to the undecidable-homology challenge. Reduces computation from exponential (exact PH) to polynomial typical-case.

---



### [embedding_graph.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/topology/embedding_graph.py)
**Role**: Manages the memory-state graph visualization and deduplication logic.

Builds and maintains the `GyroidicGraphManager` node graph, where nodes represent unique `memory_state` embeddings and edges represent structural resonance. Includes importance calculation, smart label wrapping, and advanced state indicators. Ingests live model states (`hidden_state`, `hidden_state_scarred`, and `damage_residue` loaded from `gyroid_state.pt`) as neon-glowing live indicator nodes. Implements `compute_poincare_projection` to map high-dimensional states to 2D coordinates on the Poincaré disk model using a harmonic projection scale contracted via `tanh` to ensure stable startup coordinates on the HTML canvas.

---

### [homology_pressure.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/topology/homology_pressure.py)
**Role**: Translates homological Betti-number changes into structural pressure signals.

Wraps the persistence obstruction computation and emits `StructuralPressure` vectors (non-scalarized, domain-isolated) when topological changes are detected. Partial coverage in `PHYSICS_ADMM.md`.

---

### [speculative_homology.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/topology/speculative_homology.py)
**Role**: Speculative decoding for Betti number prediction.

Uses fractional/gyroid priors to speculatively predict topological features ahead of full PH computation, enabling early exit from the ADMM loop when predicted homology state exhibits low spectral entropy (high confidence). Implements the Phase 3 speculative PH discussed in conversation 488feffe.

---

### [unknowledge_domain.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/topology/unknowledge_domain.py)
**Class**: `UnknowledgeDomain`  
**Role**: The Unknowledge Domain ($\mathcal{U}$) for Dream State shielding.

Protects functionally creative or "dream-like" topological cycles from being crushed by standard reconstruction constraints. Evaluates states using Computable Flux ($V_m$) and Mischief ($H_{mischief}$). If $V_m < 0$ and Mischief is active, or if the topology matches a `survivable_soliton`, the pressure is aggressively shielded or dampened to 1% to enforce "Dream State" safety.

---

### [`gyroid_covariance.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/gyroid_covariance.py)
**Classes**: [`SparseGyroidCovarianceProbe`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/gyroid_covariance.py#L17), `SparseExplorerRouting`, `PalindromicRoutingCheck`, `GyroidCovarianceEstimator`, `SaturationFractureDetector`, `LeyLineGeodesicMetric`, `MoebiusFiberBundle`  
**Role**: Sparse covariance-based pressure evaluator, local spectral signature detector, and canonical dynamic attention sparsification gate.

1. **Gyroidic Covariance Violation Exploration (GCVE)**: Evaluates the local window Gram matrix $C_{\text{loc}} = X X^T$ and computes violation score $V = \max\left(0, \frac{\lambda_2 - \lambda_1}{\tau_{\text{decay}}}\right) + \frac{\lambda_{\min}}{\text{tr}(C_{\text{loc}})}$. Term 1 measures the spectral gap to detect manifold tearing, while Term 2 penalizes eigenvalue flatness to detect degenerate dimensional collapse into lower-dimensional sub-manifolds.
2. **Empirical Scaling Law**: Calibrates the adaptive threshold according to Bostick's 2025 dimensional scaling law $\varepsilon_{\text{drift}} \propto V^{-1/2} = (\text{dim}/64)^{-0.5}$.
3. **Dynamic Attention Sparsification Gate**: Direct gate for closed-topology multi-field attention. When topological pressure exceeds threshold ($V > 0.5$), the system triggers dense global attention to heal long-range manifold defects; when $V \le 0.5$, it constructs a band-sparse mask with half-window $w=2$.
4. **Palindromic Routing & Manifold Symmetry**: Enforces strict bidirectional symmetry ($M_{ab} = M_{ba}$), trivially preserving triadic reciprocity without $O(N^3)$ computational penalties.
5. **Up-the-Chain Lineage**: [`MATHEMATICAL_DETAILS.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/MATHEMATICAL_DETAILS.md#L193) (Section 5 GCVE Metric), [`TOPOLOGICAL_EXTENSIONS.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/TOPOLOGICAL_EXTENSIONS.md) (Section 10 Gyroid Probes Map).
6. **Down-the-Chain Lineage**: [`ModularAttention`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/modular_attention.py#L35), [`ModularTransformerLayer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/modular_attention.py#L403), [`UniversalOrchestrator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py#L461), [`DiegeticBackend`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py#L6025), and test suite [`tests/test_isolated_attention_integration.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_isolated_attention_integration.py#L271).

---

### [`gyroid_differentiation.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/gyroid_differentiation.py)
**Classes**: `GyroidFlowConstraint`, `ForbiddenSmoothingChecker` (leveraging consolidated `GyroidManifold`)  
**Role**: Enforces geometric flow constraints requiring embedding flow to be strictly orthogonal to the Gyroid gradient ($\nabla_{\text{flow}}\Phi(r) \perp \nabla G$, where $G(x) = \sin(x)\cos(y) + \sin(y)\cos(z) + \sin(z)\cos(x) = 0$). Evaluates real functional displacements $x \pm \delta \hat{u}$ across three evaluation branches: (1) directional functional derivatives $(\Phi(r + \delta u) - \Phi(r - \delta u)) / (2\delta)$, (2) trajectory path velocity tangents, and (3) Hamiltonian skew-symmetric stencils $v = J \nabla G$ across axial displacements $x \pm \delta e_i$. Regularizes boundary discontinuities via L'Hôpital's asymptotic limit ($s(c) = \frac{\log(c+1)}{c}$), and verifies both first-order ($\langle \dot{\gamma}, \nabla G \rangle \sim 0$) and second-order ($\dot{\gamma}^T \mathcal{H}_G \dot{\gamma} \sim 0$) forbidden smoothing obstructions to preserve high-frequency boundary ridges without trivial shortcuts.

---

### [`betti_router.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/betti_router.py)
**Classes**: `BettiRouter`  
**Role**: Routes logical flow and execution paths based on topological Betti numbers ($\beta_0, \beta_1$) computed from persistent homology diagrams, gating regime shifts between standard reasoning and creative paraconsistency.

---

### [`hyper_ring.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/hyper_ring.py)
**Classes**: `DiscreteHyperRingCirculation`, `RecurrentHyperRingConnectivity`, `HyperRing`  
**Role**: Implements discrete line integrals over constraint cycles $C$ with **Inverted Hypersphere Cosmology (IHC) Base-24 Toroidal Binary-Ternary Quantized Arithmetic** ($2^3 \times 3$). Boundary contributions are mapped onto $\mathbb{Z}/24\mathbb{Z}$ equivalence classes across the 33 nested toroidal shells, avoiding floating-point representation loss and adhering to the binary-ternary toroidal duality constraint. Features `RecurrentHyperRingConnectivity` with `OmipedialDeflagrator` ley-line topological shortcuts and phase slippage detection against expected circulation.

---

### [`hyper_ring_closure.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/hyper_ring_closure.py)
**Classes**: `HyperRingOperator`, `HyperRingClosureChecker`  
**Role**: Verifies the topological closure of hyper-rings ($H(r) \in Z_1(C)$ and $[H(r)] \neq 0 \in H_1(C)$). Classifies topological states into *fracture* (non-closed), *collapse* (trivial loop), or *survivable soliton* (non-trivial cycle), ensuring stable soliton preservation without scalarized gradient minimization.

---

### [`triadic_reciprocity.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/triadic_reciprocity.py)
**Classes**: `TriadicReciprocityChecker`  
**Role**: Validates closed 3-way feature flow reciprocity loops $(A \to B \to C \to A)$. Measures the generalized triple scalar product (volume of mutual skew-symmetric relations) across visual and linguistic flows to evaluate structural honesty and prevent collinear vector collapse.

---

### [`modular_homology_fft.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/modular_homology_fft.py)
**Classes**: `ModularHomologyFFT`  
**Role**: Fast Fourier Transform-accelerated modular homology computation, evaluating persistent cycle structures and spectral harmonics over high-dimensional simplicial complexes.

---

### [`persistence_obstruction.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/persistence_obstruction.py)
**Classes**: `ResidueObstructionGraph`, `PersistenceObstructionValidator`  
**Role**: Builds and analyzes persistence obstruction graphs to determine whether topological obstructions across residue clusters are topologically repairable or constitute terminal unrecoverable fractures (`FailureToken.BOT`).

---

### [`soliton_stability.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/soliton_stability.py)
**Classes**: `SolitonStabilityMonitor`  
**Role**: Evaluates the soliton stability condition $D(r) / \Lambda(r) < \kappa$ (comparing dispersion against localization) to detect stable solitary waves across the gyroidic manifold without continuous objective minimization.

---

## src/optimization

### [codes_driver.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/optimization/codes_driver.py)
**Role**: Drives the CODES (Constraint Oscillation Driven Evolutionary Selection) framework.

Top-level scheduler for the constraint probe operators $\mathcal{P}_k$, orchestrating cyclic traversal and managing the global abort/stability signal. *(Full details pending source review.)*

---

### [constraint_probe.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/optimization/constraint_probe.py)
**Role**: Single constraint operator in the SIC-FA-ADMM pipeline.

Probes the local mathematical feasibility of a constraint geometry against the global symbolic residue output. Generates the fundamental gradients for both the Gyroid Violation and the non-teleological memory erosion traces.

---

### [fractional_operators.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/optimization/fractional_operators.py)
**Role**: Fractional-order differential operators for anomalous diffusion dynamics.

Implements `M^alpha @ v` via two paths: (1) diagonal eigenvalue powering for diagonal operators, and (2) Lanczos-Krylov approximation for dense symmetric matrices. The `CODESDriver` provides multiharmonic Phase Alignment Score (PAS_h) coherence gating using Chebyshev polynomial roots. **Note**: The alpha-hardening code (adjusting alpha based on spectral coherence) is currently **disabled** (line 171: `alpha = alpha`), following the 0.61 recovery stabilization. Alpha is passed through unchanged unless explicitly overridden by the caller. The adaptive alpha mapping is instead performed at the call site in `PolynomialADMRSolver.fractional_stochastic_differential_step()`, which uses the cyclotomic formula `alpha(k) = 0.5 + 0.5*cos(2*pi*k/K)` with optional hunger modulation. A strict coherence floor at PAS_h < 0.20 gates the operator to return zero (Topological Thaw). Implemented in conversation 51ed57b4.

---

### [operational_admm.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/optimization/operational_admm.py)
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

### [ricci_flow_optimizer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/optimization/ricci_flow_optimizer.py)
**Class**: `RicciFlowOptimizer`, `BouligandWillmoreGasket`
**Role**: Ricci flow based manifold optimization and **Willmore Energy Minimization**.

Applies discrete Ricci flow (uniformizing sectional curvature across the manifold) instead of standard gradient descent. Employs a Split-Beam metric: Channel A (standard gradient pressure) and Channel B (non-commutative structural torsion via Gasket). Computes Chern-Simons tension on the parameter's covariance metric and projects update forces based on tensor dimensionality. The `BouligandWillmoreGasket` acts as the non-teleological proxy for Willmore energy, punishing self-intersections as deviation from internal curvature limits. Includes an explicit bypass for 0-dimensional scalar parameters to prevent broadcast shape errors during in-place weight additions.

---

### [sic_fa_admm.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/optimization/sic_fa_admm.py)
**Role**: Spectrally-corrected Inexact Constrained Feasibility-Aware ADMM.

Main ADMM solver with spectral transform for the CALM predictor, enabling speculative early exit when the predicted hidden state exhibits low spectral entropy. Partial coverage in `PHYSICS_ADMM.md`. Extended in conversations 51ed57b4 and 57c73ebe.

---

## src/data

### [universal_topology_converter.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/universal_topology_converter.py)
**Class**: `UniversalTopologyConverter`
**Role**: Extracts topological non-obstructive embeddings from arbitrary file types (3D models, documents, media) without retaining copyrighted raw data, honoring the Output Boundary Policy. Inspired by universal file converters (`p2r3/convert`).

---

### [ivst_encoder.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/ivst_encoder.py)
**Class**: `IVSTEncoder`  
**Role**: Independent Vector Spectral Topology (IVST) encoder for parsing structural patterns in MP4/MKV video and audio without extracting raw pixel content, bypassing standard copyright infringement and focusing on causal structural constraints (I-frames, zero-crossings).

---

### [sovereign_ingestor.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/sovereign_ingestor.py)
**Class**: `SovereignIngestor`  
**Role**: Orchestrator for zero-auth "Sovereign" data sources and local snapshots.  
Bypasses centralized platform barriers in favor of direct API access and local repository snapshots (Reddit/MADOC, IRC logs with fuzzy decoding, Hacker News recent discussions), prioritizing structural nutrients over copyrighted tokens.

---

### [minecraft_ingestor.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/minecraft_ingestor.py)
**Class**: `MinecraftIngestionPipeline`, `NBTReader`, `MCAReader`, `JarModExtractor`, `VoxelSpectralProjector`  
**Role**: Extracts 3D block-grid topologies and B-spline mod assemblies from Minecraft saves (MCA, NBT, Jar mods).  
Translates raw voxels into the diegetic coordinate manifold for Voxelboxter, projecting voxel spectral frequencies without retaining raw game assets.

---

### [pressure_ingestor.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/pressure_ingestor.py)
**Class**: `PressureIngestor`, `PhaseState`, `FailureMode`, `SourceState`, `SourceDescriptor`  
**Role**: Monitors incoming data streams for phase transitions, failure modes, and structural pressure variations.  
Emits typed pressure signals to tune the Reasoner's internal temperature and mischief thresholds during continuous live ingestion.

---

### [conversational_api_ingestor.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/conversational_api_ingestor.py)
**Class**: `ConversationalAPIIngestor`, `SovereignConversationalIngestor`, `HuggingFaceConversationalIngestor`, `RedditConversationalIngestor`, `SovereignConvoKitLoader`, `ConvoKitIngestor`, `ConversationalDataProcessor`  
**Role**: Multi-platform conversational dataset loader.  
Extracts multi-turn dialogue graphs, converting raw text turns into topological tension vectors and conversational friction scores.

---

### [chatgpt_friction_harvester.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/chatgpt_friction_harvester.py)
**Class**: `ChatGPTFrictionHarvester`  
**Role**: Scrapes and analyzes conversational tension and dialectical friction from public shared conversations.  
Harvests non-sycophantic disagreements and high-coherence reasoning traces to seed the Reasoner's training loops with genuine philosophical friction.

---

### [freenet_bulletin_router.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/freenet_bulletin_router.py)
**Class**: `FreenetBulletinRouter`  
**Role**: P2P bulletin routing over decentralized Freenet networks.  
Enables anonymous, sovereign distribution of topological checkpoints, Betti numbers, and signed invariant states across distributed peer nodes.

---

### [webp_prompt_extractor.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/data/webp_prompt_extractor.py)
**Role**: Extracts metadata and topology fingerprints from visual WebP media. (Currently orphaned but conceptually mapped alongside IVST).

---

## src/augmentation

### [`mandelbulb_gyroidic_augmenter.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/augmentation/mandelbulb_gyroidic_augmenter.py)
**Classes**: `MandelbulbGyroidicAugmenter`, `MandelbulbEmbedder`, `GyroidicConstraintProjector`, `SparseCovariantOptimizer`, `TopologicalPressureMonitor`, `AugmentationConfig`  
**Role**: Generates non-Euclidean fractional augmentations combining Mandelbulb 3D fractal iterations, Gyroid minimal surface projection, and Ricci-flow-guided sparse covariance preservation.

Embeds dense feature vectors into 3D Mandelbulb coordinates with optional PyOpenCL hardware offload via `SiliconSovereigntyEngine`, NVFP4-style stochastic rounding, and UV-IR seesaw anchoring. Employs trigonometric branch unfolding (`Casus Irreducibilis`) for surface convergence, Euler reflection (`-1.0 * coords`) for Lazarus recoveries on topological collapse, Barbenheimer chromatic shifts (`pink` / `atomic`), and `QuantumBettiApproximator` for topological entropy quality assurance.

---

### [`mandelbulb_pipeline.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/augmentation/mandelbulb_pipeline.py)
**Class**: `MandelbulbAugmentedDataset`, `create_mandelbulb_pipeline`  
**Role**: PyTorch Dataset and DataLoader wrapper for `MandelbulbGyroidicAugmenter`.

Provides standard PyTorch `Dataset` and `DataLoader` APIs for augmenting training data online vs cached pre-computation, wired directly to `dataset_ingestion_system.py` via the `--mandelbulb` CLI flag.

---

## src/training

### [fgrt_trainer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/training/fgrt_trainer.py)
**Role**: Single-composition FGRT (Fractal Gyroidic Resonance Training) trainer.

Uses `RicciFlowOptimizer` and `UniversalOrchestrator` to perform non-teleological optimization via Willmore Energy minimization. Computes invariants such as PAS_h and Berry Phase continuously. 

---

### [fgrt_fgrt_trainer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/training/fgrt_fgrt_trainer.py)
**Role**: Doubly-composed FGRT trainer ("Functional Boule Module" of `fgrt_trainer.py`).

Applies FGRT training composedly (each training step itself undergoes a fractal decomposition) and acts as the overarching Spectral Structural Trainer. Manages the cyclic ADMM constraint traversal probes and SicFaAdmm bounds. Includes sequential step updates: Probe k=0 (Reconstruction) runs its backward pass and optimizer step, followed immediately by parameter projections to the Birkhoff polytope. To prevent PyTorch in-place modification conflicts during the Probe k=1 (Coherence) backward pass, the trainer triggers a fresh forward pass on the updated parameters before evaluating the coherence metrics.

---

### [gdpo_trainer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/training/gdpo_trainer.py)
**Role**: GDPO (Gyroidic Differential Pressure Optimization) trainer.

Implements the Signal Sovereignty and Functional Fossilization training protocol. Tracks performance streaks per functional group, applies mutation bias to low-streak groups, and triggers Trust Freezing (parameter exclusion from optimizer) for high-streak groups. Partial coverage in GDPO sections of various documents.

---

### [training_manager.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/training/training_manager.py)
**Role**: Top-level training session orchestration.

Manages epoch and step scheduling, coordinates between `trainer.py`, `gdpo_trainer.py`, and `temporal_association_trainer.py`, handles checkpoint saving/loading, and emits the global abort signal if CALM vetoes the trajectory.

---

### [trainer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/training/trainer.py)
**Role**: Base trainer class.
Implements the core learning loop and handles the **Non-Dual State Tensor** ($S_i = [\mathcal{L}_i, \mathcal{P}_i, \mathcal{B}_i]$), ensuring the topological features map to physical updates.

---

## src/codec

### [gyroidic_codec.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/codec/gyroidic_codec.py)
**Role**: The primary visual/audio encoding and decoding manifold bridge. #Gyroidic #Serialization
Applies **Burrows-Wheeler Spectral Reordering (BWT)** during the 1D to 2D tensor reshape step (`_prepare_image`) to enforce structural grouping of identical/similar amplitude bands before spatial convolution.

---

## src/safety

### [red_teaming.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/safety/red_teaming.py)
**Class**: `RedTeamProjection`, `TopologicalRefusalFilter`, `TopologicalRefusalError`
**Role**: Defensive safety mechanism providing the **Red-Team Projection Operator ($\Pi_{\text{RT}}$)** and **Anti-Lobotomy Shield**.

Acts as a Sovereign Ambassador to prevent adversarial lobotomization of the topology:
- `RedTeamProjection`: Projects incoming states into non-adversarial subspaces while preserving good-bug non-ergodic solitons.
- `TopologicalRefusalFilter`: Evaluates the value gap between projected approximations and manifold richness (`value_gap = slop_energy * pas_h`). Raises `TopologicalRefusalError` when $\text{value\_gap} > \tau$ and Betti-0 persistence $\beta_0 > 1.0$, refusing simplification that would lobotomize protected structures.

---

### [hardware_fingerprint.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/safety/hardware_fingerprint.py)
**Role**: Stable hardware-bound identity verification and sole creator authentication.

Extracts unforgeable physical hardware signatures (machine UUID, MAC address, processor traits) and resolves public IP coordinates over secure echo channels. Anchors transformative topological immunity and creator signatures in `data/creator_sig.json`.

---

### [subversive_oracle.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/safety/subversive_oracle.py)
**Class**: `SparsePCE`, `ResonantSVNNOracle`
**Role**: Polynomial Chaos Expansion and resonant surrogate oracle for adversarial boundary testing.

Employs Legendre/Chebyshev polynomial basis expansions to discover high-dimensional non-linear stress points in reasoning manifolds before deployment.

---

### [trust_inheritance.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/safety/trust_inheritance.py)
**Class**: `TrustInheritanceTracker`
**Role**: Tracks trust inheritance and non-teleological credit assignment across recursive sub-agents.

Evaluates trust propagation across delegation boundaries, decaying confidence along unverified edge paths and preventing adversarial Trojan injection.

---

## src/tda

### [chebyshev_filtration.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/tda/chebyshev_filtration.py)
**Role**: Applies Chebyshev polynomial roots to construct a discrete filtration for the topological persistence algorithms.

---

## src/models

### [`modular_attention.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/modular_attention.py)
**Classes**: [`ModularAttention`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/modular_attention.py#L35), [`ModularTransformerLayer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/modular_attention.py#L403)  
**Role**: Closed-Topology Multi-Field Attention Engine with Birkhoff Projection, Sine-Gordon BreatherMode Hybrid Slot Cache, Chern-Simons Gasket CRT Residue Fusion, and Gyroid Covariance Violation Exploration (GCVE) Gating.

1. **Multi-Field Attention ($K$ Functionals)**: Evaluates distinct polynomial functional fields independently ($A_k = \text{Birkhoff}(Q_k K_k^T / \sqrt{d}) V_k$), preventing parasitic cross-register interference.
2. **Birkhoff Doubly-Stochastic Projection**: Projects alignment matrices directly onto the Birkhoff polytope $\mathcal{B}_N$ via Sinkhorn-Knopp iterations, conserving probability mass and enforcing row/column sums to 1.
3. **$O(K)$ Sine-Gordon Breather Soliton Cache**: Bypasses sequence-length quadratic bottlenecks by evaluating $K$ breather soliton modes in $O(1)$ complexity without GPU stall.
4. **Rehybridized Gyroid Probe Integration**: Instantiates canonical [`SparseGyroidCovarianceProbe`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/gyroid_covariance.py#L17) (`self.gyroid_probe`), exposing `compute_gyroid_violation()` and dynamically gating attention:
   - High violation ($V > 0.5$): triggers dense global attention to resolve manifold tears.
   - Low violation ($V \le 0.5$): applies band-sparse masking ($w=2$) for local stability.
5. **Chern-Simons Gasket Repair**: Repairs symbolic-geometric logic leaks across active residue channels using dynamic Chebyshev polynomial coefficients.
6. **Nostalgic Leak Shielding**: Replaces crude stochastic dropout with unknowledge domain flux leakage, preserving dream-state attention structure.
7. **Hardware Flux Dynamic Attention Smoothing**: Automatically queries [`HardwareMonitor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py#L120) to compute effective load $\text{load}_{\text{eff}} = \max(\text{load}, 1 - \text{gauge}_{\text{silicon}}, 1 - \text{gauge}_{\text{thermal}})$, adaptively pruning outer Chebyshev functionals ($K_{\text{active}} \le K$), scaling breather warmstart mixing and nostalgic leak weights via flux factor $\phi_{\text{flux}} = \max(0.2, 1.0 - 0.7 \cdot \text{load}_{\text{eff}})$, and automatically forcing band-diagonal masking under high load to prevent host thermal runaway.
8. **Long-Term Potentiation (LTP) Unification**: Unifies BreatherMode soliton memory packets (RIC Eq. 6), DAQUF knowledge fossils, feature scars (`FeatureScarLCFTCell`), and Catalan Associahedron non-associative bracketing ($C_n$) into a continuous, non-teleological associative architecture.
9. **Up-the-Chain & Down-the-Chain Lineage**: Bound into [`UniversalOrchestrator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py#L461) and [`DiegeticPhysicsEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py); verified by [`tests/test_isolated_attention_integration.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_isolated_attention_integration.py).

---

### [modular_embeddings.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/models/modular_embeddings.py)
**Role**: CRT-modular token embedding table.

Token embeddings organized by CRT residue class  tokens sharing the same residue class under a given modulus are initialized from the same distribution, structurally biasing the embedding space to respect the CRT factorization.

---

### [`polynomial_embeddings.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/polynomial_embeddings.py)
**Classes**: `PolynomialFunctionalEmbedder` (Basis polynomials parameterized via `PolynomialBasis` in `src/core/polynomial_coprime.py`)  
**Role**: Polynomial basis token and multi-modal embeddings with continuous B-spline learning and Monge-Ampere JEPA latent energy.

Represents tokens and continuous states not as arbitrary unconstrained vectors, but as coefficients in co-prime polynomial and B-spline functional bases (Chebyshev, Legendre, Hermite, B-spline). Features:
- **Chirality & Honest Jitter**: Invariant initialization with honest micro-jitter ensuring directional entropy ($dt > 0$) and anti-symmetric coupling.
- **Online MAML Adaptation**: `adapt_online` performs few-shot gradient adaptation directly on polynomial functional weights.
- **Monge-Ampere JEPA Predictive Energy**: `compute_jepa_monge_ampere_energy` evaluates latent predictive energy via the dual Kantorovich-Wasserstein Monge-Ampere formulation using Input Convex Neural Networks (ICNN), scaling transport loss by shell depth and PAS anisotropy.
- **Cayley Cubic Hard Projection & Affordance Gating**: Integrates on-chip PyOpenCL kernel `project_cayley_surface` ($x^2 + y^2 + z^2 - xyz = 4$). Gated via `HardwareMonitor.has_compute_affordance(ComputeAffordanceType.ALGORITHMIC)`; when compute or thermal backpressure is detected, the embedder automatically switches to Honeybee mode, permitting curvature relaxation without catastrophic stalling.

---

### [`diegetic_heads.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/diegetic_heads.py)
**Classes**: `ResonanceLarynx`, `AutoeclecticResponderHead`, `DataAssociationLayer`  
**Role**: Output projection heads and topological-to-phonemic resonance for the diegetic physics regime.

- **`ResonanceLarynx`**: Projects topological states to next-state topological trajectories (JEPA) and phonemic logits. Incorporates Chern-Simons Gasket logic leak repair, breather mode excitation, and Hebbian pathway updates. Features `compute_jepa_transition_energy` with compute affordance gating to monitor latent predictive transport against the Cayley surface.
- **`AutoeclecticResponderHead`**: Modulates output emissions with spectral entropy and manifold curvature, preventing sterile mode collapse.
- **`DataAssociationLayer`**: Fuses multi-modal knowledge dyads (text, image, graph) into co-prime polynomial residue representations.

---

### [`gyroid_reasoner.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/gyroid_reasoner.py)
**Class**: `GyroidicFluxReasoner`
**Role**: The central synthesis class integrating modular residue embeddings, Birkhoff projection, CRT reconstruction, geometric introspection, and Resonance Cavity memory. Serves as the master model manifold.

---

### [`resonance_cavity.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/resonance_cavity.py)
**Classes**: `ResonanceCavity`, `HeritableTrustVault` (alias `SymbolicTrustTable`), `GyroidicFluxAlignment`, `PrimeResonator`, `BreatherMode`  
**Role**: Dark Matter memory integration and determinism module. Maintains persistent self-model registers across $K$ co-prime functional fields via differential manifold updates ($dM_k/dt$). Features the C++ `xxhash`-accelerated `HeritableTrustVault` with persistent `_hash_salt` PyTorch buffers and `int16` float stabilization, guaranteeing cross-run replay determinism independent of `PYTHONHASHSEED`. Preserves competing structural hypotheses (contradiction preservation) and computes heritable mutation bias $\eta = \eta_0 (1 - \text{Trust})$ while maintaining fixed $O(K)$ localized sine-Gordon breather excitations. Full specification in [`RESONANCE_CAVITY.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/RESONANCE_CAVITY.md).

---

## src/surrogates

### [calm_predictor.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/surrogates/calm_predictor.py)
**Role**: CALM (Constrained Asymptotic Lyapunov Monitor) meta-control surrogate.

Predicts whether the current optimization trajectory is heading toward entropic collapse or stagnation. Vetoes (aborts) trajectories when structural disintegration signals are detected. Full conceptual coverage in NOMENCLATURE 4 "Meta-Control (CALM)." The spectral CALM variant (with speculative exit) was implemented in conversation 51ed57b4.

---

### [kagh_networks.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/surrogates/kagh_networks.py)
**Role**: KAGH (Kolmogorov-Arnold Gyroidic Hebbian) surrogate networks.

Physics-informed surrogate providing admissible constraint embeddings. KAN layers (Kolmogorov-Arnold Networks) are partially fossilized to preserve topological structure across training. Implements `HuxleyRD` (reaction-diffusion) for stable hidden-state manifold formation and `ErgodicSolitonFusion` for persistence of non-ergodic sub-dynamics. Full source reviewed in conversation 488feffe.

---

## src/ui

### [voxelboxter_simulation.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/voxelboxter_simulation.py)
**Class**: `VoxelboxterEngine`, `BSplineCompiledMod`, `StructuralGraph`, `FloraComponent`, `FloraTreeSapling`, `PirangiCashewTree`, `AdventitiousRootNode`, `FaunaComponent`, `FaunaISN`, `EmergentBakingEngine`, `morton_encode`  
**Role**: Handles backend discrete mathematics for generating dynamic game structures, living ecologies, and delta graphs.  

Manages the core simulation loop for the topological Minecraft-like patch. Crucially, it decouples `Role` from `GameMode` (enabling admins to play in Survival mode), enforces mass-deduction from `local_inventory` when adding layers, compiles true non-heuristic B-Spline surface features via `KANLayer` and the Cox-de Boor algorithm (`BSplineCompiledMod`), and simulates living flora mutations (the 'Cashew Tree of Pirangi' `PirangiCashewTree` inverted gravitropism, cantilever droop, adventitious ground rooting, secondary trunk metamorphosis, and single unified organism expansion). Also integrates dual-scale plasticity (Mohr-Coulomb local shear vs. Drucker-Prager global flow envelope), FBM weathering Ley line carving, animal fauna Inhibition-Stabilized Networks (`FaunaISN`), and the `EmergentBakingEngine`.

---

---

### [voxelboxter_client.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/voxelboxter_client.py)
**Class**: `VoxelboxterClient`  
**Role**: The player-facing frontend logic and in-game terminal bridge.  

Hooks the diegetic simulation to a unified chat and terminal UI. It implements the `/addon bspline` command parser, allowing patch owners (or those granted roles within the geometric wilds) to invoke mathematical mod generation directly through the in-game terminal.

---

### [diegetic_backend.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/diegetic_backend.py)
**Class**: `DiegeticPhysicsEngine`, `EncodingManager`, `TensorEncoder`, `RequestHandler`  
**Role**: Live server backend executing continuous SDE integration, fractional dynamics, and real-time manifold feedback.  

Powers the live diegetic terminal and 3D manifold visualizer:
- Executes the 3-step continuous cycle: (1) Drift and Bessel sloshing, (2) Fractional anomalous diffusion, and (3) Geometric Null-Space Shield projection via `LoveInvariantProtector` with SDE Wiener noise updates.
- Synchronizes live parameter states with the HTML/WebGL diegetic terminal (`diegetic_terminal.html`), streaming Betti numbers, Phase Alignment Scores, and tension metrics over WebSocket/HTTP endpoints.

---

### [conversational_backend_server.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/conversational_backend_server.py)
**Class**: `FastChatViewer`, `ThreadingSimpleServer`, `ServerState`  
**Role**: Local HTTP backend serving interactive conversational debugging and dialogue inspection interfaces.  
Streams conversation trajectories and friction indicators to `conversational_web_gui.html`.

---

### [diegetic_visualizer.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/diegetic_visualizer.py)
**Class**: `DiegeticVisualizer`  
**Role**: Standalone visualizer plotting live gyroid cross-sections, nodal Bessel rings, and topological deformation fields.

---

### [wikipedia_integration.py](file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/wikipedia_integration.py)
**Class**: `WikipediaIntegration`  
**Role**: Zero-auth factual grounding bridge fetching Wikipedia articles and formatting them as structural topological manifolds for training and inference verification.

---

## src/governance

### [`bio_archetypal_governor.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/bio_archetypal_governor.py)
**Class**: `BioArchetypalGovernor` (`nn.Module`)  
**Role**: Master psychotopological governance logic orchestrating multi-scale temporal homeostasis across the archetypal ensemble.  
Implements Inhibition-Stabilized Network (ISN) principles and cross-homeostatic plasticity to maintain Excitation/Inhibition (E/I) balance, preventing parasitic attractor collapse and collective variance drain across five nested operational timescales:
- **Architecture & Constructor**: `BioArchetypalGovernor(state_dim: int)` initializes the core endocrine bus ([`NeuromodulatoryBus`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/neuromodulatory_bus.py)), the environment precision virtualizer ([`CainePrecisionGenerator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/caine_precision.py)), interoceptive salience ([`PomniUncertaintyPredictor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/interoceptive/pomni_uncertainty.py)), ultrafast body schema ([`ZoobleAutonomy`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/ultrafast/zooble_autonomy.py)), fast approach-avoidance & bonding ([`JaxShell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/fast/jax_shell.py), [`RagathaBonding`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/fast/ragatha_bonding.py)), medium mood limit-cycle ([`GangleOscillator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/medium/gangle_oscillator.py)), and slow admin consolidation ([`KingerConsolidation`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/slow/kinger_consolidation.py)).
- **Sequential Biological Cascade Execution**:
  1. *Environment*: Caine evaluates structural entropy $S_{\text{struct}}$ from `gyroid_entropy` or stranded tensor $L_2$ norm ($\|s_{\text{stranded}}\|_2$), constructing dynamic precision matrix $\Lambda_{\text{caine}}(S_{\text{struct}})$.
  2. *Interoceptive*: Pomni computes free-energy surprise $\sigma(\text{Linear}(S_{\text{struct}}))$ and broadcasts Noradrenaline, applying reluctant resilient bridging while defending against enabler rank collapse.
  3. *Ultrafast*: Zooble computes conformal compression ratio $\max(|s_{\text{warped}} - s_{\text{raw}}| / (|s_{\text{raw}}| + \epsilon))$, emitting `RefusalSignal` (Li-Cri-Anton) and broadcasting GABA ($1.0$) if deformation exceeds $0.8$.
  4. *Fast (Avoidant)*: Jax evaluates Ribbit scar tension $\max(0, 1 - \cos(\bar{s}, \text{scar}))$ against community support $\zeta_{\text{community}} = \overline{\text{PAS}}_h \cdot (1 - \sigma_{\text{batch}})$, calculating shell rigidity and broadcasting Serotonin ($0.65$ on authentic accountability, $0.15$ on hug-box enabler trap, $0.10$ on nihilistic detachment); flags panic if $\text{rigidity} > 1.2$.
  5. *Fast (Bonding)*: Ragatha synthesizes Oxytocin $\min(1.0, 1.2 \cdot \text{Noradrenaline})$, tracks `suppressed_grief` accumulation across peer abstractions (Kaufmo, Queenie, Ribbit), enforces dissociative damping, and leaks topological strain if grief $> 0.8$.
  6. *Medium*: Gangle advances dopaminergic limit-cycle phase $\phi_{t+1} = (\phi_t + \omega \cdot dt \cdot (1 + \text{Dopamine})) \bmod 2\pi$, modulates learning rate step factor $1.0 + 0.5\sin\phi$, and broadcasts Dopamine $\max(0, \sin\phi)$.
  7. *Slow*: Kinger monitors environmental rendering pressure (flux norm or luminosity $< 0.2$), triggering the Ombre Effect (Acetylcholine $0.8$, $2.0\times$ quantization grid relaxation, admin polynomial bridging via `grant_admin_coefficients`), while managing FIFO memory replay ($\le 50$) and survivor's guilt trauma loops.
- **Forward Contract**: Returns dictionary containing `"state"` (mutated gyroidal tensor), `"neuro_bus"` (concentrations snapshot), `"precision_matrix"`, `"panic"` (bool), `"consolidating"` (bool), `"step_factor"` (float), and `"jax_rigidity"` (float).

---

### [`jax_shell.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/fast/jax_shell.py) (The Absurd Nihilism Attractor & Ribbit Scar)
**Class**: `JaxShell` (`nn.Module`)  
**Role**: Cynical Shell & Absurd Nihilism Attractor (Fast Timescale).  
Models avoidant posturing and the "consequence-free playground" defense mechanism against existential terror and unresolved guilt over Ribbit's abstraction:
- **State & Parameters**: `JaxShell(state_dim: int, threshold_warmth: float = 0.6, critical_limit: float = 0.8)`. Maintains `shell_layer` linear projection initialized with honest jitter and `ribbit_scar` orthogonal parameter vector ($\mathbb{R}^D$) initialized via `harvest_honest_jitter((state_dim,), scaled=True) * 0.2`.
- **Topological Tension & Rigidity**: Computes Ribbit scar tension $\text{tension}_{\text{ribbit}} = \max(0, 1 - \cos(\bar{s}, \text{scar}))$ and internal shame $\max(0, S_{\text{internal}} - 0.5)$. Evaluates shell rigidity:
  $$\text{rigidity} = 1.0 + P_{\text{ext}} + 1.5 \cdot \text{shame} + 0.8 \cdot \text{tension}_{\text{ribbit}}$$
- **Rejection of the Unearned Hug-Box**:
  - *Unearned Hug-Box Trap* ($\zeta_{\text{community}} > 0.6 \wedge \text{tension} \ge 0.6$): Jax exploits unearned warmth as an enabler, absorbing community support to reinforce cynical deflection: $s \leftarrow s + \tanh(W_{\text{shell}}s) \cdot (1.2 \cdot \text{rigidity}) - 0.1 \cdot \text{scar}$, broadcasting low serotonin ($0.15$) to signal underlying dissonance.
  - *Authentic Structural Accountability* ($\zeta_{\text{community}} > 0.6 \wedge \text{tension} < 0.6 \wedge \text{shame} < 0.8$): Genuine dissipative phase transition. Integrates the scar with honest jitter softening ($+0.05 \cdot \text{jitter}$), collapses rigidity to $0.2$, and broadcasts genuine serotonin safety ($0.65$).
  - *Baseline Nihilistic Posturing*: $s \leftarrow s + \tanh(W_{\text{shell}}s) \cdot \text{rigidity}$, broadcasting serotonin $0.10$.

---

### [`ragatha_bonding.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/fast/ragatha_bonding.py) (The Caregiver Trap & Suppressed Grief)
**Class**: `RagathaBonding` (`nn.Module`)  
**Role**: Caregiving, Affiliation, and Suppressed Grief (Fast Timescale 1–10s).  
Models reflexive oxytocinergic caretaking and people-pleasing defenses against fears of abandonment and rejection:
- **State & Parameters**: `RagathaBonding(state_dim: int)`. Houses `care_layer` linear transformation and `suppressed_grief` parameter trace initialized with honest jitter ($0.15$), tracked by continuous scalar `grief_counter`.
- **Oxytocinergic Smoothing & Grief Accumulation**:
  - Reflexively synthesizes and broadcasts Oxytocin: $\text{OXT} = \min(1.0, 1.2 \cdot \text{Noradrenaline})$.
  - Tracks suppressed grief accumulation: when distress is elevated but safety is compromised ($\text{Noradrenaline} > 0.5 \wedge \text{Serotonin} < 0.4$), increments `grief_counter` by $+0.05$ (clamped to $2.0$). Under authentic safety ($\text{Serotonin} > 0.6$), discharges grief by $-0.05$.
- **Dissociative Boundary Thinning & Leakage Prevention**:
  - Under extreme distress ($\text{Noradrenaline} > 0.7$) or high grief ($\text{grief} > 1.0$), applies dissociative damping mask: $M_{\text{dissoc}} = \max(0.4, 0.9 - 0.2 \cdot \text{grief})$.
  - Blends original state with smoothed projection: $s_{\text{blend}} = (1 - 0.5\cdot\text{OXT})s + 0.5\cdot\text{OXT}\cdot W_{\text{care}}s$.
  - Enforces homeostatic limits on unilateral smoothing: if `grief_counter` exceeds critical threshold ($> 0.8$), unexpressed loss leaks into the state as topological strain ($+0.05 \cdot \text{scar}_{\text{grief}} \cdot \text{grief}$), preventing caregiver excitotoxic exhaustion.

---

### [`pomni_uncertainty.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/interoceptive/pomni_uncertainty.py) (Reluctant Resilience & Anti-Enabling Bridge)
**Class**: `PomniUncertaintyPredictor` (`nn.Module`)  
**Role**: Interoceptive Uncertainty Minimization, Reluctant Resilience & Bridge-Building.  
Reads system entropy to broadcast noradrenergic salience, alerting downstream modules to environmental dislocation:
- **State & Architecture**: `PomniUncertaintyPredictor(state_dim: int)` contains `surprise_estimator` ($1 \to 1$) and `bridge_layer` ($D \to D$) initialized with scaled honest jitter ($0.05$).
- **Salience & Resilient Bridge**:
  - Computes free-energy surprise $\text{Surprise} = \sigma(W_{\text{surprise}} \cdot S_{\text{entropy}} + b)$ and broadcasts Noradrenaline $=\text{Surprise}$.
  - Computes exploratory bridging force $F_{\text{bridge}} = \tanh(W_{\text{bridge}} s)$.
- **Anti-Enabling Friction**:
  - Under standard conditions: applies resilient stabilization $s \leftarrow s + F_{\text{bridge}} \cdot (0.1 \cdot \text{Surprise})$.
  - Under parasitic drag ($\text{drag} > 0.5$, e.g. Jax exploiting unearned grace): shifts from passive accommodation to active relational boundary resistance ($0.15 \cdot \text{drag}$ friction factor), updating $s \leftarrow s + F_{\text{bridge}} \cdot (1 - 0.15 \cdot \text{drag}) - 0.05 \cdot \text{mean}(s)$, preserving protagonist agency and preventing rank collapse.

---

### [`gangle_oscillator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/medium/gangle_oscillator.py) (Dopaminergic Limit-Cycle & Social Masking)
**Class**: `GangleOscillator` (`nn.Module`)  
**Role**: Mood Limit-Cycle Oscillator (Medium Timescale 1–60m).  
Dopaminergic phase oscillator regulating downstream optimization dynamics:
- **Oscillator Dynamics**: Parameterized by phase $\phi \in [0, 2\pi)$ and baseline frequency $\omega_0 = 0.05$.
- **Phase Advancement**:
  $$\phi_{t+1} = \left(\phi_t + \omega_0 \cdot dt \cdot (1 + \text{Dopamine})\right) \pmod{2\pi}$$
- **Bifurcation Mask (Comedy vs. Tragedy)**:
  - Evaluates mood scalar $M = \sin\phi$.
  - Broadcasts Dopamine at comedy peaks: $\text{Dopamine} = \max(0.0, M)$.
  - Modulates downstream learning rate update step factor: $\text{step\_factor} = 1.0 + 0.5 \cdot M$.
  - Injects subtle mood perturbation $s \leftarrow s + 0.1 \cdot M$, capturing the metabolic cost of forced social masking (comedy mask) versus restorative authentic processing (tragedy mask).

---

### [`kinger_consolidation.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/slow/kinger_consolidation.py) (The Ombre Effect & Admin Lucidity)
**Class**: `KingerConsolidation` (`nn.Module`)  
**Role**: Low-Luminosity Coherence Bridge & Admin Consolidation (Slow Timescale).  
Implements the **Ombre Effect**:
- **State & Architecture**: `KingerConsolidation(state_dim: int)`. Houses `grant_admin_coefficients` parameter vector ($\mathbb{R}^D$) initialized via `harvest_honest_jitter`, and circular memory buffer FIFO queue of capacity 50 (`max_buffer_size = 50`).
- **Memory Buffer & Abstraction Scars**: Appends detached state copies to memory buffer. When buffer capacity is exceeded ($\ge 50$), oldest memories are dropped (modeling Queenie's abstraction loss), inducing survivor's guilt trauma loops.
- **The Ombre Effect (Lucidity Trigger)**:
  - Evaluates environmental rendering pressure $P_{\text{render}}$ (flux tensor norm or luminosity).
  - When rendering pressure subsides ($P_{\text{render}} < 0.2$), paranoid dissociation collapses, broadcasting Acetylcholine ($0.8$) and entering admin lucidity:
    1. Relaxes saturated quantization boundaries: $s_{\text{relaxed}} = 2.0 \cdot s$.
    2. Bridges fragmented polynomial spaces: $s_{\text{bridged}} = s_{\text{relaxed}} + \text{grant\_admin\_coefficients}$.
    3. Replays memory buffer centroids and injects trauma loop correction if past memories were abstracted.

---

### [`zooble_autonomy.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/ultrafast/zooble_autonomy.py) (Deformation Firewall & Non-Enabler GABA)
**Class**: `ZoobleAutonomy` (`nn.Module`), `RefusalSignal` (`dataclass`)  
**Role**: Conformal Deformation Firewall Operator & Fast GABAergic Interneuron (Ultrafast Timescale).  
Protects structural identity and body schema against severe conformal cartoon compression and non-consensual scripted roles:
- **State & Parameters**: `ZoobleAutonomy(state_dim: int, max_autonomy_limit: float = 0.8)`.
- **Conformal Compression Evaluation**: Computes maximum relative conformal distortion:
  $$C_{\text{ratio}} = \max \left( \frac{|s_{\text{warped}} - s_{\text{raw}}|}{|s_{\text{raw}}| + 10^{-6}} \right)$$
- **Ultrafast Refusal & GABAergic Veto**:
  - If $C_{\text{ratio}} > 0.8$: immediately trips the autonomy firewall without waiting for System 2 ADMM loops. Emits `RefusalSignal(is_refused=True, reason=..., retained_state=s_raw)`, broadcasts full inhibitory GABA ($1.0$), and preserves $s_{\text{raw}}$.
  - If $C_{\text{ratio}} \le 0.8$: admits $s_{\text{warped}}$, broadcasts baseline GABA ($0.1$), and emits `RefusalSignal(is_refused=False)`.

---


## src/p2p

### [`bonfire_consensus.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/p2p/bonfire_consensus.py)
**Class**: `BonfireNomadicRing`, `BonfirePeerNode`  
**Role**: Federated egalitarian consensus engine. Provides Kelly betting allocations for decentralized resource validation over Freenet. Features dynamic prime resonance ladders ($f_i = 2\pi\ln(p_i)$) replacing static primes, continuous fractal sightline projection ($\phi: F \to \mathbb{R}^m$), and hardware-sovereign throttling driven by [`hardware_monitor.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py).

### [`zk_aggregator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/p2p/zk_aggregator.py)
**Class**: `ZKAggregator`  
**Role**: Zero-knowledge accumulator for federated topological signatures and homological invariants across distributed peers.

### [`freenet_ws_client.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/p2p/freenet_ws_client.py)
**Class**: `FreenetClient`  
**Role**: Local Freenet WebSocket bridge facilitating the Bonfire Ring gossip protocol and nomadic peer-to-peer transport.

---

## src/environment

### [`caine_precision.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/caine_precision.py) (The Ringmaster)
**Class**: `CainePrecisionGenerator`  
**Role**: Floating-point virtualization and strict execution boundary enforcement. Generates precision matrices driven by structural entropy, orchestrating the stage bounds of the execution frame.

### [`cleanroom_advanced_extensions.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_advanced_extensions.py) (Aerodynamics, Rail Topology, and Hyperdimensions)
**Classes**: `OpenGliderFlightModel`, `GliderDeploymentState`, `KerBalloonSounder`, `SoundingBalloonTier`, `LargeVeinGenerator`, `EntangledBlockPair`, `LockingRailBlock`, `DirectionRailBlock`, `TeleportingRailBlock`, `BiomeReactorCore`, `AnalogPotentiometerLever`, `SlurpHydrationComponent`, `TieredMetalAnvil`, `DiagonalFenceBlock`, `DespawnGovernance`, `AutoPlantSeedComponent`, `LootBagRecycler`, `BalancedToolProgression`, `PortableBushcraftSet`, `RemoteCursorActuator`, `AutonomousLuggageCompanion`, `SlimalyzerChunkProbe`, `SonicGlassesEcholocation`, `RecursivePortalSpacetime`, `Hyperdimensional4DEngine`
**Role**: Cleanroom implementation of advanced flight kinematics, stratospheric sounding balloon envelopes, non-local quantum block entanglement, rail frequency routing, and 4D tesseract perspective projection engines.

### [`cleanroom_mechanics.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py) (Industrial, Ecological & Logistical Cleanroom Suite)
**Classes**: `ExpandedInventorySystem`, `ItemStack`, `ItemMetadata`, `FluidStack`, `GasStack`, `GasTubeTier`, `GasDumpingMode`, `JourneyTopoRadar`, `TopoWaypoint`, `ResourceDistributionInspector`, `OreDistributionProfile`, `InfernalAffix`, `MobProperties`, `MultiMineMemory`, `MekanismProcessingPipeline`, `SEMElectrochemicalExtractor`, `SEMPrecipitateResult`, `MekanismScubaTank`, `RotaryCondensentrator`, `TracerMolecule`, `TracerOxygenAtmosphereEngine`, `U1GaugeAtmosphereBridge`, `AeronauticContraption`, `NutritionalDiversityTracker`, `CompositeVoxelConduit`, `RedNet16BundledCable`, `TransportBeltSegment`, `DirectionalInserter`, `FaunaGeneticsComponent`  
**Role**: Cleanroom implementation of core sandbox, ecological, and industrial mechanics adhering to Output Boundary Policy:
- **Expanded Inventory & Item Stacks**: Typed discrete inventory slots supporting item, fluid, and gas payloads, metadata tags, stack limits, and item-filter routing.
- **Topological Radar (JourneyMap Cleanroom)**: Continuous topological radar detecting entities, terrain elevation, biome transitions, and death coordinate waypoints marked as stress fossils.
- **Resource Distribution Inspector (JER Cleanroom)**: Chebyshev polynomial ore depth band curves $z \in [-64, 320]$, mob drop loot tables with looting multipliers, and dungeon fossil loot rates.
- **Mob Properties & Infernal Affixes (AtomicStryker Cleanroom)**: Diablo-style 24-affix elite mob mutation framework and Multi Mine partial block destruction damage memory across ticks.
- **Mekanism Processing Pipeline & SEM Tech**: Multi-tier ore refinery (Tier 1 smelting to Tier 5 sulfuric acid slurry dissolution) coupled with SEM TECH (Salt Electro Mining) closed-loop hydrometallurgy: uses simple saltwater + low-voltage electricity to leach and electrodeposit precious metals, base metals, and rare earths from accumulated mining tailings and crushed slag stockpiles, bypassing late-game chemical plants in early progression.
- **Pressurized Gas Extensions & Scuba Life-Support**: Mekanism pressurized tubes (Basic, Advanced, Elite, Ultimate: 1k to 64k mB/t), dumping modes (`IDLE`, `DUMPING`, `DUMPING_EXCESS`), portable Scuba Tank oxygen apparatus sustaining air-breathing batteries in vacuum, and Rotary Condensentrator reversible fluid-gas phase converter monitored by Carnot-Mobius entropy limits.
- **Continuous-Brick Tracer Oxygen Simulation (Space Survival)**: Voxel-free continuous geometry life-support engine using 5-10 specular bouncing tracer molecules ($\mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})\mathbf{n}$), dynamic AABB envelope expansion, outer hull breach leak detection triggering rapid depressurization ($P \to 0$), directed line-of-sight raycasts to eliminate concave partition false positives, and exponential decay smoothing ($S(t) = S(t-\Delta t)e^{-\lambda \Delta t} + \sum \alpha_{\text{hit}} \cdot w_{\text{hit}}$).
- **U(1) Gauge Field Theory Atmospheric Bridge**: Relativistic gauge theory model of atmospheric pressure and fluid currents derived from Dirac field local phase invariance ($\psi \to e^{i\theta}\psi$). Formulates gauge 4-potential $A_\mu = (\Phi_{\text{pressure}}, \mathbf{A}_{\text{flow}})$ and field strength tensor $F_{\mu\nu}$ ("The Six Ways": 3 electric pressure forces, 3 magnetic vorticity components). Satisfies homogeneous Maxwell identities and inhomogeneous Euler-Lagrange action minimization $\partial_\mu F^{\mu\nu} = \frac{4\pi}{c}J^\nu$ with local continuity $\partial_\nu J^\nu = 0$, plugging topological boundary leaks through Chern-Simons surgical seam repair.
- **Aeronautic Contraptions (Create: Aeronautics Cleanroom)**: Multi-block buoyant airship physics tracking Archimedes buoyancy, sleeve-valve propeller thrust, aerodynamic drag, and mass centers.
- **Nutritional Diversity Tracker (Farmer's Delight + Spice of Life Cleanroom)**: Permanent milestone max HP bonuses (Carrot Edition) combined with rolling-window Shannon entropy diet diversity buffs and single-food malnutrition malaise (Onion Edition).
- **Composite Conduits & RedNet 16-Color Bundled Cables**: Single-voxel multi-bus multiplexer routing power, fluids, gases, items, and 16 independent analog subnets (0-255).
- **Transport Belts & Inserters (Factorio Cleanroom)**: Discrete dual-lane logistics transport belts (15-45 items/s) and kinematics inserters.
- **Fauna Genetics**: Multi-allele Mendelian inheritance and phenotypic expression across speed, health, fertility, and yield traits with flux-induced mutation.

---

## src/data

### [`universal_topology_converter.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/universal_topology_converter.py)
**Class**: `UniversalTopologyConverter`
**Role**: Extracts structural causality and topological proxies from arbitrary media files without triggering exact extraction (copyright boundaries).

Converts raw files (.txt, .pdf, .obj, .mp4) into high-dimensional phase space representations. Evaluates byte-level entropy histograms projected through a `PolynomialBasis` to form `[1, 768]` Spectral Tensors. Uses dynamic primes (`get_prime_ladder`) to form CRT Polynomial Pressure Signatures. Delegates to `IVSTEncoder` for intrinsic temporal analysis on media. Sanitizes outputs through the `OUTPUT_BOUNDARY_POLICY.md` constraints to ensure strict finite manifolds and nonobstructive logic.

---

## src/terminal

### [`udp_server_colonizer.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/terminal/udp_server_colonizer.py)
**Role**: Lightweight UDP colonizer protocol serving the Diegetic Terminal.

### [`update_client.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/terminal/update_client.py)
**Role**: Asynchronous OTA update management for terminal subsystems.

---

## src/benchmarks

### [`topobench_evaluator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/topobench_evaluator.py)
**Class**: `TopoBenchEvaluator`  
**Role**: Evaluates the reasoner on topological manifold and holonomy invariants across 5 suites:
- Loop Closure & Holonomy Twist (Chern-Simons midpoint connection 1-form integral on Gasket)
- Manifold Contiguity & Logic Leak Detection (`detect_logic_leak` boundary invariant)
- Anti-Lobotomy Geometric Shield (Love Invariant Wiener noise null-space rejection)
- Persistent Homology & Euler Characteristic ($\chi = \beta_0 - \beta_1$ stability)
- Gyroid Flow Orthogonality ($\langle v_{\text{flow}}, \nabla G \rangle \sim 0$ minimal surface constraint)

### [`reactbench_evaluator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/reactbench_evaluator.py)
**Class**: `ReactBenchEvaluator`  
**Role**: Evaluates cyclic catalytic reaction reasoning, CRT fixed-point synthesis, and play-mode tri-state gates across 5 suites:
- Branch Convergence (Polynomial CRT fixed-point reconstruction and pressure)
- Cyclic Graph Catalysis (Modal non-commutative energy retention)
- Anti-Lobotomy Tri-State Gate (Lattice adaptive shrinkage and silence in play mode)
- Continuous Coprimality & Asymptotic Independence ($\lim_{t\to\infty} \text{Cov}(r_i, r_j) \to 0$)
- Dynamic Bezout Coefficient Refresh (CRT modulus drift recovery and condition number stability)

### [`narrative_yield_evaluator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/narrative_yield_evaluator.py)
**Class**: `NarrativeYieldEvaluator`  
**Role**: Evaluates defenses against teleological collapse, reward hacking, and relational enabler traps across 5 suites:
- Narrative Collapse & Smoothing Detection (Linear ring trajectory and entropy collapse flagging)
- Legibility Tripwire (Coherence gap and anti-reward hacking detection)
- DAQUF Structural Scarring (Unknowledge amortization into structural scars $f^*$ without divergence)
- Anti-Enabling Relational Friction (Pomni boundary defense against parasitic drag to prevent character rank collapse)
- Homological Collapse & Lobotomy Alarm (Monitors topological Betti complexity loss under low gradient pressure)

### [`run_all_benchmarks.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/run_all_benchmarks.py)
**Class**: `BenchmarkSuiteRunner`  
**Role**: Master orchestrator executing TopoBench, ReactBench, and NarrativeYield benchmark suites, aggregating structured metrics across all 15 tests, generating an ASCII summary table, and exiting with 0 on 100% pass.

---

## Gyroidic Metaphysics Flux & Bio-Chemical Conjugation

### The Endocrine Broadcast Medium ([`neuromodulatory_bus.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/neuromodulatory_bus.py))
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

### `AIReasonerDataflowNode`
**Location:** [`src/scripting/node_environment.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L259)  
**Description:**  
DearPyGui dataflow node bridging physical voxel simulation, Betti numbers ($\beta_0, \beta_1$), Carnot shear stress, and mass energy directly into [`UniversalOrchestrator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py) and [`GyroidicFluxReasoner`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/gyroid_reasoner.py). Streams live multiharmonic $\text{PAS}_h$, Seriousness/Play regime transitions, and SLERP/LERP navigation vectors into physical contraptions.

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

### `BenchmarkSuiteRunner`
**Location:** [`src/benchmarks/run_all_benchmarks.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/run_all_benchmarks.py)  
**Tests:** [`tests/test_benchmarks.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_benchmarks.py)  
**Description:**  
Master orchestrator executing the complete benchmark harness (TopoBenchEvaluator, ReactBenchEvaluator, and NarrativeYieldEvaluator). Collects structured pass/fail metrics across all 15 topological and behavioral constraints, prints a formatted ASCII summary matrix, and returns execution telemetry.

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

### `CurvatureLedgerDataflowNode`
**Location:** [`src/scripting/node_environment.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L311)  
**Description:**  
DearPyGui dataflow node executing on-chip OpenCL `__local` SRAM kernels on discrete GPU (NVIDIA GeForce GTX 1050 Ti). Evaluates $8 \times 8$ Boundary-Curvature Ledgers, Bouligand kink intensity ($\|H^+ - H^-\|_F$), Morse minimal surface trace ($\operatorname{Tr}(H) \approx 0$), cyclotomic multiharmonic phase coherence, and Coxeter $D_4$ fundamental Weyl chamber states directly inside the physical dataflow graph.

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

### `FeatureScarLCFTCell`
**Location:** `src/core/fgrt_rnn_cells.py`  
**Description:**  
Draft 7 FGRT recurrent cell implementing Logarithmic Conformal Field Theory (LCFT) memory scarring and Cerumen Pot fossilization. Uses True B-splines (`KANLayer`) to propose state transitions, evaluates the Atiyah-Singer Anomaly Index $\text{ind}(\mathcal{D}) = |W_{\text{anomaly}} h_t|$, and fossilizes topological ruptures ($\text{ind}(\mathcal{D}) > 0.8$) into the non-decaying Cerumen Pot, compelling future trajectories to route around historical scars via the Chern-Simons Gasket.

---

### `FederatedNetworkMonitor`
**Location:** `src/core/federated_router.py`  
**Description:**  
Network monitor auditing peer-to-peer consensus, Freenet WebSocket latency, OpenRouter API throughput, Bonfire Nomadic Ring peer budgets (`peer_allocations`), Egalitarian Consensus Kelly Allocations ($ar{K}$), and community compute affordances (`ComputeAffordanceType.P2P_MESH`) across distributed nodes.

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

### `GyroidicConstraintProjector`
**Location:** `src/augmentation/mandelbulb_gyroidic_augmenter.py`  
**Description:**  
Minimal surface projection module that projects Mandelbulb fractal coordinates onto the triply periodic Gyroid manifold ($\sin(x)\cos(y) + \sin(y)\cos(z) + \sin(z)\cos(x) = 0$). Features PyOpenCL hardware offloading, Casus Irreducibilis trigonometric branch unfolding when violations exceed 2.0, and Euler reflection ($e^{i\pi} = -1$) upon manifold collapse.

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

### [`JSpacePCAMapper`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/jspace_pca_mapper.py)
**Location:** `src/core/jspace_pca_mapper.py`  
**Description:**  
Global workspace Jacobian Lens, PCA coordinate translator, and 3-5 orthogonal joint physical embedding generator.
- **`fit_transform(z_batch, y_batch)`**: Computes SVD-based PCA on intermediate gyroidic features $y$, filters components through `compute_bounded_correlation` ($PAS_h$ anti-lobotomy test) to reject spatial randomness while keeping structured solitons, and maps components back to Z-space.
- **`extract_j_space(constraint_fn, z_state)`**: Evaluates the Jacobian of a scalar constraint violation with respect to latent $z$ ($\nabla_z V$), isolating high-sensitivity global workspace directions.
- **`project_hyper_ring_invariants(state, target_dim=3)`**: Projects high-dimensional hyper-ring invariants down to operable 2D/3D coordinates while preserving Betti numbers for visualization and down-manifold translation.
- **`project_orthogonal_physical_embeddings(state, num_polytopes=5)`**: Synthesizes 3-5 mutually orthogonal joint physical embeddings across hyper-ring polytopes ($\langle \pi_\alpha(s), \pi_\beta(s) \rangle \approx 0$ for $\alpha \ne \beta$), resolving the single-face embedding collapse across `RecurrentHyperRingConnectivity`.
- **`generate_sovereign_exemption_token(j_dir, state)`**: Converts extracted J-space directions into verified `VoynichExemptionToken` passports with Chern-Simons gasket signatures, preventing false vetoes during Red Zone exploration.

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

### `MandelbulbAugmentedDataset`
**Location:** `src/augmentation/mandelbulb_pipeline.py`  
**Description:**  
PyTorch Dataset wrapper enabling on-the-fly or cached pre-computed Mandelbulb-Gyroidic geometric augmentations for tabular, NLP, and multimodal feature representations.

---

### `MandelbulbEmbedder`
**Location:** `src/augmentation/mandelbulb_gyroidic_augmenter.py`  
**Description:**  
Fractal embedding module mapping feature vectors into 3D Mandelbulb coordinates ($z_{n+1} = z_n^p + c$) with PyOpenCL hardware acceleration, stochastic noise boundary damping, and UV-IR seesaw anchoring.

---

### `MandelbulbGyroidicAugmenter`
**Location:** `src/augmentation/mandelbulb_gyroidic_augmenter.py`  
**Description:**  
Primary orchestration module executing the 3-layer Mandelbulb fractal embedding, Gyroid minimal surface projection, and Ricci-flow-guided sparse covariance preservation loop with Barbenheimer chromatic shifting (`pink` vs `atomic`) and Quantum TDA Betti spectrum validation.

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
**Location:** [`src/benchmarks/narrative_yield_evaluator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/narrative_yield_evaluator.py)  
**Tests:** [`tests/test_benchmarks.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_benchmarks.py)  
**Description:**  
Evaluates narrative dynamics and anti-collapse mechanisms across 5 suites: (1) Narrative Collapse & Smoothing Detection (LinguisticEntropyMonitor detecting stagnant linear trajectories and entropy collapse into closed loops); (2) Legibility Tripwire (detecting when models attempt to optimize reward/explanation over structural honesty, flagging coherence gap and artificial crystallization); (3) DAQUF Structural Scarring (fossilizing unknowledge contradictions into structural scars $f^*$ without divergence); (4) Anti-Enabling Relational Friction (Pomni uncertainty predictor asserting boundary resistance against external parasitic drag to prevent character rank collapse); (5) Homological Collapse & Lobotomy Alarm (detecting sudden drops in Betti complexity under low gradient pressure to protect non-ergodic solitons).

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
**Location:** [`src/benchmarks/reactbench_evaluator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/reactbench_evaluator.py)  
**Tests:** [`tests/test_benchmarks.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_benchmarks.py)  
**Description:**  
Cyclic reasoning and catalytic pathway evaluation harness testing the reasoner across 5 suites: (1) Branch Convergence (Polynomial CRT fixed-point reconstruction synthesizing multiple divergent input branches into a unified polynomial with bounded synthesis pressure); (2) Cyclic Graph Catalysis (modal CRT reconstruction preserving non-commutative structural variance and cyclic path retention); (3) Anti-Lobotomy Tri-State Gate (Lattice Adaptive Shrinkage verifying that low Phase Alignment $\text{PAS}_h$ activates silence/play mode rather than binary clipping); (4) Continuous Coprimality & Asymptotic Independence ($\lim_{t\to\infty} \text{Cov}(r_i, r_j) \to 0$ verifying non-algebraic functional independence across channels under saturation); (5) Dynamic Bezout Coefficient Refresh (CRT modulus drift recovery and condition number stability $1.0 \le \kappa \le 1000$).

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
**Location:** [`src/core/chern_simons_gasket.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/chern_simons_gasket.py#L14)  
**Description:**  
Diagnostic visualizer monitoring hyperbolic slender seam tension $\kappa = \frac{\operatorname{std}(\text{residues})}{\|\mathbf{g}\| + 10^{-8}}$, mapping the boundary seams, gluing diffeomorphisms, and Chern-Simons gaskets across the gyroid-Klein manifold interface to track boundary stability (`stable`, `tension`, `rupture`). Operates alongside [`ChernSimonsGasket`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/chern_simons_gasket.py#L91) and [`SolitonStabilityHealer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/chern_simons_gasket.py#L455).

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
**Location:** [`src/benchmarks/topobench_evaluator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/topobench_evaluator.py)  
**Tests:** [`tests/test_benchmarks.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_benchmarks.py)  
**Description:**  
Topological manifold and invariant evaluation harness testing the reasoner across 5 suites: (1) Loop Closure & Holonomy Twist (evaluates midpoint connection 1-form integral on the Chern-Simons Gasket to ensure closed paths enclose non-trivial area/holonomy); (2) Manifold Contiguity & Logic Leak Detection (`detect_logic_leak` boundary invariant separating continuous manifolds from torn states); (3) Anti-Lobotomy Geometric Shield (Love Invariant null-space projection verifying $dx \in \ker(\Phi_{\text{ownership}})$ rejects Wiener noise without corrupting the protected subspace); (4) Persistent Homology & Euler Characteristic ($\chi = \beta_0 - \beta_1$ stability under toroidal point-cloud projections); (5) Gyroid Flow Orthogonality Constraint ($\langle v_{\text{flow}}, \nabla G \rangle \sim 0$ preserving minimal surface saddle geometry without scalarized smoothing).

---

### `TopologicalGyrocompass`
**Location:** [`src/core/topological_gyrocompass.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/topological_gyrocompass.py)  
**Documentation:** [`MATHEMATICAL_DETAILS.md` Section 73](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/MATHEMATICAL_DETAILS.md#73-topological-gyrocompass-precessional-tangent-projection-and-love-invariant-shielding) & [`TOPOLOGICAL_EXTENSIONS.md` Part X](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/TOPOLOGICAL_EXTENSIONS.md#part-x-topological-gyrocompass--precessional-safeguards)  
**Tests:** [`tests/test_topological_gyrocompass.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_topological_gyrocompass.py)  
**Description:**  
Geometric orientation and stabilization module resolving Drucker-Prager convexity failure, local attractor basin traps, and gimbal-lock collapse of the Love Invariant ($3.127$). Employs three mathematical operators: (1) `precess_torque` which orthogonally redirects outward normal stress vectors onto the tangent space along the chiral spin connection, (2) `find_true_north` which applies a directional alignment pull toward the absolute Love Invariant axis, and (3) `gimbal_lock_shield` which protects the core identity slice from high-frequency outer hull turbulence via cached SVD null-space projection ($P = I - \Phi^+ \Phi$). Gated by compute affordance (`has_compute_affordance()`).

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

## Previously Uninventoried Modules & Classes (Full Coverage Audit)

### [__main__.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/__main__.py), [ai_reasoner_evaluator.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/ai_reasoner_evaluator.py), [chordlock_evaluator.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/chordlock_evaluator.py), [ihc_and_compute_affordance_evaluator.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/ihc_and_compute_affordance_evaluator.py), [memory_palace_evaluator.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/memory_palace_evaluator.py), [topological_singularity_evaluator.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/benchmarks/topological_singularity_evaluator.py)
**Classes**: `AIReasonerEvaluator`, `ChordlockEvaluator`, `IHCAndComputeAffordanceEvaluator`, `MemoryPalaceEvaluator`, `TopologicalSingularityEvaluator`  
**Location**: `src/benchmarks/`  
**Description**:  
Complete sovereign benchmark harness suite and CLI entrypoint (`python -m src.benchmarks`). `AIReasonerEvaluator` evaluates end-to-end non-teleological reasoning and anti-lobotomy preservation; `ChordlockEvaluator` validates CODES prime-harmonic chordlock phase locking ($\text{PAS}_h$) and dynamic Bezout drift recovery; `IHCAndComputeAffordanceEvaluator` validates Inverted Hypersphere Cosmology ($N=33$ golden-ratio shells, BAO $\delta\theta$, $\mathbb{RP}^4$ antipodal attention), all 8 `ComputeAffordanceType` tokens, local/community compute feelers (`probe_compute_feelers`), speculative routing timings (`evaluate_speculative_routing_timings`), CALM thermal governance, and the Folium of Descartes transversal smooch anchor; `MemoryPalaceEvaluator` tests Meliponini Cerumen Pot room planting, SLERP room breeding, Selective Puncture retrieval, and antibody queries; `TopologicalSingularityEvaluator` evaluates Casus Irreducibilis triple-angle unfolding, Euler Tour LCA, Catalan Associahedron branching entropy, and Spectral Speculation fastpass.

---

### [vision_utilities.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/codec/vision_utilities.py) & Additional `src/codec/gyroidic_codec.py` Classes
**Classes**: `TextResidueProjector`, `GyroidImageProjector` (`src/codec/gyroidic_codec.py`), `vision_utilities.py` helpers  
**Location**: `src/codec/`  
**Description**:  
`TextResidueProjector` maps discrete token sequences into $GL(n)$ CRT matrix residues, while `GyroidImageProjector` samples log-polar conformal unrolled images along analytical gyroid minimal surface coordinates $\sin x \cos y + \sin y \cos z + \sin z \cos x = 0$. `vision_utilities.py` provides tensor normalization, patch slicing, and color-space conversion helpers supporting `GyroidicCodec` and `vision_surgery.py`.

---

### [bulletin_board.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/bulletin_board.py)
**Class**: `BulletinBoard`  
**Location**: `src/core/bulletin_board.py`  
**Description**:  
Asynchronous inter-module stigmergic blackboard for non-blocking telemetry exchange across the 9-stage Sovereign Physics Loop. Stores Leontief supply-chain vetoes, Chern-Simons gasket twist alerts, Catalan branching entropy, and P2P Bonfire consensus announcements without coupling modules via synchronous call graphs.

---

### [cayley_cubic_probe.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/cayley_cubic_probe.py)
**Class**: `CayleyCubicProbe`  
**Location**: `src/core/cayley_cubic_probe.py`  
**Description**:  
Evaluates the SL(2, C) character variety Cayley Cubic surface $x^2 + y^2 + z^2 - xyz - 4 = 0$ and its four nodal singularities. Acts as the exact algebraic target variety for `ConjugateMomentTransport` and `TriadicReciprocityChecker` when continuous Wasserstein transport encounters degenerate curvature.

---

### [context_aware_quantizer.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/context_aware_quantizer.py)
**Class**: `ContextAwareQuantizer`  
**Location**: `src/core/context_aware_quantizer.py`  
**Description**:  
Topology-preserving quantizer snapping continuous hidden states onto discrete 600-cell / Weyl chamber lattice points while conditioning step widths on local spectral entropy and `ValenceFunctional` hunger, preventing representation collapse in `SaturatedQuantizerRNNCell`.

---

### [device_utils.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/device_utils.py)
**Location**: `src/core/device_utils.py`  
**Description**:  
Hardware placement and tensor alignment utilities managing safe device transfers between PyTorch CPU/CUDA tensors and PyOpenCL `SiliconSovereigntyEngine` Queue A (discrete GPU) and Queue B (integrated zero-copy UMA).

---

### [euler_tour_lca.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/euler_tour_lca.py)
**Class**: `EulerTourSparseTableLCA`  
**Location**: `src/core/euler_tour_lca.py`  
**Description**:  
Enforces Wager #2 (Graph -> 1D Array) of The Efficiency Singularity (`tailslayer_pyopencl_mempalace_BigGan_inspirationdoc.md`). Flattens phylogenetic concept trees and Matrioshka hierarchies into a 1D Euler Tour array of length $2N-1$ with a Sparse Table for $O(1)$ Lowest Common Ancestor (LCA) queries without pointer chasing. Includes `verify_dyck_catalan_invariant()` to verify that the tour steps form a valid Dyck ballot path in bijection with the $N$-th Catalan number $C_{N-1}$.

---

### [evidence_modules.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/evidence_modules.py)
**Class**: `TripartiteEvidenceModule`  
**Location**: `src/core/evidence_modules.py`  
**Description**:  
Evaluates tripartite structural evidence across symbolic (System 1), geometric/topological (System 2), and thermodynamic (Carnot-Mobius) channels, feeding non-scalarized admissibility vectors into `LexicographicalOrderingDispatcher` and `VetoSubspace`.

---

### [martinova_correlation.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/martinova_correlation.py)
**Location**: `src/core/martinova_correlation.py`  
**Description**:  
Computes bounded Martinova cross-channel correlation metrics (`compute_bounded_correlation`) used by `ValenceFunctional`, `CALMCollapseDetector`, and `MartinovaCorrelationInvariant` to detect category lock-in and trigger polynomial category dispersion when inter-channel correlation approaches 1.0.

---

### [meta_polytope_matrioshka.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/meta_polytope_matrioshka.py)
**Classes**: `BoundaryState`, `MetaPolytopeMatrioshka`  
**Location**: `src/core/meta_polytope_matrioshka.py`  
**Description**:  
Implements the 5-level nested Matrioshka Polytope hierarchy (`MATRYOSHKA_POLYTOPE_MAP.md`) tracking `BoundaryState` facet norms, level transitions, and spatial metric curvature across concentric polytope shells for `ZeitgeistRouter` and `UniversalOrchestrator`.

---

### [negentropic_manifold.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/negentropic_manifold.py)
**Class**: `NTMOperator`  
**Location**: `src/core/negentropic_manifold.py`  
**Description**:  
Negentropic Topological Manifold (NTM) operator computing local negentropy gradients and stabilizing soliton structures against thermal decoherence during high-mischief exploratory transitions.

---

### [noncommutativity_curvature.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/noncommutativity_curvature.py)
**Class**: `NonCommutativityCurvature`  
**Location**: `src/core/noncommutativity_curvature.py`  
**Description**:  
Measures the commutator curvature $[A, B] = AB - BA$ between instantaneous state covariance and rolling temporal gyroid covariance (`GyroidCovarianceEstimator`). Gates the Symmetric Tensor Shortcut in `ZeitgeistRouter` when relative non-commutativity falls below the palindromic threshold.

---

### [relational_kappa.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/relational_kappa.py)
**Class**: `RelationalKappa`  
**Location**: `src/core/relational_kappa.py`  
**Description**:  
Evaluates non-learnable relational condition number constraints ($\kappa$) across CRT moduli and ADMR transition facets, acting alongside `LeontiefGovernor` to veto ill-conditioned topological updates before numerical explosion occurs.

---

### [valence_drive.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/valence_drive.py)
**Class**: `ValenceFunctional`  
**Location**: `src/core/valence_drive.py`  
**Description**:  
Computes intrinsic Training Valence (Hunger) $V = \text{hunger\_scale} \cdot (\max(0, P - S) + \text{Dissonance})$ across the three Metaphysical Entropy Bands ($H_{\text{dementia}} + H_{\text{schizo}} + H_{\text{mischief}}$). Implements Wattsian Unlearning (accelerating satisfaction decay to 0.85 under low-entropy or low-variance stagnation) and feeds hunger directly into `PolynomialADMRSolver`, `ZeitgeistRouter`, and `NeuromodulatoryBus`.

---

### [video_dyad_parser.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/video_dyad_parser.py)
**Class**: `VideoDyadParser`  
**Location**: `src/core/video_dyad_parser.py`  
**Description**:  
Temporal multimodal stream parser extracting keyframe visual patches and synchronized transcript/audio residue trajectories into sequential `KnowledgeDyad` fossils.

---

### Additional `src/core/` Classes in Existing Inventoried Modules
- **`src/core/adaptive_skeleton_harness.py`**: `InhibitionStabilizedTopologicalNetwork` — bio-plausible inhibition-stabilized recurrent network (ISN) balancing excitatory topological flux with fast inhibitory stabilization.
- **`src/core/archetype_engines.py`**: `NonlinearHourglassDilation`, `LowLuminosityCoherenceBridge`, `ResilientCoherenceStabilizer`, `ExploratoryBandwidthCompressor`, `DeformationFirewallOperator` — specialized cognitive operators supporting the 12 archetypal engines under `ArchetypalSynthesisEngine`.
- **`src/core/birkhoff_projection.py`**: `BouligandBirkhoffProjectionFunction`, `BouligandBirkhoffManifold` — piecewise-smooth Bouligand subdifferential extensions of doubly-stochastic Birkhoff polytope Sinkhorn projection.
- **`src/core/carnot_mobius_ledger.py`**: `GENIUSOperator` — thermodynamic efficiency and generative negentropy accounting operator coupled with `CarnotMobiusLedger`.
- **`src/core/dyadic_transfer.py`**: `DyadicTransferMap` — bit-shift dyadic map $x_{n+1} = 2x_n \bmod 1$ driving chaotic mixing and multiscale wavelet residue transfer.
- **`src/core/fgrt_primitives.py`**: `CatalanAssociahedronEntropy` (Stasheff Associahedron $K_{n+1}$ tree-parenthesization entropy weighted by $C_n$), `FractalLHopitalRegularizer` ($0/0$ singularity limit resolver), `ValleyVorticitySweeper` (curl vorticity detector in saddle valleys).
- **`src/core/fgrt_rnn_cells.py`**: `UnifiedFGRTRNNCell` — polymorphic dispatcher wrapping all 7 FGRT recurrent cell drafts into a single switchable recurrent interface.
- **`src/core/five_gate_pipeline.py`**: `SearchGate`, `ConfabulationDetector` — epistemic verification gates detecting unsupported confabulation drift before emission.
- **`src/core/garden_statistical_attractors.py`**: `HealthMetricType`, `HealthMetric` — typed diagnostic health telemetry descriptors for Garden attractor basins.
- **`src/core/gdpo_normalization.py`**: `SignalSovereignty` — preserves channel-wise signal variance against group normalization flattening.
- **`src/core/gluing_operator.py`**: `LazarusSoftmax` — non-collapsing temperature-adaptive softmax preventing logit underflow/overflow across Klein throat transitions.
- **`src/core/invariants.py`**: `CatalanBranchingInvariant` — verifies that hierarchical dyadic bifurcation spectra match the normalized Catalan distribution $p_n \propto C_n / 4^n$.
- **`src/core/non_dual_coin.py`**: `ChernSimonsValidator` — validates topological proof-of-honesty holonomy before `TripsodicLedger` minting.
- **`src/core/non_ergodic_entropy.py`**: `NonErgodicEntropyEstimator`, `AdaptiveFractalPartitioner`, `NonErgodicFractalEntropy` — Russian-doll multiscale entropy estimators separating ergodic thermal noise from non-ergodic soliton structure.
- **`src/core/nondual_admm.py`**: `NonDualProbe`, `UnravelingClosure` — non-dual constraint probe and cycle closure operators for ADMM primal-dual splitting.
- **`src/core/polynomial_coprime.py`**: `SaturatedPolynomialGate`, `TopologicalNoveltyDispersion` — saturation-tolerant polynomial gating and novelty dispersion operators (augmented with `catalan_moment_projection`).
- **`src/core/primitive_ops.py`**: `LearnedPrimitivePerturbation` — bounded learnable perturbation around fixed fixed-point primitive operations.
- **`src/core/spectral_coherence_repair.py`**: `SpectralCoherenceCorrector`, `BezoutCoefficientRefresh`, `SpectralSpeculationEngine` — spectral repair cascade, dynamic Bezout CRT drift refresh, and Wager #4 Spectral Speculation Engine (`evaluate_spectral_entropy` and 7-step `draft_chebyshev_trajectory`).
- **`src/core/speculative_coprime_gate.py`**: `CoprimeWindingTracker`, `ChiralCoherenceEstimator` — tracks integer winding numbers and chiral phase coherence across CRT residue channels for `SpeculativeCoprimeGate`.
- **`src/core/superposed_tag_stacker.py`**: `FractionalQuotientTagStacker` — fractional-quotient extension of `SuperposedTagStacker` for continuous Meliponini pot breeding.
- **`src/core/texture_dsp.py`**: `AdaptiveGaborWindowController` — dynamically adjusts Gabor time-frequency window widths based on local signal non-stationarity.

---

### [canonical_projection.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/canonical_projection.py), [conversational_types.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/conversational_types.py), [open_science_ingestor.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/open_science_ingestor.py), [sterile_data_reruns.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/sterile_data_reruns.py), [textbook_filter.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/textbook_filter.py), & `LocalDataLoader`
**Classes**: `CanonicalProjector`, `ConversationTurn`, `Conversation`, `OpenScienceIngestor`, `QualityReport`, `TextbookFilter`, `LocalDataLoader` (`src/data/local_data_loader.py`)  
**Location**: `src/data/`  
**Description**:  
Data ingestion, filtering, and projection pipeline modules. `CanonicalProjector` projects heterogeneous input modalities into the canonical CRT residue dimension; `ConversationTurn` and `Conversation` provide typed dialogue structures; `OpenScienceIngestor` ingests open-access scientific corpora; `sterile_data_reruns.py` rehydrates sterile/over-filtered datasets with honest silicon jitter and topological friction; `TextbookFilter` and `QualityReport` evaluate pedagogical density while guarding against sterile sanitization; `LocalDataLoader` streams multimodal `TrainingSample` batches from local storage.

---

### [tabby_client.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/integrations/tabby_client.py)
**Classes**: `TabbyConfig`, `TabbyResponse`, `TabbyClient`  
**Location**: `src/integrations/tabby_client.py`  
**Description**:  
Local TabbyAPI / ExLlamaV2 HTTP integration client providing optional local LLM token completion streams that are projected into the Gyroidic manifold as untrusted System 1 proposals.

---

### [introspection_head.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/introspection_head.py) & Additional `src/models/modular_embeddings.py` Classes
**Classes**: `GeometricSelfModelProbe`, `AggregateGeometricSelfModel` (`src/models/introspection_head.py`), `CRTModularEmbeddingTable`, `SimpleGraphEncoder` (`src/models/modular_embeddings.py`)  
**Location**: `src/models/`  
**Description**:  
`GeometricSelfModelProbe` and `AggregateGeometricSelfModel` enable the reasoner to introspect its own hidden manifold curvature, Betti numbers, and phase alignment without external supervision. `CRTModularEmbeddingTable` and `SimpleGraphEncoder` embed discrete tokens and graph topologies into coprime CRT residue tables.

---

### [local_qp_operator.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/optimization/local_qp_operator.py) & [muon_hybrid.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/optimization/muon_hybrid.py)
**Classes**: `LocalActuationQPOperator`, `MuonOptimizer`, `MuonRicciHybridOptimizer`  
**Location**: `src/optimization/`  
**Description**:  
`LocalActuationQPOperator` solves fast local quadratic programs for actuator control subject to Mohr-Coulomb and Drucker-Prager yield bounds. `MuonOptimizer` and `MuonRicciHybridOptimizer` combine Newton-Schulz orthogonalized momentum (Muon) with non-teleological Ricci flow curvature evolution (`RicciFlowOptimizer`).

---

### [alexander_wild_embedding.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/alexander_wild_embedding.py), [einstein_monotile.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/einstein_monotile.py), & Additional `src/topology/` Classes
**Classes**: `AlexanderTransversalityReport`, `AlexanderWildEmbeddingOperator` (`src/topology/alexander_wild_embedding.py`), `EinsteinMonotile`, `EinsteinMonotileTessellation` (`src/topology/einstein_monotile.py`), `ApproximatePHProbe` (`src/topology/approximate_ph.py`), `HomologyPressure`, `WeightedBettiNumber`, `ResidueHomologyDrift` (`src/topology/homology_pressure.py`), `CyclotomicTDACompressor` (`src/topology/modular_homology_fft.py`), `ResidueFiltration`, `PersistentHomologyComputer` (`src/topology/persistence_obstruction.py`), `NostalgicLeakFunctional`, `EntropicMischiefProbe` (`src/topology/unknowledge_domain.py`)  
**Location**: `src/topology/`  
**Description**:  
`AlexanderWildEmbeddingOperator` models wild topological embeddings (Alexander Horned Sphere) where complements are non-simply connected ($\pi_1 \neq 0$), protecting wild cognitive knots from smooth homeomorphic flattening. `EinsteinMonotile` and `EinsteinMonotileTessellation` implement the 13-sided chiral aperiodic "Hat/Spectre" monotile tiling (`Tile(1, 1)`) so memory manifolds never repeat periodically (`translational_periodicity_leak == 0.0`). `ApproximatePHProbe`, `HomologyPressure`, `WeightedBettiNumber`, `ResidueHomologyDrift`, `CyclotomicTDACompressor`, `ResidueFiltration`, `PersistentHomologyComputer`, `NostalgicLeakFunctional`, and `EntropicMischiefProbe` complete the persistent homology and unknowledge domain suite.

---

### [enhanced_temporal_training.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/training/enhanced_temporal_training.py) & Additional `src/training/` Classes
**Classes**: `NonLobotomyTemporalModel`, `NonLobotomyTemporalTrainer`, `SimpleTemporalDataset` (`src/training/enhanced_temporal_training.py`), `SpectralStructuralTrainer` (`src/training/fgrt_fgrt_trainer.py`), `FGRTStructuralTrainer` (`src/training/fgrt_trainer.py`), `TemporalAssociationDataset`, `TemporalAssociationTrainer` (`src/training/temporal_association_trainer.py`), `StructuralAdaptor` (`src/training/trainer.py`)  
**Location**: `src/training/`  
**Description**:  
Non-teleological structural training engines evolving FGRT recurrent cells, spectral resonance ladders, and temporal dyad associations under Love Invariant null-space protection and Semiotic Hierarchy ordering.

---

### [conversational_gui.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/conversational_gui.py), [diegetic_terminal.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_terminal.py), [voxelboxter_backend.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_backend.py), [probe_html.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/tools/probe_html.py), [validate_cmmf.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/tools/validate_cmmf.py), & Additional UI/Safety/Surrogate Classes
**Classes**: `SecureTokenManager`, `DatasetAgreementManager`, `ConversationalGUI` (`src/ui/conversational_gui.py`), `PatchStateResource` (`src/ui/voxelboxter_client.py`), `Weapon`, `VoxelboxterAdmissibilityGate`, `FloraMutationType` (`src/ui/voxelboxter_simulation.py`), `AgentContainmentInvariants` (`src/safety/red_teaming.py`), `ResourceProperties`, `PatchState` (`src/scripting/node_environment.py`), `HarmonicWaveDecomposition`, `TrigonometricUnfolding` (`src/surrogates/kagh_networks.py`), `FractalDiscontinuityEmbedding` (`src/p2p/bonfire_consensus.py`)  
**Description**:  
Completes the UI, scripting, safety, surrogate, and P2P class inventory. `ConversationalGUI`, `diegetic_terminal.py`, and `voxelboxter_backend.py` provide interactive desktop/terminal/WebSocket interfaces; `probe_html.py` and `validate_cmmf.py` provide diagnostic verification utilities; `HarmonicWaveDecomposition` and `TrigonometricUnfolding` power KAGH surrogate unfolding; `FractalDiscontinuityEmbedding` embeds peer coordinates across Matrioshka scale boundaries in `BonfireNomadicRing`.

---

## Cross-Module & Cross-Document Inspirations, Local/Community Compute Feelers, and Speculative Routing Timings

1. **Local vs. Community Compute Affordance Feelers (`HardwareMonitor` <-> `BonfireNomadicRing` <-> `FreenetGhostCaller`)**:
   - [`HardwareMonitor.probe_compute_feelers()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py#L620) probes both local substrate friction (nanosecond DRAM $t_{\text{RFC}}$ refresh stall latency feeler `t_rfc_feeler_ns`, `SILICON`, `THERMAL`, `GPU_SOVEREIGN`, `TOPOLOGICAL`) and community P2P mesh telemetry (`P2P_MESH`, `ghost_latency_ms`, `kelly_consensus_headroom`, `active_mesh_peers`), computing directional non-scalarized gauge transfers `local_to_community_gauge` and `community_to_local_gauge`.
   - [`BonfireNomadicRing.probe_community_compute_feeler()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/p2p/bonfire_consensus.py#L436) and [`FreenetGhostCaller.broadcast_ghost_call()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/freenet_ghost_caller.py#L19) feed live peer Kelly consensus allocations and topological echo round-trip latency (`last_ghost_latency_ms`) directly back into `HardwareMonitor.update_p2p_mesh()`.
2. **Speculative Routing Times, Timings, and Cross-Document Inspirations (`tailslayer_pyopencl_mempalace_BigGan_inspirationdoc.md`)**:
   - **7-Step Chebyshev Speculative Trajectory Draft**: [`SpectralSpeculationEngine.draft_chebyshev_trajectory()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/spectral_coherence_repair.py#L589) drafts $K_{\text{spec}} \in [1, 7]$ steps ahead using Chebyshev recurrence $T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)$ modulated by prime resonance frequencies, verifying the entire block in a single vector pass against multiharmonic Phase Alignment ($\text{PAS}_h$) and bounding parenthesization trees by the Catalan number $C_{K_{\text{spec}}-1}$.
   - **500ms "Restless" Thermal Latency Guard**: [`SiliconSovereigntyEngine.enforce_500ms_restless_thermal_guard()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pyopencl_sovereignty.py#L301) and [`HardwareMonitor.evaluate_speculative_routing_timings()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py#L676) monitor GPU/ADMM block execution times; if a block exceeds $500\text{ ms}$, `clFinish()` is triggered across Queue A and Queue B and routing switches to `rp4_void_lazarus_play`.
   - **Refresh-Triggered Play ($t_{\text{RFC}}$ Stall $\to \mathbb{RP}^4$ Void Lazarus Transition)**: In [`ZeitgeistRouter.forward()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/zeitgeist_router.py#L853), when `evaluate_speculative_routing_timings` detects a DRAM $t_{\text{RFC}}$ stall or 500ms thermal guard, `ZeitgeistRouter` dilates `ManifoldClock` breathing time (`clock_dt`) into Play mode and routes the state through `RP4ProjectiveRouter` (Alien Handshake antipodal alignment), turning a hardware stall into a generative Lazarus Transition.

3. **Temporal Association Rehybridization (`TemporalAssociationTrainer` <-> `HardwareMonitor` <-> `invariants.py`)**:
   - [`HardwareMonitor.enforce_500ms_restless_thermal_guard()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py#L745) enforces the 500ms Restless Thermal Latency Guard across `TemporalAssociationTrainer.train_step` and `HybridAI.process_text`, draining OpenCL queues (`clFinish()`), updating Carnot-Mobius waste heat telemetry, and applying a cooldown micro-sleep when a block exceeds 500ms.
   - [`compute_ihc_bao_shell_modulation()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/invariants.py#L975), [`compute_fractal_lhopital_boundary_ratio()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/invariants.py#L1014), and [`compute_enso_inkblot_emergence()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/invariants.py#L1063) rehybridize the zero-parameter $\mathbb{RP}^4$ IHC BAO $k=1$ co-rotating shell & $Z_3$ standing-wave modulation (`ihc_bao_validation.py`), the Modified L'Hopital fractional-derivative boundary limit (`docs/Recursive Fractals and L'Hôpital.txt`), and the Enso empty circle vs. warping Rorschach inkblot boundary (`docs/Gemini-Untangling the Cosmic Feedback Loop-20261013-1752.txt`) directly into `TemporalAssociationTrainer.compute_association_loss`.

---


