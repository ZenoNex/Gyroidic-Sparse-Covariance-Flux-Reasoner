# Open Questions & Theoretical Boundaries

This document formalizes the technical boundaries and unresolved mathematical gaps in the **Gyroidic Flux Reasoner** architecture. To maintain structural honesty, we acknowledge where the system relies on heuristics or approximations.

---

## 1. Symbolic Co-Primality vs. Transversality
**Challenge**: Exact algebraic co-primality is lost once functionals are saturated and evolved.
**Formalization**: We reframe co-primality as **Symbolic Transversality**—a property of "generic position under saturation."
- **Definition**: A set of functionals is symbolically co-prime if no finite projection produces a residue merge that survives selection.
- **Invariant**: We optimize for **non-mergeability under projection** across fractal clusters rather than $GCD=1$ in a ring.

## 1.1 Initialization Sensitivity
**Challenge**: Without warm-start data, how do we bootstrap functionals $\theta_k$?
**Refinement**: 
- **Orthogonal Bootstrapping**: Initialize with random orthogonal coefficients (via Gram-Schmidt) to ensure initial co-primality.
- **Biomimetic Priors**: For gyroid probes, initialize with ideal minimal-surface covariance patterns rather than Gaussian noise.

---

## 2. The Failure-Space Topology
**Challenge**: Defining a residue-space adjacency graph for homological tracking is heuristic.
**Formalization**: We reframe the **Residue Obstruction Graph** as a **Bipartite Obstruction Complex**.
- **Nodes**: One set of nodes represents symbolic residues; the other represents violated physical constraints $\{\psi\}$.
- **Adjacency**: $i \sim \psi \iff$ Residue $i$ contributes to violation $\psi$.
- **Tracking**: Betti numbers are tracked over the **space of breakdowns** rather than the space of representations. This provides persistence generators for inconsistencies across scales.

---

## 3. Gyroid Violation: Necessary but Not Sufficient
**Challenge**: The gyroid violation score $\psi$ is mathematically "loose."
**Formalization**: $\psi$ is a **necessary but not sufficient filter** for physical admissibility.
- **Analogy**: It functions like a CFL condition in a PDE solver—it does not provide a truth, but it defines the **boundary of the forbidden**.
- **Epistemic Role**: It identifies *that* a symbolic embedding is inadmissible, triggering a veto, without claiming to know the "correct" manifold realization.
- **Theoretical Boundary**: There is no algebraic proof of completeness for the gyroid metric. We accept this "sane" heuristic but require **empirical validation**: the system must detect >95% of simulated manifold tears.

---

## 4. Continuous Co-primality & Transversality
**Status**: ✅ **IMPLEMENTED (Phase 3)**

**Implementation**: `ContinuousCoprimality` uses discrete entropy quantization (binary outcomes, bincount, log2) to compute entropy pressure:
- $E(r_i, r_j) = H(r_i + r_j) - H(r_i) - H(r_j)$
- Checks asymptotic independence: $\lim_{t\to\infty} \text{Cov}(r_i^{(t)}, r_j^{(t)}) = 0$

**Entropy Quantization**: Uses the same discrete method as `HypergraphOrthogonalityPressure`:
- Binary quantization: `values > 0` → discrete outcomes
- Integer keys via powers of 2
- Bincount for discrete probabilities
- log2 for entropy computation

**No continuous approximations** - all entropy is properly quantized.

---

## 5. Potential Semantic Backsliding
**Risk**: Future implementers may reintroduce gradient-based "improvement" metrics into System 2.
**Guardrail**: The **Four Non-Negotiable Laws** (see `INVARIANT_OPTIMIZATION.md`) must be audited during every evolutionary cycle. Any detection of cross-instance skill accumulation should be treated as a system fracture.

---

## 6. Constraint Probe Architecture (Phase 1)
**Status**: ✅ **IMPLEMENTED**

**Implementation**: System 2 now uses constraint probe operators with:
- Local feasibility probes per constraint (no global objective)
- Cyclic constraint traversal
- Bounded oscillation detection (no convergence guarantee)
- Failure token system for rupture conditions

## 7. Topological Guarantees (Phase 2)
**Status**: ✅ **IMPLEMENTED**

**Implementation**:
- Hyper-ring closure: `H(r) = ∮_C ∇_top Φ(r)` with closure checks
- Persistence obstruction graphs: Filtration-based persistent homology
- Soliton stability: Dispersion/localization ratio checks

## 8. Advanced Constraints (Phase 3)
**Status**: ✅ **IMPLEMENTED**

**Implementation**:
- Structural irreducibility: Evidence module orthogonality checks
- Gyroidic differentiation: Flow constraints `∇_flow Φ(r) ⟂ ∇G`
- Meta-invariant: Topology expansion monitoring `d/dt E_r[dim H_1(C_t)] >= 0`

---

**Conclusion**: We accept these gaps as the **price of non-ergodicity**. We do not aim for a "solved" system, but for a **survivable** one.

**Recent Updates (January 2026)**: Phase 1-3 implementations have addressed several open questions:
- Constraint probe architecture replaces global minimization
- Topological guarantees provide soliton stability checks
- Continuous co-primality uses proper discrete entropy quantization
- Meta-invariant prevents topology collapse

---

## 9. Structural Design Decisions (Phase 5)
**Status**: ✅ **IMPLEMENTED (January 18, 2026)**

These questions from the original plan have been resolved:

### 9.1 Performance: Persistent Homology
**Original Question**: Exact vs approximate PH? GPU acceleration?
**Resolution**: **Approximate PH, treat as trigger not metric.**
- Implemented: `ApproximatePHProbe` in `src/topology/approximate_ph.py`
- Uses relative barcode change (not exact Betti numbers)
- Landmark subsampling for sparse filtration
- Exactness is epistemically misleading

### 9.2 Hyper-Ring Computation
**Original Question**: Continuous or discrete? What resolution?
**Resolution**: **Discrete always. Adaptive resolution.**
- Implemented: `DiscreteHyperRingCirculation` in `src/topology/hyper_ring.py`
- ∮ Φ ≈ Σ ⟨Φ(C_i), ΔC_i⟩
- Increase resolution only on phase slippage or soliton nucleation
- Fixed high resolution = fake precision + compute waste

### 9.3 Evidence Modules (E_α)
**Original Question**: Data-derived, learned, or user-specified?
**Resolution**: **Tripartite: data + architectural + adversarial.**
- Implemented: `TripartiteEvidenceModule` in `src/core/evidence_modules.py`
- Evidence must disagree internally or PASₕ goes blind
- Mutual predictability detection with warnings

### 9.4 Soliton Threshold (κ)
**Original Question**: Fixed, learned, or adaptive?
**Resolution**: **Relational and history-dependent, never learned.**
- Implemented: `RelationalKappa` in `src/core/relational_kappa.py`
- κ(t) = μ_rupture(t) + λ · σ_rupture(t)
- λ is architectural temperament (chosen, not learned)
- Learning κ turns solitons into rewards

### 9.5 Meta-Invariant Enforcement
**Original Question**: How to force expansion when topology collapses?
**Resolution**: **Do not force expansion. Poison collapse paths.**
- Implemented: `CollapsePathPoisoner` in `src/core/collapse_poisoner.py`
- Techniques: Constraint anti-alignment, cycle debt, synthetic inconsistency, dimensional shearing
- Topology survives by resisting ease, not by being rewarded

---

## 10. Failure-Mode Table

| Failure Mode | Pre-Collapse Signature | Detection | Response |
|--------------|----------------------|-----------|----------|
| Mode Collapse | β_1 → 0, evidence entropy → 0 | Architectural detector | Inject anti-aligned constraints |
| Soliton Reward Loop | κ tracks performance | correlation check | Reset κ history |
| Evidence Blindness | Evidence correlation > 0.7 | mutual predictability | Inject adversarial evidence |
| Hyper-Ring Slippage | Non-zero circulation | circulation ≠ expected | Refine resolution |
| Cycle Debt Explosion | Homotopy repeats > 0.5 | Debt metric | Dimensional shearing |
| PH Trigger Fatigue | Rupture rate → 0 | Relative change | Recalibrate landmarks |
| κ Flatline | σ_rupture → 0 | Kappa volatility | Inject perturbations |

---

## All Open Questions Status

| Question | Status |
|----------|--------|
| Symbolic Co-Primality | Reframed as Symbolic Transversality |
| Failure-Space Topology | Bipartite Obstruction Complex |
| Gyroid Violation | Necessary boundary of forbidden |
| Continuous Co-primality | ✅ Discrete entropy quantization |
| Constraint Probe | ✅ No global objective |
| Topological Guarantees | ✅ Hyper-ring, persistence, soliton |
| Advanced Constraints | ✅ Irreducibility, differentiation, meta-invariant |
| PH Performance | ✅ Approximate, trigger-based |
| Hyper-Ring Integral | ✅ Discrete, adaptive resolution |
| Evidence Modules | ✅ Tripartite heterogeneous |
| Soliton Threshold | ✅ Relational κ |
| Meta-Invariant Enforcement | ✅ Collapse path poisoning (`CollapsePathPoisoner`) |

---

## 9.6 Collapse Path Poisoner — Implementation Detail

**Status**: ✅ **IMPLEMENTED**  
**Source**: [`src/core/collapse_poisoner.py`](../src/core/collapse_poisoner.py) — aliased as `AdversarialStressTester`

The "collapse path poisoning" referenced in the Meta-Invariant entry is **defensive**, not offensive. It does not corrupt training data; it *injects synthetic topological ruptures* to verify that:

1. `SpeculativeHomologyEngine` correctly detects Betti number changes when a hole is injected.
2. `DyadicTransferMap` remains robust to the perturbation.

**Two mechanisms**:

- **Synthetic Rupture**: `generate_synthetic_rupture(manifold)` uses Gram-Schmidt orthogonalization to construct a perturbation vector perpendicular to **every** basis vector of the current manifold, guaranteeing it cannot be absorbed without a topology change.
- **Cycle Debt**: `compute_cycle_debt(state)` uses cosine similarity to a rolling history buffer of 100 states. If the current state matches a previous state at cosine similarity > 0.9, it is counted as a homotopy repeat. `debt = repeats / history_size`, triggering a warning when `debt > 0.5`.

**Usage**: Called during periodic topological health checks in the training loop.

---

## Phase 10 Open Questions: TailSlayer / Meliponini / BigGAN / Coherence Asymmetry

### Q10.1 — DRAM XOR Offsets as CRT Moduli
Can the XOR-mapped physical address offsets of DRAM channels (e.g., AMD Ryzen `0x003fc0` boundary scheme) be formally approximated as CRT moduli for hardware-layer polytope routing — specifically, can the physical address bits that select Bank A vs. Bank B be treated as the parity check `r & 1` (INVARIANT_OPTIMIZATION Tripwire 8 §8.4)? Does this mapping violate any symplectic constraints from §41 (Symplectic Gluing)?

**Status**: Open. Requires hardware profiling of the GTX 1050 Ti memory controller's XOR scheme.

### Q10.2 — Meliponini Packing Fraction Threshold: Dynamic or Fixed?
Is the Meliponini Packing Fraction $\phi$ threshold ($\phi < \phi_{RCP} \approx 0.64$) dynamically computed from the `elipsodistrophy` Atrophy signal, or is it a fixed architectural constant? The Atrophy metric already provides a continuous proxy for $\phi$ (TOPOLOGICAL_EXTENSIONS §Part VII §2). If dynamic: what is the mapping function from Atrophy → $\phi_{effective}$? Is it linear, threshold-stepped, or Sigmoid?

**Status**: Open. Atrophy is currently used as a veto signal but not mapped to $\phi$.

### Q10.3 — Mamba/SSM Compression and Betti Number Preservation
Can Mamba-style State Space Model compression (O(N) fixed state) fully preserve Betti numbers $\beta_0, \beta_1$ of the concept manifold without running the full Persistent Homology pipeline, or does the BreatherMode O(K) fossilization (RESONANCE_INTELLIGENCE_CORE Eq 11.2) require independent topological verification after each fossilization event?

**Status**: Open. The SSM recurrent state is a smooth average — it loses the topological fingerprint. BreatherMode preserves the scar but the PH cost of verification could exceed $O(K)$ if $K$ is large.

### Q10.4 — Elipsodistrophy → Topology Mode Switching Threshold
At what Elipsodistrophy Atrophy level should the system automatically switch from Apis ($\phi \to 1.0$) to Meliponini ($\phi < 0.64$) manifold topology — is this the same threshold as `topological_pressure > 0.5` in `GYROID_REASONER`? If so, a single threshold gates two architectural regime changes (pressure response + topology mode). Is this a Scalarization Trap (INVARIANT_OPTIMIZATION Tripwire 3) — encoding two independent pressures as a single float comparison?

**Status**: Suspected issue. The `topological_pressure` float may be a hidden scalar aggregation of distinct Betti-number and spectral signals.

### Q10.5 — Drucker-Prager Convexity Under High Mischief
The Drucker-Prager smooth envelope ($\alpha I_1 + \sqrt{J_2} - k = 0$) assumes convexity of the global yield surface. Under high $V_m$ (Mischief) augmentation, can the yield surface remain convex, or does the ChernSimonsGasket $\kappa$ curvature break convexity locally? If $\kappa$ is high at multiple boundary crossing points simultaneously (extreme slider zone, RESONANCE_INTELLIGENCE_CORE §11.4), does the Drucker-Prager envelope still provide a valid global flow path, or does it degenerate into a non-convex multi-modal surface?

**Status**: Open. The interaction between $V_m$, $\kappa$, and the DP yield surface is not yet formalized.

### Q10.6 — SAR* Computability from Internal Signals
Is the SAR* (revolutionary threshold from Coherence Asymmetry theory, VETO_SUBSPACE §11.3) computable from the Reasoner's internal signals — specifically, can it be expressed as a function of $\text{PAS}_h$ (harmonic phase alignment), Elipsodistrophy Atrophy, and the veto count (number of TopologicalRefusalError events in the last $N$ steps)? If yes, the VetoSubspace could self-monitor its own approach to the SAR* threshold and pre-emptively inject Mischief before the external meritocratic probe reaches the critical pressure.

**Status**: Open. SAR formula is defined (VETO_SUBSPACE §11.3) but not connected to computable internal signals.

### Q10.7 — SLERP/LERP Mode Labeling in Diagnostic Payload
Should the ZeitgeistRouter's diagnostic output explicitly label the current navigation style as SLERP (`interior`) vs. LERP (`grazing`) vs. wandering glitch (`undefined`) in the payload for downstream consumers? This would expose "interpolation glitch potential" to any system that uses the `nc_curvature` signal. Would this exposure constitute a Tripwire 4 violation (Silent Failure — no intermediate visibility) if `nc_curvature` is already in the payload?

**Status**: Near-resolved. `nc_curvature` is already in the diagnostics table (ZEITGEIST_ROUTER §7). Adding a semantic mode label (`slerp`, `lerp`, `void`) would not add new gradient information — it would add human-readable state classification. Likely safe.

### Q10.8 — Anti-Disentanglement Constraint Enforcement
The MATHEMATICAL_DETAILS §55.5 specifically forbids making the $K$ polynomial functionals orthogonal in activation space (that would be disentanglement = loss of holistic glitch diversity). Is there currently any optimizer or regularization pathway that implicitly pushes functionals toward orthogonality (e.g., the independence criterion in §7.6 Continuous Co-Primality)? If the co-primality condition enforces $\lim_{t\to\infty} \text{Cov}(r_i, r_j) = 0$, is this equivalent to asymptotic functional orthogonalization? This may be the most critical unresolved tension in the system.

**Status**: Open. The co-primality asymptotic independence condition requires careful disambiguation from the holistic glitch interdependence requirement. These may be in direct conflict.



