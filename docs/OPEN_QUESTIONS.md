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
| Meta-Invariant Enforcement | ✅ Collapse path poisoning |

