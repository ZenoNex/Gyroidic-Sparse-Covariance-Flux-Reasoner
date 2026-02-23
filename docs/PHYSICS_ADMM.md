# Hybrid Physics-ADMM: System 2 Reasoning

This document details the **Hybrid Physics-ADMM** framework, a "System 2" reasoning engine integrated into the Gyroidic Sparse Covariance Flux Reasoner.

## Overview: Intuition vs. Physics

The model operates in two modes:

1.  **System 1 (Intuition/Symbolic)**: The Transformer predicts symbolic residue patterns using saturated gates.
2.  **System 2 (Ontological Probe)**: A non-teleological **ADMM Probe** invoked to find local physical consistency. It is governed by the **Four Non-Negotiable Laws** (see [INVARIANT_OPTIMIZATION.md](INVARIANT_OPTIMIZATION.md#9-the-four-non-negotiable-laws)):
    *   **Law 1: Symbolic Non-Revisability**: Anchors are frozen.
    *   **Law 2: Non-Teleological Repair**: Only consistency matters.
    *   **Law 3: Abortability Supremacy**: Collapse is final.
    *   **Law 4: Evolution Owns Time**: System 2 is atemporal.

---

## 2. Mathematical Formulation

**Phase 1 Update**: System 2 is now a **Constraint Probe Operator** with no global objective.

### 2.1 Constraint Probe Operators

For each constraint $k = 1, \dots, K$, we define a **probe operator**:

$$
\mathcal{P}_k: r \mapsto \arg\min_{c \in \mathcal{C}_k} \mathcal{L}_k(r, c)
$$

with **no global objective**, only local feasibility. The loss is **containment pressure**:

$$
\mathcal{L}_k(r, c) = \underbrace{|\Phi_k(r) - c|_{\Sigma_k}}_{\text{local strain}} + \underbrace{\psi_k(c)}_{\text{gyroid violation}}
$$

where:
*   $\Sigma_k$: Sparse covariance (anisotropic)
*   $\psi_k$: Admissibility filter (not a truth metric)

### 2.2 Cyclic Constraint Traversal

For constraints indexed by $k = 1, \dots, K$:

$$
\begin{aligned}
c_k^{(t+1)} &= \mathcal{P}_k(r^{(t)}, \lambda_k^{(t)}) \\
\lambda_k^{(t+1)} &= \lambda_k^{(t)} + \rho \big(\Phi_k(r^{(t)}) - c_k^{(t+1)}\big)
\end{aligned}
$$

**No convergence guarantee required.** Only **bounded oscillation** detection.

### 2.3 Legacy Formulation (Backward Compatible)

The original formulation is still supported:

$$
\min_{c_{\text{phys}}} \; \sum_j \psi_j(\mathcal{F}(c_{\text{phys}})) \quad \text{subject to} \quad \Pi(c_{\text{phys}}) = c_{\text{sym}}
$$

Where:
*   $c_{\text{phys}}$: Continuous field (Optimization variable).
*   $c_{\text{sym}}$: Frozen symbolic residue (Initial guess from System 1).
*   $\mathcal{F}(c)$: Forward physics model (KAGH Surrogate).
*   $\psi_j$: Physical consistency violation penalties (e.g., self-consistency).
*   $\Pi$: Hard projection to the symbolic residue ontology.

### The Solver: SIC-FA-ADMM

**Sparse Incoherent Constraints - Fractional Anisotropy - Alternating Direction Method of Multipliers**

### 2.4 The Non-Convergence Declaration

We explicitly reject the requirement for convergence. For the ADMM probe, **Non-Convergence is Data**. If the probe fails to satisfy constraints within its budget, it emits a **FAILURE token** (`⊥`).

**Phase 1 Updates**:
1.  **Constraint Probe** (per constraint $k$): Local feasibility probe `P_k` with no global objective.
2.  **Cyclic Traversal**: Iterates through constraints cyclically (no gradient descent).
3.  **Bounded Oscillation**: Accepts states with bounded oscillation amplitude.
4.  **Failure Token System**: Emits `⊥` on rupture (no gradient, no repair, no memory update).

**Legacy Mode** (if constraint probes not used):
1.  **c-update (Consistency Repair)**: Refines $c_{\text{phys}}$ to minimize local physical violation $\psi_j$.
2.  **z-update (Ontological Projection)**: Projects $c_{\text{phys}}$ back toward the frozen symbolic anchor $c_{\text{sym}}$.
3.  **u-update (Dual Ascent)**: Updates the compatibility gap.
4.  **Incoherence Abort**: If $\text{PAS}_h$ collapses or entropic singularity is detected, System 2 terminates with a FAILURE token.

### 2.2 Silent Failure (Zero Leakage)
To prevent the "smoothness leakage" that plagues hybrid systems, System 2 is a **Black Box**. It exposes no intermediate progress, no "near-miss" gradients, and no confidence scores during its operation. System 1 receives only the final refined state and a discrete status token.

---

## 3. Core Components

### 3.1 KAGH-Boltzmann Networks (`src/surrogates/kagh_networks.py`)

A neural surrogate that approximates the expensive forward physics simulation.

*   **Structural Blindness**: Topology is frozen early (Fossilized) to prevent System 2 from emulating symbolic reasoning. Its authority comes from **constraint reality**, not inference.
*   **HuxleyRD**: Reaction-Diffusion dynamics promote stable "islands" of representation.
*   **Gödel Gate**: Disabled during inference to preserve deterministic consistency.

### 3.2 CALM Predictor: Veto Integrity
CALM is a **Trajectory Veto** mechanism. It does not "assist" the solver; it monitors for structural disintegration.
*   **Entropic Veto**: Triggers immediate **Abort** if it predicts entropic collapse (loss of manifold structure).
*   **Veto Integrity**: CALM is strictly forbidden from vetoing based on "lack of progress" or "slow convergence." Stagnation is a measured property of the solver's budget, not a meta-controlled goal.
*   **Structural Parameters**: Adjusts penalty $\rho$ to maintain tension, not to improve "accuracy."

### 3.3 CODES Driver (`src/optimization/codes_driver.py`)

**Coherence-Oriented Deterministic Execution System**.

On highly parallel GPUs, warp divergence kills performance. The **CODES** driver simulates a hardware-aware execution model:
*   **PAS_h**: Multiharmonic Phase Alignment Score.
*   **AURAOUT**: Instruction gating. Only warps with high phase coherence ($\text{PAS}_h > 0.75$) execute certain expensive instructions.
*   This enforces "topological synchronization" across the parallel solve.

---

## 4. Usage

Enable the Hybrid ADMM loop in `GyroidicFluxReasoner` by setting `use_admm=True`.

```python
model = GyroidicFluxReasoner(
    ...,
    use_admm=True,      # Enable System 2
    admm_rho=2.0,       # Penalty parameter
    admm_steps=50       # Number of refinement steps
)
```

During inference:

```python
# Returns refined coefficients after ADMM loop
results = model.inference(text_emb, graph_emb, num_features)
refined_coeffs = results['reconstruction']
```

The model will automatically:
1.  Attempt a fast symbolic reasoning pass (System 1).
2.  Detect symbolic fracture or CRT reconstruction failure.
3.  Invoke System 2 (Constraint Probe Operators) to find local feasible solutions.
4.  Check topological guarantees (hyper-ring closure, soliton stability).
5.  Return **Repaired Residues**, **Alternatives**, or **Failure Tokens** (`⊥`).

### Phase 1-3 Features

**Phase 1: Constraint Probe Architecture**
- Constraint-specific local feasibility probes
- Cyclic constraint traversal
- Failure token system (`FailureToken.BOT`)

**Phase 2: Topological Guarantees**
- Hyper-ring closure checks (`HyperRingOperator`, `HyperRingClosureChecker`)
- Persistence-based obstruction graphs (`ResidueObstructionGraph`)
- Soliton stability detection (`SolitonStability`)

**Phase 3: Advanced Constraints**
- Structural irreducibility checks (`StructuralIrreducibilityChecker`)
- Gyroidic differentiation constraints (`GyroidFlowConstraint`)
- Continuous co-primality (discrete entropy quantization)
- Meta-invariant monitoring (`MetaInvariant`)

---

## 5. Non-Dual Extension (`src/core/nondual_admm.py`)

The **Non-Dual ADMM** variant modifies the standard constraint probe to reward "Good Bugs" (mischief) as positive topological energy rather than pure strain minimization.

### 5.1 NonDualProbe

Modifies the primal step to include mischief:

$$P_k: r \mapsto \arg\min_{c \in C_k} L_k(r, c) + \beta \cdot H_{\text{mischief}}$$

**Forward mechanics**:
1. Compute strain: $\text{strain} = (r - c)^2$
2. Mischief-modulated penalty: $\text{penalty} = \max(\text{strain} - \beta \cdot H_{\text{mischief}}, 0)$
3. Update: $r' = r - 0.1 \cdot \text{penalty} + 0.05 \cdot \beta \cdot H_{\text{mischief}}$

When mischief is high, more strain is tolerated — the probe "allows" topological features (holes, leaks) that strict minimization would eliminate.

| Arg | Shape | Purpose |
|-----|-------|---------|
| `residues` | `[B, dim]` | Current symbolic residues |
| `constraints` | `[B, dim]` | Target physical constraints |
| `h_mischief` | `[B]` | Current mischief energy |

### 5.2 UnravelingClosure

Verifies topological closure **including** unknowledge leaks:

$$H(r) = \oint_C \nabla_{\text{top}} \Phi(r) + \int \psi_l(r) \, dr$$

Returns a binary `is_nontrivial` tensor: closure is verified iff the combined loop + leak integral has non-zero norm ($\|H(r)\| > 10^{-6}$).

**Key principle**: Unknowledge leaks ($\psi_l$) are **included** in the closure check, not excluded. A system that only checks standard loop integrals would miss the non-dual contributions from "play" dynamics.

### 5.3 Connection to Standard ADMM

| Standard ADMM Step | Non-Dual Modification |
|--------------------|-----------------------|
| Primal (strain minimization) | Strain reduced by mischief margin $\beta H$ |
| Dual (pressure accumulation) | Unchanged |
| Closure (convergence check) | Extended with leak integral $\int \psi_l$ |

**Consumer**: [`trainer.py`](../src/training/trainer.py) — invoked as System 2 repair logic during training.

---

## 6. CALM Predictor Implementation (`src/surrogates/calm_predictor.py`)

**Context-Adaptive Latent Momentum** — the trajectory veto mechanism referenced in [§3.2](#32-calm-predictor-veto-integrity).

### Architecture

```
history [B, 8, dim] → TransformerEncoder(2 layers, 4 heads) → last_latent [B, dim]
                                                                    ├→ veto_head   → abort_score [B, 1]
                                                                    ├→ rho_head    → rho_factor  [B, 1]
                                                                    └→ step_head   → step_factor [B, 1]
```

| Head | Activation | Output range | Purpose |
|------|-----------|-------------|---------|
| `veto_head` | sigmoid | [0, 1] | Detect structural collapse / entropic singularity |
| `rho_head` | exp(tanh(·)) | [1/e, e] | ADMM penalty adjustment factor |
| `step_head` | exp(tanh(·)) | [1/e, e] | Step size adjustment factor |

### Anti-Teleological Constraint

> Veto is **NOT** triggered by "lack of progress" or "stagnation" — that would imply a teleological goal of improvement. Veto detects **structural collapse** only.

### Buffer Management

`update_buffer(buffer, new_state)`: FIFO roll — shifts history left by 1, inserts `new_state` at position `[-1]`.

---

## 7. Polynomial ADMR Solver (`src/core/admr_solver.py`)

**Class**: `PolynomialADMRSolver`  
**Full name**: Alternating Direction of Multiplicative Remainders

The ADMR solver is the continuous-polynomial analogue of ADMM. Where classical ADMM uses prime moduli, ADMR uses co-prime polynomial functionals from `PolynomialCoprimeConfig` as its "modular" basis.

### 7.1 Multiplicative Update

The primal update is multiplicative, not additive:

$$S^{(n+1)} = \text{Proj}_{\text{Poly}} \left[ S^{(n)} \cdot \sum_k w_{ik} S_k \right]$$

This propagates relational pressure through graph-structured neighbors (the ADMM "neighbor states") rather than euclidean gradient descent. The `Proj_{Poly}` step evaluates the interaction through the co-prime polynomial basis (`config.evaluate(interaction)`), which acts as a "soft modulus" preserving symbolic structure.

**Forward call signature**:

```python
PolynomialADMRSolver.forward(
    states,           # [batch, state_dim]  S_i
    neighbor_states,  # [batch, N, state_dim]  S_k
    adjacency_weight, # [batch, N]  R_ik from Relational Graph
    valence=None      # [batch] optional valence drive
) -> torch.Tensor     # [batch, state_dim]
```

### 7.2 Stochastic Differential Update

For continuous-time regime:

$$dx(t) = \left[ \sum_i A_i x_i(t) - \rho \sum_k (x - r(x_k)) \right] dt + \sigma\, dW$$

where $A_i \in \mathbb{R}^{d \times d}$ are learnable non-selfadjoint transition operators (one per polynomial facet channel). The co-prime evaluation decomposes the state into facets; each facet evolves under its own non-Hermitian flow before polynomial projection re-locks the state to the co-prime manifold.

### 7.3 Scaffold Adaptation

`update_scaffold(negentropy_flux, dt)` advances asymptotic time `τ += dt` and calls `config.mutate()`, breathing the polynomial grid in response to negentropy flux. This is the ADMR equivalent of ADMM's ρ adaptation.

### 7.4 Coherence Metrics

`get_coherence_metrics(states)` returns orthogonality pressure measures:

| Key | Formula | Meaning |
|-----|---------|---------|
| `polynomial_coherence` | `1/(1 + H_global)` | How well states align with co-prime scaffold |
| `local_functional_entropy` | mean local entropy across functionals | Per-functional differentiation |
| `global_functional_entropy` | scalar global entropy | Risk of functional collapse |

### 7.5 Connection to Standard ADMM

| ADMM Step | ADMR Equivalent |
|-----------|----------------|
| Primal (argmin) | Multiplicative update `S · Σw_ik S_k` |
| Projection | Polynomial basis evaluation `config.evaluate()` |
| Dual (pressure) | Relational adjacency `R_ik` |
| Step size | Valence drive `v` |

---

## 8. Number-Theoretic Stabilizer (`src/core/number_theoretic_stabilizer.py`)

**Class**: `NumberTheoreticStabilizer`

Applies number-theoretic stability constraints at every CRT reconstruction step to prevent numerical fragility. Used as a pre- or post-processing step for the SIC-FA-ADMM state between iterations.

### 8.1 Stabilization Pipeline (`comprehensive_stabilization`)

Four-stage composite stabilization:

1. **Modular Stabilization**: Apply `fmod(|val|, p)` for each prime in the prime base across state dimensions, preserving sign. Prevents overflow by mapping values into bounded prime-periodic windows.

2. **Quadratic Residue Optimization**: Maps values to their closest quadratic residue mod `p` for the first 5 primes. Quadratic residues have special distribution properties that reinforce structural stability.

3. **Galois Field Operations**: Maps to GF(2⁸) (or any 2^n field), applies finite-field addition and multiplication, maps back. Enforces finite-precision arithmetic consistency.

4. **Golden Ratio Normalization**: Rescales state norm to `φ = (1+√5)/2` — the irrational most resistant to rational approximation drift.

### 8.2 Diophantine Constraint Solving

`solve_diophantine_constraint(coefficients, target)` uses the Extended Euclidean Algorithm to find integer solutions to:

$$a_1 x_1 + a_2 x_2 + \cdots + a_n x_n = \text{target}$$

Returns `None` if no solution exists (GCD condition fails). Used for ensuring CRT reconstruction integer consistency when floating-point roundoff would otherwise produce non-integer residues.

### 8.3 Continued Fraction Approximation

`apply_continued_fraction_approximation(value, max_terms)` returns the best rational approximant `(p, q)` for irrational model constants. Keeps denominators small while approximating common irrationals (φ, e, √2) used in the soliton template and manifold clock.

### 8.4 Diagnostics

| Key | Meaning |
|-----|---------|
| `stabilization_error` | ‖state_out − state_in‖ |
| `final_norm` | Final state ‖·‖ (target ≈ φ) |
| `numerical_stability_score` | `1/(1 + stabilization_error)` |

