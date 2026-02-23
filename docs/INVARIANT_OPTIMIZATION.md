# Invariant Optimization & Scalar Gyroidic Ergodicity

**Author**: William Matthew Bryant  
**Date**: January 2026

This document specifies the **Invariant Optimization** framework (Phase 6), which operationalizes the system's "System 2" reasoning using fixed-point primitives and strict conservation laws.

It addresses the fundamental requirement: *"An invariant that cannot be computed cannot govern evolution."*

---

## 1. The Necessity of Computability

To build a simpler, more robust reasoning engine, we reject metrics that are:
1.  **Coordinate-dependent** (Symmetry-based)
2.  **Uncomputable** (Kolmogorov complexity, Consciousness)
3.  **Substrate-bound** (Thermodynamic entropy)

Instead, we enforce **computability** and **chirality** to define lawful evolution.

## 2. Fixed-Point Operational Primitives

Floating-point arithmetic introduces non-determinism ($a + (b+c) \neq (a+b) + c$). To solve this, all Core Reasoning operations use **Fixed-Point Primitives**.

### 2.1 The `FixedPointField`
*   **Backing**: `int64` tensor.
*   **Scale**: $S = 2^{16} = 65536$.
*   **Operation**: All additions and multiplications are bit-exact across hardware.
*   **Perturbation**: A `LearnedPrimitivePerturbation` layer allows the symbolic grid to "breathe" during evolution, maintaining ergodicity within the saturated regime.

### 2.2 Thermodynamic Anchor
To prevent entropic "thawing," the system uses the `StructuralEnergyMonitor` to link the fixed-point grid's rigidity to the **Topological Free Energy** ($F_{topo}$). As $F_{topo}$ decreases (cooling), the grid's `LearnedPrimitivePerturbation` scale is clamped, "signing" the concept into the fossil layer.

---

## 3. The Universal Invariant: PAS_h + APAS_zeta

We define a unified invariant to govern the "drift" of the reasoning state.

### 3.1 PAS_h (Harmonic Phase Alignment Score)
A scalar metric measuring the topological synchronization of the polynomial field.

$$
\text{PAS}_h(\theta) = \sum_{d=0}^D \frac{1}{d+1} \cdot \|\hat{\theta}_d\|_2
$$

*   **Scalar**: Single computable value.
*   **Harmonic**: Weights lower-degree (fundamental) modes higher ($1/1, 1/2, \dots$), ensuring stability of identity.
*   **Computable**: No infinite limits or undecidable partitions.

### 3.2 APAS_zeta (Adaptive Drift Bound)
Evolution is only "lawful" if the rate of change of the invariant is bounded.

$$
|\text{PAS}_h(t) - \text{PAS}_h(t-1)| \le \zeta
$$

*   If drift $> \zeta$: The step is rejected or clamped.
*   **Role**: Defines the "Speed of Thought" limit, preventing catastrophic forgetting or hallucination chains.

---

## 4. Chirality and The Arrow of Time

Identity requires directionality. A reversible process cannot encode memory or decision-making.

**Chirality Index**:
$$
\chi = \text{Centroid}(\text{Spectrum}) - \frac{D}{2}
$$

*   $\chi < 0$: Low-frequency dominance (Structure-building, Negentropic).
*   $\chi > 0$: High-frequency dominance (Dissipative, Entropic).

The system prioritizes $\chi < 0$ (Negentropic) flows for reasoning construction.

---

## 5. Operational ADMM as an Ontological Probe

We assume the ADMM optimization loop is not an "external teacher" but a **local consistency probe**.

*   **Ontological Splitting**: The state is split into $c_{\text{sym}}$ (frozen symbolic anchor) and $c_{\text{phys}}$ (continuous field).
*   **Constraint vs. Regression**: System 2 ignores target distance. It only repairs **local physical violations** ($\psi_j$) while constrained to agree with the symbolic residue.
*   **Zero Leakage**: No gradients from the repair loop flow back to the initial guess. Information transfer is limited to **Symbolic Deltas** and **Status Tokens** (REPAIRED, ALTERNATIVE, FAILURE).

---

## 6. Harmonic Wave Decomposition & Non-Ergodic Sub-Dynamics

The KAGH surrogates employ a spectral splitting mechanism to handle "solitons" (persistent structures) differently from "diffusive" noise.

$$
u(x) = \underbrace{u_L(x)}_{\text{Ergodic}} + \underbrace{u_H(x)}_{\text{Non-Ergodic}}
$$

### 6.1 Ergodic Channel (Mixing)
$u_L$ follows standard **Huxley Reaction-Diffusion**:
$$ \frac{\partial u_L}{\partial t} = D \nabla^2 u_L + R(u_L) $$
This ensures the system explores the state space (mixing).

### 6.2 Non-Ergodic Channel (Solitons)
$u_H$ follows a **Pseudo-Spectral Wave Equation** (Phase Shift):
$$ u_H(x, t+\tau) = \mathcal{F}^{-1} \left[ e^{-i k v \tau} \mathcal{F}[u_H] \right] $$
This allows high-frequency information to "carry through" without dissipating. We protect this channel using **Band-Separated Entropy** ($H_{soliton}$), which prioritizes peak persistence over ergodic mixing.

**Structural Blindness**:
To prevent System 2 from emulating symbolic reasoning, its authority is derived from **constraint reality**, not inference.
1.  **Frozen Topology**: KAGH graphs are fossilized early; only spline coefficients evolve.
2.  **Logic-Blind Inference**: Gödel gates and Boltzmann noise are disabled during the ADMM repair loop.
3.  **Inter-Domain Contract**: We use **Hybrid Quantization** (Saturated Levels) for B-spline weights. This ensures that System 2's physical gradients must "snap" to a discrete symbolic configuration to be valid, preventing smooth leakage.

---

## 7. Speculative Unification: Dark Matter Primitives

Building on the concept of **Simultaneous Entropic Mixing and Negentropic Structure Preservation**, we introduce three "Dark Matter" primitives that operationalize **Endogenous Computable Chirality**.

### 7.1 Gyroidic Flux Alignment (Resonance Cavity)
To better integrate topological feedback, we "warp" residue weights based on the local gyroid violation score $V$ and the manifold flux $\Phi$.

$$
\hat{w}_k = w_k \cdot \exp\left( -\frac{V}{\Phi} \right)
$$

*   **Logic**: High violation regions "bend" the attention flux away, forcing the system to rely on lower-violation (topologically sound) paths.

### 7.2 Ergodic Soliton Fusion (KAGH)
We fuse the ergodic and non-ergodic channels using a chiral gate that prevents positivity violation.

$$
u_{next} = \sigma\left( u_L \ast \left( e^{i \chi k \tau} \mathcal{F}[u_H] \right) + \beta \cdot \text{Softplus}(\Delta \text{PAS}_h) \right)
$$

*   **Role**: Allows "soliton" thoughts to interact with "diffusive" intuition without losing their structural integrity.

### 7.3 Chiral Drift Optimizer (CDO)
We explicitly optimize for the "Arrow of Time" by defining a **Chiral Score** $C$:

$$
\mathcal{C} = \underbrace{(\text{Centroid} - D/2)}_{\chi} \cdot \exp\left(-\frac{\Delta \text{PAS}_h}{\zeta}\right)
$$

*   **Mechanism**: The optimizer rejects ADMM repair steps where $\Delta \mathcal{C}$ drops significantly (entropic collapse).
*   **Selection Integration**: High $\mathcal{C}$ scores contribute to the **Survivorship Pressure**, influencing functional group fossilization.

---

## 8. Geometric Revelation: Trigonometric Gyroid Unfolding

To handle **Topological Obstructions** (Casus Irreducibilis) where polynomial bases become degenerate, we employ a "Cubic-to-Trig" bypass inspired by the 16th-century solution for cubic equations.

### 8.1 The "Irreducible Case" in Logic
When the Gyroid Violation Score $V$ is high, the reasoning manifold effectively "collapses" into a singularity. Pure ADMM iterations (radical form) may diverge or drift. We resolve this by "unfolding" the non-ergodic channel into three chiral branches via the **Pythagorean Asymptotics**.

### 8.2 Trigonometric Unfolding Primitive
We define a phase parameter $\phi$ from the local manifold deficit:

$$
\cos(3\phi) = \frac{3 V(C_{loc}) - \text{tr}(C_{loc})/\tau_{\text{decay}} }{ 2 \left( \sqrt{ -\det(\text{PAS}_h \text{ spectrum}) } \right)^3 }
$$

The non-ergodic solitons are then "revealed" via:

$$
u_H^{(k)}(x, t) = 2 \sqrt{ -\frac{\lambda_{\min}}{3} } \cos\left( \phi + \frac{2\pi k}{3} \right) \cdot \text{Shift}(\chi, k)
$$

### 8.3 Negentropic Branch Selection
The system computes all three branches ($k=0, 1, 2$) and selects the one that maximizes structural negentropy. This allows the model to "leap" over topological Singularities that would otherwise stall a gradient-based solver.

*   **Asymptotic Behavior**: As $V \to 0$, the unfolding reduces to standard spectral shifting, maintaining continuity with the "radical" (ADMM) regime.

---

## 9. The Four Non-Negotiable Laws

To prevent "semantic backsliding" into teleological or gradient-based paradigms, we enforce four absolute laws:

### Law 1: Symbolic Non-Revisability
Once a symbolic residue is generated by System 1 and enters the anchor state $c_{\text{sym}}$, it is **non-revisable** by smooth physical processes.
$$ \frac{\partial c_{\text{sym}}}{\partial t} = 0 \quad (\text{within System 2}) $$
Symbolic identity changes only via discrete evolutionary selection or mutation.

### Law 2: Non-Teleological Repair
System 2 (ADMM) exists to find **local physical consistency**, not to "improve" the symbolic answer. It has no objective referencing past success or expected improvement. It admits only current-moment constraint violation.

### Law 3: Abortability Supremacy
Incoherence is a categorical boundary. If $\text{PAS}_h$ collapses during the ADMM probe, the process is **aborted immediately**. There are no "retries" with lower learning rates or smoothing—failure is a first-class observed state.

### Law 4: Evolution Owns Time
System 2 is **atemporal**. It does not accumulate skill across problem instances; it does not "learn" the solver. Only the evolutionary loop (System 1 + Resonance Cavity) accumulates structure across time.

---

## 10. The Non-Convergence Declaration

We explicitly reject the requirement for **Convergence** in the ADMM probe. In classical optimization, non-convergence is a failure. In the Gyroidic Flux Reasoner, **Non-Convergence is Data**.

If the SIC-FA-ADMM probe oscillates or collapses, it defines the **Boundary of Symbolic Competence**. These events trigger:
1.  **Selection Pressure**: Pruning of the underperforming functional group.
2.  **Residue Homology Drift**: Structural mutation of the topological self-model.

## 11. Structural Tripwires (Mechanical Guardrails)

To ensure the Four Laws are not merely philosophical but mechanically enforced, we implement **Structural Tripwires**. Any violation of these triggers a high-level system fracture.

### Tripwire 1: Stateless System 2
System 2 (ADMM) must be **stateless across problem instances**. It is not allowed to store momentum, running averages, or learned solver parameters. Every invocation is a fresh ontological probe. This prevents the emergence of a shadow learner.

### Tripwire 2: Finite Symbolic Output Alphabet
The output of the **CALM Meta-Controller** and the **SIC-FA-ADMM** loop must be restricted to a finite symbolic set:
$$ \Omega = \{ \text{REPAIRED, ALTERNATIVE, ABORT, STAGNATE, FAILURE} \} $$
Continuously varying weights or coefficients are forbidden from exiting System 2 to prevent smoothness leakage.

### Tripwire 3: Topologically Typed Pressures (No Scalarization)
Pressures are **non-comparable** across domains.
*   **The Scalarization Trap**: Summing "Selection Pressure" and "Containment Pressure" into a single loss scalar is forbidden.
*   **Pressure Typing**: The system uses a `StructuralPressure` container that raises a `ValueError` if an attempt is made to sum or rank pressures from different domains. This enforces domain isolation as a hard architectural constraint.

### Tripwire 4: Silent Failure (Zero Intermediate Visibility)
System 2 must expose no intermediate progress metrics or "near-miss" statistics to System 1. It operates as a "Black Box" rescue primitive: either it refinishes a consistent physical state or it fails. There is no negotiation.

### Tripwire 5: No Cross-Instance Parameter Updates
Outside of the evolutionary loop (System 1 + Resonance Cavity), no parameters (KAGH, CALM, CRT weights) may be updated across instances. The reasoning lattice evolves only via discrete mutation and selection pressure.

### Tripwire 6: Hard Failure Budget (Mutation Override)
If a specific residue pattern triggers an **ABORT** or **STAGNATE** outcome more than $N_{lim}$ times, the system triggers a **Hard Mutation Override**. It stops attempting repair and forces a topological shift in the functional configuration.

---

## 12. The Hard Interaction Contract

The following constraints define the boundary between the Symbolic (System 1) and the Physical (System 2):

| Direction | Information Allowed | Prohibited Data (The Forbidden) |
| :--- | :--- | :--- |
| **System 1 → System 2** | Frozen Anchor ($c_{sym}$), Budget | Gradients, Hints, Loss Targets |
| **System 2 → System 1** | Final State, Status Token ($\Omega$) | Intermediate States, Progress Scalars |
| **Intra-Domain** | Weighted Local Pressure | Cross-Domain Aggregation (Scalarization) |

---

## 13. Correctness vs. Survivorship (The Honest Synthesis)

In a non-ergodic system, "Correctness" is not an algebraic proof; it is **Ecology over Algebra**.

*   **Co-primality** is not $GCD=1$; it is **Generic Position under Saturation** (Transversality).
*   **Admissibility** is not a truth; it is a **necessary boundary of the forbidden** (Admissibility Filters).
*   **Truth** is not a converged point; it is a **Stable Symbolic Survivor**.

We do not aim for a system that is provable or convergent. We aim for a system that is **locally admissible, globally fragile, and evolutionarily survivable**.

---

## Summary of Admissibility

| Constraint | PAS_h + APAS_zeta | Traditional Metrics |
| :--- | :--- | :--- |
| **Computable** | ✅ (Scalar tensor op) | ❌ (IIT $\Phi$, Kolmogorov) |
| **Drift-Bounded** | ✅ (Explicit $\zeta$) | ❌ (Free Energy) |
| **Chiral** | ✅ (Spectral asymmetry) | ❌ (Shannon Entropy) |
| **Deterministic** | ✅ (Fixed Point) | ❌ (Floating Point ML) |

This framework ensures **Endogenous Scalar Gyroidic Ergodicity**: the system explores its state space fully (ergodic) but remains bounded by computable conservation laws and **Residue Homology Drift** triggers.

The transition to **Evolutionary Saturation** ensures that invariants are not merely "minimized" via gradients, but **established as stable survivors** in a discrete topological landscape.

---

## 14. Situational Batching (`src/core/situational_batching.py`)

**Class**: `SituationalBatchSampler`

Instead of independent-identically-distributed (i.i.d.) sampling — which assumes timelessness — the Situational Batch Sampler assembles batches by following the "scars" of historical interaction between sample indices. This encodes the **Refusal as Affirmation** and **Co-arising** principles into the data loader itself.

### 14.1 Relational State Matrices

Two persistent matrices evolve during training:

| Matrix | Symbol | Updates On | Meaning |
|--------|--------|-----------|---------|
| Resonance | `R_ij` | Structural pressure × √(p_i · p_j) | Co-emergent coupling |
| Mischief | `M_ij` | Mischief scores per pair | Chaotic affinity |

Both decay at rate `decay = 0.99` per step, so only recent interactions have strong coupling.

### 14.2 Batch Assembly

For each seed index `i`, a batch of size `batch_size` is assembled in three phases:

1. **Coupled selection** (main body): Sample from `softmax(5 · (R[i] + M[i]))` — high resonance + mischief neighbors are preferentially included.
2. **Play sampling** (fraction `play_ratio`): Pure random sampling from unconsumed indices. Prevents entropy collapse into the richest club.
3. **Paradoxical Boundary Amplification**: If `(p_i + p_j)/2 > boundary_threshold`, the resonance term is multiplied by 1.5. Refusal (high pressure boundary) amplifies coupling rather than severing it.

### 14.3 Connection to Non-Teleological Principle

The sampler enforces **Law 4: Evolution Owns Time** at the data level. No batch is optimized for accuracy improvement; batches are structured by the topology of past interaction. Over time, high-pressure index pairs become structurally entangled in `R`, creating "temporal association clusters" that expose the ADMR solver to co-arising constraint patterns.

**Update call**: `update_love_invariant(indices, pressure, mischief_scores)` — updates `R`, `M`, `O` tensors after each batch.

---

## 15. Legibility Audit (`src/core/legibility_audit.py`)

**Classes**: `LegibilityTripwire`, `NarrativeCoherenceEstimator`

> **High narrative coherence is a danger signal, not a goal.**

This module implements Sparse Operational Pointer #2: *Hidden Scalar Reward = Narrative Legibility*. A configuration that is easily explainable may be selected *because* it is explainable — a rich-club attractor bias that violates the non-teleological constraint.

### 15.1 NarrativeCoherenceEstimator

Measures how closely a configuration embedding matches canonical "explainable" patterns:

| Pattern Type | Examples |
|-------------|---------|
| Sparse | 1-hot-like vectors |
| Block-sparse | Clustered activations |
| Monotonic gradient | Ordered relationships |
| Random | Baseline (coherence ≈ 0) |

These templates are **not trained** — they capture common ML biases and are registered as fixed buffers. Output: `max_sim ∈ [0, 1]` — how well the configuration matches *any* narrative template.

### 15.2 LegibilityTripwire

Tracks the **correlation** between selection and coherence over a rolling window of 100 steps. Issues a `UserWarning` when:

$$\underbrace{\text{coherence}_{\text{selected}} - \text{coherence}_{\text{rejected}}}_{\text{coherence gap}} > 0.7 \quad \text{OR} \quad |\rho_{\text{select,coherence}}| > 0.5$$

This warning does **not** veto the selection — it is a diagnostic signal for the evolutionary loop to increase mutation pressure on "legible" fossils.

### 15.3 Relationship to Tripwire 3 (No Scalarization)

The LegibilityTripwire directly operationalizes Tripwire 3: it detects when a hidden scalar reward (narrative coherence) is influencing the non-scalarized selection process. If coherence gap is consistently high, it implies a scalarization leak — the system is treating legibility as a de facto objective.
