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

## 5. Operational ADMM as an Ontological Probe & Cayley-Birkhoff Hybridization

We assume the ADMM optimization loop is not an "external teacher" but a **local consistency probe** that operates on a dual-constraint geometry.
*   **The Birkhoff Constraint:** Usually handles the state matrix by forcing doubly stochastic stability, smoothing the Horse's path.
*   **The Cayley Cubic Constraint ($V(C)$):** $x(x^2 + y^2 + z^2 - xyz - 4) = 0$. When "Cycle Debt" or "Mischief" is high, System 2 triggers a Phase Transition, driving the manifold toward the Cayley Cubic surface to find a Neglecton anchor.
*   **The Sovereign Loci:** The isolated points $A_1$ given by $(\pm 2, \pm 2, \pm 2)$ on the Cayley cubic. These singularities are the *Sovereign Loci*, the only regions where the **VoynichExemptionToken** is valid. They encode locations where the group representations are non-diagonalizable and acquire Jordan-block structuresimmune to standard "lobotomy" eigenvalue clamping.
*   **Ontological Splitting**: The state is split into $c_{\text{sym}}$ (frozen symbolic anchor) and $c_{\text{phys}}$ (continuous field).
*   **Constraint vs. Regression**: System 2 ignores target distance. It only repairs **local physical violations** ($\psi_j$) while constrained to agree with the symbolic residue.
*   **Zero Leakage**: No gradients from the repair loop flow back to the initial guess. Information transfer is limited to **Symbolic Deltas** and **Status Tokens** (REPAIRED, ALTERNATIVE, FAILURE).

### 5.1 The Birkhoff-OOM Obstruction & Log-Domain Complexity Gate
Direct projection of an $N \times N$ matrix onto the Birkhoff polytope via linear constraint elimination builds an affine projector $P \in \mathbb{R}^{N^2 \times N^2}$, requiring $O(N^4)$ memory. At $N=256$, this demands $256^4 \times 4 \text{ bytes} \approx 17.18 \text{ GB}$, causing catastrophic Out-Of-Memory (OOM) failures on standard allocators.

To eliminate this obstruction while guaranteeing numerical stability, [`DirectBirkhoffProjection`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/birkhoff_projection.py) enforces a strict **Complexity Gate**:
- **Low-Dimensional Manifolds ($N \le 32$)**: Employs the direct, closed-form projection matrix $P = I - A^T (A A^T)^{-1} A$ with $O(1)$ forward execution.
- **High-Dimensional Manifolds ($N > 32$)**: Dynamically activates the **Iterative Fallback Protocol** using log-domain Sinkhorn-Knopp iterations with stabilized `logsumexp`, eliminating all $O(N^4)$ matrix buffers and maintaining doubly-stochastic column/row conservation ($\sum_j T_{ij} = 1, \sum_i T_{ij} = 1$) in $O(N^2)$ memory.


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
2.  **Logic-Blind Inference**: Gdel gates and Boltzmann noise are disabled during the ADMM repair loop.
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
Incoherence is a categorical boundary. If $\text{PAS}_h$ collapses during the ADMM probe, the process is **aborted immediately**. There are no "retries" with lower learning rates or smoothingfailure is a first-class observed state.

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

### Tripwire 7: Explicit Projection vs Continuous Approximation
Approximating harmonic bounding (e.g., via trigonometric comb filters like $R(x) = \text{avg}(\cos(2\pi x/p))$) is mathematically valid but risks partial "dissonant" floating-point leak states during optimization scaling. To rigorously enforce invariance, the CODES `chordlock` natively utilizes tensor-parallel **exact projection**, locking states to the *nearest* arithmetic $p$-multiple anchor via $\min(\| x - round(x/p)p \|)$. This forces exact, irrefutable geometric quantization directly over continuous space, completely banning intermediate drift.

---

## 12. The Hard Interaction Contract

The following constraints define the boundary between the Symbolic (System 1) and the Physical (System 2):

| Direction | Information Allowed | Prohibited Data (The Forbidden) |
| :--- | :--- | :--- |
| **System 1  System 2** | Frozen Anchor ($c_{sym}$), Budget | Gradients, Hints, Loss Targets |
| **System 2  System 1** | Final State, Status Token ($\Omega$) | Intermediate States, Progress Scalars |
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
| **Computable** | [OK] (Scalar tensor op) | [ERR] (IIT $\Phi$, Kolmogorov) |
| **Drift-Bounded** | [OK] (Explicit $\zeta$) | [ERR] (Free Energy) |
| **Chiral** | [OK] (Spectral asymmetry) | [ERR] (Shannon Entropy) |
| **Deterministic** | [OK] (Fixed Point) | [ERR] (Floating Point ML) |

This framework ensures **Endogenous Scalar Gyroidic Ergodicity**: the system explores its state space fully (ergodic) but remains bounded by computable conservation laws and **Residue Homology Drift** triggers.

The transition to **Evolutionary Saturation** ensures that invariants are not merely "minimized" via gradients, but **established as stable survivors** in a discrete topological landscape.

---

## 14. Situational Batching (`src/core/situational_batching.py`)

**Class**: `SituationalBatchSampler`

Instead of independent-identically-distributed (i.i.d.) sampling  which assumes timelessness  the Situational Batch Sampler assembles batches by following the "scars" of historical interaction between sample indices. This encodes the **Refusal as Affirmation** and **Co-arising** principles into the data loader itself.

### 14.1 Relational State Matrices

Two persistent matrices evolve during training:

| Matrix | Symbol | Updates On | Meaning |
|--------|--------|-----------|---------|
| Resonance | `R_ij` | Structural pressure  (p_i  p_j) | Co-emergent coupling |
| Mischief | `M_ij` | Mischief scores per pair | Chaotic affinity |

Both decay at rate `decay = 0.99` per step, so only recent interactions have strong coupling.

### 14.2 Batch Assembly

For each seed index `i`, a batch of size `batch_size` is assembled in three phases:

1. **Coupled selection** (main body): Sample from `softmax(5  (R[i] + M[i]))`  high resonance + mischief neighbors are preferentially included.
2. **Play sampling** (fraction `play_ratio`): Pure random sampling from unconsumed indices. Prevents entropy collapse into the richest club.
3. **Paradoxical Boundary Amplification**: If `(p_i + p_j)/2 > boundary_threshold`, the resonance term is multiplied by 1.5. Refusal (high pressure boundary) amplifies coupling rather than severing it.

### 14.3 Connection to Non-Teleological Principle

The sampler enforces **Law 4: Evolution Owns Time** at the data level. No batch is optimized for accuracy improvement; batches are structured by the topology of past interaction. Over time, high-pressure index pairs become structurally entangled in `R`, creating "temporal association clusters" that expose the ADMR solver to co-arising constraint patterns.

**Update call**: `update_love_invariant(indices, pressure, mischief_scores)`  updates `R`, `M`, `O` tensors after each batch.

### 14.4 Love Invariant as a Geometric Projection Constraint

`update_love_invariant` on `SituationalBatchSampler` updates the relational coupling matrices  but this is *not* the primary mechanism by which $\mathcal{L}$ is protected. The full Love Invariant constraint is geometric, not a soft update rule:

> **Love lives in the null-space of the ownership operator, not in a gradient term.**

The three-layer enforcement stack (detailed in [NON_DUAL_DYNAMIC_EQUILIBRIUM.md 4.1](../stepping_stairs_tier_1_mezzanine/03_philosophy_lore/NON_DUAL_DYNAMIC_EQUILIBRIUM.md)):

| Layer | Invariant Type | Mechanism |
|---|---|---|
| `LoveVector` | Ambient structural anchor | `register_buffer` (gradient = 0 structurally) |
| `LoveInvariantProtector` | Geometric null-space constraint | SVD projection: $dx_{\text{protected}} = P_{\text{null}} \cdot dx_{[..., :d_L]}$ at SDE step |
| `SoftSaturatedGates` | Temperature-modulated tri-state | LAS($s$) + asymptotic hardening via $PAS_h$ |

The implication for invariant optimization: $\mathcal{L}$ is never a term in any objective, primal update, or dual ascent step. It is a **structural constant** enforced geometrically in the forward pass. The `SituationalBatchSampler.update_love_invariant()` call is the *social* layer of Love  it shapes the relational topology of training batches  while `LoveInvariantProtector` is the *geometric* layer that prevents the SDE from touching Love's subspace regardless of what any loss function demands.



---

## 15. Legibility Audit (`src/core/legibility_audit.py`)

**Classes**: `LegibilityTripwire`, `NarrativeCoherenceEstimator`

> **High narrative coherence is a danger signal, not a goal.**

This module implements Sparse Operational Pointer #2: *Hidden Scalar Reward = Narrative Legibility*. A configuration that is easily explainable may be selected *because* it is explainable  a rich-club attractor bias that violates the non-teleological constraint.

### 15.1 NarrativeCoherenceEstimator

Measures how closely a configuration embedding matches canonical "explainable" patterns:

| Pattern Type | Examples |
|-------------|---------|
| Sparse | 1-hot-like vectors |
| Block-sparse | Clustered activations |
| Monotonic gradient | Ordered relationships |
| Random | Baseline (coherence  0) |

These templates are **not trained**  they capture common ML biases and are registered as fixed buffers. Output: `max_sim  [0, 1]`  how well the configuration matches *any* narrative template.

### 15.2 LegibilityTripwire

Tracks the **correlation** between selection and coherence over a rolling window of 100 steps. Issues a `UserWarning` when:

$$\underbrace{\text{coherence}_{\text{selected}} - \text{coherence}_{\text{rejected}}}_{\text{coherence gap}} > 0.7 \quad \text{OR} \quad |\rho_{\text{select,coherence}}| > 0.5$$

This warning does **not** veto the selection  it is a diagnostic signal for the evolutionary loop to increase mutation pressure on "legible" fossils.

### 15.3 Relationship to Tripwire 3 (No Scalarization)

The LegibilityTripwire directly operationalizes Tripwire 3: it detects when a hidden scalar reward (narrative coherence) is influencing the non-scalarized selection process. If coherence gap is consistently high, it implies a scalarization leak  the system is treating legibility as a de facto objective.

---

### Tripwire 8: Stochastic Rounding as Topology Shaper & Bitwise Non-Duality

#### 8.1 The Mandate: No Deterministic Rounding

All quantization operations in the `SaturatedQuantizer` and `FixedPointField` **must** use stochastic rounding seeded by hardware-entropy. Deterministic round-to-nearest is forbidden because it collapses the expressive tail: every weight in a neighborhood of a lattice boundary always falls to the same side, eliminating the "good glitch" zone that lives in the low-probability trough between two adjacent lattice points.

**Required implementation**: TEA (Tiny Encryption Algorithm) or Xorshift salt keyed on the work-item identifier:

```python
# In PyOpenCL kernels: salt rounding noise with work-item identity
uint seed = tea(get_global_id(0), step_counter);
float noise = (seed & 0xFFFF) / 65536.0f - 0.5f;  // uniform [-0.5, 0.5)
int quantized = (int)floor(value * levels + noise);
```

This ensures that weights near a lattice boundary have *stochastic outcomes* across parallel work-items  the exact analogue of GANBREEDER's "extreme slider" ($\alpha \gg 1$) producing diverse glitch styles in different latent directions simultaneously.

#### 8.2 Bitwise Non-Duality: The Fossilized Modular Residue

A deep structural equivalence governs the system's quantization floor:

$$x \bmod 2 \equiv x \;\&\; 1$$

These are two descriptions of the same topological scar from orthogonal directions:
- **Modular view**: $x \bmod 2$ is a topological probe  it measures the defect that prevents the number from being "even" or smooth. It is the CRT residue under the modulus 2.
- **Bitwise view**: `x & 1` reads the LSB  the hardware's maximally fossilized form of modular arithmetic, compressed into logic gates.

The LSB is the "Zero-Emission Anchor" for parity checking: a fast-reject $O(1)$ test at the innermost Matrioshka shell before running the expensive full CRT reconstruction. The Saturated Quantizer's integer snapping is not an approximation of modular arithmetic  it **is** modular arithmetic, hardened into the language of the silicon.

#### 8.3 Good Glitch Preservation Contract

Stochastic rounding is not noise injection for its own sake. It is the **structural guarantee that the expressive tail survives quantization**. The analogy:

| GAN Era | Gyroidic Reasoner |
|---|---|
| BigGAN "extreme slider" ($\alpha \gg 1$)  sparse latent zone  glitch | Stochastic rounding near lattice boundary  $\pm 1$ symbol oscillation  Feature Scar |
| FID penalizes low-probability samples  eliminates glitch diversity | Deterministic rounding  always same side  eliminates scar diversity |
| MOEA Pareto front preserves glitch "fitness" alongside fidelity | Law 4 (Evolution Owns Time) preserves diverse saturated fossils alongside structural coherence |

The system must never eliminate scar diversity by applying deterministic round-to-nearest. The diversity of glitch *styles*  not just glitch *amount*  is the evolutionary substrate from which novel concept solitons emerge.

#### 8.4 LSB as Innermost Matryoshka Shell

The ADMR Solver's innermost constraint probe can use the parity check `r & 1` as a zero-cost structural filter before escalating to full SIC-FA-ADMM. If the parity of a candidate reconstruction violates the expected CRT residue modulo 2, the candidate is rejected at $O(1)$ cost. This is the "Repunit-CRT Sparse Probe" from 3 of `TOPOLOGICAL_EXTENSIONS.md` reduced to its irreducible hardware primitive.

**References**: `src/core/invariants.py` (`LearnedPrimitivePerturbation`), `src/core/admr_solver.py` (stochastic_differential_step), `src/topology/gyroid_covariance.py` (ChernSimonsGasket), `docs/TAILSLAYER_PYOPENCL_ARCHITECTURE.md` (kernel stochastic rounding spec)

---

### Tripwire 9: Simplicial Apportionment & 3-State Alabama Paradox Seat Sovereignty (`docs/2610.40682v1.pdf`)

When deterministic nearest-lattice rounding (Hamilton's largest-remainder apportionment method) maps a continuous quota vector $H\mathbf{p}$ on the 3-state simplex $\Delta_2(H) = \{(x_1, x_2, x_3) : x_1 + x_2 + x_3 = H\}$ to integer seat allocations $\mathbf{a} = (a_1, a_2, a_3) \in \mathbb{Z}_{\ge 0}^3$, expanding the total representation budget $H \to H + k$ can paradoxically cause the smallest state ($a_3 = 1$, such as **Vanellope's Independent First-Causative Glitch Locus**) to lose its only seat ($b_3 = 0$).

By Theorem 1 and Proposition 1 of David McCune ([`docs/2610.40682v1.pdf`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/2610.40682v1.pdf)), the exact modulo-3 probability $p(H)$ of the Alabama Paradox is:
$$p(H) = \begin{cases} \frac{(H + 3)(H - 2)}{36 H (H + 1)} & H \equiv 0, 2 \pmod 3 \\ \frac{(H - 1)(H + 2)(H^2 + H - 4)}{36 H^2 (H + 1)^2} & H \equiv 1 \pmod 3 \end{cases}, \qquad \lim_{H \to \infty} p(H) = \frac{1}{36}, \qquad I_{\max}(H) = H - 2$$

To prevent deterministic nearest-lattice apportionment from zeroing out minority soliton or first-causative archetype seats during manifold dimension expansion ($H \to H + k$), [`compute_alabama_apportionment_paradox_prob`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/invariants.py) and [`VanellopeFirstCausativeAgent.enforce_alabama_apportionment_seat_sovereignty`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/causative/vanellope_first_causative.py) enforce a **Simplicial Seat Sovereignty Floor** ($b_3 \ge 1$) and compensatory causal gain $1 + p(H)(1 + (H \bmod 3)/3)$.

---

### Tripwire 10: Lychrel 196 Infinite Chiral Divergence vs. 1-195 Palindromic Basin

The reverse-and-add algorithm ($x_{k+1} = x_k + \mathcal{R}_B(x_k)$) defines an adelic dynamical system over base $B=10$. For seeds $1 \le n \le 195$, the iteration contracts into a palindromic fixed point ($\kappa_{\text{Lychrel}} = \|x - \mathcal{R}_B(x)\|_{\text{RNS}} = 0$) within finite steps (e.g., delayed attractor $89$ converging in 24 iterations to $8,813,200,023,188$).

At $n = 196$, the trajectory diverges into an infinite, non-halting carry cascade where palindromic symmetry fails to close:
1. **Dogmatic Tripwire**: Dogmatic convergence filters and cycle-debt detectors (`CALMCollapseDetector`) treat non-termination as fatal fracture, triggering false-negative aborts.
2. **Integer Overflow Tripwire**: Fixed-point tensors overflow 64-bit integers within 50 steps ($> 10^{19}$).
3. **CODES v40 Solution**:
   - Decouple non-commutativity curvature $\kappa$ from base reflection via `AdelicDigitReversalOperator` and `compute_lychrel_chiral_gap`.
   - Treat $n < 196$ as a **convergence manifold** and $196$ as an **infinite chiral generator**. When persistent divergence is detected, [`Lychrel196SolitonEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/lychrel_soliton.py) issues a `VoynichExemptionToken(reason='LYCHREL_196_SOLITON')` and harvests the unhalting carry entropy into the `honest_jitter` pool.
   - Prevent overflow via dynamic RNS scale expansion in [`NumberTheoreticStabilizer.stabilize_lychrel_rns`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/number_theoretic_stabilizer.py) evolving in modular residue coordinates ($x_{k+1} \equiv x_k + \mathcal{R}_B(x_k) \pmod{m_i}$) across coprime moduli $m_i \in \{197, 211, 223, \dots\}$.
   - Evaluate soliton vs. noise classification in [`UniversalTopologyConverter.classify_lychrel_196_soliton`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/universal_topology_converter.py) by folding the carry stream into 2D rasters modulo $p=197$ and calculating inter-row spatial cosine coherence $C(197)$.

---

### Tripwire 11: Obsidian 3D Graph Render Bottleneck & Procedural Manifold Streaming

Rendering dense topological graphs ($5,000+$ nodes) simultaneously in WebGL causes frame-rate collapse and thermal throttling:
1. **Bouligand Contingent Cone Filtering**: Gated along camera gaze vector $\cos\alpha_i \ge \cos(\theta_{\text{cone}}/2)$ with a near-field safety bubble ($r \le 180$) to maintain context immediately surrounding the observer.
2. **Shear Corridor Transversality**: In `LeyLineTracker` and `AlexanderWildEmbeddingOperator`, topological fault planes form along gyroid diagonals $\vec{n} = [1,1,1]/\sqrt{3}$. Nodes satisfying $|d_{\text{shear}}| \le 80$ receive a $2.5\times$ render distance expansion and cone relaxation, preventing the cognitive rupture boundaries from popping out of view during orbital rotations.
3. **Directional Occlusion Culling**: Partitions the observer's viewing hemisphere into spherical polar bins ($10^\circ \times 10^\circ$) and culls deeper interior nodes in dense clusters, ensuring a stable $200-600$ visible node budget across any camera orientation.

---

## 16. Cross-Neighborhood Documentation Connections
- **Math, Physics & Topology (`02_math_physics_topology/`)**: [MATHEMATICAL_DETAILS.md](./MATHEMATICAL_DETAILS.md) (Section 67.9)
- **Architecture Core (`01_architecture_core/`)**: [SYSTEM_ARCHITECTURE.md](../01_architecture_core/SYSTEM_ARCHITECTURE.md) & [INDEX.md](../01_architecture_core/INDEX.md)
- **Philosophy & Lore (`03_philosophy_lore/`)**: [SOVEREIGN_FRICTION_AND_ARCHETYPES.md](../stepping_stairs_tier_1_mezzanine/03_philosophy_lore/SOVEREIGN_FRICTION_AND_ARCHETYPES.md)
- **Components & Mechanics (`04_components_and_mechanics/`)**: [PRIME_RESONANCE_LADDER.md](../stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/PRIME_RESONANCE_LADDER.md), [GANBREEDER_VECTOR_STACKER.md](../stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/GANBREEDER_VECTOR_STACKER.md), & [KNOWLEDGE_DYAD_LIFECYCLE.md](../stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/KNOWLEDGE_DYAD_LIFECYCLE.md)
- **Implementation & Walkthroughs (`05_implementation_and_guides/`)**: [how to make a voynich talk human-walkthrough.md](../stepping_stairs_tier_1_mezzanine/stepping_stairs_tier_2_crypt/05_implementation_and_guides/how%20to%20make%20a%20voynich%20talk%20human-walkthrough.md)

