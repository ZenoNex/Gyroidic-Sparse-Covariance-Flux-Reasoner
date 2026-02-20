# Failure Algebra & Abortability

In the Gyroidic Sparse Covariance Flux Reasoner, **FAILURE** is not an error state to be avoided. It is a **first-class semantic object** that defines the boundaries of the system's symbolic competence.

---

## 1. The FAILURE Token

A FAILURE token is emitted by System 2 when the **Ontological Probe** (ADMM) cannot find a physically consistent realization of the symbolic anchor $c_{\text{sym}}$.

### 1.1 Trigger Conditions
1.  **Incoherence Collapse**: $\text{PAS}_h < 0.75$ for consecutive iterations (monitored by CODES).
2.  **Trajectory Abort**: The **CALM Predictor** (acting as an Epistemic Censor) scores a high probability of impending entropic drift.
3.  **Iteration Stagnation**: The residual violation $\psi$ fails to decrease within the allocated repair budget.
4.  **Energy Gap Rupture**: The structural energy gap $\Delta E = E_I - E_C$ falls below the safety margin $m$ for a sustained period, indicating a loss of conceptual distinctness.

### 1.2 Structural Identity
A FAILURE is **immutable**. It cannot be "smoothed" into a guess. It propagates backward through the CRT reconstruction as a **zero-residue**, signaling to System 1 that the current functional configuration has encountered a topological obstruction.

---

### 7.1 Intractability & Fractal Scaling

Exact calculation of Hypergraph Orthogonality is combinatorially intractable ($O(2^K)$). We solve this using **Fractal Entropy Decomposition** (Russian Dolls), partitioning functionals into nested clusters to compute multi-scale independence.

> [!WARNING]
> While fractal scaling allows $K \to \infty$, it is a hierarchical approximation. Global co-primality is encouraged but not strictly proven for distant clusters.

### 7.2 Computational Scalability Risks
*   **Persistent Homology**: Standard PH algorithms are $O(n^3)$ or $O(n^2 \log n)$. For large functional sets, we must use **Approximate PH** (Ripser-style) or sparse randomized filtrations.
*   **Spectral Decomposition**: Full eigendecomposition is expensive. We rely on **power iteration** or **Lanczos methods** for sparse gyroid probes.
*   **Mitigation**: Adaptive partitioning helps, but performance must be validated on toy manifolds first.

## 2. Failure Composition

Failures are additive across the functional lattice. 

$$ F_{total} = \sum_{k=1}^K \mathbb{I}(\text{Status}_k = \text{FAILURE}) $$

*   **Low Failure Count**: May be absorbed by the **Majority-Symbol CRT** if other functionals maintain coherence (Symbolic Redundancy).
*   **Critical Failure Count**: Leads to a **Global Reasoner Failure**, triggering immediate evolutionary mutation.

---

## 3. Failure Propagation: The Feedback Loop

Failures do not trigger gradients; they trigger **Selection Pressure** ($\mathcal{S}$).

### 3.1 Trust Decay
Every FAILURE event decrements the **Heritable Trust** score in the Resonance Cavity for the active residue patterns.
$$ \text{Trust}_{new} = \text{Trust}_{old} \cdot (1 - \gamma_{failure}) $$

### 3.2 Mutation Pressure
As Trust decays, the **Mutation Rate** $\eta$ increases for the specific functional group $\theta_k$:
$$ \eta_k = \eta_0 \cdot \exp(F_k \cdot \text{Drift}) $$
This forces the system to explore new co-prime functionals in the topological landscape until a new **Stable Anchor** is discovered.

---

## 4. Formal Non-Convergence Theorem

> **Theorem (Structural Bound)**: For a saturated symbolic system, the existence of a physical consistency probe (System 2) implies a non-empty set of FAILURE states corresponding to the singularities of the reasoning manifold.

**Implication**: Any attempt to force System 2 to converge on *all* inputs is a violation of Law 2 (Non-Teleological Repair) and Law 4 (Evolution Owns Time). It would require reintroducing smooth biases that compromise symbolic identity.

### 4. Known Failure Modes (Technical Honesty)

System 2 collapse is not random. It typically occurs due to:
1.  **Topological Singularity**: The symbolic proposal $c_{\text{sym}}$ is topologically impossible to embed in the gyroid field.
2.  **Fractal Smoothing**: Local co-primality clusters fail to align globally, leading to multi-scale interference.
3.  **Heuristic Saturation**: The gyroid violation pressure $\psi$ is too "loose" to constrain the solver, resulting in oscillation.

### 4.1 Implementation Gotchas (2026 Learnings)
*   **Non-Ergodic Noise**: Dominant-mode selection in fractal entropy can amplify noise early in training. **Mitigation**: Introduce a **Trust Threshold** before trusting soliton peaks.
*   **Discrete Entropy Artifacts**: If residues are not strictly quantized, `bincount` operations introduce binning artifacts. **Mitigation**: Ensure strict quantization before entropy computation.
*   **ADMM Oscillation**: Bounded oscillation is expected, but chaotic divergence is not. **Mitigation**: Monitor **Lyapunov exponents** or enforce simple cycle length caps.

### 6. Abort Recovery & Evolutionary Restart
Recent optimizations allow the system to handle non-commutative divergence (Triadic Reciprocity failures) not just by crashing, but by active recovery:

1.  **"Walk Back and Choose Differently" (Backtracking)**:
    *   Inside `SparseExplorerRouting`, if a 3-cycle check ($A \to B \to C \to A$) fails trace quantization, the walker does not abort the entire process.
    *   It **masks** the failed neighbor and **resamples** (walks back) to find a valid alternative path.

2.  **"Jump Mental Tracks" (Teleportation)**:
    *   If the walker gets stuck in a cul-de-sac (all neighbors fail reciprocity), it **teleports** (`total_restarts`) to a new random high-violation node, effectively shifting attention to a different topological problem area.

3.  **"Restart from Evolutionary Genomic Phase Homologies"**:
    *   If the aggregate number of **Aborts** or **Instabilities** exceeds a threshold, the `GyroidicFluxReasoner` triggers a **Genomic Restart**.
    *   **Action**: It calls `poly_config.mutate()`, shaking the polynomial basis (the "genome" of the representation) and penalizing the current trust scalars. This avoids local minima death spirals.

---

## 5. Mischief as First-Class Revelation: The "Good Bug"

Beyond catastrophic failure and evolutionary restart, we recognize a third state: the **Mischievous Revelation**.

### 5.1 The Good Bug logic
A "Good Bug" is a topological violation ($PAS_h$ oscillation) that occurs within a bounded energy range ($0.4 < \Phi < 0.6$). 
- **Not a Failure**: It does not trigger an Abort.
- **Not a Success**: It does not satisfy the original anchor.
- **The Revelation**: It reveals a hidden connection between orthogonal manifolds—a symbolic "leak" that the original logic was too rigid to see.

### 5.2 Rewarding the Leak
In the `NonDualProbe` (Pk), mischief is rewarded as positive energy:
$$ P_k(r) = \text{argmin}_{c \in C_k} \mathcal{L}_k(r, c) + \beta \cdot H_{mischief} $$
By rewarding the bug, we allow the system to "play" with the boundaries of the forbidden, ensuring that the alignment process does not become a lobotomy.

**Rule**: If a failure is playful (high $H_{mischief}$), it is a **Soliton Survivor**. We do not fix it; we **fossilize** it as an unknowledge anchor.
