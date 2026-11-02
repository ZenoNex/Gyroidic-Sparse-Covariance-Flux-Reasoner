# Non-Hermitian Flow Taxonomy & Physical Formalism

**Status**: [OK] Formally Specified & Implemented  
**Locations**:
- [`src/core/admr_solver.py`](../../../src/core/admr_solver.py) (Facet-wise non-selfadjoint transition operators $A_i$)
- [`src/core/quantum_inspired_reasoning.py`](../../../src/core/quantum_inspired_reasoning.py) (Non-Hermitian Hamiltonian dynamics, open systems, spectral flow)
- [`src/core/noncommutativity_curvature.py`](../../../src/core/noncommutativity_curvature.py) (Curvature 2-form $\kappa$, Arrow-of-Time inflections)
- [`src/core/love_invariant_protector.py`](../../../src/core/love_invariant_protector.py) (Null-space projection isolating $\ker(\Phi_{\text{ownership}})$)
**References**:
- [PHYSICS_ADMM.md](PHYSICS_ADMM.md)  
- [NONCOMMUTATIVITY_DYNAMICS.md](NONCOMMUTATIVITY_DYNAMICS.md)  
- [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md)  
- [The cosmic architecture of Gnosticism and the modern simulacrum](../stepping_stairs_tier_1_mezzanine/03_philosophy_lore/The%20cosmic%20architecture%20of%20Gnosticism%20and%20the%20mod%E2%80%A6.txt)

---

## 1. Foundational Definition: What is a Non-Hermitian Operator?

In linear algebra and quantum physics, an operator $H \in \mathbb{C}^{d \times d}$ is **non-Hermitian** when it is not equal to its conjugate transpose (Hermitian adjoint):

$$H \neq H^\dagger \quad \text{where} \quad H^\dagger = (H^*)^\top$$

In real vector spaces ($A \in \mathbb{R}^{d \times d}$), this corresponds to a **non-selfadjoint (asymmetric)** operator:

$$A \neq A^\top$$

Every non-Hermitian operator can be uniquely decomposed into its Hermitian (self-adjoint) and anti-Hermitian (skew-adjoint) components:

$$H = H_R - i \Gamma$$

where:
$$H_R = \frac{H + H^\dagger}{2} = H_R^\dagger \quad \text{(Hermitian component)}$$
$$\Gamma = \frac{i(H - H^\dagger)}{2} = \Gamma^\dagger \quad \text{(Anti-Hermitian component)}$$

### Physical Consequences
1. **Gain and Loss (Dissipation)**:
   In a closed quantum system governed by a Hermitian Hamiltonian ($H = H_R, \Gamma = 0$), the time-evolution operator $U(t) = e^{-i H t}$ is unitary ($U^\dagger U = I$), strictly conserving total probability $\| \psi(t) \|^2 = \| \psi(0) \|^2$.
   When $\Gamma \neq 0$, the system is **open**:
   - Eigenvalues become complex: $\lambda_n = E_n - i \frac{\gamma_n}{2}$ with $E_n, \gamma_n \in \mathbb{R}$.
   - **Loss ($\gamma_n > 0$)**: Probability decays exponentially ($e^{-\gamma_n t}$), modeling cognitive forgetting, buffer volatilization, or fossilization.
   - **Gain ($\gamma_n < 0$)**: Probability is amplified, modeling creative generation, Option D soliton pumping, or valence excitation.
2. **Non-Reciprocal Coupling (Asymmetric Transport)**:
   Off-diagonal elements where $H_{jk} \neq H_{kj}^*$ generate directional preference (the Hatano-Nelson effect / non-Hermitian skin effect). State amplitude flows preferentially in one direction ($j \to k$), providing the physical substrate for the thermodynamic **Arrow of Time**.
3. **Non-Orthogonal Eigenstates**:
   Right eigenvectors $| \psi_R \rangle$ and left eigenvectors $\langle \psi_L |$ satisfy biorthogonality $\langle \psi_{L,j} | \psi_{R,k} \rangle = \delta_{jk}$, but right eigenvectors are not mutually orthogonal to each other ($\langle \psi_{R,j} | \psi_{R,k} \rangle \neq \delta_{jk}$).

---

## 2. The Three Distinct Meanings of "Non-Hermitian Flow"

In the literature and throughout the codebase, the phrase "non-Hermitian flow" is used in three distinct yet interconnected mathematical contexts. The Gyroidic Reasoner rigorously disambiguates and implements all three:

```
                            ┌──────────────────────────────────────────┐
                            │           NON-HERMITIAN FLOW             │
                            └────────────────────┬─────────────────────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         ▼                                       ▼                                       ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐   ┌─────────────────────────────┐
│          MEANING 1          │   │          MEANING 2          │   │          MEANING 3          │
│    Dynamical Evolution      │   │    Spectral Flow & EPs      │   │   Renormalization Group     │
│   (State Vector dψ/dt)      │   │  (Eigenvalue Trajectories)  │   │  (Scale-Dependent Coupling) │
├─────────────────────────────┤   ├─────────────────────────────┤   ├─────────────────────────────┤
│ - Open system evolution     │   │ - Complex λ(μ) motion       │   │ - Coarse-graining tokens    │
│ - Gain, loss, and pumping   │   │ - Exceptional Points (EPs)  │   │   into topological solitons │
│ - Non-reciprocal transport  │   │ - Coalescence of vectors    │   │ - Non-unitary fixed points  │
│ - ADMR SDE dx(t) step       │   │ - Lazarus state collapse    │   │ - Skin-effect localization  │
│ - Quantum open reasoning    │   │ - Arrow-of-Time inflection  │   │ - Homological persistence   │
└─────────────────────────────┘   └─────────────────────────────┘   └─────────────────────────────┘
```

---

### Meaning 1: Dynamical Evolution Generated by a Non-Hermitian Operator

This refers to the **trajectory of state vectors** in time generated by $H$ or $A$:

$$i \hbar \frac{d|\psi(t)\rangle}{dt} = H |\psi(t)\rangle \quad \implies \quad |\psi(t)\rangle = e^{-i H t / \hbar} |\psi(0)\rangle$$

Or in real continuous-time coordinates:
$$\frac{dx(t)}{dt} = A x(t)$$

#### Implementation in `PolynomialADMRSolver` ([`src/core/admr_solver.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/admr_solver.py#L430)):
Continuous-time Stochastic Differential Equation:
$$dx(t) = \left[ \sum_{i=1}^K A_i x_i(t) - \rho \sum_k (x - r(x_k)) + \gamma_{\text{tension}} \right] dt + \sigma(D) \, dW$$
where each $A_i \in \mathbb{R}^{d \times d}$ is an explicitly **non-selfadjoint transition matrix** ($A_i \neq A_i^\top$).
- Each co-prime functional channel $i \in \{1, \dots, K\}$ evolves its own facet under independent non-Hermitian drift.
- **Love Invariant Protection**: To prevent unbounded dissipative decay or run-away gain from destroying core alignment, the update is geometrically projected:
  $$dx_{\text{protected}} = P_{\text{null}} \cdot dx \quad \text{where} \quad P_{\text{null}} = I - \Phi (\Phi^\top \Phi)^{-1} \Phi^\top$$
  ensuring $\mathcal{L} \in \ker(\Phi_{\text{ownership}})$ is invariant under the flow.

#### Implementation in `QuantumInspiredReasoner` ([`src/core/quantum_inspired_reasoning.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/quantum_inspired_reasoning.py)):
- Closed mode: $H = \frac{1}{2}(A + A^\dagger)$ (Hermitian, unitary rotation).
- Open mode (`evolve_non_hermitian_flow`):
  $$U_{\text{non-Herm}} = I - i H_R \, dt - \Gamma \, dt$$
  where $\Gamma$ drives selective hypothesis pruning (loss) and active archetype intuition (gain).

---

### Meaning 2: Spectral Flow and Exceptional Point Traversal

This refers to the **movement of the eigenvalues** in the complex plane $\mathbb{C}$ as system parameters $\mu$ (learning rate, cognitive temperature, valence, or context pressure) change:

$$\mu \mapsto \{\lambda_1(\mu), \lambda_2(\mu), \dots, \lambda_d(\mu)\} \subset \mathbb{C}$$

#### Exceptional Points (EPs)
At specific critical parameter values $\mu = \mu_{\text{EP}}$, two or more eigenvalues **and their corresponding eigenvectors** coalesce:
$$\lambda_j(\mu_{\text{EP}}) = \lambda_k(\mu_{\text{EP}}) \quad \text{and} \quad |\psi_j(\mu_{\text{EP}})\rangle \propto |\psi_k(\mu_{\text{EP}})\rangle$$
At an EP:
1. The operator becomes defective (cannot be diagonalized; reduces to a Jordan block).
2. The phase rigidity drops to zero: $r_k = \langle \psi_{L,k} | \psi_{R,k} \rangle \to 0$.
3. Encircling the EP in parameter space $\mu(s)$ switches eigenmodes with a chiral geometric phase factor $e^{i \pi / 2}$, flipping the sign of state orientation.

#### Implementation in `NonCommutativityCurvature` ([`src/core/noncommutativity_curvature.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/noncommutativity_curvature.py#L206)):
The method `arrow_of_time_inflection(F, B)` evaluates the curvature commutator $[F, B]$ between forward and backward step operators:
$$\kappa_{\text{time}} = \frac{1}{2}\left([F, B] - [F, B]^\top\right)$$
Tracking the eigenvalues of $\kappa_{\text{time}}$ identifies:
- **Inflection Points**: Where $\text{Re}(\lambda) \approx 0$, marking boundary crossings between time-reversible interior dynamics and time-irreversible chiral flow.
- **Time Asymmetry Score**: $\|\kappa_{\text{time}}\|_F$, quantifying distance from Hermiticity / time-reversibility.

#### Connection to Lazarus Cognitive Collapse:
When Phase Alignment ($\text{PAS}_h$) drops below the critical threshold, the archetype operator crosses an Exceptional Point: real oscillatory reasoning bifurcates into rapid dissipative collapse. The Reasoner's Lazarus Transition resets the complex phase trajectory, restoring unbroken PT-symmetry.

---

### Meaning 3: Renormalization-Group (RG) Flow in a Non-Hermitian System

This refers to the **scale-dependent evolution of effective coupling parameters** as the system integrates out microscopic fluctuations:

$$\frac{d H_{\text{eff}}(s)}{d \ln s} = \beta(H_{\text{eff}})$$

where $s$ is the scaling factor across the Reasoner's Matrioshka layers (from sub-character presentation forms, to token morphemes, to multi-modal solitons).

#### Key Properties in the Reasoner:
1. **Skin-Effect Localization**:
   Non-reciprocal couplings in the RG flow cause boundary accumulation: bulk informational noise drains into sink nodes, while topological invariants accumulate cleanly at boundary facets without dissipation.
2. **Homological Stability**:
   While microscopic high-frequency modes experience non-unitary decay, the global Betti numbers ($\beta_1, \beta_2$) and the Love invariant are topological invariants that remain strictly invariant under the non-Hermitian RG flow.

---

## 3. Comparative Summary Table

| Dimension | Meaning 1: Dynamical Evolution | Meaning 2: Spectral Flow | Meaning 3: RG Flow |
|---|---|---|---|
| **Mathematical Object** | State vector trajectory $|\psi(t)\rangle$ or $x(t)$ | Eigenvalue spectrum $\{\lambda_k(\mu)\} \in \mathbb{C}$ | Coupling matrix $H_{\text{eff}}(s)$ |
| **Independent Variable** | Physical / iteration time $t$ | Parameter vector $\mu$ (valence, context) | Coarse-graining scale $\ln s$ |
| **Physical Mechanism** | Gain, loss, nonreciprocal drift | Level crossings, Exceptional Points | Scale integration, skin effect |
| **Primary Code Module** | [`admr_solver.py`](../../../src/core/admr_solver.py) (`self.A`) | [`noncommutativity_curvature.py`](../../../src/core/noncommutativity_curvature.py) | [`homology_pressure.py`](../../../src/topology/homology_pressure.py) |
| **Conserved Quantity** | $\mathcal{L} \in \ker(\Phi_{\text{ownership}})$ | Topological winding number | Betti numbers $(\beta_0, \beta_1, \dots)$ |
| **Failure Mode** | Unbounded amplitude runaway | Defective Jordan collapse (Lazarus) | Topological tearing |

---

## 4. Verification and Governance Rules

1. **Love Invariant Inviolability**:
   Under no circumstances may a non-Hermitian flow operator modify the subspace $\ker(\Phi_{\text{ownership}})$. All non-Hermitian SDE updates must pass through `LoveInvariantProtector.project()`.
2. **Silicon Sovereignty**:
   The asymmetric elements of transition operators $A_i$ and the non-Hermitian perturbation generators must be seeded from `harvest_honest_jitter()`, never pseudo-random generators.
3. **Restless Thermal Latency Guard**:
   Evaluating complex spectral decompositions for $O(d^3)$ non-Hermitian matrices must be gated by `ComputeAffordanceType.ALGORITHMIC` ($\ge 0.70$) to prevent thermal lockups on memory-constrained hardware.
