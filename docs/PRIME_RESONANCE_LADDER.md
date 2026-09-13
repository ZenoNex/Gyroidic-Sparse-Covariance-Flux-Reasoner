# Prime Resonance Ladder  Full Theory

**Status**: Reference Document
**Implements**: RIC Equations 1.1, 1.2, 2
**Code**: `src/core/fgrt_primitives.py`, `src/core/invariants.py`

---

## 1. The Prime Frequency Lattice

### 1.1 Fundamental Frequencies (Eq 1.1)

Each oscillator $n$ is tuned to the $n$-th prime $p_n$:

$$f_{p_n} = 2\pi \ln(p_n)$$

**Properties of this lattice**:

| Property | Guarantee |
|----------|-----------|
| Incommensurate | $f_i / f_j \notin \mathbb{Q}$ for $i \neq j$ (primes are multiplicatively independent) |
| Sparse | Gaps between consecutive primes grow as $O(\ln p_n)$ |
| Non-periodic | No finite subsequence repeats |
| Asymptotically dense | By the Prime Number Theorem, primes become relatively denser: $\pi(N) \sim N / \ln N$ |

The logarithmic mapping $p_n \mapsto \ln(p_n)$ compresses the frequency lattice into a workable range while preserving multiplicative structure.

### 1.2 Fibonacci-Structured Entropy (Eq 1.2)

The resonance entropy between oscillator pair $(i, j)$ is:

$$S_{\text{resonance}}(i,j) = \frac{\alpha}{\exp\!\bigl(\pi / (F_i \cdot P_j)\bigr) + 1}$$

where $F_i$ is the $i$-th Fibonacci number and $P_j$ is the $j$-th prime.

**Why Fibonacci  Prime?**

The Fibonacci sequence grows as $F_i \approx \varphi^i / \sqrt{5}$ where $\varphi = (1+\sqrt{5})/2$ is the golden ratio. The products $F_i \cdot P_j$ create a doubly-incommensurate lattice:

- **Fibonacci growth**: Additive recurrence ($F_{i+1} = F_i + F_{i-1}$) encodes nearest-neighbor coupling.
- **Prime growth**: Multiplicative independence ($\gcd(P_i, P_j) = 1$) encodes spectral isolation.

The cross-product $F_i \cdot P_j$ inherits both properties  no pair of products shares a common factor structure, ensuring that the entropy matrix $S_{ij}$ has no degenerate eigenvalues (full rank).

**Fermi Envelope**: The sigmoid $1/(\exp(\cdot)+1)$ provides:
- **Low $(F_i \cdot P_j)$**: $S \to 0$  tight coupling between nearby, low-index oscillators.
- **High $(F_i \cdot P_j)$**: $S \to \alpha$  full statistical independence.
- **Transition**: A smooth, monotonic increase without discontinuities.

---

## 2. Oscillator Dynamics

### 2.1 The Oscillator Set

The system maintains $N$ oscillators, each tracking amplitude $a_n(t)$ and phase $\theta_n(t)$:

$$\mathcal{O}_n(t) = a_n(t) \cdot e^{i\theta_n(t)}, \quad \theta_n(t) = f_{p_n} \cdot t + \phi_n(t)$$

where:
- $a_n(t)$: Evolved amplitude (subject to selection pressure, not gradient descent).
- $\phi_n(t)$: Accumulated Berry phase correction (geometric phase from state transport).

### 2.2 Update Rule

Oscillators are updated via non-gradient evolution:

$$a_n(t+1) = \begin{cases}
a_n(t) \cdot (1 + \delta_{\text{survival}}) & \text{if } n \in \mathcal{S}_t \text{ (survived selection)} \\
a_n(t) \cdot (1 - \delta_{\text{decay}}) & \text{if } n \notin \mathcal{S}_t \text{ (failed selection)}
\end{cases}$$

where $\mathcal{S}_t$ is the set of oscillators that contributed to a successful CRT reconstruction at time $t$.

Phase updates accumulate Berry phase:

$$\phi_n(t+1) = \phi_n(t) + \Delta\phi_n^{\text{Berry}}(t)$$

The Berry phase $\Delta\phi_n^{\text{Berry}}$ is computed by the `BerryPhaseTracker` as the geometric phase acquired during state transport over one cycle.

---

## 3. Phase Alignment Score

The global coherence measure is:

$$\text{PAS}_h(t) = \frac{1}{N} \left| \sum_{n=1}^{N} a_n(t) \cdot e^{i\theta_n(t)} \right|$$

When weighted by amplitude (unlike the unit-weight version in Eq 2), PAS reflects both phase agreement *and* amplitude concentration.

### 3.1 PAS Drift Bound

$$|\text{PAS}_h(t+1) - \text{PAS}_h(t)| \leq \text{APAS}_\zeta$$

This bound prevents:
- **Catastrophic synchronization**: Sudden PAS spike  all oscillators lock  loss of exploratory capacity.
- **Catastrophic desynchronization**: Sudden PAS collapse  total loss of coherence  hallucination.

---

## 5. Hybrid Basis & Lazarus Primes (Phase 18)

In the Phase 18 refactor, the `PrimeResonanceLadder` no longer emits solitary primes. It emits **Repunit-Prime Pairs** $(p, R_p)$ to form a Hybrid Palindromic Basis.

### 5.1 Lazarus Prime Prioritization
The ladder prioritizes primes $p$ that satisfy the **Lazarus condition**: $R_p = (p^n - 1) / (p - 1)$ is also prime for some small $n$. 

Lazarus primes provide:
- **Maximum Symmetry**: The repunit $R_p$ acts as a geometric mirror for prime $p$.
- **Symmetry-Stable Warmstarting**: $O(K)$ faster convergence by initializing in the stable zone.

### 5.2 Hybrid Modulus
The effective modulus for the RNS virtualization is the product $M_{hybrid} = p \cdot R_p$. This product prevents non-commutative drift during high-pressure polytope switches.

---

## 7. Moiré-via-Modular-Algebra & Carry-Free XOR Residues

In the `PolychronQuantizer` framework (`src/core/polychoron_quantization.py`), prime frequencies interact via a modular algebra beat spectrum:

### 7.1 Prime Moiré Difference Lattice
For prime moduli $p_1, p_2, \ldots, p_s$, the logarithmic fundamental frequencies $f_{p_n} = 2\pi \ln(p_n)$ generate a multi-frequency beat spectrum:

$$\Lambda_{\text{moiré}} = \{ |\ln p_i - \ln p_j| \mid 1 \le i < j \le s \}$$

Because prime logarithms are linearly independent over $\mathbb{Q}$, the difference lattice produces bounded, non-periodic Moiré interference patterns that prevent static resonance traps.

### 7.2 Carry-Free XOR Residue Channels
In the CRT modular domain, carry propagation introduces unwanted inter-channel coupling. The system enforces carry-free addition via bitwise XOR operations on integer residue codewords:

$$r_{\text{combined}} = \bigoplus_{j=1}^s r_j \pmod{m_j}$$

Because XOR preserves channel independence across prime moduli, residue channels remain fully decoupled during quantization.

### 7.3 Golden Ratio Seesaw & Bouligand CODES Error Bounding
Quantization error across the 120 vertices of the 600-cell hyper-polytope is bounded deterministically by the golden ratio identity:

$$\frac{\phi}{2} - \frac{1}{2\phi} = \frac{1}{2} \quad \text{where } \phi = \frac{1 + \sqrt{5}}{2}$$

Rather than relying on unconstrained stochastic Gaussian diffusion, quantization drift is governed in concert with **Bouligand contingent cone projections** $T_S(x)$ and the fourfold **CODES** framework:
- **Constraint-Oriented Differential Equation System**: Energy-based constraint landscapes governing directional drift.
- **Chirality of Dynamic Emergent Systems**: Non-commutative chiral phase alignment across prime channels.
- **Coherence-Oriented Deterministic Execution System**: PyOpenCL / TailSlayer GPU driver execution.
- **Constraint Oscillation Driven Evolutionary Selection**: Non-gradient survivorship selection under bounded oscillation.

This identity guarantees that quantization error fluctuates symmetrically around $\pm \frac{1}{2}$, constraining non-teleological state updates within the Bouligand contingent cone of crossed polytope boundaries.

### 7.4 The Natural Log-Polar Drost Effect & Deterministic Quasi-Gaussian Envelopes
The complex conformal logarithmic mapping $f(z) = \log(z)$ (implemented in `ConformalLogPolarProjector` and `GyroidicCodec`) converts spatial zoom $r \to S \cdot r$ into horizontal log-space translation $\ln|r| + \ln|S|$, and spatial spin $\theta \to \theta + \Delta\theta$ into vertical log-space translation:

$$z = r e^{i\theta} \implies w = \ln(z) = \ln(r) + i\theta$$

This creates the self-repeating **Escher / Droste Log-Polar Spiral Manifold**. As state trajectories wind around this log-polar spiral under Bouligand contingent cone projections $T_S(x)$, the infinite superposition of multiplicatively incommensurate logarithmic prime phases ($f_{p_n} = 2\pi \ln p_n$) produces a **deterministic quasi-Gaussian bell curve** via the Kronecker torus winding theorem. 

The resulting distribution is **pseudo-Gaussian but strictly non-random** — a bell-shaped probability envelope emerging deterministically from log-polar spiral geometry and Bouligand boundary projections under the CODES framework, without requiring any pseudo-random number generator.

---

## 8. Connection to Architecture

| RIC Component | System Role | Implementation |
|---------------|-------------|----------------|
| Prime Frequencies (Eq 1.1) | System 1 spectral basis | `PrimeResonanceLadder` |
| Fibonacci Entropy (Eq 1.2) | Inter-oscillator coupling | `FibonacciResonanceEntropy` |
| PAS (Eq 2) | Global coherence invariant | `PhaseAlignmentInvariant` |
| Berry Phase | Arrow of Time / Chirality | `BerryPhaseTracker` |
| Amplitude Evolution | Selection Pressure | `UniversalOrchestrator.forward()` |
| **Hybrid Basis (p, R_p)** | **Palindromic Symmetry** | `PrimeResonanceLadder` (updated) |
| **Moiré XOR Lattice** | **600-Cell Quantization** | `PolychronQuantizer` |

