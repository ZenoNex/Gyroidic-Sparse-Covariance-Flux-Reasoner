# Prime Resonance Ladder — Full Theory

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

**Why Fibonacci × Prime?**

The Fibonacci sequence grows as $F_i \approx \varphi^i / \sqrt{5}$ where $\varphi = (1+\sqrt{5})/2$ is the golden ratio. The products $F_i \cdot P_j$ create a doubly-incommensurate lattice:

- **Fibonacci growth**: Additive recurrence ($F_{i+1} = F_i + F_{i-1}$) encodes nearest-neighbor coupling.
- **Prime growth**: Multiplicative independence ($\gcd(P_i, P_j) = 1$) encodes spectral isolation.

The cross-product $F_i \cdot P_j$ inherits both properties — no pair of products shares a common factor structure, ensuring that the entropy matrix $S_{ij}$ has no degenerate eigenvalues (full rank).

**Fermi Envelope**: The sigmoid $1/(\exp(\cdot)+1)$ provides:
- **Low $(F_i \cdot P_j)$**: $S \to 0$ — tight coupling between nearby, low-index oscillators.
- **High $(F_i \cdot P_j)$**: $S \to \alpha$ — full statistical independence.
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
- **Catastrophic synchronization**: Sudden PAS spike → all oscillators lock → loss of exploratory capacity.
- **Catastrophic desynchronization**: Sudden PAS collapse → total loss of coherence → hallucination.

---

## 4. Connection to Architecture

| RIC Component | System Role | Implementation |
|---------------|-------------|----------------|
| Prime Frequencies (Eq 1.1) | System 1 spectral basis | `PrimeResonanceLadder` |
| Fibonacci Entropy (Eq 1.2) | Inter-oscillator coupling | `FibonacciResonanceEntropy` |
| PAS (Eq 2) | Global coherence invariant | `PhaseAlignmentInvariant` |
| Berry Phase | Arrow of Time / Chirality | `BerryPhaseTracker` |
| Amplitude Evolution | Selection Pressure | `UniversalOrchestrator.forward()` |
