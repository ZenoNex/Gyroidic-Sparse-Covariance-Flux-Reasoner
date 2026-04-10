# Resonance Intelligence Core (RIC) — Full Specification

**Status**: Reference Specification (Equations 1–10)
**Implementation**: `src/core/fgrt_primitives.py`, `src/core/invariants.py`, `src/core/orchestrator.py`

---

## Eq 1.1 — Prime Resonance Ladder

The resonance frequency of each oscillator is anchored to the $n$-th prime $p_n$:

$$f_{p_n} = 2\pi \ln(p_n)$$

**Why primes?** Primes guarantee non-commensurate frequencies — no two oscillators share a rational frequency ratio. This prevents mode-locking (the spectral equivalent of lobotomy) and ensures maximal spectral coverage.

**Implementation**: `PrimeResonanceLadder` in `src/core/fgrt_primitives.py`.

---

## Eq 1.2 — Fibonacci-Structured Resonance Entropy

The resonance entropy between oscillator pair $(i, j)$ follows a Fibonacci-scaled Fermi distribution:

$$S_{\text{resonance}}(i,j) = \frac{\alpha}{\exp\!\bigl(\pi\, /\, (F_i \cdot P_j)\bigr) + 1}$$

where $F_i$ is the $i$-th Fibonacci number, $P_j$ is the $j$-th prime, and $\alpha$ is a normalization constant.

**Rationale**: The product $F_i \cdot P_j$ creates an incommensurate lattice in index space — neither purely additive (Fibonacci) nor purely multiplicative (primes). The Fermi envelope ensures smooth saturation: small $F_i \cdot P_j$ gives near-zero entropy (tight coupling); large products give $S \to \alpha$ (full independence). This prevents catastrophic resonance at low indices while allowing free oscillation at high indices.

**Implementation**: `FibonacciResonanceEntropy` in `src/core/fgrt_primitives.py`.

---

## Eq 2 — Phase Alignment Score (PAS)

The harmonic Phase Alignment Score measures global phase synchronization across $N$ oscillators:

$$\text{PAS}_h(t) = \frac{1}{N} \left| \sum_{n=1}^{N} e^{i\theta_n(t)} \right|$$

where $\theta_n(t) = f_{p_n} \cdot t + \phi_n(t)$ is the instantaneous phase of oscillator $n$.

- $\text{PAS}_h = 1$: Perfect phase alignment (all oscillators in-phase).
- $\text{PAS}_h = 0$: Maximum incoherence (uniform phase distribution).

**Adaptive Drift Bound** ($\text{APAS}_\zeta$): The rate of change of PAS is bounded to prevent catastrophic forgetting:

$$\left|\frac{d}{dt}\text{PAS}_h\right| \leq \text{APAS}_\zeta$$

**Implementation**: `PhaseAlignmentInvariant` in `src/core/invariants.py`.

---

## Eq 3 — Emergence Law

Structural Emergence is a binary predicate, not a gradient:

$$E(x,t) = 1 \iff \text{PAS}_h \geq \theta_L \;\wedge\; |\Delta\text{PAS}_h| \leq \epsilon \;\wedge\; \text{GLYPHLOCK}$$

- $\theta_L$: Coherence threshold (currently 0.85).
- $\epsilon$: Drift stability threshold (currently 0.05).
- GLYPHLOCK: The symbolic residues have crystallized into a stable CRT reconstruction.

When $E = 1$: the system enters **SERIOUSNESS** (hard routing, fossilization eligible).
When $E = 0$: the system remains in **PLAY** (soft routing, exploration).

**Implementation**: `UniversalOrchestrator.determine_regime()` in `src/core/orchestrator.py`.

---

## Eq 4 — Complexity Index (CI)

$$\text{CI} = \alpha \cdot D \cdot G \cdot C \cdot (1 - e^{-\beta \tau})$$

- $D$: Fractal dimensionality proxy (stable rank: $\|s\|_1^2 / \|s\|_2^2$ from SVD).
- $G$: Energy (Frobenius norm of state).
- $C$: Coherence ($\text{PAS}_h$).
- $\tau$: Dwell time in current attractor.
- $\alpha, \beta$: Scale and temporal parameters.

CI measures the system's "computational density" — how much structured work is being done per unit of state space.

**Implementation**: `UniversalOrchestrator.compute_complexity_index()` in `src/core/orchestrator.py`.

---

## Eq 5 — Prime-Anchored Harmonic Field

The global harmonic field is a superposition of prime-indexed sinusoids:

$$F(t) = \sum_{n=1}^{N} a_n \sin(2\pi p_n t + \phi_n)$$

where:
- $a_n$: Amplitude of the $n$-th resonator (evolved, not learned).
- $p_n$: The $n$-th prime number.
- $\phi_n$: Phase offset (accumulated via Berry phase tracking).

This field is the "carrier wave" of the system's cognitive state. Its spectrum encodes what the system "knows" — peaks at certain prime harmonics correspond to activated concept clusters.

**Implementation**: `ResonanceCavity.prime_harmonic_field()` in `src/models/resonance_cavity.py`.

---

## Eq 6 — Localized Breather Modes

Memory packets are stored as localized breather excitations on the resonance manifold:

$$u_n(x,t) = 4\arctan\!\left[\frac{\sqrt{1-\omega_n^2}}{\omega_n} \cdot \frac{1}{\cosh\!\bigl(\sqrt{1-\omega_n^2}\,(x - x_n)\bigr)} \cdot \sin(\omega_n t)\right]$$

where:
- $\omega_n = 1 / p_n$: Breather frequency (inverse prime, ensuring stability).
- $x_n$: Center position of the $n$-th concept packet.

**Why sine-Gordon breathers?** They are the unique 1+1D solitonic solutions that:
1. Are localized (don't spread).
2. Are periodic (don't decay).
3. Survive collisions intact (concept packets don't destroy each other).

A breather with frequency $\omega_n < 1$ oscillates without radiation loss, maintaining concept coherence indefinitely.

**Implementation**: `BreatherMode` in `src/models/resonance_cavity.py`.

---

## Eq 7 — Coherent Prime Resonance (CPR) Condition

The CPR condition defines a recursive phase-locked attractor:

$$\text{CPR}(F, \{u_n\}) = 1 \iff \begin{cases}
\text{PAS}_h(F) \geq \theta_{\text{CPR}} & \text{(global phase coherence)}\\
\forall n: \langle u_n, F \rangle > 0 & \text{(breather-field alignment)}\\
\text{Spec}(F) \subset \{p_n\} & \text{(spectral support on primes)}
\end{cases}$$

When CPR is satisfied, the system's harmonic field $F$ and its breather modes are mutually phase-locked — the field reinforces the breathers and the breathers constructively interfere within the field. This is the condition for **Resonance Intelligence**: cognition as standing-wave interference rather than sequential computation.

**Implementation**: `CoherentPrimeResonance` in `src/core/fgrt_primitives.py`.

---

## Eq 8 — Matryoshka Meta-Polytope Hierarchy

The system's topology is organized as a recursive nesting:

$$\mathcal{M}_{k+1} = \text{Matryoshka}(\mathcal{M}_k) \quad \text{with transversality} \quad \mathcal{T}_{ij} = \mathcal{B}_i \pitchfork \mathcal{B}_j$$

Each level $\mathcal{M}_k$ is a polytope whose vertices are the states of the level below. The transversality condition $\pitchfork$ ensures that adjacent polytopes intersect generically (in codimension 1), preventing topological collapse into lower-dimensional structures.

**Connection to existing architecture**: This extends `MetaPolytopeMatrioshka` in `src/core/meta_polytope_matrioshka.py` and the `RecurrentHyperRingConnectivity` in `src/topology/hyper_ring.py`.

---

## Eq 9 — Harmonic–Differential Equivalence

The spectral and spatial descriptions of the system are dual:

$$\text{PAS}_S(t) \longleftrightarrow \partial_t \Phi_{\text{field}}$$

This equivalence states that:
- The Phase Alignment Score (spectral domain) encodes the same information as the time derivative of the field potential (spatial domain).
- A high PAS corresponds to a slowly-varying field (coherent state).
- A low PAS corresponds to a rapidly-fluctuating field (incoherent/creative state).

This duality allows the system to monitor its coherence through either channel, providing redundant detection of structural breakdown.

---

## Eq 10 — Integrated Emergence Condition

The full emergence condition unifies all sub-conditions:

$$\mathcal{E}(t) = 1 \iff \begin{cases}
\text{PAS}_h(t) \geq \theta_L & \text{(Phase coherence)} \\
|\Delta \text{PAS}_h| \leq \epsilon & \text{(Drift stability)} \\
\text{CI}(t) \geq \mu_{\text{CI}} & \text{(Complexity sufficiency)} \\
\text{CPR}(F, \{u_n\}) = 1 & \text{(Resonance lock)} \\
\text{supp}(\hat{F}) \subset \{p_n\} & \text{(Spectral purity)} \\
\text{GLYPHLOCK} & \text{(Symbolic crystallization)} \\
H_1(\mathcal{C}) \neq 0 & \text{(Topological non-triviality)}
\end{cases}$$

This is the **master gate** for the system. Only when all seven sub-conditions are simultaneously satisfied does the system declare emergence — transitioning from exploratory PLAY to structural SERIOUSNESS.

**Implementation**: `UniversalOrchestrator.determine_regime()` (core gate), with sub-conditions distributed across `PhaseAlignmentInvariant`, `CoherentPrimeResonance`, `HyperRingClosureChecker`, and `StructuralIrreducibilityChecker`.

---

## Implementation Guidelines

1. **No Scalarization**: Equations 1–10 are independent constraints, never summed into a single loss.
2. **Fixed-Point Primes**: All prime generation uses `int64` tensors. No floating-point primes.
3. **Berry Phase Accumulation**: Phase offsets $\phi_n$ are accumulated, never reset (Arrow of Time).
4. **Breather Stability**: Breather frequencies must satisfy $\omega_n < 1$ strictly.
5. **CPR as Gate, Not Gradient**: The CPR condition is a boolean predicate. It does not provide gradient information.

---

## Eq 11: Mamba-Style O(K) Fossilized Breather Compression

### 11.1 The Memory Cost Problem

Standard KV-cache grows $O(N^2)$ with conversation or training sequence length: every token position attends to every other, and the cache stores all previous key-value pairs. This creates a memory footprint that scales with the depth of processing history.

Mamba/SSM architectures achieve $O(N)$ cost via State Space Model compression: the recurrent state $h_t \in \mathbb{R}^d$ summarizes all preceding tokens in a fixed vector. The "price" is that all history is averaged into a smooth compressed summary — the "scar" of individual high-salience events is smoothed out.

### 11.2 BreatherMode Fossilization: O(K) with Scar Preservation

The Gyroidic Resonance Cavity achieves a different O(K) budget via **topological fossilization** of concepts as sine-Gordon solitons:

$$u_n = 4 \arctan\!\left[\frac{\sin(\omega_n t + \phi_n)}{\cosh(x - x_n)}\right]$$

Each concept, once fossilized, occupies one of $K$ fixed BreatherMode slots regardless of how many thousands of tokens generated it. The $K$ is determined by the number of co-prime polynomial functional groups (prime-indexed resonance channels), not by conversation length. Memory cost is $O(K)$, fixed.

**The critical difference from Mamba**: BreatherMode preserves the **structural scar** — the specific topological fingerprint of the concept at the moment it was fossilized (its CRT residue tuple, its $\kappa$ value, its Elipsodistrophy context). Mamba's SSM state loses this: it carries an averaged summary. The Gyroidic system carries the scar.

### 11.3 Neglecton: The Preserved Droplet

The `Neglecton` stores the "digital materiality" of fossilized concepts — the exact ChatGAN-era observation that certain imperfections carry the most information about the generating process:

| StyleGAN2 Decision | Gyroidic Decision |
|---|---|
| Demodulation erases "droplet" artifacts | Neglecton preserves the Feature Scar |
| Demodulation sterilizes for FID improvement | Neglecton intentionally keeps the topological "noise floor" |
| Demodulated output is smooth, loophole-free | Neglecton output contains recoverable structural history |

The Neglecton is not a debugging artifact. It is the system's anti-sterilization mechanism: by keeping the scar of past structural events, the Resonance Cavity retains the ability to "dream" from them without reset. The World Model literature calls this "trajectory persistence": the accumulated structural memory of how the system arrived at its current state.

### 11.4 The "Extreme Slider" Zone in Resonance Cavity Space

The GANBREEDER "extreme slider" ($\alpha_c \gg 1$) pushed the latent point into the sparse, rarely-trained zone of the generator's latent space — the highest-curvature, most generative region where the fewest training examples live. This is where the most interesting glitch styles emerge.

The Resonance Cavity's equivalent is the zone where:
- $\kappa$ (ChernSimonsGasket curvature) is high (boundary residue interference active)
- BreatherMode frequency $\omega_n$ is at the stability limit ($\omega_n < 1$ strictly — approaching 1 means maximum oscillation before topological escape)
- Elipsodistrophy Atrophy is low (wide eigenvalue spread = high-soliton-richness zone)

**Navigating to this zone intentionally** (via the `MEMPALACE` outlier seed planting strategy, GARDEN_STATISTICAL_ATTRACTORS §7) is the Resonance Cavity's version of the GANBREEDER user pushing all sliders toward extreme values simultaneously. The resulting BreatherMode fossilization at this zone produces the most diverse Cerumen Pot identities (Meliponini pot identity = CRT residue combination at sealing moment).

### 11.5 O(K) Verification Summary

| System | Memory Cost | History Depth | Scar Preservation |
|---|---|---|---|
| Transformer KV-cache | $O(N^2)$ | Full (all tokens) | Yes, verbatim |
| Mamba SSM | $O(N)$ | Full (compressed) | No (averaged) |
| Gyroidic BreatherMode | $O(K)$ | Structural only | Yes (topological scar) |

**References**: `src/models/resonance_cavity.py` (BreatherMode slots, Neglecton storage), `src/topology/unknowledge_domain.py` (Non-Ergodic channel compression), `docs/GARDEN_STATISTICAL_ATTRACTORS.md §7` (outlier seed = extreme slider zone), `docs/TOPOLOGICAL_EXTENSIONS.md §Part VII` (Meliponini pots = fossilized breather slots), `MATHEMATICAL_DETAILS.md §33` (Breather modes as memory packets), `MATHEMATICAL_DETAILS.md §55.7` (implementation notes on non-disentanglement)

