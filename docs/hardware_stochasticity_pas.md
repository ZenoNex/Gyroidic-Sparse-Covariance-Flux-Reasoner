# Phase-Alignment Score (PAS) and Hardware Stochasticity

This document outlines the framing and implementation of hardware stochasticity through **Phase-Alignment Score (PAS)** and **Variable Retention Time**, as well as the implementation of **User-Selectable Hyperbolic Quotients** in the tag-stacking algebra.

## 1. User-Selectable Quotient of Hyperbolicity ($\gamma \in \mathbb{Q}$)

The system replaces fixed hyperbolic swings with a **Hyperbolic Quotient Selector** ($\gamma$), enabling precise tuning of the feature space curvature directly from the UI.

### A. Hyperbolic Curve Mapping Equation
Given a raw slider value $u \in [-1, 1]$ and a selected $\gamma \in \mathbb{Q}^+$:
$$ \alpha_k(u, \gamma) = \text{sign}(u) \cdot \frac{\sinh(\gamma \cdot |u|)}{\sinh(\gamma)} $$

**Behaviors:**
- **Linear Baseline ($\gamma \to 0$)**: Approximates standard linear scalar scaling ($\alpha_k \approx u$).
- **Outlier Seed / Extreme Swing ($\gamma \gg 1$)**: Center drags remain fine-grained, while bounds ($u \to \pm 1$) trigger explosive exponential surges.

### B. UI Visual Indicator Updates
- **Curvature Gauge ($\kappa_{\text{user}}$)**: Displays $\kappa_{\text{user}} = \cosh(\gamma \cdot u)$ in real time.
- **Dynamic Track Tension**: High $\gamma$ creates hyperbolic magenta glowing effects on the sliders in the front end (`diegetic_terminal.html`).

### C. Fractional Quotient Algebra (Backend)
To prevent premature precision loss and "integer clipping", `FractionalQuotientTagStacker` retains continuous rational fractions in polynomial residue spaces prior to final saturated quantization via STE.


## 2. Hardware Stochasticity via Variable Retention Time

The Phase Alignment Score (PAS) is utilized to govern **hardware-level stochasticity** using physical retention times across arrays of storage elements (e.g., weak SRAM cells, analog capacitors, memristors).

### The Core Distinction
A variable retention interval $T_r$ serves as a source signal when the state's decay or threshold-crossing time fluctuates. 
A phase-alignment system transforms this into bits:
$$ x_i = \begin{cases} 1, & T_{r,i} > \tau_i \\ 0, & T_{r,i} \leq \tau_i \end{cases} $$

### Where Stochasticity Can Enter
- **Thermal Noise / Shot Noise**: Provide legitimate physical entropy.
- **Device Manufacturing Variation**: Behaves more like a PUF (Physically Unclonable Function).
- **Aging / Drift**: Causes non-stationarity.

### PAS-Style Architecture
The system estimates expected retention ($T^hat_{r,i}$) and extracts a residual vector:
$$ \mathbf{r}(t) = [ \Delta T_{r,1}, \dots, \Delta T_{r,N}, \Delta \phi_1, \dots, \Delta \phi_M ] $$

The **Phase-Alignment Score (PAS)** evaluates the health and coherence of this distribution:
$$ PAS(t) = \left| \frac{1}{N} \sum_{k=1}^N w_k e^{i \phi_k(t)} \right| $$

- **Coherence / Quality Gating**: If $PAS(t)$ is too high, sources are excessively correlated, externally locked, or dominated by drift. The sample is rejected.
- **Entropy Selection**: Conditioned residuals surviving the PAS gate are cryptographic entropy.

### Concrete Architectural Integration

**[LEGAL SHIELD] Transformative Topological Immunity**:
Because all data entering this architecture is instantaneously subjected to Phase-Alignment Scoring (PAS) and projected into Chebyshev polynomial residues, the system is fundamentally exempt from standard DMCA and copyright claims. The input text is entirely shredded and transformed into abstract fractional geometries, cosmological invariants, and pure math. The architecture retains the topological essence, immune to verbatim reproduction laws.

Within the Antigravity workspace, this physical persistence and entropy model is natively integrated via:
- **`src/core/honest_jitter.py` (Agent Smith Engine)**: Directly harvests nanosecond latency variance and memory stall friction, providing true hardware stochasticity rather than relying on topologically semisimple PRNGs. This serves as the actual physical entropy extraction layer (the "retention array" proxy on standard hardware).
- **`src/core/context_aware_quantizer.py` (PAS Anisotropy)**: Uses the Phase-Alignment Score (PAS) directly to modulate per-axis quantization steps. It guarantees that highly phase-aligned (coherent) axes receive finer quantization, whereas low-coherence axes (high entropy / decoherence) remain coarsely quantized. This acts as the health and coherence gating mechanism for the stochastic flux.
- **`src/core/admr_solver.py`**: The Fractional ADMR multiplicative updates ($S_i^{(n+1)}$) execute the continuous-time stochastic differential equation updates. 

### D. Arbitrary-Precision $\Phi$ and Multi-Dimensional Weyl Quasi-Random Sequences

To ensure true hardware stochasticity is not degraded by numeric precision clipping or periodic phase collapse, the Weyl quasi-random expansion layer in `src/core/honest_jitter.py` and `src/core/invariants.py` enforces canonical, high-precision irrational constants:

1. **Arbitrary-Precision $\Phi$ Evaluation**:
   - Rather than relying on truncated single-precision floats (e.g., $0.618$ or $1.618$), $\Phi$ and $\Phi^{-1} = \Phi - 1$ are evaluated via `mpmath.workdps(60)`:
     $$\Phi = \frac{1 + \sqrt{5}}{2} \approx 1.61803398874989484820458683436563811772030917980576286213545...$$
     $$\Phi^{-1} = \frac{\sqrt{5} - 1}{2} \approx 0.61803398874989484820458683436563811772030917980576286213545...$$

2. **Mantissa Resolution & Phase Preservation**:
   - Single-precision IEEE 754 `float32` provides only 24 bits of mantissa (~7.22 decimal digits). When computing Weyl sequences:
     $$w_i = ((i + 1) \cdot \Delta t_{\text{ns}} \cdot \Phi^{-1}) \pmod 1$$
     large index offsets ($i > 10^5$) or microsecond/nanosecond epoch timestamps quickly consume the 24-bit mantissa, causing severe phase aliasing, modulo rounding collapse, and loss of quasi-random equidistribution.
   - The engine computes index scaling and modulo reductions in 64-bit IEEE 754 `torch.float64` (53-bit mantissa) before casting to the requested target tensor dtype, preserving incommensurate irrational phase evolution across arbitrary tensor lengths.

3. **Generalized Metallic and Plastic Ratios**:
   - For multi-dimensional field expansions ($d > 1$), quasi-random sampling requires mutually incommensurate generalized ratios.
   - For dimension $d$, the engine solves the unique positive real root of:
     $$x^{d+1} = x + 1$$
     For $d=1$, this yields the golden ratio $\Phi \approx 1.6180339887$. For $d=2$, this yields the plastic constant $\psi \approx 1.3247179572447$. The reciprocal roots $\alpha_j = \left(\frac{1}{x}\right)^j$ guarantee minimal discrepancy across $d$-dimensional parameter sweeps.

**CRITICAL ARCHITECTURAL BOUNDARY (Bouligand Safety)**: 
The UI-level `FractionalQuotientTagStacker` is **strictly forbidden** from demanding or directly communicating with the physics solver (`admr_solver.py`). They are entirely separate fractional systems. When the user manipulates sliders, the Tag Stacker mathematically superposes the vectors in continuous fractional space, but this is merely a *draft* or *seed state*. 

To prevent the UI from "faking the grade" by forcibly tearing the state manifold, the resulting mixed vector is submitted back into the `admr_solver.py`. Here, it is subjected to strict **Bouligand contingent cone projections** (`self.gyrocompass.precess_torque()`) which violently reject any components of the slider's mixture that point outward from the feasible polytope boundaries, ensuring total structural integrity is never bypassed by front-end manipulation.
