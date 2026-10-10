# ZeitgeistRouter: CRT Polytope Switching & Non-Abelian Braid Steering

**Phase**: 18  
**Status**: [OK] Fully Implemented  
**Source**: [src/core/zeitgeist_router.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/zeitgeist_router.py)  
**References**:
- [ai project report II-VI](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/stepping_stairs_tier_2_crypt/summaries/ai%20project%20report_2-2-2026.txt)
- [SYSTEM_ARCHITECTURE 9.4-9.5](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/SYSTEM_ARCHITECTURE.md)
- [BIOMIMETIC_SYNTHESIS_REPORT 4.4](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/BIOMIMETIC_SYNTHESIS_REPORT.md)
- [NONCOMMUTATIVITY_DYNAMICS.md](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/NONCOMMUTATIVITY_DYNAMICS.md)
- [CONTEXT_AWARE_QUANTIZER.md](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/CONTEXT_AWARE_QUANTIZER.md)

---

## 1. Overview

The `ZeitgeistRouter` implements **CRT Polytope Switching and Non-Abelian Braid Steering**—the mechanism by which the reasoning engine navigates between culturally and logically non-commensurable meaning systems without forcing scalar reconciliation between them.

A reasoner operating across diverse semiotic registers (e.g. formal logic, qualitative analogy, specialized technical vocabulary) cannot project all registers onto a single Cartesian coordinate axis without flattening distinct structural subtleties. The Zeitgeist Router assigns each register a modular residue from a co-prime prime ladder and allows the system to switch between polytopes non-commutatively through Artin braid group generators ($B_M$) and Chern-Simons topological phases.

---

## 2. Formal Basis

### The Stratified System State

The complete system state is a quadruple:

$$S_t = (x_t,\; \alpha_t,\; \ell_t,\; u_t)$$

| Component | Type | Description |
|---|---|---|
| $x_t$ | `[batch, dim]` or `[dim]` tensor | Current representation vector |
| $\alpha_t$ | `[M, M]` symmetric tensor | **Symmetric CRT index** ($M_{ij} = M_{ji}$) |
| $\ell_t$ | `int` | Matrioshka shell depth |
| $u_t$ | `BoundaryState` or `None` | Last facet crossing event from Matrioshka loop |

### The Symmetric Tensor CRT Index

The CRT index $\alpha$ is structured as a **Symmetric Tensor** $M_{\alpha} \in \mathbb{R}^{M \times M}$:
* The **diagonal elements** $M_{ii}$ contain the modular residues $r_i \in \{0, \dots, p_i - 1\}$ for each prime $p_i$ in the ladder.
* The **off-diagonal elements** $M_{ij} = \frac{r_i + r_j}{2}$ form a palindromic mirror that stabilizes the routing against non-commutative drift.

The Chinese Remainder Theorem maps the diagonal residues $(r_1, \ldots, r_M)$ bijectively to a unique integer index $\alpha \in [0, M_{\text{total}})$:

$$\alpha = \sum_{i=1}^M r_i \cdot M_i \cdot y_i \pmod{M_{\text{total}}}$$

where $M_{\text{total}} = \prod_{i=1}^M p_i$, $M_i = M_{\text{total}} / p_i$, and modular inverses $y_i$ are computed via Fermat's Little Theorem:

$$y_i = M_i^{p_i - 2} \pmod{p_i}$$

---

## 3. Four-Mode Dispatch Architecture

The router classifies each step into one of four mutually exclusive modes:

```
                          ZeitgeistRouter.forward(x, state, boundary, tadc_kwargs)
                                                     │
                             Archetypal Synthesis & TADC Context Injection
                                                     │
                             Algebraic Geometry: Rational Snapping & D-Module
                                                     │
                                           Is D-Module Critical or
                                           Lazarus Void? (cohom_dim > dim/2)
                                              ├── YES ──> [UNDEFINED] (Topological Refusal)
                                              └── NO
                                                     │
                                       Facet Check: |n_i · x - c_i| < eps
                                              ├── NO  ──> [INTERIOR] (Intra-polytope traversal)
                                              └── YES
                                                     │
                                          Braid Group B_M Transition &
                                          CRT Switch Delta Computation
                                              ├── Alpha Changed? ── YES ──> [SWITCHING]
                                              └── Alpha Unchanged ───────> [GRAZING]
                                                     │
                                    Non-Abelian Temporal Inverse Kinematics
                                                     │
                                  Returns: (mode, new_state, diagnostics, x_steered)
```

| Mode | Condition | Scalar Metrics | $\alpha_t$ Update | Behavior |
|---|---|---|---|---|
| `interior` | No facet grazing ($|n_i \cdot x - c_i| \ge \varepsilon$) | Allowed | Unchanged | Stable intra-polytope reasoning; time step dilates ($dt \uparrow$) |
| `grazing` | Grazing zone entered, but residue delta rounds to 0 | Prohibited | Unchanged | High facet tension; pure pressure dynamics |
| `switching` | Grazing zone entered, residue delta changes diagonal | Prohibited | **Updated** | Non-commutative transition across polytope boundaries |
| `undefined` | Boundary critical OR Lazarus Void ($H^k > \text{dim}/2$) | Prohibited | Unchanged | Topological refusal (NaN guard); avoids catastrophic flattening |

### Non-Commutativity Invariant

The core structural guarantee enforced by this module is non-commutativity of trajectory switching:

$$\text{route}(x,\; \text{route}(y, S_0)) \neq \text{route}(y,\; \text{route}(x, S_0)) \quad \text{for distinct } x, y$$

This path-dependence emerges from the combination of state-dependent switch gating, Braid group generator permutations, and the nostalgic leak buffer.

---

## 4. Braid Group Automaton ($B_M$) & Non-Abelian Braiding

Rather than executing instantaneous coordinate leaps, polytope transitions are governed by the **Artin Braid Group $B_M$**:

### 1. Burkov Matrix Representation
The `BraidGroupMatrices` module generates $(M \times M)$ non-Abelian representation matrices for each generator $\sigma_i$ ($1 \le i < M$) via Burkov expansion:

$$\sigma_i = \begin{pmatrix} I_{i-1} & 0 & 0 \\ 0 & \begin{pmatrix} 1 - q & q \\ 1 & 0 \end{pmatrix} & 0 \\ 0 & 0 & I_{M - i - 1} \end{pmatrix}, \quad \sigma_i^{-1} = \begin{pmatrix} I_{i-1} & 0 & 0 \\ 0 & \begin{pmatrix} 0 & 1 \\ 1/q & 1 - 1/q \end{pmatrix} & 0 \\ 0 & 0 & I_{M - i - 1} \end{pmatrix}$$

Using parameter $q = 1.0$, the generators execute non-commutative strand swaps across CRT channels.

### 2. Greedy Word Reduction (`braid_reduce`)
Active generator sequences are reduced using canonical braid relations:
* **Inverse Law**: $\sigma_i \cdot \sigma_i^{-1} = e$
* **Far Commutativity**: $\sigma_i \cdot \sigma_j = \sigma_j \cdot \sigma_i$ for $|i - j| > 1$
* **Braid Relation (Type-II Reidemeister Trace)**: $\sigma_i \sigma_{i+1} \sigma_i = \sigma_{i+1} \sigma_i \sigma_{i+1}$

### 3. Word Length Ceiling & Topological Refusal
If the accumulated braid word length exceeds $2M$, the router triggers a topological refusal reset (`new_word = []`), preventing knot entanglement divergence.

### 4. Chern-Simons Phase Accumulation
Every applied generator increments the Chern-Simons phase angle:

$$\Delta \theta_{CS} = \frac{|\sigma_i| \cdot \pi}{M}$$

anchoring phase shifts to the Prime Resonance Ladder.

---

## 5. Non-Abelian Temporal Inverse Kinematics

The forward pass does not simply output discrete modes; it actively steers the representation vector $x \to x_{\text{steered}}$ using `temporal_inverse_kinematics`:

1. **Compass Rotation**: The accumulated Chern-Simons phase $\theta_{CS}$ forms a 2D rotation matrix applied to the first two coordinate dimensions of $x$:
   $$R(\theta_{CS}) = \begin{pmatrix} \cos\theta_{CS} & -\sin\theta_{CS} \\ \sin\theta_{CS} & \cos\theta_{CS} \end{pmatrix}$$
2. **Word Pressure Damping**: The total length of the active braid word exerts damping pressure:
   $$\text{damping} = 1.0 - 0.2 \tanh\left(\frac{|\text{word}|}{2M}\right)$$
   producing continuous path-dependent steering across the manifold.

---

## 6. Archetypal Synthesis & TADC Context Integration

The router integrates with `ArchetypalSynthesisEngine` and accepts psychological context values via `tadc_kwargs`:

```python
arch_results = self._archetype.run_archetypes(
    current_state=x,
    stranded_states=torch.empty((0, x.shape[-1]), device=x.device),
    current_mischief=tadc_kwargs.get('mischief', 0.5),
    phase_alignment=tadc_kwargs.get('pas_h', 0.5),
    love_strengths=tadc_kwargs.get('love', torch.tensor([1.0], device=x.device)),
    void_frictions=tadc_kwargs.get('friction', torch.tensor([0.0], device=x.device)),
    global_dt=1.0,
    env_luminosity=tadc_kwargs.get('luminosity', 1.0),
    volitional_scalar=tadc_kwargs.get('volition', 0.0),
    system_entropy=tadc_kwargs.get('entropy', 0.1),
    memory_trauma=tadc_kwargs.get('trauma', 0.1),
    dissonance=tadc_kwargs.get('dissonance', 0.1),
    lucidity_idx=tadc_kwargs.get('lucidity', 1.0),
    raw_unquantized_state=x
)
```

If the archetypal engine reports `system_collapsed = True`, the router immediately initiates a fast-track exit to `undefined` mode with diagnostic flag `abstraction_event = True`.

---

## 7. Exact Algebraic Geometry: D-Modules & Lazarus Void

Rather than relying on heuristics, boundary ruptures are evaluated using exact algebraic geometry via `NumericalDModuleManager` and `RationalSnappingLayer`:

1. The state vector $x$ is normalized and snapped onto rational coordinate boundaries via `RationalSnappingLayer`.
2. The D-module manager computes the **cohomological dimension** ($H^k$) across facet projections $g$.
3. If $H^k > \frac{\text{dim}}{2}$ or `is_lazarus_void` is flagged, the system classifies the state as entering an unresolvable topological void, immediately emitting mode `undefined`.

---

## 8. Poincare Gravity Wells & Fossil Landmarks

The router links persistent storage directly to spatial navigation (`Bridge 4`):

1. **Registration**: When an anomaly or dyad is fossilized, its Blake2s digest ID is registered via `register_fossil_landmark(blake2s_id, intensity)`.
2. **Moving Average Bias**: The ID is mapped to a bias vector across the $M$ coprime channels, updating `gravity_well_bias`:
   $$\text{bias}_{t} = 0.9 \cdot \text{bias}_{t-1} + 0.1 \cdot \text{bias}_{\text{new}}$$
3. **Trajectory Pull**: During switching calculations, the gravity well bias pulls the braided switch delta:
   $$\Delta_{\text{final}} = \Delta_{\text{braided}} + 0.3 \cdot \text{gravity\_well\_bias}$$
   ensuring that fossilized knowledge dyad landmarks physically curve future thought trajectories.

---

## 9. 4D Log-Polar Space Carving & Betti Routing

1. **Log-Polar Radial Compression**:
   Multiplicative Matrioshka depth zooming is converted into additive shifting via:
   $$x_{\text{lp}} = \frac{x}{\|x\|} \log(\|x\| + 1.0)$$
   protecting `switch_gate` from explosive singularities while preserving angular orientation.
2. **Betti-Aware Routing**:
   `BettiRouter` evaluates topological homology across sectors and injects homological bias:
   $$\Delta_{\text{soft}} = \Delta_{\text{soft}} + 0.4 \cdot \text{betti\_bias}$$

---

## 10. Curvature Tracking & The Love Invariant

1. **Rolling Covariance Estimator (`GyroidCovarianceEstimator`)**:
   Tracks the temporal covariance drift between the instantaneous covariance ($x^T x$) and rolling manifold covariance.
2. **High-Load Bypass**:
   Under extreme load ($\text{load} > 0.8$) or severe grazing pressure ($P > 0.9$), expensive curvature calculations are bypassed (`nc_curvature = 0.0`).
3. **The Love Invariant Shortcut**:
   If relative non-commutativity curvature drops below $0.35$, the router recognizes harmonic stabilization and collapses the state to pure palindromic trace-stable symmetry:
   $$M_{\alpha} = \frac{r_{\text{col}} + r_{\text{row}}}{2}$$
4. **Nostalgic Leak Buffer (`digimon_buffer`)**:
   Retains historical non-commutative illusions ($\psi_l = 0.1 \cdot \text{digimon\_buffer}$) to prevent sterile mathematical forgetting:
   $$\text{buffer}_{t} = 0.95 \cdot \text{buffer}_{t-1} + 0.05 \cdot M_{\alpha}$$

---

## 11. Hardware Stall as Polytope Switch: The DRAM $t_{\text{RFC}}$ Analogue

The ZeitgeistRouter mode transitions correspond to physical memory hardware migrations:

| Hardware Phenomenon | ZeitgeistRouter Event | Operational Mode |
|---|---|---|
| Channel A active, normal memory access | CRT polytope stable, $\alpha_t$ invariant | `interior` |
| Channel A $t_{\text{RFC}}$ refresh stall begins | CRT facet grazing, boundary pressure rise | `grazing` |
| Migration to Channel B alternate bank | Coprime modulus shift across braid generator | `switching` |
| Refresh finishes, alternate bank active | New $\alpha_t$ resolved on target polytope | `interior` (new polytope) |
| Both channels stall ($P^2 \approx 0$) | Topological refusal / Lazarus Void | `undefined` |

### SLERP vs. LERP Navigation Geometries
* **SLERP (Spherical Linear Interpolation)** (`interior` mode): Traverses the great circle along the high-density Birkhoff polytope surface.
* **LERP (Linear Interpolation)** (`grazing` mode): Cuts through the chord of the hypersphere, crossing the low-probability center void and producing controlled non-commutative glitches.
* **`undefined` mode**: Complete topological refusal that activates the Lazarus Preparation Window, enabling parallel work units to re-anchor into fresh lattice coordinates.

---

## 12. Complete Diagnostics Reference

The dictionary returned in `forward()` (and stored in `get_diagnostics()`) emits:

| Field | Type | Description |
|---|---|---|
| `mode` | `str` | Dispatch classification (`interior`, `grazing`, `switching`, `undefined`) |
| `prev_alpha_diag` | `List[int]` | Modular residues $(r_1, \ldots, r_M)$ prior to the current step |
| `new_alpha_diag` | `List[int]` | Modular residues after the current step |
| `prev_crt_index` | `int` | Reconstructed CRT scalar prior to the step |
| `new_crt_index` | `int` | Reconstructed CRT scalar after the step |
| `alpha_changed` | `bool` | True if a full polytope switch occurred |
| `level` | `int` | Active Matrioshka shell depth |
| `step` | `int` | Monotonic call sequence counter |
| `grazing_dims` | `int` | Number of facets currently in the grazing zone |
| `grazing_pressure` | `float` | Mean absolute deviation from facet thresholds |
| `facet_norms_mean` | `float` | Mean absolute projection magnitude across normals |
| `betti_routing_bias` | `float` | Accumulated Chern-Simons phase contribution |
| `clock_dt` | `float` or `None` | Dilated/contracted breathing time from `ManifoldClock` |
| `valence` | `float` or `None` | Structural hunger/dissonance score from `ValenceFunctional` |
| `nc_curvature` | `float` or `None` | Non-commutativity curvature norm |
| `braid_word` | `List[int]` | Current reduced braid word sequence |
| `cs_phase` | `float` | Total accumulated Chern-Simons rotation angle |
| `word_length` | `int` | Current number of active braid generators |
| `gasket_tension` | `float` | Normalized braid word tension ratio ($|\text{word}| / M$) |
| `state` | `Dict` | Serialized dictionary representation of `ZeitgeistState` |

---

## 13. Method Signatures & Usage

### Module Signature
```python
class ZeitgeistRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        moduli: Optional[Tuple[int, ...]] = None,
        grazing_eps: float = 0.05,
        critical_boundary_threshold: float = 0.5,
        use_noncommutativity_check: bool = True,
    ): ...

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[ZeitgeistState] = None,
        boundary=None,
        tadc_kwargs: Optional[Dict] = None
    ) -> Tuple[str, ZeitgeistState, Dict, torch.Tensor]: ...

    def register_fossil_landmark(self, blake2s_id: str, intensity: float = 1.0) -> None: ...
    def get_diagnostics(self) -> Dict: ...
```

### Integration Call
```python
mode, new_state, diagnostics, x_steered = zeitgeist_router(
    seed_state,
    current_state,
    boundary=last_boundary_state,
    tadc_kwargs={
        "mischief": 0.4,
        "pas_h": 0.8,
        "luminosity": 1.0,
        "entropy": 0.2
    }
)
```
Single-dimensional vectors (`[dim]`) are seamlessly supported and returned with 1D dimensionality preserved.
