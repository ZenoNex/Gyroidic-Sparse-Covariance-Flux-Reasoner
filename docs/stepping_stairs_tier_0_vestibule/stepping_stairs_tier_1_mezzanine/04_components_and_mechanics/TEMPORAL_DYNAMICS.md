# Temporal Dynamics

> Negentropic developmental scaling and pressure-modulated manifold time.

---

## 1. NTMOperator (Negentropic Trigonometric Manifold)

**Source**: [`src/core/negentropic_manifold.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/negentropic_manifold.py) (68 lines)

Governs the "Negentropic Trunk" — developmental scaffolding that modulates polynomial basis frequencies as the system matures.

### Basis Warping

$$w_j(\tau) = \frac{\cos(\omega_j \cdot \tau + \phi_j)}{1 + \bar{N}}$$

where $\bar{N}$ = mean negentropy (informational density).

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `tau` | scalar | Asymptotic clock (monotonically increases via $\tau \leftarrow \tau + dt$) |
| `frequencies` | `[D+1]` | Harmonic diversity: `linspace(0.5, 2.5, degree + 1)` |
| `phases` | `[D+1]` | Random phase offsets $\phi_j$ seeded via Silicon Sovereignty `harvest_honest_jitter((degree + 1,), scaled=True)` |

**Key behavior**: As negentropy increases (system matures), amplitude damping $\frac{1}{1 + \bar{N}}$ reduces oscillation — basis stabilizes — "saturation."

### Negentropy Flux Tripsody
The negentropy flux doesn't just scale variables; it exhibits a **tripsody** — a tripartite rhapsodic oscillation. As negentropy increases, the flux induces temporary phase-locking, slowing down the scaffold mutation and allowing the polynomial basis to traverse topological singularities safely before expanding again.

### Asymptotic State

`get_asymptotic_state()` returns:
- `asymptotic_time_tau`: current $\tau$
- `structural_heat`: $\exp(-\tau \cdot 0.05)$ — entropy dissipation rate (decays exponentially with maturity $\tau$)

---

## 2. ManifoldClock

**Source**: [`src/core/manifold_time.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/manifold_time.py) (175 lines)

Implements **Breathing Time** where the coordinate time step `dt` is dynamically pressure-modulated:

$$dt = \text{clamp}\left(dt_{\text{base}} \cdot \frac{\text{Play}}{1 + \text{Seriousness}},\, dt_{\text{min}},\, dt_{\text{max}}\right)$$

### Thermodynamic Mapping

$dt$ acts as the effective inverse temperature ($\beta$) governing the topological free energy $F_{\text{topo}} = -\frac{1}{\beta} \log(Z)$:

| Regime | Pressure $P$ | $dt$ | $\beta$ (Inverse Temp) | Behavior |
|--------|--------------|------|------------------------|----------|
| **Play** | Low | Large | Low | Playful flux, rapid exploratory phase |
| **Seriousness** | High | Small | High | Structural freezing, fine-grained protective care |

$$\text{Seriousness} = \tanh(\lambda_s \cdot P), \quad \text{Play} = e^{-\lambda_p \cdot P}$$

### Constructor Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `dt_base` | 1.0 | Baseline coordinate time step |
| `dt_min` | 0.001 | Numerical floor preventing total freezing |
| `dt_max` | 2.0 | Numerical ceiling preventing dynamical instability |
| `lambda_seriousness` | 2.0 | Sensitivity to pressure ($\lambda_s$: high values cause rapid step shrinkage) |
| `lambda_play` | 0.5 | Sensitivity to smoothness ($\lambda_p$: high values cause rapid step expansion) |

### State Tracking & Telemetry

| Buffer / Metric | Type | Purpose |
|-----------------|------|---------|
| `coordinate_time` | Buffer (`float`) | Accumulated variable-rate time $t = \sum dt$ |
| `proper_time` | Buffer (`float`) | Invariant monotonic step counter $\tau = \sum 1.0$ |
| `current_dt` | Buffer (`float`) | Most recent discrete time increment $dt$ |
| `accumulated_seriousness` | Buffer (`float`) | Time-weighted seriousness integral $\int \text{Seriousness} \cdot dt$ |
| `accumulated_play` | Buffer (`float`) | Time-weighted play integral $\int \text{Play} \cdot dt$ |
| `dt_ratio` | Property (`float`) | Normalized time scale ratio $dt / dt_{\text{base}}$ |
| `dilation` | Metric (`float`) | Relativistic time dilation ratio $\tau / t$ |

**The "Watts Move"**: Take the universe seriously enough to dance (fine steps), not seriously enough to freeze (infinite steps).

---

### 2.1 TwoCopsSchedule (Temporal Decoupling & TEMPOLOCK)

**Location**: [`src/core/manifold_time.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/manifold_time.py#L125-L175) and [`src/core/orchestrator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py#L107-L111)

Decouples high-frequency intuitive heuristics from exact constraint verification via the **Two Cops** architecture:

1. **System 1 (Fast Cop)**:
   - High-frequency Material Point Method (MPM)-style continuous evaluation.
   - Stepped continuously at every iteration using `ManifoldClock.tick(pressure)`.
   - In [`UniversalOrchestrator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py#L448-L480), the fast cop executes `micro_steps` iterations of rapid heuristic drafting, injecting hardware-anchored mischief perturbations, applying surface erosion filtering, evaluating recurrent feature scar dynamics (`rnn_core`), and posting residue states to the `BulletinBoard`.
2. **System 2 (Slow Cop)**:
   - Low-frequency Finite Element Method (FEM)-style exact geometric and yield constraint verification.
   - Only executed when a macro sync condition is triggered.
3. **Shared Bulletin Board**:
   - Asynchronous force and state exchange register updated via an Exponential Moving Average (EMA) with $\alpha = 0.3$:
     $$\text{bulletin\_board} \leftarrow 0.7 \cdot \text{bulletin\_board} + 0.3 \cdot \bar{F}_{\text{sys2}}$$
4. **TEMPOLOCK Prime Synchronization Law**:
   - When `use_tempolock=True`, macro synchronization is gated by the `PrimeResonanceLadder(num_resonators=20)`.
   - The slow cop executes if and only if the discrete step counter shares a lattice divisor with any active prime resonance frequency:
     $$\text{should\_sync} = \bigvee_k \left(\text{step\_counter} \pmod{p_k} == 0\right), \quad p_k \in \{2, 3, 5, 7, 11, 13, \dots\}$$
   - When `use_tempolock=False`, synchronization falls back to periodic uniform macro steps: $\text{step\_counter} \pmod{\text{macro\_steps}} == 0$.
5. **Dynamic Trigger Conditions in Orchestration**:
   - In `UniversalOrchestrator.step()`, System 2 execution is triggered if:
     $$\text{Trigger} = \text{should\_sync} \lor \text{is\_red\_zone} \lor \text{is\_good\_bug}$$
     where `is_red_zone` is signaled by `NonCommutativityCurvature` ($\kappa > 0.35$).
   - Upon triggering, System 2 enforces the Love Invariant ($3.127$ norm anchor), applies a True North pull via `TopologicalGyrocompass`, quantizes coordinates onto the 600-Cell polychoron lattice, applies the gluing operator $\Psi$, and updates the board:
     $$\text{schedule.update\_board}(S_{\text{final}} - S_{\text{init}})$$

---

## 3. System Wiring & Temporal Flow

```mermaid
graph TD
    P["Structural Pressure P<br/>(1.0 - cos_sim)"] --> MC["ManifoldClock<br/>dt = clamp(dt_base * Play / (1 + Seriousness))"]
    MC -->|dt| NTM["NTMOperator<br/>w_j(tau) = cos(omega_j * tau + phi_j) / (1 + N)"]
    MC -->|dt| TCS["TwoCopsSchedule<br/>(TEMPOLOCK sync via PrimeResonanceLadder)"]
    TCS -->|Fast Cop (Sys 1)| MPM["MPM Continuous Evaluation<br/>(micro_steps heuristic drafting)"]
    TCS -->|Slow Cop (Sys 2)| FEM["FEM Constraint Rigor<br/>(Love, Gyrocompass, 600-Cell, Gluing)"]
    FEM -.->|EMA alpha=0.3| BB["Shared Bulletin Board"]
    BB -.->|Coupling force| MPM
    NTM -->|w(tau) basis warping| POLY["PolynomialCoprimeConfig / ADMR"]
    MC -->|dt| RHR["RecurrentHyperRingConnectivity<br/>flow_step(states, connectivity, dt)"]
    MC -->|dt| DHR["DiscreteHyperRingCirculation<br/>Base-24 Toroidal Line Integral"]
    DEFLAG["OmipedialDeflagrator<br/>Delta D_i = |R - R_hat| * amp"] -->|Defect pressure| VAL["ValenceFunctional<br/>(Manifold Hunger)"]
    VAL -->|Hunger tensor| ADMR["PolynomialADMRSolver<br/>fractional_stochastic_differential_step(dt, hunger)"]
    DEFLAG -->|NonCommutativityCurvature<br/>[Op_A, Op_B]| TRANS["Symbolic Transversality (kappa)"]
```

---

## 4. Omipedial Deflagration Scout

**Source**: [`src/core/deflagration_scout.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/deflagration_scout.py) (96 lines)

Implements "Omipedial Interstitiality" — **defect scouting** that amplifies sparse anomalies and enables signal propagation across topological gaps.

### Defect Detection

$$\Delta D_i = \left| R_{ij} - \hat{R}_{ij} \right| \cdot \text{amplification}$$

where $R_{ij}$ is the actual resonance flux and $\hat{R}_{ij}$ is the predicted/expected flux. High $\Delta D_i$ indicates rare anomalies that are rewarded to prevent "Goo" (phase flattening).

Defect count tracks accumulated anomaly density:
```python
self.defect_count.add_(defects.mean().detach())
```

### Martinova Correlation & Structural Gaps
Integrates bounded correlation analysis (`from core.martinova_correlation import compute_bounded_correlation`):
$$\text{structural\_gaps} = (\text{corr}(R) \le -0.8)$$
Points exhibiting strong negative correlation ($\le -0.8$) are flagged as topological tears or structural gaps requiring repair or bridging.

### Omipedial Jump

When the ley line potential exceeds the jump threshold ($> 0.8$), the scout enables **topological shortcuts** — signal propagation across holes where resonance potential is high but adjacency is sparse:

$$\text{jumps} = (\text{ley\_potential} > \text{threshold\_jump})$$

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `threshold_jump` | 0.8 | Minimum ley potential for topological shortcut |
| `amplification` | 2.0 | Anomaly amplification factor |

### Symbolic Transversality & Non-Commutativity Curvature

`compute_transversality(archetype_state, voynich_state)` measures path dependence across disparate symbolic domains via `NonCommutativityCurvature(dim=dim)`:
1. Projects representations into rank-1 operator matrices:
   $$Op_A = \mathbf{s}_{\text{arch}}^T \mathbf{s}_{\text{arch}}, \quad Op_B = \mathbf{s}_{\text{voy}}^T \mathbf{s}_{\text{voy}}$$
2. Evaluates the Lie bracket commutator:
   $$[A, B] = Op_A Op_B - Op_B Op_A$$
3. Measures the non-commutativity curvature $\kappa$ and updates the internal Exponential Moving Average (EMA).

---

## 5. Fractional-Order Riemann-Liouville SDE Bridge & HyperRing Flow

### Connection to Polynomial Scaffold & ADMR

The defect signal $\Delta D_i$ feeds into [`src/core/valence_drive.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/valence_drive.py) as `current_pressure`, which computes Manifold Hunger. This hunger tensor is passed to [`src/core/admr_solver.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/admr_solver.py)'s `fractional_stochastic_differential_step()` where it modulates the distributed fractional order $\alpha(k)$ per coprime functional channel:

$$\alpha(k) = 0.5 + 0.5 \cos\left(\frac{2\pi k}{K}\right)$$

- **Low-frequency channels ($k \approx 0$)**: $\alpha(k) \to 1.0$ (near-Markovian, rapid updating).
- **High-frequency anomalies ($k \approx K/2$)**: $\alpha(k) \to 0.5$ (anomalous subdiffusion, heavy memory with viscoelastic drag via the Riemann-Liouville kernel computed through Lanczos-Krylov iterations).
- **Viscoelastic Boundary Drag**:
  $$\alpha(k) \leftarrow \alpha(k) \cdot \left(0.5 + 0.5 \tanh(\text{dist\_to\_boundary})\right)$$
- **Hunger modulation**: High hunger shifts $\alpha$ globally toward $1.0$, accelerating updates across all channels to escape starvation:
  $$\alpha(k) \leftarrow \alpha(k) + (1.0 - \alpha(k)) \cdot \bar{h} \cdot 0.3$$

### Tripsodic Negentropy Oscillation & Effective Time Step

In `PolynomialADMRSolver.fractional_stochastic_differential_step()`, the coordinate time step $dt$ undergoes tripsodic negentropy modulation:

$$\text{effective\_dt} = dt \cdot \frac{1}{1 + \|\mathbf{J}_{\text{frac}}\|} \cdot \left(1 + 0.5 \cos(\pi \|\mathbf{J}_{\text{frac}}\|)\right)$$

where $\mathbf{J}_{\text{frac}}$ is the fractional drift tensor. When mischief violation $V_m$ is present:
$$\text{effective\_dt} \leftarrow \text{effective\_dt} \cdot (1 + V_m)$$

### Hans-Joachim Hein Kähler Geometry Boundary Decay

To prevent numerical rupture near topological limits ($r_{\text{max}} = 10.0$):
$$d_{\text{boundary}} = \text{clamp}(r_{\text{max}} - \|x\|_2,\, 10^{-5})$$
$$\beta_{\text{decay}} = \left(\frac{d_{\text{boundary}}}{r_{\text{max}}}\right)^{1.345}$$
$$dx \leftarrow dx \cdot \beta_{\text{decay}}$$

The update norm is further clamped to at most $10\%$ of the remaining distance to the boundary: $\|dx\| \le 0.1 \cdot d_{\text{boundary}}$.

### Topological Flow in HyperRing Circulation

In [`src/topology/hyper_ring.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/topology/hyper_ring.py), two complementary modules govern topological circulation:

1. **`RecurrentHyperRingConnectivity.flow_step()`**:
   Integrates coordinate time $dt$ from `ManifoldClock` directly into polytope state propagation:
   $$S_{t + dt} = S_t + dt \cdot \sum_j H_{ij} S_j$$
   where $H_{ij} = \Omega_{ij} \tanh(f_i - f_j) + 1.5 \cdot \text{jumps}$.
   Under high pressure, $dt \to dt_{\text{min}}$, freezing the polytope circulation into a rigid, highly protective regime; under relaxed playful conditions, $dt \to dt_{\text{max}}$, allowing free, rapid circulation across the hyper-ring manifold.

2. **`DiscreteHyperRingCirculation.compute_circulation()`**:
   Computes the discrete line integral over constraint cycles using IHC Base-24 Toroidal Modular Arithmetic:
   $$\oint_H \langle \nabla \Phi(C_i),\, \Delta C_i \rangle$$
   with adaptive resolution refinement triggered when phase slippage exceeds $\theta_{\text{slip}} = 0.1$.

---

## 6. Unknowledge Dilation & Metaphysical Entropy Bands

**Sources**: [`src/training/trainer.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/training/trainer.py#L364-L368), [`src/core/valence_drive.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/valence_drive.py#L30-L96), [`src/core/zeitgeist_router.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/zeitgeist_router.py#L812-L840), and [`src/ui/diegetic_backend.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py#L2365-L2385)

### Unknowledge Dilation in Training
In the training loop (`Trainer.train_step()`), the clock is ticked using the mischief violation score $V_m$:
```python
dt = self.clock.tick(v_m)
self.energy_monitor.set_temperature(self.clock.dt_ratio)
```
- High violation $V_m$ contracts coordinate time $dt \to dt_{\text{min}}$ (freezing parameter drift during crisis).
- Low violation relaxes coordinate time $dt \to dt_{\text{max}}$ (accelerating exploration).
- The structural energy monitor temperature dynamically synchronizes to the time dilation ratio `dt_ratio`.

### Metaphysical Entropy Bands ($H_{\text{meta}}$)
The system decomposes disorder into tripartite spectral entropy bands:
$$H_{\text{meta}} = H_{\text{dementia}} + H_{\text{schizo}} + H_{\text{mischief}}$$

- **Dementia Band ($H_{\text{dementia}}$)**: Decays historical anchors that lack current resonance.
- **Schizo Band ($H_{\text{schizo}}$)**: Fragments rigid fixed categories into playful multi-modal clusters.
- **Mischief Band ($H_{\text{mischief}}$)**: Rewards topological violations (Good Bugs) to prevent scale-induced lobotomy.

### Wattsian Unlearning in Valence Baseline
In `ValenceFunctional`:
- Under normal dynamics, historical satisfaction decays with $\lambda = 0.99$.
- Under stagnation ($\text{entropy} < 0.05$ or pressure variance $< 10^{-6}$), decay sharply accelerates to $0.85$, unlearning complacency to spike hunger and escape cognitive flatlines.

### Zeitgeist & Diegetic Pressure Clamping
- In `DiegeticBackend`, structural pressure is derived from normalized cosine divergence:
  $$P = 1.0 - \frac{\mathbf{S}_{\text{meta}} \cdot \mathbf{I}_{\text{input}}}{\|\mathbf{S}_{\text{meta}}\| \|\mathbf{I}_{\text{input}}\|}$$
- In `ZeitgeistRouter`, grazing pressure drives `ManifoldClock.tick()`. When load exceeds $0.8$ or grazing pressure exceeds $0.9$, an emergency high-pressure bypass deactivates costly non-commutativity curvature calculation to preserve throughput.



