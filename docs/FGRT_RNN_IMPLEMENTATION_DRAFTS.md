# FGRT Recurrent Neural Network Cell Implementations

> Authoritative architectural specification of the 7 topological recurrent cell models in the Fiberalized Gyroidic Recurrent Topology (FGRT).  
> **Source**: [`src/core/fgrt_rnn_cells.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_rnn_cells.py) (246 lines)  
> **Theoretical Foundation**: [FGRT_FORMALIZATION.md](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/FGRT_FORMALIZATION.md)  
> **Status**: Fully Implemented & Verified

---

## 1. Overview & Recurrent Cell Taxonomy

Standard recurrent networks update a hidden state $h_t = f(W_x x_t + W_h h_{t-1})$ over an assumed Euclidean flat space. FGRT recurrent cells replace this with manifold traversal over a non-orientable Klein-Gyroid slip-space ($\mathcal{G} \cup \mathcal{K}$), tracking:
1. **Geometric Berry Phase Accumulation** ($\gamma_t$).
2. **First Stiefel-Whitney Class Obstruction** ($w_1(E) \in H^1(M; \mathbb{Z}_2)$).
3. **Logarithmic Conformal Field Theory (LCFT) Scars** fossilized in non-decaying memory tensors.
4. **Thermodynamic Affordance Protection** via [`HardwareMonitor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py).

```mermaid
graph TD
    subgraph Recurrent_Cells["7 FGRT Recurrent Cells"]
        C1["1. BerryPhaseGRUCell<br/>(Phase Track & Stiefel-Whitney Flip)"]
        C2["2. ChiralGatedRNNCell<br/>(Bostick Chiral Projections)"]
        C3["3. ComplexFGRTRNNCell<br/>(C^768 Native Phase & Roots of Unity)"]
        C4["4. SaturatedQuantizerRNNCell<br/>(KAN B-Splines + Weyl Snapping)"]
        C5["5. PyOpenCLHardwareSovereigntyCell<br/>(Queue B Matrix Breeding + Affordance Guard)"]
        C6["6. LazarusSuperpositionRNNCell<br/>(Continuous Prime Superposition)"]
        C7["7. FeatureScarLCFTCell<br/>(Cerumen Pot LCFT Fossilization)"]
    end
    C1 --> Gasket["Chern-Simons Gasket<br/>(Hypersurface Glue)"]
    C2 --> Gasket
    C3 --> Gasket
    C4 --> Gasket
    C5 --> Gasket
    C6 --> Gasket
    C7 --> Gasket
```

---

## 2. Detailed Cell Specifications

### Cell 1: `BerryPhaseGRUCell` (Berry-Phase Modulated GRU)
**Class**: [`BerryPhaseGRUCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_rnn_cells.py#L15)  
**Concept**: Minimal topological integration augmenting standard GRU gating with Geometric Berry Phase tracking.

- **State Representation**: Coupled tuple $(h_t, \gamma_t)$ where $h_t \in \mathbb{R}^D$ and $\gamma_t \in \mathbb{R}^1$ represents the accumulated holonomy.
- **Contorsion Twist**: Evaluates local Riemannian twist:
  $$\Delta\gamma_t = \tanh\left(W_{\text{contorsion}} [x_t, h_{t-1}]\right), \quad \gamma_t = \gamma_{t-1} + \Delta\gamma_t$$
- **Stiefel-Whitney Parity Flip**: When crossing a non-orientable Klein boundary ($\cos(\gamma_t) < 0$):
  $$h_{t-1} \longleftarrow h_{t-1} \cdot \left(1 - 2 \cdot \mathbb{I}_{[\cos(\gamma_t) < 0]}\right)$$
  Inverts state orientation and backpropagates through the orientation jump.
- **GRU Step**: $h_t = \text{GRUCell}(x_t, h_{t-1}^{\text{flipped}})$.

---

### Cell 2: `ChiralGatedRNNCell` (Chiral Gated Recurrent Cell)
**Class**: [`ChiralGatedRNNCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_rnn_cells.py#L46)  
**Concept**: Replaces scalar forget gates with Bostick chiral projection manifolds.

- **Chiral Vectors**: Maintains $K$ learnable chiral direction vectors $\{\chi_k\}_{k=1}^K \subset \mathbb{R}^D$.
- **Chiral Projection Gate**:
  $$\text{proj}_k = \langle h_{t-1}, \chi_k \rangle, \quad \Gamma_\chi = \sigma\left(\frac{1}{K}\sum_{k=1}^K \text{proj}_k\right)$$
- **Orientation-Dependent Retention**:
  $$h_t = (1 - \Gamma_\chi) \odot h_{t-1} + \Gamma_\chi \odot \tanh\left(W_x x_t + W_h h_{t-1}\right)$$
  Information is retained permanently unless its chiral orientation destructively interferes with the incoming data manifold.

---

### Cell 3: `ComplexFGRTRNNCell` (Complex-Valued Native Phase Cell)
**Class**: [`ComplexFGRTRNNCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_rnn_cells.py#L72)  
**Concept**: Operates natively in $\mathbb{C}^D$ to eliminate conditional branching during orientation flips.

- **State Representation**: $h_t \in \mathbb{C}^D$, implemented as paired real/imaginary dimensions $[B, 2D]$.
- **Gyroidic Connection**: Computes affine projection followed by cyclotomic phase advancement:
  $$\text{magnitude} = \sqrt{\text{Re}(u)^2 + \text{Im}(u)^2 + \epsilon}, \quad \theta = \text{atan2}(\text{Im}(u), \text{Re}(u)) + \frac{\pi}{4}$$
- **Atiyah-Singer Index Flip**: Traversal through $e^{i\pi} = -1$ executes smooth, analytic phase reversal without discrete if/else conditionals.

---

### Cell 4: `SaturatedQuantizerRNNCell` (B-Spline Saturated Quantizer Core)
**Class**: [`SaturatedQuantizerRNNCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_rnn_cells.py#L106)  
**Concept**: Combines Kolmogorov-Arnold B-splines with discrete Weyl-vertex quantization to prevent representation collapse ("diffusion toxin").

- **KAN Spline Transition**: Pushes $[x_t, h_{t-1}]$ through [`KANLayer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/surrogates/kagh_networks.py#L79) using true Cox-de Boor B-splines ($k=3$).
- **Speculative Ghost Drafting**: Evaluates surrogate admissibility via [`KAGHBlock`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/surrogates/kagh_networks.py#L537).
- **Weyl Snapping**: Quantizes outputs to discrete steps with Straight-Through Estimator (STE):
  $$h_t^{\text{quant}} = \frac{\text{round}(h_t \cdot L)}{L}, \quad h_t \longleftarrow h_t + \left(h_t^{\text{quant}} - h_t\right)_{\text{detached}}$$

---

### Cell 5: `PyOpenCLHardwareSovereigntyCell` (Queue B Hardware Sovereignty)
**Class**: [`PyOpenCLHardwareSovereigntyCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_rnn_cells.py#L142)  
**Concept**: Offloads temporal matrix mixing to the asynchronous GPU breeding queue of the Silicon Sovereignty Engine.

- **Hardware Dispatch**: Dispatches $h_{t-1}$ and $W_x x_t$ to [`SiliconSovereigntyEngine.matrix_mix_breeding`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pyopencl_sovereignty.py) on Queue B (Non-Ergodic / Soliton channel).
- **Thermodynamic Affordance Protection**:
  ```python
  from src.core.hardware_monitor import HardwareMonitor
  if not HardwareMonitor.get_instance().has_compute_affordance():
      # VETO: Fallback to local CPU linear mixing to avoid thermal runaway
      mixed = (1.0 - alpha) * h_prev + alpha * x_proj
  ```
- **Honest Jitter Scars**: Natural timing jitter and scar injection when activations exceed saturation thresholds ($> 0.88$).

---

### Cell 6: `LazarusSuperpositionRNNCell` (Continuous Dark Matter Superposition)
**Class**: [`LazarusSuperpositionRNNCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_rnn_cells.py#L186)  
**Concept**: Continuous superposition without non-linear activations in the recurrent path, using incommensurate prime frequencies.

- **Authoritative Prime Ladder**: Anchored to [`PrimeResonanceLadder`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_primitives.py#L217):
  $$f_t = 2\pi \ln(p_t) \quad \text{where } p_t \text{ is the } (t \bmod N)\text{-th prime}$$
- **Linear Superposition**:
  $$h_t = h_{t-1} + \sin(f_t) \odot \left(W_x x_t\right)$$
  Information is never destroyed or compressed; non-linear decoding occurs solely at sequence readout.

---

### Cell 7: `FeatureScarLCFTCell` (Logarithmic CFT Memory Scarring — Recommended)
**Class**: [`FeatureScarLCFTCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_rnn_cells.py#L211)  
**Concept**: The definitive FGRT memory cell implementing LCFT non-decaying scars and Cerumen Pot fossilization.

- **State Representation**: Coupled pair $(h_t, \mathcal{C}_t)$ where $\mathcal{C}_t \in \mathbb{R}^D$ is the **Cerumen Pot**.
- **KAN State Proposal**:
  $$u_t = \tanh\left(\text{KANLayer}([x_t, h_{t-1}])\right)$$
- **Atiyah-Singer Index Evaluation**:
  $$\text{ind}(\mathcal{D}) = \left|W_{\text{anomaly}} u_t\right|$$
- **Good Bug Fossilization**: When $\text{ind}(\mathcal{D}) > 0.8$, a topological rupture occurs:
  $$\mathcal{C}_t = \mathcal{C}_{t-1} + \mathbb{I}_{[\text{ind}(\mathcal{D}) > 0.8]} \odot u_t$$
- **Chern-Simons Gasket Bypass**:
  $$h_t = u_t \odot (1 - \text{mask}_{\text{scar}}) + \mathcal{C}_t \odot \text{mask}_{\text{scar}}$$
  Future trajectories are compelled to route around historical scars, cementing structural memory without catastrophic forgetting.

---

## 3. Comparison & Selection Matrix

| Cell Name | State Type | Non-Linearity | Best Used For |
|---|---|---|---|
| `BerryPhaseGRUCell` | $(h, \gamma)$ | Tanh / GRU | Lightweight baseline with topological orientation flips |
| `ChiralGatedRNNCell` | $h$ | Chiral $\sigma(\langle h, \chi \rangle)$ | Bostick attractor fields and directional exploration |
| `ComplexFGRTRNNCell` | $h \in \mathbb{C}^D$ | Complex rotation | Cyclotomic polynomials and smooth index inversion |
| `SaturatedQuantizerRNNCell` | $h$ | KAN B-Spline + Rounding | Preventing diffusion toxin model collapse |
| `PyOpenCLHardwareSovereigntyCell` | $h$ | OpenCL matrix-mix | GPU acceleration under positive compute affordance |
| `LazarusSuperpositionRNNCell` | $h$ | Linear sine wave | Harmonic superposition and uncompressed memory |
| `FeatureScarLCFTCell` | $(h, \mathcal{C})$ | KAN B-Spline + Cerumen | Full LCFT topological scarring and structural memory |

---

## 4. Cross-Module References

### Immediate Neighborhood (FGRT Core)
- [`fgrt_primitives.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/fgrt_primitives.py): Math primitives for Gyroid, Berry phase, and prime ladders.
- [`fgrt_trainer.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/training/fgrt_trainer.py): Ricci flow optimizer and Bouligand-Willmore structural training.
- [`hardware_monitor.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py): Compute affordance and thermal runaway backpressure.
- [`carnot_mobius_ledger.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/carnot_mobius_ledger.py): Thermodynamic waste heat calculations.

### Extended Neighborhoods
- [`temporal_association_trainer.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/training/temporal_association_trainer.py): Sequential association and temporal dependency training.
- [`calm_predictor.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/surrogates/calm_predictor.py): Autoregressive trajectory veto predictor ($\gamma(t)$).
- [`bio_archetypal_governor.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/bio_archetypal_governor.py): Inhibition-Stabilized Network (ISN) continuous-time recurrent governance.
- [`orchestrator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py): Cognitive loop orchestrator integrating `FeatureScarLCFTCell`.
- [`diegetic_heads.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/diegetic_heads.py): Resonance Larynx character-level autoregressive generation.
- [`topological_ingestion_validator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/topological_ingestion_validator.py): Ingestion boundary sequence validator.

---

## 5. Architectural De-Isolation: Interfacing with Non-Neighborhood Recurrent Components

A significant historical architectural issue was that the 7 FGRT recurrent cells operated in an **isolated island**: while they solved non-orientable Klein-Gyroid manifold navigation and LCFT scarring mathematically, other subsystems dealing with sequence modeling, time, and recurrence were constructed independently without cross-neighborhood integration.

```mermaid
graph TD
    subgraph Ingestion_Neighborhood["1. Ingestion Neighborhood"]
        TIV["Topological Ingestion Validator"]
        LegacyLSTM["Legacy TextbookFilter LSTM (Deprecated)"]
    end

    subgraph FGRT_Core["2. FGRT Recurrent Island"]
        FGRT_Cells["7 FGRT Recurrent Cells<br/>(fgrt_rnn_cells.py)"]
    end

    subgraph Training_Neighborhood["3. Training Neighborhood"]
        TAT["Temporal Association Trainer<br/>(A -> B -> C Chains)"]
        RC["Resonance Cavity<br/>(Trust Memory)"]
    end

    subgraph Surrogate_Neighborhood["4. Surrogate Neighborhood"]
        CALM["CALM Trajectory Predictor<br/>(Autoregressive Collapse Veto)"]
        KAGH["KAGH B-Spline Surrogate"]
    end

    subgraph Governance_Neighborhood["5. Governance Neighborhood"]
        ISN["Bio-Archetypal Governor<br/>(Continuous-Time ISN Recurrence)"]
    end

    subgraph Orchestration_Neighborhood["6. Orchestration & Output"]
        ORCH["Cognitive Orchestrator<br/>(Polymorphic Core)"]
        LARYNX["Resonance Larynx<br/>(Autoregressive Speech)"]
    end

    TIV --> FGRT_Cells
    FGRT_Cells --> TAT
    TAT --> RC
    FGRT_Cells -.-> CALM
    KAGH --> FGRT_Cells
    ISN <--> FGRT_Cells
    FGRT_Cells --> ORCH
    ORCH --> LARYNX
```

### 5.1 Ingestion Neighborhood: Sequential Text Classification vs Topological Manifolds
- **The Divergence**: Early prototypes utilized a classical recurrent LSTM in [`textbook_filter.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/textbook_filter.py) to score educational content along a 1D token stream. This introduced teleological bias and probabilistic collapse.
- **The Integration**: The system transitioned to [`TopologicalIngestionValidator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/topological_ingestion_validator.py). Instead of unrolling tokens through an empirical LSTM, incoming sequential text streams are projected into Chebyshev polynomial functional residues and validated across `BerryPhaseGRUCell` or `FeatureScarLCFTCell` passes, ensuring that topological holes and invariants are preserved at the ingestion boundary.

### 5.2 Training Neighborhood: Temporal Association Training Unrolling
- **The Isolation Gap**: [`TemporalAssociationTrainer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/training/temporal_association_trainer.py) feeds ordered sequences (e.g. concept sequences $A \to B \to C$) and association windows into the reasoner. However, it historically treated temporal training as static multi-vector snapshots passed into `GyroidicFluxReasoner` rather than stepping states through recurrent cells.
- **The Integration**: The temporal association pipeline leverages FGRT cells for sequential dependency accumulation:
  - For continuous associative clustering: `LazarusSuperpositionRNNCell` linearly stacks incoming concepts via incommensurate prime frequencies ($f_t = 2\pi\ln(p_t)$) without non-linear gradient attenuation.
  - For causal state fossilization: `FeatureScarLCFTCell` captures persistent cause-and-effect transitions as topological scars in the Cerumen Pot, preventing catastrophic forgetting of foundational temporal sequences.

### 5.3 Surrogate Neighborhood: CALM Trajectory Vetoes & KAGH B-Splines
- **The Isolation Gap**: [`calm_predictor.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/surrogates/calm_predictor.py) acts as an unrolled autoregressive trajectory predictor monitoring spectral entropy and Betti number collapse ($\Delta\beta_k$) over a prediction horizon $H$. It ran completely decoupled from the recurrent hidden states of FGRT cells.
- **The Integration**:
  - `SaturatedQuantizerRNNCell` directly integrates `KAGHBlock` and `KANLayer` to generate speculative state drafts.
  - CALM serves as the **Trajectory Supervisor** for all 7 FGRT cells: as states $h_t$ and Berry phases $\gamma_t$ evolve across non-orientable Klein-Gyroid boundaries, CALM monitors the emergent trajectory $\gamma(t)$. If a Stiefel-Whitney parity flip produces an unmanageable divergence, CALM triggers a soft veto before the recurrent step compromises downstream stability.

### 5.4 Governance Neighborhood: Bio-Archetypal ISN Recurrence
- **The Isolation Gap**: The 6 archetypes (Jax, Pomni, Ragatha, Zooble, Kinger, Mandy) in [`bio_archetypal_governor.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/governance/bio_archetypal_governor.py) form an Inhibition-Stabilized Network (ISN). The ISN runs a continuous-time recurrent step:
  $$v_{t+1} = v_t + dt \cdot \left(-v_t + W_{\text{ISN}} f(v_t) + I_{\text{ext}}\right)$$
  This constitutes a biologically inspired recurrent neural network, yet it operated without utilizing FGRT topological mechanics.
- **The Integration**: `ChiralGatedRNNCell` provides the exact mathematical substrate for the archetypal ISN:
  - Bostick chiral projection vectors $\{\chi_k\}$ correspond directly to archetypal sector basins (e.g. Pomni's boundary resistance vs. Jax's parasitic attractor).
  - States are preserved or inhibited based on chiral orientation alignment, preventing parasitic mode dominance while maintaining homeostatic $E/I$ balance.

### 5.5 Orchestration & Output Neighborhood: Dynamic Cell Polymorphism & The Larynx
- **The Isolation Gap**: In [`orchestrator.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/orchestrator.py), `FeatureScarLCFTCell` was hard-coded at line 209 (`self.rnn_core`), leaving the other 6 cells dormant in the main cognitive loop. Meanwhile, [`ResonanceLarynx`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/models/diegetic_heads.py) autoregressively generated character tokens without receiving topological scar feedback from the recurrent core.
- **The Integration**:
  - **Affordance-Driven Polymorphic Core**: The orchestrator dynamically selects the appropriate recurrent cell based on runtime conditions:
    1. **GPU Sovereign Mode**: When GPU affordance is positive, select `PyOpenCLHardwareSovereigntyCell` to offload matrix mixing to OpenCL Queue B.
    2. **Low-Compute / Thermal Throttled Mode**: Fallback to `BerryPhaseGRUCell` for lightweight $O(1)$ orientation tracking.
    3. **Deep Abstraction / Long-Term Fossilization**: Engage `FeatureScarLCFTCell` to route state vectors around historical cerumen scars.
    4. **Harmonic Synthesis**: Engage `LazarusSuperpositionRNNCell` for non-compressed dark matter interference.
  - **Larynx Feedback Sealing**: The accumulated cerumen pot and phase track $(\gamma_t, \mathcal{C}_t)$ are passed into `ResonanceLarynx`, modulating the character-level generation loop with historical topological scars to prevent consonant clustering and vowel starvation.

### 5.6 Thermodynamic Neighborhood: Carnot Möbius Ledger Integration
- **The Integration**: Whenever non-orientable topological flips occur in `BerryPhaseGRUCell` (Stiefel-Whitney sign inversion) or in `ComplexFGRTRNNCell` (Atiyah-Singer cyclotomic index flip), the topological contorsion twist $\Delta\gamma_t$ is logged directly into [`CarnotMobiusLedger`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/carnot_mobius_ledger.py), tracking entropy generation and enforcing strict Landauer thermodynamic dissipation bounds.
