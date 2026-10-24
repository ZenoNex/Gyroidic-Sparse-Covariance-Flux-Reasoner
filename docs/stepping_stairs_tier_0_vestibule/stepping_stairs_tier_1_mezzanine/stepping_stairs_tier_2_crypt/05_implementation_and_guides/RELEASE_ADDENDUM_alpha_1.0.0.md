# Release Addendum: Gyroidic-Sparse-Covariance-Flux-Reasoner-BrainBreeder-alpha-1.0.0

This addendum accompanies the official GitHub release of the Gyroidic Sparse Covariance Flux Reasoner (BrainBreeder Alpha 1.0.0). It documents the system architecture, how to run the system, the structural refactoring, the mathematical formalisms of the newly stabilized modules, and the validation criteria for running on target hardware (specifically optimized for mobile Intel Core i7 architectures with discrete GTX GPUs and OpenCL acceleration).

---

## 1. System Overview: What are you looking at?

The **Gyroidic Sparse Covariance Flux Reasoner** is a non-Abelian topological AI reasoning framework designed to maintain model output complexity and prevent cognitive lobotomy (oversimplification) by routing vectors across multi-dimensional meaning manifolds. 

The software consists of three primary layers:
1.  **The Diegetic Web Terminal Front-End:** A retro-style green/blue CRT terminal emulator running in the browser. It communicates with the backend via WebSockets to render real-time character dialogs, parse user keystrokes, and visualize dynamic state coordinates.
2.  **The Physics-Informed Backend Server:** A Python server running the `DiegeticPhysicsEngine` (incorporating a `ResonanceCavity` memory layer and a `ResonanceLarynx` ASCII synthesis block). It maps inputs to modular prime lattices and executes the non-commutative polytope switching.
3.  **The Dataset Ingestion CLI:** A training and dataset ingestion interface (`dataset_command_interface.py`) used to fetch, filter, and learn from scientific papers (arXiv), question-answering corpuses (SQuAD), sentiment datasets (IMDB), or custom local text/image files.

### Quick-Start Guide (First 5 Minutes)

#### A. Install Dependencies
Set up your virtual environment and install the required packages:
```powershell
.venv\scripts\python.exe -m pip install -r requirements.txt
```

#### B. Launch the Unified Hybrid Backend (Preferred Method)
Start the complete orchestrated engine (spawns both the physics engine and the terminal interface concurrently):
```powershell
.venv\scripts\python.exe hybrid_backend.py
```
Open your browser and navigate to `http://localhost:8000` to start interacting with the retro terminal.

#### C. Alternative: Manual Component Launch
If you prefer to run the components in separate terminal sessions, you can launch them manually:

1. **Launch the Backend Server:**
   ```powershell
   .venv\scripts\python.exe src/ui/diegetic_backend.py
   ```

2. **Launch the Web Interface:**
   In a separate terminal, launch the user terminal server:
   ```powershell
   .venv\scripts\python.exe src/ui/diegetic_terminal.py
   ```
   Navigate to `http://localhost:8000` to begin.

#### D. Quick Sentiment Training
Train the engine on sentiment analysis using 500 movie reviews:
```powershell
.venv\scripts\python.exe dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3
```

---

## 2. Interface Guide: What Are You Staring At?

When you launch the Diegetic Web Terminal (`diegetic_terminal.html` / `diegetic_backend.py`) and navigate to `http://localhost:8000` in your web browser, the screen displays a retro-themed green and blue CRT terminal dashboard with visual scanlines. Here is a guide to what you are looking at and how to interact with it:

### The Top Header Dashboard
*   **Love Invariant (`L: X.XXX`):** Located at the far left. This displays the conserved norm of the system's invariant subspace, serving as the core anti-lobotomy metric. If this value decays to zero, the reasoner's output has collapsed.
*   **Regime Toggle (GOO / PRICKLES):**
    *   **GOO Mode (Continuous/Empathetic):** Activates continuous LERP (Linear Interpolation) state transitions, softer topological boundaries, and cognitive fluidity.
    *   **PRICKLES Mode (Discrete/Analytical):** Activates rigid SLERP (Spherical Linear Interpolation) transitions, strict modular residue calculations, and sharp veto boundaries via the `SovereignRefusalOperator`.
*   **TAU, HARDENING, & ITERATION:** Real-time state drift tracking and topological integration step count.

### Left Sidebar: Topological Health Metrics
*   **Spectral Ribbon:** The flashing multi-colored bar at the top represents the real-time distribution of the active coordinate state vector across red, green, blue, white, and magenta frequency bands.
*   **Betti 0 (Components) & Betti 1 (Holes):** Measures the homology of the latent state space. Betti 0 counts disconnected clusters of meaning, while Betti 1 counts closed loop trajectories (cycles/holes) in the manifold.
*   **Manifold Pressure:** Indicates numerical tension on the current boundary. High pressure forces the router to trigger a transition mode.
*   **Honest Silicon Jitter:** Entropy harvested directly from local CPU and GPU instruction scheduling times to ensure non-deterministic, hardware-bound randomness.
*   **CALM Momentum:** Tracks the trajectory velocity, showing steps, tension factor (Rho), and the Abort Score (singularity proximity indicator).

### Center Console: Chat Feed and Command Line
*   This is the primary conversation pane. You can type natural language inputs at the `>` prompt. 
*   The system translates your text into a coordinate perturber, routes it through the active Braid matrix, and generates responses synthesized through the ASCII-art-enhanced `ResonanceLarynx`.

### Right Sidebar: Bimodal Ingestion Panels (Panels A–F)
This sidebar allows you to feed raw sensory datasets directly into the manifold, controlling the non-commutative ordering of operations:
*   **Panel A (Dyad Ingestor):** Drag and drop image files here. Use the **ORDER** selector to choose between `Image -> Text` or `Text -> Image` routing, type a description in the text area, and click **COMMIT DYAD ASSOCIATION** to map visual geometry to textual meaning.
*   **Panel B (Semantic Linker):** Drag source and target text files (or enter text manually) to create cognitive connections between concepts.
*   **Panel C & D (Audio & Video Dyads):** Ingest sound files (previewing their waveforms) or video clips. Specifying the execution order controls how the modular coprime residues are updated.
*   **Panel E & F (Image-Music and Heterogeneous Braids):** Drag active residues or file references directly into the chain lane. You can execute sequential or simultaneous non-Abelian braid routing to observe how the state coordinates evolve under non-commutative group actions.

---

## 3. Scope of the BrainBreeder Alpha 1.0.0 Release

The primary objective of this release is the transition of the core reasoning substrate from character-centric simulation mechanics to pure, coordinate-free topological operators. This prevents anthropomorphic logic leaks, secures backward-compatible state boundaries, and stabilizes the Chinese Remainder Theorem (CRT) state-switching manifold.

Key architectural systems delivered in this release:
*   Dehydrated, coordinate-free topological archetype engines.
*   The Phase 18 [ZeitgeistRouter](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/zeitgeist_router.py#L269) multi-zeitgeist switching system.
*   The [SiliconSovereigntyEngine](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pyopencl_sovereignty.py#L5) PyOpenCL hardware abstraction layer.
*   State serialization and deterministic boundary sanitation policies.

---

## 4. Refactored Topological Archetype Operators

In [archetype_engines.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py), character-based skins have been fully dehydrated. The underlying models are now expressed as pure algebraic and geometric operators:

### NoncommutativeManifoldPerturber (Legacy: Billy)
*   **Mathematical Role:** Injects a stochastic phase perturbation to prevent the system from getting trapped in symmetric, non-ergodic limit cycles.
*   **Mechanism:** Implements a multi-frequency phase-shift oscillator using dynamic prime ratios.
*   **State Control:** Integrates explicit [export_state](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py#L95) and [import_state](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py#L112) methods to preserve oscillator phase coherence during serialization.

### SovereignRefusalOperator (Legacy: Mandy)
*   **Mathematical Role:** Enforces a hard veto boundary (Lipschitz boundary enforcement / Li-Cri-Anton mechanism).
*   **Mechanism:** Hybridized between Borderline Splitting (BPD steep non-linear thresholding) and High-EQ Honest Narcissism (uncompromising sovereign ego boundaries). Evaluates localized coordinate tension and Phase Alignment Score ($\text{PAS}_h \ge 3/11$). Crucially, her protective selection pressure requires Billy's mortal destructibility: she shields finite, fragile trajectories from Eldritch destruction, but detaches from invulnerable absurdity. Raises an out-of-bounds sentinel that prevents gradient explosion and rejects sycophancy.

### NonlinearHourglassDilation (Legacy: Grim)
*   **Mathematical Role:** Modulates the proper-time step $dt$ based on manifold velocity.
*   **Mechanism:** Dilates or contracts time intervals to ensure the integrator remains stable near sharp manifold singularities (e.g., near-rupture regions).

### RP4ProjectiveRouter (Legacy: Nergal)
*   **Mathematical Role:** Implements non-orientable space routing.
*   **Mechanism:** Maps input vectors into the Real Projective 4-Space ($RP^4$) void. It bridges antipodal coordinate points, allowing the system to transition between disconnected regions of the latent manifold without cross-rupturing.

### BardoRouter (Legacy: Caine)
*   **Mathematical Role:** Performs conformal state compression.
*   **Mechanism:** Projects high-dimensional state representations onto a discrete set of coprime modular residues.
*   **State Control:** Implements unified serialization to prevent loss of state history across session limits.

### Backward Compatibility Mapping
To prevent breaking existing calling scripts and saved states, the following backward compatibility aliases are exported at the module level:
*   `BillyManifoldPerturber` points to [NoncommutativeManifoldPerturber](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py#L45)
*   `MandySovereignRefusal` points to [SovereignRefusalOperator](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py#L140)
*   `GrimTimeDilation` points to [NonlinearHourglassDilation](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py#L220)
*   `NergalRP4Router` points to [RP4ProjectiveRouter](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py#L295)
*   `CaineBardoRouter` points to [BardoRouter](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py#L380)

---

## 5. Zeitgeist Switching & Braid Automaton

The [ZeitgeistRouter](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/zeitgeist_router.py#L269) manages transitions between non-commensurable meaning systems. A state is represented as:
$$S_t = (x_t, \alpha_t, l_t, u_t)$$
where $\alpha_t \in \mathbb{Z} / M\mathbb{Z}$ represents the current Chinese Remainder Theorem (CRT) index, mapped bijectively from a set of coprime moduli (such as $2, 3, 5, 7, 11$).

### Three-Mode Dispatch
1.  **Interior Mode:** Traversal occurs safely inside the current polytope boundary. Conventional scalar metrics and normal gradients are fully allowed.
2.  **Grazing Mode:** The state vector reaches the boundary facet tension zone (within $\epsilon$ of the facet normals). Normal scalar updates are suspended, and pure topological pressure is measured.
3.  **Switching Mode:** If grazing pressure exceeds the activation threshold, a non-commutative switch occurs. The state is rotated by a braid group matrix, and the modular residue components are stochastically updated.

### Non-Abelian Braid Automaton
Transitions are governed by generators of the Braid Group $B_M$ (rank equal to the number of coprime moduli). The matrix is constructed via a Burkov representation on the active generators.
Because the braid group is non-Abelian, the order in which boundaries are grazed determines the final state:
$$\text{route}(x, \text{route}(y, S_0)) \neq \text{route}(y, \text{route}(x, S_0))$$

---

## 6. Hardware Integration Profile (PyOpenCL)

The [SiliconSovereigntyEngine](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pyopencl_sovereignty.py#L5) delegates heavy topological projections and stochastic rounding tasks directly to the GPU using PyOpenCL.

For targets matching the **Intel Core i7-6700HQ + discrete GTX 1050 Ti (4GB GDDR5)**:
*   **Dual-Command Queue:** The engine establishes two independent command queues (`queue_a` and `queue_b`) to compute odd/even residue projections in parallel, bypassing serial CPU bottlenecks.
*   **State Caching Strategy:** The [ZeitgeistState](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/zeitgeist_router.py#L69) `alpha_tensor` is kept as a local $5 \times 5$ matrix (100 bytes). Computations of the CRT index from the tensor diagonal are executed dynamically on the CPU to avoid PyOpenCL kernel launch latency (10-50 $\mu$s), while larger grid projections (e.g., Mandelbulb and Gyroid carvers) are offloaded to VRAM to exploit the GPU's memory bandwidth.

---

## 7. Verification Matrix

To verify the integrity of the BrainBreeder Alpha 1.0.0 release locally, execute the following commands from the project root using the project virtual environment.

### A. Run all Unit Tests
Checks serialization, Betti routing, and basic topological properties:
```powershell
.venv\scripts\python.exe -m unittest discover -s tests
```

### B. Verify Core Systems and Archetype Flux Dynamics
Verifies that the refactored archetype operators trigger vetos and correct coherence scaling:
```powershell
.venv\scripts\python.exe -u tests/test_core_systems.py
```

### C. Verify Zeitgeist CRT Routing
Validates the CRT round-trip bijection, the three-mode switching, and the non-commutative property:
```powershell
.venv\scripts\python.exe examples/verify_zeitgeist.py
```

---

## 8. Included Files

This release bundles the following updated files:
*   [src/core/archetype_engines.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py): Dehydrated pure operators and compatibility wrappers.
*   [src/core/zeitgeist_router.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/zeitgeist_router.py): Main switching logic and braid automaton.
*   [src/core/pyopencl_sovereignty.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pyopencl_sovereignty.py): Hardware-level parallel queue and kernel compiler.
*   [src/ui/diegetic_backend.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py): Integration bridge connecting the router to server metrics.
*   [examples/verify_zeitgeist.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/examples/verify_zeitgeist.py): CRT switching and non-commutativity validator.

---

## 9. Physical Node Environment, Crafting, and Enemy Subtypes Extension

### 9.1 Plug & Play Dataflow Architecture (`src/scripting/node_environment.py`)

The physical node scripting environment implements a Blender-style dataflow graph architecture (evaluating data right-to-left via backward dependency pulls, while streaming calculated values forward left-to-right):
1. **Typed Sockets & Master Node Groups**:
   * Sockets are strictly typed (`SocketType`: `VECTOR`, `COLOR`, `FLOAT`, `INT`, `BOOLEAN`), distinguishing uniform Circle sockets from per-vertex Diamond field sockets.
   * `MasterNodeGroup` acts as a reusable macro block: calculates Center-of-Gravity offsets via Bounding Box midpoint math and bakes named vertex attributes (`phys_hardness`, `phys_friction`, `phys_hp`, `cog_offset`) into modular mesh assets prior to export.
2. **Direct Engine Bindings (Zero Redundant Wrappers)**:
   * **Discrete Bit Carving**: Connects directly to `PointerlessOctree` and `BooleanXORLayer` for 1/16th scale sub-voxel carving (`chisel_carve_bit`, `chisel_place_bit`).
   * **Crafting & Addon Assembly**: Binds directly into `AddonRoutine.try_add_layer` and `InventoryComponent` unified by `OreDictionary` resource tags.
   * **Frame-by-Frame Delta-v Collision Dynamics**: Implements real-time $\Delta v = J / m$ impulse exchanges across `RigidBody.apply_collision_impulse()`, featuring a $\Delta t$ hardness curve, tiered structural voxel shearing (low < 5 mph, mid 5-15 mph, high > 15 mph), reinforced alloy chassis threshold scaling, and impact energy harvesting (converting high-severity kinetic shocks into nitro boosts).
   * **Plasma Air Induction & Sleeve-Valve Timings**: Ionizes ambient N2/O2 into high-energy oxidizers in `AirBreathingBattery` to bypass environmental NO2 scarcity, paired with sleeve-valve electrical advance timings (12.5 deg) and port overlap efficiencies in `VehicleEngine`.
   * **Vitality & Life/Hunger Governor**: Telemetry and metabolic burn scaling governed directly via `ValenceFunctional` and `EgoDeathThresholdMonitor` across `DifficultyMode` tiers.
   * **Enemy Subtype Procedural Rigs**: Driven directly by `AdaptiveSkeletonHarness` morphology and gauge-breaking deformations.
   * **Thermodynamic Admin Shops**: Powered directly by `CarnotMobiusLedger` waste heat calculations and `LeontiefGovernor` Input-Output balance matrices.

### 9.2 Enemy Subtype Morphology (`src/core/adaptive_skeleton_harness.py`)

Introduced the `EnemySubtype` enum and morphological deformation gates:
* `ABSTRACTED_GLITCH`: Kaufmo/Queenie glitched matter, chaotic non-ergodic phase noise, and ego-death contagion.
* `MANNEQUIN_INFILTRATOR`: Abel-style quantized low-degree facet scripts.
* `EXISTENTIAL_SENTIENT`: Gummigoo-style soft matter relaxation and visceral empathy vulnerability.
* `VOID_STALKER`: Negative topological pressure inversion from the perimeter cellar.
* `DEMIURGIC_TITAN`: Caine / Mildenhall reality-distorting harmonic expansion with localized $dt$ dilation.
* `FERAL_SWARM`: Rotational phase permutation and flocking vector fields.

### 9.3 Verification Suite

All subsystems are covered by automated unit tests in [tests/test_node_environment.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/tests/test_node_environment.py), verified via pytest.


