# Gyroidic Scripting Systems

This document explains the three distinct tiers of scripting available within the Gyroidic Sparse Covariance Flux Reasoner ecosystem. Rather than a monolithic scripting language, the system partitions control into **Semantic**, **Structural**, and **Physical** tiers, enforcing the philosophy of "Patch Sovereignty" across its biomimetic components.

---

## Tier 1: Semantic Scripting (Diegetic Terminal)

The **Diegetic Terminal** (`http://localhost:8000`) is the highest-level interface, driven by `src/ui/diegetic_terminal.html` and `src/ui/diegetic_backend.py`. It provides control over the semantic meaning and cognitive "focus" of the Reasoner.

### Capabilities
- **Manifold Injection**: Users can inject raw semantic concepts, text, or multimodal dyad associations (images, audio).
- **Control Flags**: Adjust the Reasoner's regime (GOO vs. PRICKLES) and dyad commutativity.
- **Cognitive Routing**: By associating concepts, you script the "thought process" of the Reasoner as it navigates the embedding manifold.

*For full details on semantic routing, see [INTERFACE_LAYER.md](INTERFACE_LAYER.md).*

---

## Tier 2: Structural Scripting (Voxelboxter CLI & REPL)

**Voxelboxter** (`src/ui/voxelboxter_client.py`) acts as the tangible, PyBevy-based voxel client where abstract math turns into structural physics. It features a chat-based command line interface that provides Structural Scripting.

### Commands
- `/addon bspline [latent_dim]`: Injects a Compiled B-Spline mathematical mod layer into the environment.
- `/addon node`: Launches the **Physical Node Editor** (see Tier 3).
- `/create`: Switches the engine to CREATION mode (Admin editing unlocked).
- `/play`: Switches the engine to PLAY mode (Survival logic and physics enabled).

### Sovereign Execution (`/eval` and `/exec`)
Voxelboxter enforces **Patch Sovereignty**, meaning that if you have `ADMIN` or `BUILDER` roles, you possess unrestricted access to the underlying engine state. 

- `/eval [expression]`: Evaluates a raw Python expression.
- `/exec [statement]`: Executes raw Python statements.

**Context Injected**: Both commands are injected with the active PyBevy `state` (the `PatchStateResource`), `torch`, and `logging`.

*Example:*
```python
/eval len(state.graph.blocks)
/exec state.fingerprint_energy = 100.0
```

> **Note on Security**: These commands use unrestricted `eval()` and `exec()`. This is by design. The system does not sandbox the Administrator; to restrict `locals` would limit the kind of mods that could be generated, breaking the integrity of the sovereign user.

---

## Tier 3: Physical Scripting (Node Environment)

The most direct, hardware-adjacent scripting layer is the **Physical Node Editor** (`src/scripting/node_environment.py`), built using `dearpygui`.

When you run `/addon node` in Voxelboxter, the DearPyGui node editor opens. This allows you to construct "Virtual Links" between functional nodes that represent mathematical tensors, signal processors (Texture DSP), and sinks.

### D-Wave Collective Computation Pool
To prevent UI blocking during heavy DSP node computations, the Physical Scripting tier uses an **asynchronous evaluation worker thread** inspired by D-Wave collective computing.

- The node graph is evaluated constantly in a side-chain thread.
- It tracks the "dirty" state of sliders (e.g., the Latent Dimension of the BSpline Generator) and connection topologies.
- When an output (like a `BSpline Tensor Out`) is linked to the `Voxelboxter Graph Sink`, the worker pool dynamically recompiles the `BSplineCompiledMod` and pushes it directly into the running PyBevy `PatchStateResource.routine`.

This creates a seamless flow where adjusting a slider in the physical node editor instantaneously recompiles the voxel world's structural mathematics without blocking the rendering thread.
