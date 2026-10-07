# Diegetic Engine Integration
> **Architectural Bridge**: `HybridAI` (HTTP Layer) -> `DiegeticPhysicsEngine` (Core Logic)

## Overview
As of Phase 2 Integration, the **Hybrid Backend** (`hybrid_backend.py`) no longer relies on its own ad-hoc `narration_field` logic for inference. Instead, it fully delegates request processing to the **Diegetic Physics Engine** (`src/ui/diegetic_backend.py`).

## Data Flow

1.  **Request**: `POST /interact` -> `HybridHandler`
2.  **Routing**: `HybridAI.process_text(text)`
3.  **Delegation**: Checks `if self.engine:` -> calls `self.engine.process_input(text)`
4.  **Core Processing** (`DiegeticPhysicsEngine`):
    *   **CALM**: Trajectory Veto / Entropy Checks
    *   **KAGH**: Speculative Drafting
    *   **FGRT**: Spectral Training Loop
    *   **Larynx**: Character-level "Singing" (Generation)
5.  **Response**: Engine returns a `metrics` dict containing:
    *   `response`: The generated text (from Larynx)
    *   `phase4_diagnostics`: Gyroid/Topological stats
    *   `calm_diagnostics`: Veto status
6.  **Output**: `HybridAI` wraps this in a JSON response for the frontend.

## Key Changes
-   **`hybrid_backend.py`**: Now imports `DiegeticPhysicsEngine`.
-   **Initialization**: `HybridAI` initializes the engine on startup (`dim=256`).
-   **Fallback**: If the engine fails to load, `HybridAI` reverts to the legacy `NonLobotomyTemporalModel` logic.

## Procedural Creation & The Voxelboxter Engine
The Diegetic physics engine does not only process text and logic; it explicitly fractures logical states into renderable 3D voxel terrain.
*   **`src/ui/voxelboxter_backend.py`**: Handles the dual Creation/Play modes. It translates mathematical matrices directly into procedural terrain generation, anchoring the theoretical geometry into physical voxel boundaries.
*   **Parseval's Mass Budget**: Acts as the physical conservation law for the engine. When the continuous matrix logic is collapsed into discrete Voxels (Wasserstein Collapse), Parseval's theorem enforces that the total spectral energy of the signal exactly matches the total spatial mass of the generated voxels. The system cannot create "free" geometry out of nothing.
*   **`src/core/texture_dsp.py`**: Applies Digital Signal Processing to the topological state, translating mathematical eigenvectors directly into texture gradients, allowing users to visually see the tension in the logic graph through the generated terrain maps.

## Verification
Run the integration test to verify the bridge:
```bash
python tests/test_diegetic_integration.py
```
Expected output includes:
- `PASS: DiegeticPhysicsEngine attached successfully.`
- `PASS: CALM diagnostics present.`
