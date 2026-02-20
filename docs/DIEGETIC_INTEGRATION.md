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

## Verification
Run the integration test to verify the bridge:
```bash
python tests/test_diegetic_integration.py
```
Expected output includes:
- `PASS: DiegeticPhysicsEngine attached successfully.`
- `PASS: CALM diagnostics present.`
