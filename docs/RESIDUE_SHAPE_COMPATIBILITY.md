Residue Shape Compatibility and Forward Commutativity

Overview
- This document explains how the repair pipeline maintains consistent behavior when historical checkpoints contain parameters with residue-dependent shapes that differ from the current runtime configuration (e.g., K changes or derived residue_dim changes).

Key Concepts
- State Dimension (state_dim): The working vector length of the system state (e.g., 64).
- K (num_functionals): The number of functionals used to construct residues.
- Residue Dimension (residue_dim): Derived at runtime as padded_dim // K.

Runtime Residue Pipeline
1) Padding to multiple of K
   - If state_dim % K != 0, reflective padding is applied to reach padded_dim such that padded_dim % K == 0.
2) Residue view
   - The state is reshaped into [B, K, residue_dim].
3) Transformation in residue space
   - Each phase (Bezout CRT correction, Chern-Simons gasket, Soliton healing, Soft Saturated Gates) operates on [B, K, residue_dim].
4) Flatten and truncate
   - Results are flattened back to [B, padded_dim] and truncated to [B, state_dim], restoring the original working dimension.

Polynomial Coefficients Adaptation (Chern-Simons)
- Base coefficients [K, D] are adapted to residue_dim:
  - If D > residue_dim: truncate coefficients to [:, :residue_dim]
  - If D < residue_dim: expand via polynomial evaluation at additional points to create [:, residue_dim]
- This ensures coefficient compatibility with current residue_dim at runtime without dependence on historical values.

Forward Commutativity Under Non-Strict Loads
- Non-strict loading (strict=False) permits missing/unexpected keys in checkpoints (e.g., bezout_refresh.last_residues saved as [5,13] vs current [5,5]).
- Since residues are constructed from the current state every run, and phases adapt to residue_dim dynamically, the phase transformations commute with respect to historical shape differences:
  - T(pad(view(state))) ≈ pad(view(T(state))) after flatten/truncate to state_dim
  - Observable behavior at the working state_dim is preserved, even if a historical auxiliary parameter keyed to an older residue shape is skipped during load.

Practical Implications
- Checkpoints from older configurations with different residue shapes can be loaded non-strictly without breaking the pipeline.
- Diagnostics (e.g., Bezout condition_number) and transformations are driven by the current runtime residues and configuration.
- Persistence verification will pass: A freshly initialized engine and a loaded engine exhibit equivalent resonance/weights at the working dimension after the repair pipeline executes.

Verification
- Run python -m examples.verify_persistence
  - Expect informational logs for missing/unexpected keys
  - No runtime errors due to residue shape mismatches
  - Confirmed consistent metrics and persistence pass
