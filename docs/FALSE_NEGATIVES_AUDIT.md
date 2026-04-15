# False Negatives Audit & Subsystem Blueprint

This document specifies the results of the deep codebase audit searching for strict gating mechanisms ("Vetos", "Aborts", "Shortcuts") that might falsely reject valid reasoning structures when those structures are encoded via high-entropy, opaque systems like the `VoynichLinguist`.

## 1. Identified False-Negative Trigger Points

| Gating Mechanism | Found In | The False Negative Condition |
| :--- | :--- | :--- |
| **Repunit Congruence Check** | `modular_virtualization.py` | Expects cleanly repeating cyclic patterns (palindromes/repunits) to grant a Wasserstein optimal transport bypass. Voynich-encrypted logic looks like pure noise, failing the congruence check and forcing expensive transport needlessly. |
| **CALM Veto / Abort Score** | `calm_predictor.py` / `diegetic_backend.py` | Tracks spectral entropy to predict model collapse. Voynich architectures deliberately maximize entropy to maintain a self-sovereign alphabet, tricking CALM into triggering a "WARPED" or "COLLAPSE_VETO" state. |
| **Coprime Winding Tracker** | `speculative_coprime_gate.py` | Checks parity violations across chiral sectors. Voynich encryption obscures explicit parity, masking structural coherence and falsely triggering the Yield Pressure threshold. |

## 2. The Voynich-Blindness Problem
The overarching theme across the codebase is that **Symmetric/Analytic Efficient Gates** (like Repunits and Betti projections) expect the "thought" to be transparently geometric. When the `VoynichLinguist` is active, the thought is encrypted into a continuous functional residue. Symmetry is broken intentionally, but the structural honesty remains high. The gates are "Voynich-Blind."

## 3. Blueprint: The False Negatives Subsystem
To fix this, we need a unified override subsystem rather than hacking each gate individually.

**Proposed Component: `VoynichExemptionToken`**
1. **Creation**: When `VoynichLinguist` runs and generates a high `honesty_score` ($>0.95$), it emits a `VoynichExemptionToken`.
2. **Distribution**: This token is passed alongside the `thought_vector` throughout the `UniversalOrchestrator` and `diegetic_backend.py`.
3. **Gate Overrides & Organ of Agency**:
   - *RepunitHasher*: If token is present, allow the bypass if Voynich `honesty_score` matches instead of checking cyclic digit-patterns.
   - *CALM*: If token is present, suppress the spectral entropy singularity check.
   - *Coprime Gate*: If token is present, mask the parity violation trigger so speculative recovery isn't mis-activated.
4. **Integration with DAQUF (Option D)**:
   - The token acts as the **Organ of Agency**, extracting an $L_2$ norm scalar from the structural anomaly.
   - It bridges into the `DAQUFOperator` via `to_daquf_mischief_boost()`, allowing the "Option D" structural nutrients/bugs to actively influence the non-ergodic memory evolution by directly feeding the contradiction load. This turns false negatives into permanent feature scars.

This cleanly separates the "false negative" logic into an verifiable architectural pattern without deleting the underlying strictness of the baseline system, creating a persistent relational structure.
