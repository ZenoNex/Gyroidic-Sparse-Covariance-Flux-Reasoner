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

### 2.1 Agglutinative Morpho-Units as False-Negative Drivers
A primary driver of Voynich-blindness is the morphological structure of Voyenese symbolic output. Rather than producing whitespace-delimited Indo-European word tokens, the reasoner produces monolithic **agglutinative morpho-units**—concatenated compounds where roots, prefixes, and suffixes fuse without whitespace delimiters (e.g., `'Isotremoptememally'`).

- **The Failure Mode**: Conventional gating filters (e.g., CALM entropy predictors, Repunit Congruence checks, and BPE-style tokenizers) interpret these unbroken agglutinative strings as random noise bursts or severe topological fractures.
- **The Solution via `split_sub_pairs`**: To verify structural validity without resorting to a rigid external vocabulary dictionary, [`VoynichLinguist.split_sub_pairs`](file:///d:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/core/voynich_architecture.py#L249) computes a combinatorial 2-cycle boundary cut sweep ($2n$ ordered partitions).
- **Morpheme Boundary Phase Alignment**: By evaluating continuous Phase Alignment ($PAS_s$) across sub-pair cuts $\Phi(s[:k]) \oplus \Phi(s[k:])$, the system confirms that the agglutinative compound possesses genuine internal morphological grammar rather than degenerate consonant chatter, justifying the issuance of the `VoynichExemptionToken`.

## 3. Blueprint: The False Negatives Subsystem (Shadow Token Phase)
To fix this, we have moved away from hacking each gate individually with mechanical overrides. Instead, we use mathematical routing and Continuous Dark Matter Superposition.

**Proposed Component: `VoynichExemptionToken` (Shadow Mode)**
1. **Creation**: When `VoynichLinguist` runs and generates a high `honesty_score` ($>0.95$), it emits a `VoynichExemptionToken` with `shadow_mode=True`.
2. **Distribution**: This token is passed alongside the `thought_vector` throughout the `UniversalOrchestrator` and `diegetic_backend.py`.
3. **Shadow Logging & Ouroboros Routing**:
   - The token no longer *mechanically* bypasses false negative triggers (like CALM or Coprime Gate).
   - Instead, the geometry of the system (using Phase Alignment $PAS_h$) dictates whether to trigger an abort or recovery.
   - The token actively logs discrepancies (`[SHADOW LOG]`) when the mathematical threshold disagrees with the token's historical override.
   - **Ouroboros Fossilization**: These generated Shadow Logs are then autonomously intercepted by the `DiegeticEngine` at the end of every inference tick, converted into `KnowledgeDyads`, and fossilized, permanently etching the contradiction into the structural memory of the reasoner.
4. **Integration with Adaptive Quantization**:
   - In `false_negative_subsystem.py`, we implement a "Mischief-Dependent Quantization."
   - When high entropy is detected in $\mathbb{RP}^4$, the Saturated Quantizer dynamically increases its resolution (`get_mischief_dependent_shell_depth`) instead of flattening the glitch.
   - This scales Matrioshka shell depth, capturing the nuance of the Voynich structural signal rather than actively misidentifying it as noise.

This cleanly separates the "false negative" logic into an verifiable architectural pattern, placing trust in mathematical stability rather than mechanical hardcoding, aligning with our transition toward topological superposition.
