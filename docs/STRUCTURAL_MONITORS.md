# Structural Monitors & Yield Criteria

> Adversarial stress testing, entropy monitoring, soliton detection, yield projections, and hunger drives.

---

## 1. AdversarialStressTester

**Source**: [`src/core/collapse_poisoner.py`](../src/core/collapse_poisoner.py) (155 lines)

Generates synthetic "rupture" events to verify the Speculative Homology Engine detects structural failures correctly.

### Capabilities

| Method | Purpose |
|--------|---------|
| `generate_synthetic_rupture(manifold)` | Creates orthogonal perturbation via Gram-Schmidt that should change Betti numbers |
| `compute_cycle_debt(state)` | Measures topological "boredom" — cosine similarity to history (>0.9 = repeat) |
| `forward(manifold, trigger_rupture)` | Returns `{cycle_debt, debt_warning}` ± optional rupture tensor |

**Cycle Debt**: `debt = repeats / filled_history`. When `debt > 0.5`, the system is looping and needs structural perturbation.

---

## 2. LinguisticEntropyMonitor

**Source**: [`src/core/narrative_collapse.py`](../src/core/narrative_collapse.py) (108 lines)

Monitors reasoning chain smoothness to detect hallucination loops and artificial smoothing.

| Check | Method | Trigger |
|-------|--------|---------|
| **Entropy** | `calculate_entropy(hidden)` → softmax entropy | `entropy < 0.01` → smoothing warning |
| **Trajectory** | `detect_smoothing(current, recent)` → cosine sim of consecutive deltas | `smoothness > 0.99` → linear trajectory |

Uses `ResidueObstructionGraph` for PAS_h integration. Forward returns `{entropy, smoothing_warning, is_linear, smoothness_score}`.

---

## 3. RelationalKappa

**Source**: [`src/core/relational_kappa.py`](../src/core/relational_kappa.py) (149 lines)

Context-dependent soliton threshold — never learned, always architectural.

$$\kappa(t) = \mu_{\text{rupture}}(t) + \lambda \cdot \sigma_{\text{rupture}}(t)$$

| Parameter | Value | Constraint |
|-----------|-------|-----------|
| `λ (temperament)` | 4.5 | **NOT** `nn.Parameter` — chosen, not learned |
| `window_size` | 50 | Rolling statistics window |
| `min_history` | 5 | Minimum before κ computation |

**Key invariant**: Learning κ turns solitons into rewards → rewarded solitons become attractors → attractors destroy rupture sensitivity.

| Method | Returns |
|--------|---------|
| `is_soliton(tension)` | `(bool, κ)` — exceeds threshold? |
| `check_kappa_flatline()` | `True` if σ → 0 (forced topological thaw) |

---

## 4. Yield Criteria

**Source**: [`src/core/yield_criteria.py`](../src/core/yield_criteria.py) (76 lines)

Dual-regime plasticity for information flow:

### MohrCoulombProjection (Sharp/Local)
$$\tau = c + \sigma \tan\phi$$

Projects pressure onto the MC yield surface. When `|τ| > strength`, the system ruptures locally — the brittle edge is **preserved**, not smoothed.

### DruckerPragerProjection (Smooth/Global)
$$\alpha I_1 + \sqrt{J_2} - k = 0$$

Provides a convex envelope over MC rupture sites. Smoothly scales back pressure exceeding `k`.

| Class | Regime | Behavior at yield |
|-------|--------|------------------|
| `MohrCoulombProjection` | Local/brittle | Sharp rupture (preserves edges) |
| `DruckerPragerProjection` | Global/smooth | Convex envelope (navigability) |

---

## 5. ValenceFunctional

**Source**: [`src/core/valence_drive.py`](../src/core/valence_drive.py) (54 lines)

Measures the manifold's "hunger" — the drive to resolve structural dissonance.

$$V = \eta \cdot \max(0, \; P_{\text{current}} - S_{\text{satisfaction}})$$

where $S$ is an EMA baseline updated as `S ← 0.99·S + 0.01·P`.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `decay` | 0.99 | EMA decay for satisfaction baseline |
| `hunger_scale` (η) | 1.0 | Amplification of hunger signal |

Returns scalar hunger tensor. Zero hunger = system is at or below its historical satisfaction level.

---

## 6. Integration Map

```mermaid
graph TB
    AST["AdversarialStressTester<br/>(synthetic rupture)"] -->|tests| SHE["Speculative Homology Engine"]
    LEM["LinguisticEntropyMonitor<br/>(smoothing detection)"] -->|flags| DPE["DiegeticPhysicsEngine"]
    RK["RelationalKappa<br/>(soliton threshold)"] -->|gates| SOL["SolitonStability"]
    MC["MohrCoulomb<br/>(local yield)"] -->|shear planes| SCCCG["SpeculativeCoprimeGate"]
    DP["DruckerPrager<br/>(global yield)"] -->|envelope| SCCCG
    VF["ValenceFunctional<br/>(hunger)"] -->|drives| TRAIN["StructuralAdaptor"]
```
