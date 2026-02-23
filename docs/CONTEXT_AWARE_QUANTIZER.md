# Context-Aware Quantizer (CAQ)

**Phase**: 17  
**Status**: ✅ Implemented  
**Source**: [`src/core/context_aware_quantizer.py`](../src/core/context_aware_quantizer.py)  
**References**: [ai project report §3](ai%20project%20report_2-2-2026.txt) · [VETO_SUBSPACE_ARCHITECTURE §5](VETO_SUBSPACE_ARCHITECTURE.md) · [SYSTEM_ARCHITECTURE §9.3](SYSTEM_ARCHITECTURE.md)

---

## Overview

The `ContextAwareQuantizer` (CAQ) realizes the core **Matrioshka Quantized Windows** equation from the project report:

$$x_{t+1} = Q_{\mathcal{Z}_t}\!\left(F\!\left(Q_{\mathcal{Z}_t}(x_t)\right)\right)$$

Quantization is applied **twice** per step — at input and after the feature transform $F$ — so the system always reasons inside a discrete lattice, never in the continuous intermediate space. The quantization grid $\mathcal{Z}_t$ is context-dependent: it adapts to the current Matrioshka shell depth and the per-axis Phase Alignment Score.

---

## Architecture

### Dependency Chain

```
ContextAwareQuantizer
    └── MetaPolytopeMatrioshka  ← CRT shell switching logic
            └── BoundaryState   ← emitted on shell crossings
```

CAQ wraps `MetaPolytopeMatrioshka` and adds **per-axis PAS anisotropy** on top. The Matrioshka handles CRT-level transitions; CAQ handles the step-size geometry.

---

## Step-Size Formula

The effective quantization step for axis $j$ at shell depth $\ell$ is:

$$\Delta_{j\ell} = e^{\delta_{\log}[\ell,\, j]} \cdot \frac{\Delta_0}{2^\ell}$$

where $\delta_{\log} \in \mathbb{R}^{(L+1) \times d}$ is a learnable `nn.Parameter` initialized to zero (so the initial step is exactly the depth-geometric base).

**PAS anisotropy** further sharpens the grid on structurally coherent axes:

$$\tilde{\Delta}_{j\ell} = \frac{\Delta_{j\ell}}{1 + \lambda \cdot \text{PAS}_j}$$

where $\text{PAS}_j \in [0,1]$ is the per-axis Phase Alignment Score from CALM, and $\lambda$ = `pas_anisotropy` (default 2.0).

| PAS_j | Effect |
|---|---|
| 1.0 (fully coherent) | Step shrinks by $1/(1+\lambda)$ — fine lattice |
| 0.0 (incoherent) | Step unchanged — coarse lattice |

High-coherence axes deserve finer resolution because the system can trust them; incoherent axes get coarse quantization to prevent noise from freezing the lattice.

---

## Shell Transition Logic

After per-axis quantization, CAQ passes the result to `MetaPolytopeMatrioshka`, which updates the CRT index (`_alpha`) and shell depth (`_level`) based on quantization energy:

| Condition | Transition |
|---|---|
| `energy > 0.1 · Δ` and `level < max_depth` | Dive deeper (increase ℓ) — finer shell |
| `energy < 0.01 · Δ` and `level > 0` | Surface (decrease ℓ) — coarser shell |
| Otherwise | Stay at current ℓ |

The CRT index transitions cyclically based on the state mean: `new_alpha = (alpha + int(x.mean() * 10)) % len(crt_moduli)`.

---

## BoundaryState Emission

When a **shell crossing** occurs (`_level` changes), CAQ constructs a `BoundaryState`:

$$\Sigma_{ij} = u_i \cdot n_j \quad \text{(outer product of state direction and facet normal)}$$

The facet normal is approximated from the quantization error direction — the axis with the largest error component is taken as the crossing direction.

`BoundaryState.is_critical()` returns `True` when:
- `‖Σ‖_F > threshold` (default 0.5), or
- `level ≥ max_level` (shell escape ceiling reached)

This `BoundaryState` is passed downstream to:
1. **`ZeitgeistRouter.forward(boundary=...)`** — triggers the `undefined` mode NaN guard
2. **`_last_matrioshka_diag['boundary_state']`** — cached for the next `process_input` cycle

---

## Persistent Shell State

CAQ maintains two Python ints across calls (not registered as torch buffers — they are control flow, not model weights):

```python
self._alpha: int = 0   # Current CRT index
self._level: int = 0   # Current shell depth
```

Call `reset_shell_state()` at session start or after a critical veto to return to the outermost shell.

---

## Parameters

| Name | Default | Description |
|---|---|---|
| `dim` | required | State vector dimensionality |
| `max_depth` | 5 | Number of Matrioshka shells (0 = coarsest, max_depth = finest) |
| `base_step` | 0.1 | Outermost shell base quantization step |
| `pas_anisotropy` | 2.0 | PAS scaling factor λ |

**`delta_log`** shape: `[max_depth+1, dim]` — initialized to zero, trained to discover which axes benefit from depth-dependent refinement.

---

## Diagnostics

`get_diagnostics()` returns (from last `forward()` call):

| Key | Description |
|---|---|
| `level` | Current Matrioshka shell depth |
| `alpha` | Current CRT index |
| `step_mean` | Mean effective step size |
| `step_min` / `step_max` | Step size range |
| `shell_crossed` | Whether a BoundaryState was emitted |
| `quant_error_rms` | RMS of `x − Q(x)` |

---

## Engine Integration

**Init** (`DiegeticPhysicsEngine.__init__`, Phase 17 block):

```python
if EXTENSIONS_AVAILABLE:
    self.caq = ContextAwareQuantizer(
        dim=dim, max_depth=5, base_step=0.1, pas_anisotropy=2.0
    )
```

**Call site**: The CAQ is called during the Phase 2.5/2.6 Matrioshka processing block in `process_input`. Its emitted `BoundaryState` is stored in `_last_matrioshka_diag` and consumed by the Phase 2.7 `ZeitgeistRouter`.

---

## Related Documents

- [SYSTEM_ARCHITECTURE §9.3–9.4](SYSTEM_ARCHITECTURE.md) — Matrioshka nested polytopes and quantization formalism
- [VETO_SUBSPACE_ARCHITECTURE §5](VETO_SUBSPACE_ARCHITECTURE.md) — BoundaryState and is_critical() threshold
- [ZEITGEIST_ROUTER.md](ZEITGEIST_ROUTER.md) — Phase 18 consumer of the BoundaryState emitted here
- [MATHEMATICAL_DETAILS §23.7](MATHEMATICAL_DETAILS.md) — Full BoundaryState tensor formalism
