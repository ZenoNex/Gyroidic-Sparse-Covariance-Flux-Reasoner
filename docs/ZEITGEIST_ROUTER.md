# ZeitgeistRouter — CRT Polytope Switching

**Phase**: 18  
**Status**: ✅ Implemented (2026-02-22)  
**Source**: [`src/core/zeitgeist_router.py`](../src/core/zeitgeist_router.py)  
**References**: [ai project report §II–VI](ai%20project%20report_2-2-2026.txt) · [SYSTEM_ARCHITECTURE §9.4](SYSTEM_ARCHITECTURE.md) · [BIOMIMETIC_SYNTHESIS_REPORT §4.4](BIOMIMETIC_SYNTHESIS_REPORT.md)

---

## Overview

The `ZeitgeistRouter` implements **CRT Polytope Switching** — the mechanism by which the engine navigates between culturally non-commensurable meaning systems without forcing scalar reconciliation between them.

The fundamental problem it solves: a reasoner operating in multiple semiotic registers (formal logic, everyday analogy, domain-specific jargon) cannot reduce all registers to a single coordinate axis without losing what makes each register meaningful. The Zeitgeist Router instead assigns each register a **CRT index** α ∈ ℤ = ∏ ℤ_{p_i} and allows the system to switch between polytopes non-commutatively.

---

## Formal Basis

### The Stratified System State

The full system state is a quadruple:

$$S_t = (x_t,\; \alpha_t,\; \ell_t,\; u_t)$$

| Component | Type | Meaning |
|---|---|---|
| $x_t$ | `[batch, dim]` tensor | Current representation vector |
| $\alpha_t$ | tuple of residues $(r_1, \ldots, r_m)$ | Active polytope index (Zeitgeist) |
| $\ell_t$ | int | Matrioshka shell depth |
| $u_t$ | `BoundaryState` or `None` | Last facet crossing event |

### The CRT Index

The Chinese Remainder Theorem guarantees a bijection between the residue tuple and a unique integer α ∈ [0, M):

$$\alpha = \sum_{i=1}^m r_i \cdot M_i \cdot y_i \pmod{M}$$

where:
- $M = \prod_i p_i$ (total polytope space)
- $M_i = M / p_i$ (partial product)
- $y_i = M_i^{p_i - 2} \bmod p_i$ (modular inverse via Fermat's little theorem, valid for prime $p_i$)

The primes $(p_1, \ldots, p_m)$ are shared with `MetaPolytopeMatrioshka` and are dynamically generated (no hardcoded lists) to ensure uniqueness of the index across the full Matrioshka depth.

> **Note on implementation**: `pow(M_i, -1, p_i)` was intentionally replaced by `pow(M_i, p_i-2, p_i)` (Fermat's little theorem) for compatibility across all Python 3 versions.

---

## Three-Mode Dispatch

The router classifies each forward pass into one of four mutually exclusive modes:

```
               ┌─────────────────────────────────────────────────┐
               │             ZeitgeistRouter.forward              │
               │                                                  │
  x_t ──────►  │  facet check: |⟨nᵢ, x̂⟩ − cᵢ| < ε?            │
               │        │                                        │
               │   NO   ▼   YES                                  │
               │  ┌─────────────┐   ┌──────────────────────┐    │
               │  │  INTERIOR   │   │  boundary critical?  │    │
               │  │  α unchanged│   │  (stress norm > thr) │    │
               │  └─────────────┘   └──┬──────────────────-┘    │
               │                   NO  │  YES                    │
               │             ┌─────────┘  └──────────────────┐   │
               │             ▼                               ▼   │
               │  ┌──────────────────┐      ┌─────────────────┐ │
               │  │  α changed?      │      │   UNDEFINED     │ │
               │  │  SWITCHING/GRAZING│      │  NaN guard      │ │
               │  └──────────────────┘      └─────────────────┘ │
               └─────────────────────────────────────────────────┘
```

| Mode | Condition | Scalars OK | α updates |
|---|---|---|---|
| `interior` | No facet grazing | ✅ Yes | No |
| `grazing` | Grazing zone hit, Δα = 0 after round | ❌ No | No |
| `switching` | Grazing zone hit, Δα ≠ 0 | ❌ No | **Yes** |
| `undefined` | `BoundaryState.is_critical()` AND grazing | — | No (topological refusal) |

### Non-Commutativity — The Core Invariant

The key structural property enforced by this module:

$$\text{route}(x,\; \text{route}(y, S_0)) \neq \text{route}(y,\; \text{route}(x, S_0)) \quad \text{for distinct } x, y$$

This arises naturally from the switch computation: $\Delta\alpha_i = \text{round}(\sigma(\text{gate}(x))_i \cdot p_i) \bmod p_i$. Because `gate(x)` is state-dependent, computing $x$ first changes the starting state for $y$, producing a different path through $\mathbb{Z}$.

> **Cold-start note**: Immediately after weight initialization, the gate may produce the same Δα for different inputs (discrete round() can land at the same integer for nearby sigmoid outputs). Non-commutativity diverges as the gate specializes during training.

---

## Learned Parameters

| Parameter | Shape | Purpose |
|---|---|---|
| `facet_normals` | `[M, dim]` | Outward facet normal per modulus — initialized via Gram-Schmidt |
| `facet_thresholds` | `[M]` | Scalar threshold $c_i$ per modulus — initialized to 0 |
| `switch_gate` | `Linear(dim → M)` | Per-modulus switching pressure network |

**Initialization logic**: Gram-Schmidt orthonormalization of M random vectors in $\mathbb{R}^{dim}$ (for M < dim), or random unit vectors (for M ≥ dim). This ensures initial transversality — the system starts with facets pointing in distinct, non-overlapping directions.

---

## ZeitgeistState

`ZeitgeistState` is an **immutable dataclass** — every update returns a new object. This prevents accidental mutation of the persistent session state.

```python
@dataclass
class ZeitgeistState:
    alpha   : Tuple[int, ...]   # CRT residues (r_1, ..., r_m)
    level   : int               # Matrioshka shell depth ℓ_t
    moduli  : Tuple[int, ...]   # Fixed primes — session invariant
    boundary: Optional[object]  # Last BoundaryState (type-erased)
    mode    : str               # Current classification
    step    : int               # Monotonic update counter
```

**Factories**:
- `ZeitgeistState.initial(moduli)` — alpha = (0,…,0), level = 0, mode = 'interior'
- `state.switched(new_alpha, new_level, mode, boundary)` — returns new state, step += 1

**Serialization**: `state.to_dict()` emits all fields plus `crt_index` and `is_undefined`. This dict is embedded in the `metrics['zeitgeist']['diagnostics']` payload.

---

## Wired Orphan Modules

Two previously documented-but-disconnected modules are now active inside `ZeitgeistRouter`:

### ManifoldClock (Breathing Time)
- **Source**: `src/core/manifold_time.py`
- **Call**: `self._clock.tick(grazing_pressure)` on every `forward()`
- **Effect**: High grazing pressure → small dt (seriousness / defensive mode). Interior mode → large dt (play / exploratory mode).
- **Emitted in**: `diagnostics['clock_dt']`

### ValenceFunctional (Hunger Drive)
- **Source**: `src/core/valence_drive.py`
- **Call**: `self._valence(pressure_t)` — measures the dissonance gap between current pressure and saturated trust
- **Effect**: High valence = urgency to resolve structural dissonance
- **Emitted in**: `diagnostics['valence']`

Both modules fail gracefully (try/except on import and call) — the router runs normally without them.

---

## Engine Integration

### Initialization (`DiegeticPhysicsEngine.__init__`)

```python
# Phase 18 — uses the same CRT primes as MetaPolytopeMatrioshka
_mpm_moduli = tuple(MetaPolytopeMatrioshka(max_depth=5, base_dim=dim).crt_moduli)
self.zeitgeist_router = ZeitgeistRouter(dim=dim, moduli=_mpm_moduli)
self._zeitgeist_state = ZeitgeistState.initial(moduli=_mpm_moduli)
```

The moduli are shared with the Matrioshka stack so the CRT index space is consistent across the full Phase 17+18 pipeline.

### Call Site (`process_input`, Phase 2.7)

Located **after** `seed_state` is formed (post harmonic decomposition), **before** constraint pressure injection:

```python
_zg_mode, self._zeitgeist_state, _zg_diag = self.zeitgeist_router(
    seed_state,
    self._zeitgeist_state,
    boundary=self._last_matrioshka_diag.get('boundary_state', None),
)
```

`_zeitgeist_state` is **persistent across `process_input` calls** — the engine remembers which polytope it was in at the end of the previous interaction.

### Metrics Payload

```python
metrics['zeitgeist'] = {
    'mode':        _zg_mode,          # 'interior' | 'grazing' | 'switching' | 'undefined'
    'alpha':       [...],             # current residue tuple as list
    'crt_index':   ...,               # unique int ∈ [0, M)
    'step':        ...,               # monotonic counter
    'diagnostics': {                  # full _zg_diag dict
        'prev_alpha', 'new_alpha', 'alpha_changed',
        'grazing_dims', 'grazing_pressure', 'facet_norms_mean',
        'clock_dt', 'valence', 'nc_curvature',
        'state',                      # full ZeitgeistState.to_dict()
    }
}
```

---

## Diagnostics Reference

| Key | Type | Description |
|---|---|---|
| `mode` | str | Current dispatch mode |
| `prev_alpha` | list | Residues before this step |
| `new_alpha` | list | Residues after this step |
| `alpha_changed` | bool | Whether a polytope switch occurred |
| `grazing_dims` | int | Number of facets currently grazed |
| `grazing_pressure` | float | Mean facet projection deviation |
| `facet_norms_mean` | float | Mean absolute facet projection |
| `clock_dt` | float? | Breathing time step from ManifoldClock |
| `valence` | float? | Hunger score from ValenceFunctional |
| `nc_curvature` | float? | NonCommutativity curvature norm (switching/grazing only) |

---

## Verification

```bash
python examples/verify_zeitgeist.py
```

20 checks covering: module import, CRT bijection round-trip, router forward pass (all modes), non-commutativity (inflated gate), diegetic_backend integration markers.

---

## Related Documents

- [SYSTEM_ARCHITECTURE §9.3–9.5](SYSTEM_ARCHITECTURE.md) — Matrioshka polytopes and Meta-Polytope dynamics
- [NONCOMMUTATIVITY_DYNAMICS.md](NONCOMMUTATIVITY_DYNAMICS.md) — The curvature tensor behind order-dependence
- [CONTEXT_AWARE_QUANTIZER.md](CONTEXT_AWARE_QUANTIZER.md) — Phase 17 quantizer feeding boundary states to Phase 18
- [TEMPORAL_DYNAMICS.md](TEMPORAL_DYNAMICS.md) — ManifoldClock and ValenceFunctional
- [VETO_SUBSPACE_ARCHITECTURE.md](VETO_SUBSPACE_ARCHITECTURE.md) — BoundaryState and is_critical()
