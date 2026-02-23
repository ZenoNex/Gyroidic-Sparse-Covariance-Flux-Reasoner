# Non-Commutativity Dynamics

**Status**: ✅ Implemented  
**Source**: [`src/core/noncommutativity_curvature.py`](../src/core/noncommutativity_curvature.py)  
**References**: [VETO_SUBSPACE_ARCHITECTURE](VETO_SUBSPACE_ARCHITECTURE.md) · [ai project report §"Non-commutativity curvature tensor"](ai%20project%20report_2-2-2026.txt) · [ZEITGEIST_ROUTER.md](ZEITGEIST_ROUTER.md)

---

## Motivation

Standard reasoning systems treat operators as commutative: apply transformation A then B, or B then A — same result. In the Gyroidic Flux Reasoner, this assumption is **explicitly rejected**.

When two update operators $A, B$ fail to commute, the order of application changes the state trajectory. This is not a bug — it is the **structural mechanism behind multi-zeitgeist reasoning** (Phase 18) and the foundation of the non-teleological arrow of time (Phase 6).

The `NonCommutativityCurvature` module provides a **runtime measurement** of how much the current pair of operators violates commutativity, so the system can detect when it is entering a non-commutative regime and modulate its behavior accordingly.

---

## Mathematical Foundation

### The Commutator

Given operators $A, B \in \mathbb{R}^{d \times d}$:

$$[A, B] = AB - BA$$

If $[A, B] = 0$, the operators commute and order is irrelevant. If $[A, B] \neq 0$, the system is in a **non-commutative regime** — path matters.

### The Curvature 2-Form

The raw commutator mixes symmetric (scaling) and antisymmetric (rotational) contributions. The curvature extracts only the **pure rotation** — the part that genuinely measures non-commutativity:

$$\kappa_{ij} = \frac{1}{2}\left([A, B]_{ij} - [A, B]_{ji}\right)$$

This is the **antisymmetric curvature 2-form** $\kappa = \frac{1}{2}([A,B] - [A,B]^\top)$. Geometrically, it is a differential 2-form — the curvature of the connection defined by $A$ and $B$ on the operator bundle.

### Curvature Norm and Relative Curvature

$$K_{\text{norm}} = \|\kappa\|_F \qquad \text{(total curvature)}$$

$$K_{\text{rel}} = \frac{K_{\text{norm}}}{\|A\|_F \cdot \|B\|_F} \qquad \text{(scale-invariant, } \in [0, \infty))$$

$K_{\text{rel}}$ is the preferred diagnostic metric because it is unit-free and comparable across different operator magnitudes.

### Veto Pressure

The curvature is mapped to a scalar pressure signal via:

$$p_{\text{NC}} = \sigma\!\left(3 \cdot K_{\text{rel}} - 1\right) \in (0, 1)$$

This sigmoid places the **threshold at $K_{\text{rel}} = 1/3$**: below this, pressure is below 0.5 (weak); above it, pressure rises rapidly toward 1 (strong non-commutativity).

---

## API Reference

### `NonCommutativityCurvature(dim, curvature_threshold=0.3, ema_decay=0.95)`

| Parameter | Default | Description |
|---|---|---|
| `dim` | required | Dimensionality of the operator space |
| `curvature_threshold` | 0.3 | $K_{\text{rel}}$ above which operators are "strongly non-commutative" |
| `ema_decay` | 0.95 | Decay for the running EMA of curvature history |

**Registered buffers**:
- `curvature_ema` — exponential moving average of `curvature_norm`
- `max_observed` — maximum curvature norm seen in this session

### `compute_curvature(A, B, update_ema=True) → Dict`

Full curvature pipeline. Returns:

| Key | Description |
|---|---|
| `kappa` | `[dim, dim]` antisymmetric curvature 2-form |
| `curvature_norm` | Scalar Frobenius norm of κ |
| `commutator` | `[dim, dim]` raw $[A,B]$ |
| `is_strongly_noncommutative` | bool — $K_{\text{rel}} >$ threshold |
| `curvature_ema` | Running EMA scalar |
| `relative_curvature` | Scale-invariant $K_{\text{rel}}$ |
| `max_observed` | Session maximum curvature |

### `curvature_pressure(A, B) → Tensor`

Convenience scalar in $(0,1)$. Used as a veto signal to the lattice — high pressure indicates the system should be cautious about committing to either operator ordering.

### `compute_wedge_components(kappa, basis_vectors=None) → Tensor`

Decomposes κ into wedge product coefficients $\kappa_{ij} e_i \wedge e_j$. Returns upper-triangular entries `[dim*(dim-1)/2]`. With a custom basis, projects via $E^\top \kappa E$ first.

---

## Integration with ZeitgeistRouter

`NonCommutativityCurvature` is used as an **optional diagnostic** inside `ZeitgeistRouter`:

```python
# Instantiated in ZeitgeistRouter.__init__ (if use_noncommutativity_check=True)
self._nc_curvature = NonCommutativityCurvature(dim=dim)

# Called in forward() during 'switching' or 'grazing' modes only
nc_curvature = float(self._nc_curvature(x, x).mean().item())
```

The diagnostic value is emitted in `diagnostics['nc_curvature']` (the payload field in `metrics['zeitgeist']`).

> **Note**: The current call passes `(x, x)` — measuring self-curvature of the state against itself. This serves as a proxy for the curvature of the facet-normal operators induced by $x$. A future enhancement would pass the actual pair of update operators (e.g., pre- and post-switch facet normals).

### Why Only in Switching/Grazing Modes?

Curvature computation requires a matrix multiply of shape `[dim, dim]` — non-trivial cost. In `interior` mode, non-commutativity is irrelevant (the system is safely inside a polytope, not at an operator boundary), so the computation is skipped.

---

## Relationship to the Arrow of Time

Non-commutativity is the **structural source** of the system's directionality. If all operators commuted, there would be no notion of "this ordering vs. that ordering" — the system would be reversible and could not encode memory of its path.

The curvature 2-form $\kappa$ directly measures how much the current operating regime *breaks time-reversal symmetry*. High $\kappa$ = **the system is writing irreversible history**. Low $\kappa$ = the system is in a reversible (interior) regime where path doesn't matter.

This connects to:
- **Chirality Index** (INVARIANT_OPTIMIZATION §4) — the arrow of time in the spectral domain
- **Fossilization** — making irreversible the accumulated structural history
- **Polytope Switching** (Phase 18) — the non-commutative transitions that constitute the system's narrative path

---

## Veto Lattice Connection

`curvature_pressure()` returns a scalar ∈ (0,1) suitable for use as a soft veto signal. High pressure means:

> "The current update ordering is path-dependent enough that we should not assume the result is robust to reordering."

This can be used by the veto subspace (VETO_SUBSPACE_ARCHITECTURE) to suppress or modulate downstream operations that would commute the operators incorrectly.

---

## Related Documents

- [ZEITGEIST_ROUTER.md](ZEITGEIST_ROUTER.md) — Phase 18 consumer of nc_curvature diagnostics
- [VETO_SUBSPACE_ARCHITECTURE.md](VETO_SUBSPACE_ARCHITECTURE.md) — Veto lattice and BoundaryState
- [INVARIANT_OPTIMIZATION.md](INVARIANT_OPTIMIZATION.md) — Chirality and Arrow of Time
- [SYSTEM_ARCHITECTURE §9.4](SYSTEM_ARCHITECTURE.md) — Meta-Polytope dynamics and non-commutative switching
