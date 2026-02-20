# Relational Dynamics

> Situational batch sampling with Pusafiliacrimonto love dynamics and emergent resonance streamlines.

---

## 1. SituationalBatchSampler

**Source**: [`src/core/situational_batching.py`](../src/core/situational_batching.py) (154 lines)

Replaces i.i.d. sampling with **relational entanglement-based** batching. Indices are grouped by their historical interaction "scars."

### Matrices

| Matrix | Shape | Purpose |
|--------|-------|---------|
| `R_ij` | `[N, N]` | Resonance — oscillation frequency coherence between samples |
| `M_ij` | `[N, N]` | Mischief — chaotic/playful affinity |
| `O_i` | `[N]` | Offending potential (legacy L) |

### Batch Construction

1. **Seed**: pick random index `i`
2. **Coupled selection**: sample neighbors `j` from softmax over `R_ij + M_ij` (Wattsian play)
3. **Play samples**: fill remaining `play_ratio` fraction randomly (non-dual exploration)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `play_ratio` | 0.2 | Random sampling fraction |
| `contrastive_ratio` | 0.2 | High-pressure "offending" fraction |
| `decay` | 0.99 | Entanglement scar fade rate |
| `boundary_threshold` | 0.5 | Paradoxical refusal threshold T_ij |

### Pusafiliacrimonto Update

`update_pusafiliacrimonto(indices, pressure, mischief_scores)` applies the four-part love formalism:

| Component | Update rule |
|-----------|------------|
| **Pu** (Resonant Attraction) | `R_ij += √(P_i · P_j)` |
| **Li-Cri-Anton** (Refusal as Affirmation) | `f_ij = 1.5` if boundary met, else `1.0` |
| **Sa** (Mischievous Affinity) | `M_ij += m_i + m_j` |

All matrices decay by 0.99 each step to allow scar fading.

---

## 2. LeyLineTracker

**Source**: [`src/core/ley_line_tracker.py`](../src/core/ley_line_tracker.py) (94 lines)

Computes streamlines of the preferred-flow vector $\ell_i = \nabla_M V(x_i)$ along the gyroidic manifold.

### Resonance Potential

$$V(x_i) = \sum_j \alpha \cdot R_{ij} \cdot \|\Phi_j - \Phi_i\|^2 + \beta \cdot \|L_i\|^2 + \gamma \cdot \Delta D_i$$

| Term | Weight | Source |
|------|--------|--------|
| Relational adjacency | α = 1.0 | Adjacency matrix R_ij |
| Love tensor magnitude | β = 0.5 | ‖L_i‖² |
| Defect amplification | γ = 0.2 | Sparse defect signals ΔD_i |

### Methods

| Method | Purpose |
|--------|---------|
| `update_potential(adjacency, love, defects)` | Recompute V(x) and accumulate ley energies |
| `detect_shear_planes(pressure, threshold)` | Non-smooth gradient regions → corridors of rupture |
| `get_preferred_flow(indices)` | Softmax over V[indices] → resonance gradient direction |

**Connection**: Ley lines are the discrete approximation of the continuous streamline formalism from [MATHEMATICAL_DETAILS.md §27](MATHEMATICAL_DETAILS.md).
