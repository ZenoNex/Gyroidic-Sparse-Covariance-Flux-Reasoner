# Topology & Mathematical Invariants



## Topological Manifolds & Primitives

This section was auto-generated to document classes discovered in the codebase that were previously floating free from the architectural map.

### `BirkhoffPolytopeSampler`
**Location:** `src\core\polynomial_coprime.py`

**Description:**
Sample doubly-stochastic matrices from the Birkhoff polytope.

Uses Sinkhorn-Knopp projection to ensure:
    - Row sums = 1
    - Column sums = 1
    - All entries  0

---

### `ConstraintManifold`
**Location:** `src\optimization\constraint_probe.py`

**Description:**
Represents the constraint manifold C = C_sym  C_phys  C_ext

---

### `MartinovaCorrelationInvariant`
**Location:** `src\core\invariants.py`

**Description:**
Martinova Correlation Invariant.
Wraps compute_bounded_correlation to provide a standard module interface.

---

### `ObscuredBirkhoffManifold`
**Location:** `src\core\birkhoff_projection.py`

**Description:**
Obscured Birkhoff Polytope B_N^o.

Fuses the Sturmfels direct projection with the evolved obstruction level.

---

### `SparseHigherOrderTensorDynamics`
**Location:** `src\core\sparse_higher_order_tensors.py`

**Description:**
Implements sparse higher-order tensor interactions using Matrioshka shells.
Only computes N-th order dynamics along active polytope facets.

---

### `TensorEncoder`
**Location:** `src\ui\diegetic_backend.py`

**Description:**
Custom JSON encoder to handle PyTorch tensors and numpy arrays.

---

