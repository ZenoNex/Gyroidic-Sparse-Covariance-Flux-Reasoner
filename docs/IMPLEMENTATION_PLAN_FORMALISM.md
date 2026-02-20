# Implementation Plan: Formal Constraint Probe Architecture

**Author**: Implementation Plan  
**Date**: January 2026  
**Status**: **IN PROGRESS** (Updated January 18, 2026)

This document outlines the implementation plan to integrate the dense formalism into the existing Gyroidic Sparse Covariance Flux Reasoner architecture.

---

## Overview

The formalism introduces several critical concepts that transform System 2 from a "repair engine" into a **non-teleological constraint probe operator** with topological closure guarantees. This plan maps each formal concept to concrete implementation changes.

---

## 1. System 2 as Constraint Probe Operator (No Global Objective)

### Current State
- `OperationalAdmm` minimizes a global objective with violation penalties
- Uses gradient descent with step size
- Convergence-oriented (max_iters with early stopping)

### Required Changes

**1.1 Local Feasibility Probe Operator**
- **File**: `src/optimization/operational_admm.py`
- **Change**: Replace global minimization with per-constraint local probes
- **Implementation**:
  ```python
  class ConstraintProbeOperator(nn.Module):
      """
      P_k: r -> argmin_{c in C_k} L_k(r, c)
      No global objective, only local feasibility.
      """
      def __init__(self, constraint_index: int, sparse_covariance: torch.Tensor):
          self.k = constraint_index
          self.Sigma_k = sparse_covariance  # Anisotropic covariance
          
      def forward(self, residue: torch.Tensor) -> torch.Tensor:
          # Local strain: |Phi_k(r) - c|_{Sigma_k}
          # Gyroid violation: psi_k(c)
          # No global minimization
          ...
  ```

**1.2 Containment Pressure Loss**
- **Change**: Separate `local_strain` and `gyroid_violation` terms
- **File**: `src/optimization/operational_admm.py` (lines 108-114)
- **New Structure**:
  ```python
  local_strain = torch.norm(Phi_k(r) - c, p=2, weight=Sigma_k)
  gyroid_violation = psi_k(c)  # Admissibility filter, not truth metric
  L_k = local_strain + gyroid_violation
  ```

**1.3 Remove Convergence Guarantees**
- **Change**: Replace convergence checks with **bounded oscillation** detection
- **Implementation**: Track oscillation amplitude, accept if bounded within threshold

---

## 2. ADMM-like Cyclic Constraint Traversal (No Descent)

### Current State
- Sequential ADMM updates (c, z, u)
- Gradient-based descent steps
- Convergence-oriented iteration

### Required Changes

**2.1 Cyclic Constraint Indexing**
- **File**: `src/optimization/operational_admm.py`
- **Change**: Iterate over constraints `k = 1, ..., K` cyclically
- **Implementation**:
  ```python
  for t in range(max_iters):
      k = (t % K) + 1  # Cycle through constraints
      c_k = self.probe_operators[k](r, lambda_k)
      lambda_k = lambda_k + rho * (Phi_k(r) - c_k)
  ```

**2.2 Remove Gradient Descent**
- **Change**: Replace gradient steps with constraint-specific projections
- **File**: `src/optimization/operational_admm.py` (lines 117-119)
- **New**: Direct projection onto constraint manifold `C_k`

**2.3 Bounded Oscillation Detection**
- **Change**: Track oscillation amplitude instead of convergence
- **Implementation**: 
  ```python
  oscillation_amplitude = torch.std([c_k[t] for t in recent_steps])
  if oscillation_amplitude < threshold:
      accept_state()
  ```

---

## 3. Hyper-Ring Closure Condition

### Current State
- No explicit topological closure checks
- Homology computed via NetworkX (discrete graphs)

### Required Changes

**3.1 Hyper-Ring Operator**
- **New File**: `src/topology/hyper_ring_closure.py`
- **Implementation**:
  ```python
  class HyperRingOperator(nn.Module):
      """
      H(r) = ∮_C ∇_top Φ(r)
      """
      def forward(self, residue: torch.Tensor, constraint_manifold: torch.Tensor):
          # Compute topological gradient
          grad_top = self.compute_topological_gradient(residue, constraint_manifold)
          # Line integral around constraint boundary
          hyper_ring = self.line_integral(grad_top, constraint_manifold)
          return hyper_ring
  ```

**3.2 Closure Condition Check**
- **Implementation**:
  ```python
  def check_closure(self, hyper_ring: torch.Tensor) -> Tuple[bool, str]:
      """
      Closure iff:
      - H(r) in Z_1(C) (closed)
      - [H(r)] != 0 in H_1(C) (non-trivial)
      """
      is_closed = self.is_in_cycle_group(hyper_ring)
      is_trivial = self.is_trivial_cycle(hyper_ring)
      
      if not is_closed:
          return False, "fracture"
      elif is_trivial:
          return False, "collapse"
      else:
          return True, "survivable_soliton"
  ```

**3.3 Integration with ADMM**
- **File**: `src/optimization/operational_admm.py`
- **Change**: Check hyper-ring closure after each constraint probe
- **Action**: If closure fails, emit appropriate status token

---

## 4. Structural Irreducibility (Orthogonality Criterion)

### Current State
- `HypergraphOrthogonalityPressure` exists but doesn't check evidence module orthogonality
- No explicit projection operators per evidence cluster

### Required Changes

**4.1 Evidence Module Projections**
- **New File**: `src/core/structural_irreducibility.py`
- **Implementation**:
  ```python
  class EvidenceModuleProjection(nn.Module):
      """
      pi_alpha: C -> R^{d_alpha}
      Projects constraint manifold onto evidence cluster alpha.
      """
      def __init__(self, evidence_cluster: torch.Tensor, projection_dim: int):
          self.E_alpha = evidence_cluster
          self.d_alpha = projection_dim
          self.proj_matrix = nn.Linear(constraint_dim, projection_dim)
          
      def forward(self, constraint_state: torch.Tensor):
          return self.proj_matrix(constraint_state)
  ```

**4.2 Structural Irreducibility Check**
- **Implementation**:
  ```python
  def is_structurally_irreducible(
      self, 
      residue: torch.Tensor,
      evidence_modules: List[EvidenceModuleProjection]
  ) -> bool:
      """
      Check: <pi_alpha Phi(r), pi_beta Phi(r)> = 0 for alpha != beta
      and rank(⊕_alpha pi_alpha Phi(r)) > 1
      """
      projections = [pi(Phi(r)) for pi in evidence_modules]
      
      # Check orthogonality
      for i, proj_i in enumerate(projections):
          for j, proj_j in enumerate(projections):
              if i != j:
                  dot_product = torch.dot(proj_i, proj_j)
                  if abs(dot_product) > 1e-6:
                      return False
      
      # Check rank
      stacked = torch.stack(projections)
      rank = torch.linalg.matrix_rank(stacked)
      return rank > 1
  ```

**4.3 Integration with System 1**
- **File**: `src/models/gyroid_reasoner.py`
- **Change**: Add irreducibility check before CRT reconstruction
- **Action**: If reducible, trigger System 2 repair or emit failure token

---

## 5. Persistence-Based Residue Obstruction Graph

### Current State
- `ResidueHomologyDrift` tracks homology over time
- No explicit filtration-based persistent homology
- Obstruction graph is discrete (NetworkX)

### Required Changes

**5.1 Filtration Construction**
- **New File**: `src/topology/persistence_obstruction.py`
- **Implementation**:
  ```python
  class ResidueFiltration(nn.Module):
      """
      C_epsilon = {c in C | L(r, c) <= epsilon}
      """
      def __init__(self, residue: torch.Tensor, constraint_manifold: torch.Tensor):
          self.r = residue
          self.C = constraint_manifold
          
      def build_filtration(self, epsilon_values: torch.Tensor):
          """
          Build simplicial complex at each epsilon.
          """
          complexes = []
          for eps in epsilon_values:
              C_eps = self.filter_by_loss(eps)
              complex = self.build_simplicial_complex(C_eps)
              complexes.append(complex)
          return complexes
  ```

**5.2 Persistent Homology Computation**
- **Implementation**:
  ```python
  def compute_persistent_homology(
      self, 
      filtration: List[SimplicialComplex]
  ) -> Dict[int, List[Tuple[float, float]]]:
      """
      PH_k(r) = H_k(C_epsilon) as epsilon increases.
      Returns birth-death pairs for each dimension.
      """
      # Use existing PH library (e.g., ripser, gudhi) or custom implementation
      ...
  ```

**5.3 Obstruction Graph Construction**
- **File**: `src/topology/persistence_obstruction.py`
- **Implementation**:
  ```python
  class ResidueObstructionGraph:
      """
      G = (V, E) where:
      V = {r_i}
      E_{ij} <-> exists epsilon: beta_1^{ij}(epsilon) != 0
      """
      def build_graph(
          self, 
          residues: List[torch.Tensor],
          epsilon_range: torch.Tensor
      ) -> nx.Graph:
          G = nx.Graph()
          G.add_nodes_from(range(len(residues)))
          
          for i, r_i in enumerate(residues):
              for j, r_j in enumerate(residues):
                  if i != j:
                      # Compute joint persistent homology
                      beta_1_ij = self.compute_joint_loop_count(r_i, r_j, epsilon_range)
                      if beta_1_ij > 0:
                          G.add_edge(i, j, weight=beta_1_ij)
          
          return G
  ```

**5.4 Integration with Homology Pressure**
- **File**: `src/topology/homology_pressure.py`
- **Change**: Use obstruction graph instead of constraint graph
- **Benefit**: More principled cycle detection based on persistent homology

---

## 6. Soliton Stability Functional

### Current State
- No explicit soliton detection
- Gyroid violation is local, not global stability

### Required Changes

**6.1 Dispersion Computation**
- **New File**: `src/topology/soliton_stability.py`
- **Implementation**:
  ```python
  class SolitonStability(nn.Module):
      """
      D(r) = ∫_C |∇Φ(r)|^2 dμ
      """
      def compute_dispersion(
          self, 
          residue: torch.Tensor,
          constraint_manifold: torch.Tensor
      ) -> torch.Tensor:
          # Compute gradient of embedding
          grad_phi = torch.autograd.grad(
              outputs=self.embedding_fn(residue),
              inputs=residue,
              create_graph=True
          )[0]
          
          # Integrate |grad|^2 over constraint manifold
          dispersion = torch.sum(grad_phi.pow(2) * constraint_manifold)
          return dispersion
  ```

**6.2 Localization Computation**
- **Implementation**:
  ```python
  def compute_localization(
      self,
      residue: torch.Tensor,
      constraint_manifold: torch.Tensor,
      eta: float = 0.5
  ) -> torch.Tensor:
      """
      Lambda(r) = sup_{U subset C} mu(U) s.t. ∫_U |Phi(r)|^2 dμ >= eta
      """
      # Find largest region U where energy >= eta
      # Use greedy or optimization-based approach
      ...
  ```

**6.3 Soliton Condition Check**
- **Implementation**:
  ```python
  def is_soliton(
      self,
      residue: torch.Tensor,
      kappa: float = 0.1
  ) -> bool:
      """
      D(r) / Lambda(r) < kappa
      """
      D = self.compute_dispersion(residue)
      Lambda = self.compute_localization(residue)
      ratio = D / (Lambda + 1e-8)
      return ratio < kappa
  ```

**6.4 Integration with System 2**
- **File**: `src/optimization/operational_admm.py`
- **Change**: Check soliton condition after constraint probe
- **Action**: If soliton condition fails, mark as unstable (not necessarily failure)

---

## 7. Gyroidic Differentiation Constraint

### Current State
- `SparseGyroidCovarianceProbe` detects violations but doesn't enforce flow constraints
- No explicit gyroid gradient computation

### Required Changes

**7.1 Gyroid Implicit Surface**
- **New File**: `src/topology/gyroid_differentiation.py`
- **Implementation**:
  ```python
  class GyroidImplicitSurface(nn.Module):
      """
      G(x) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
      """
      def forward(self, x: torch.Tensor) -> torch.Tensor:
          # x: [batch, 3] or [batch, seq_len, 3]
          sin_x, cos_x = torch.sin(x[..., 0]), torch.cos(x[..., 0])
          sin_y, cos_y = torch.sin(x[..., 1]), torch.cos(x[..., 1])
          sin_z, cos_z = torch.sin(x[..., 2]), torch.cos(x[..., 2])
          
          G = sin_x * cos_y + sin_y * cos_z + sin_z * cos_x
          return G
      
      def gradient(self, x: torch.Tensor) -> torch.Tensor:
          """∇G(x)"""
          x.requires_grad_(True)
          G = self.forward(x)
          grad_G = torch.autograd.grad(G, x, create_graph=True)[0]
          return grad_G
  ```

**7.2 Flow Constraint Enforcement**
- **Implementation**:
  ```python
  class GyroidFlowConstraint(nn.Module):
      """
      ∇_flow Φ(r) ⟂ ∇G
      """
      def __init__(self):
          self.gyroid = GyroidImplicitSurface()
          
      def check_constraint(
          self,
          residue: torch.Tensor,
          embedding: torch.Tensor
      ) -> torch.Tensor:
          """
          Check: <∇_flow Φ(r), ∇G> = 0
          """
          grad_flow = self.compute_flow_gradient(residue, embedding)
          grad_G = self.gyroid.gradient(embedding)
          
          dot_product = torch.sum(grad_flow * grad_G, dim=-1)
          return torch.abs(dot_product)  # Should be ~0
  ```

**7.3 Forbidden Smoothing Condition**
- **Implementation**:
  ```python
  def check_forbidden_smoothing(
      self,
      residue_1: torch.Tensor,
      residue_2: torch.Tensor
  ) -> bool:
      """
      Check: NOT exists gamma: [0,1] -> C s.t.
      Phi(r1) ~ Phi(r2) and gamma subset G^perp
      """
      # Check if smooth path exists in gyroid-orthogonal space
      # If yes, return True (forbidden)
      # If no, return False (allowed)
      ...
  ```

**7.4 Integration with Constraint Probe**
- **File**: `src/optimization/operational_admm.py`
- **Change**: Add gyroid flow constraint to `gyroid_violation` term
- **File**: `src/topology/gyroid_covariance.py`
- **Change**: Extend `SparseGyroidCovarianceProbe` with flow constraints

---

## 8. Continuous Co-Primality (Non-Algebraic)

### Current State
- `PolynomialCoprimeConfig` uses algebraic co-primality
- No entropy-based independence tracking

### Required Changes

**8.1 Entropy Pressure Computation**
- **New File**: `src/core/continuous_coprimality.py`
- **Implementation**:
  ```python
  class ContinuousCoprimality(nn.Module):
      """
      E(r_i, r_j) = H(r_i + r_j) - H(r_i) - H(r_j)
      """
      def compute_entropy_pressure(
          self,
          residue_i: torch.Tensor,
          residue_j: torch.Tensor
      ) -> torch.Tensor:
          H_i = self.entropy(residue_i)
          H_j = self.entropy(residue_j)
          H_ij = self.entropy(residue_i + residue_j)
          
          E = H_ij - H_i - H_j
          return E
  ```

**8.2 Asymptotic Independence Check**
- **Implementation**:
  ```python
  def check_asymptotic_independence(
      self,
      residue_i: torch.Tensor,
      residue_j: torch.Tensor,
      time_steps: int = 100
  ) -> bool:
      """
      Check: lim_{t->∞} Cov(r_i^(t), r_j^(t)) = 0
      """
      covariances = []
      for t in range(time_steps):
          r_i_t = self.evolve(residue_i, t)
          r_j_t = self.evolve(residue_j, t)
          cov = torch.cov(torch.stack([r_i_t, r_j_t]))
          covariances.append(cov)
      
      # Check if covariance decays to zero
      final_cov = covariances[-1]
      return torch.abs(final_cov) < 1e-6
  ```

**8.3 Integration with Polynomial Config**
- **File**: `src/core/polynomial_coprime.py`
- **Change**: Add entropy-based co-primality check alongside algebraic check
- **Action**: Use both criteria (algebraic + continuous) for co-primality validation

---

## 9. Failure Token (Not Error)

### Current State
- `OperationalAdmm` returns status: 0=REPAIRED, 1=ALTERNATIVE, 2=FAILURE
- Failure is treated as error condition

### Required Changes

**9.1 Rupture Functional**
- **New File**: `src/core/failure_token.py`
- **Implementation**:
  ```python
  class RuptureFunctional(nn.Module):
      """
      R(r) = 1 if exists k: L_k(r, c_k) = ∞
      R(r) = 0 otherwise
      """
      def forward(
          self,
          residue: torch.Tensor,
          constraint_losses: Dict[int, torch.Tensor]
      ) -> torch.Tensor:
          """
          Check if any constraint loss is infinite or exceeds threshold.
          """
          for k, loss_k in constraint_losses.items():
              if torch.isinf(loss_k) or loss_k > self.rupture_threshold:
                  return torch.tensor(1.0)
          return torch.tensor(0.0)
  ```

**9.2 Failure Token Emission**
- **File**: `src/optimization/operational_admm.py`
- **Change**: On `R(r) = 1`, emit `⊥` token (not error, not gradient, not repair)
- **Implementation**:
  ```python
  if rupture_functional(residue, constraint_losses) == 1:
      return FailureToken.BOT, status=2
      # No gradient, no repair, no memory update
  ```

**9.3 Failure Token Type**
- **New File**: `src/core/failure_token.py`
- **Implementation**:
  ```python
  class FailureToken:
      BOT = "⊥"
      REPAIRED = "repaired"
      ALTERNATIVE = "alternative"
      
      def __init__(self, token_type: str):
          self.type = token_type
          self.gradient_enabled = False
          self.repair_enabled = False
          self.memory_update_enabled = False
  ```

---

## 10. Meta-Invariant (Non-Teleological)

### Current State
- No explicit meta-invariant tracking
- Topology can collapse (no expansion guarantee)

### Required Changes

**10.1 Topology Expansion Monitor**
- **New File**: `src/core/meta_invariant.py`
- **Implementation**:
  ```python
  class MetaInvariant(nn.Module):
      """
      d/dt E_r[dim H_1(C_t)] >= 0
      Topology must never collapse toward single basin.
      """
      def __init__(self):
          self.register_buffer('prev_h1_dim', torch.tensor(0.0))
          
      def check_invariant(
          self,
          current_h1_dim: torch.Tensor,
          residue_distribution: torch.Tensor
      ) -> Tuple[bool, torch.Tensor]:
          """
          Check: d/dt dim H_1 >= 0
          """
          expected_dim = torch.mean(current_h1_dim)
          rate = expected_dim - self.prev_h1_dim
          
          violation = (rate < 0).float()
          self.prev_h1_dim = expected_dim
          
          return rate >= 0, violation
  ```

**10.2 Integration with Evolution**
- **File**: `src/models/gyroid_reasoner.py`
- **Change**: Monitor meta-invariant during forward pass
- **Action**: If invariant violated, trigger topology expansion (not collapse)

---

## Implementation Priority & Dependencies

### Phase 1: Core Constraint Probe (High Priority)
1. **Constraint Probe Operator** (Section 1)
2. **Cyclic Constraint Traversal** (Section 2)
3. **Failure Token** (Section 9)

### Phase 2: Topological Guarantees (Medium Priority)
4. **Hyper-Ring Closure** (Section 3)
5. **Persistence Obstruction Graph** (Section 5)
6. **Soliton Stability** (Section 6)

### Phase 3: Advanced Constraints ✅ COMPLETE (January 2026)
7. **Structural Irreducibility** (Section 4) - Implemented in `structural_irreducibility.py`
8. **Gyroidic Differentiation** (Section 7) - Implemented in `gyroid_differentiation.py`
9. **Continuous Co-Primality** (Section 8) - Implemented in `polynomial_coprime.py`
10. **Meta-Invariant** (Section 10) - Implemented in `meta_invariant.py`

### Phase 4: Sparse Operational Pointers ✅ COMPLETE (January 2026)
11. **Rejection-Only Selection** - `trainer.py` (Pointer #1)
12. **Legibility Audit** - `legibility_audit.py` (Pointer #2)
13. **Saturation-Gated Fossilization** - `polynomial_coprime.py` (Pointer #3)
14. **Coherence-as-Signal** - `meta_invariant.py` (Pointer #4)
15. **Violation Scouting** - `gyroid_covariance.py` (Pointer #8)
16. **Narration Suppression** - `introspection_head.py` (Pointer #11)
17. **Narrative Collapse Detection** - `narrative_collapse.py` (Pointer #12)

---

## Testing Strategy

### Unit Tests
- Each new module gets isolated unit tests
- Test edge cases (infinite losses, trivial cycles, etc.)

### Integration Tests
- Test constraint probe with existing ADMM
- Test hyper-ring closure with homology computation
- Test failure token propagation through system

### Verification Scripts
- `examples/verify_hyper_ring_closure.py`
- `examples/verify_soliton_stability.py`
- `examples/verify_structural_irreducibility.py`

---

## Breaking Changes & Migration

### API Changes
- `OperationalAdmm.forward()` signature changes (constraint-specific)
- New return type: `FailureToken` instead of just status codes
- `gyroid_reasoner.py` forward pass may return additional topological diagnostics

### Backward Compatibility
- Add `legacy_mode` flag to maintain old behavior during transition
- Deprecation warnings for old API usage

---

## Open Questions for Review

> **All questions resolved as of January 18, 2026**

1. ✅ **Performance**: Use **approximate PH** (Ripser-style, trigger-based). Exactness is epistemically misleading.
   - Implemented: `ApproximatePHProbe` in `src/topology/approximate_ph.py`

2. ✅ **Hyper-Ring Computation**: **Discrete always.** Adaptive resolution on phase slippage.
   - Implemented: `DiscreteHyperRingCirculation` in `src/topology/hyper_ring.py`

3. ✅ **Evidence Modules**: **Tripartite** (data + architectural + adversarial). Must disagree internally.
   - Implemented: `TripartiteEvidenceModule` in `src/core/evidence_modules.py`

4. ✅ **Soliton Threshold**: **Relational κ(t) = μ + λσ**, never learned. λ is architectural temperament.
   - Implemented: `RelationalKappa` in `src/core/relational_kappa.py`

5. ✅ **Meta-Invariant Enforcement**: **Poison collapse paths**, don't force expansion.
   - Implemented: `CollapsePathPoisoner` in `src/core/collapse_poisoner.py`

---

## Next Steps

**PHASES 1-6 COMPLETE**

All implementation phases finished:
- Phase 1: Core Constraint Probe ✅
- Phase 2: Topological Guarantees ✅
- Phase 3: Advanced Constraints ✅
- Phase 4: Sparse Operational Pointers ✅
- Phase 5: Structural Design Decisions ✅
- Phase 6: Non-Ergodic Fractal Entropy ✅

### Phase 6: Non-Ergodic Fractal Entropy (January 18, 2026)

Optimized Fractal Entropy Decomposition using non-ergodic intra-domain methods:

18. **Band-Separated Entropy** - `src/core/non_ergodic_entropy.py`
    - Ergodic/transitional/soliton bands computed independently
19. **Adaptive Partitioning** - Spectral coherence-based block boundaries
20. **Soliton Preservation** - Dominant mode representatives (not mean)

---

**Status**: COMPLETE (January 18, 2026)


