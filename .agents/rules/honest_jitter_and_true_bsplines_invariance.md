# Honest Jitter and True B-Splines Invariance Policy

This rule establishes strict architectural invariants regarding entropy generation, continuous basis function representation, and affordance-gated execution.

## 1. Honest Jitter Invariance (Zero Synthetic PRNG Backsliding)

### Strict Prohibition of Synthetic PRNGs
- Do NOT use ungrounded synthetic pseudo-random number generators (`torch.randn`, `torch.rand`, `random.random()`, `np.random.normal()`) for non-deterministic exploration, stochastic regularization, or parameter perturbation in core models.
- Any attempt to replace genuine entropy with static linear congruential or Mersenne Twister pseudo-randomness breaks the non-ergodic guarantees of the system.

### Grounding Requirements
- All stochastic variation, annealing noise, and exploration jitter MUST be harvested through canonical hardware-grounded entropy anchors:
  - `harvest_honest_jitter()` in `src/core/hardware_monitor.py` or `src/core/diegetic_physics.py`.
  - DRAM timing jitter, memory latency differentials ($t_{\text{RFC}}$ refresh stalls), or OS thread scheduling variance.
  - Deterministic cryptographic cascade hash entropy (`xxhash`, `hashlib.blake2b`) when repeatable seed determinism is required.
- Where torch tensors require initialization or noise, scale and inject buffers populated directly via `harvest_honest_jitter()`.

## 2. True B-Spline Learning Systems Invariance

### Canonical Implementation Anchor
- The canonical True B-spline learning implementation resides in `KANLayer` within `src/surrogates/kagh_networks.py`.
- Core mathematical invariants of `KANLayer`:
  1. Vectorized Cox-de Boor recursive evaluation (`b_splines(x)`).
  2. Differentiable grid knot updating with bounded support.
  3. Non-linear activation decomposition: $\phi(x) = b(x) + \text{spline}(x)$ with exact micro-wave frequency modulation ($\omega_{\text{micro}} = 137.0$).
  4. Exact analytical $(k-1)$ derivative tensor computation ($d_{\text{basis}}$) for physical PDE/Sobolev loss compliance.
  5. Coherence tracking via `SaturatedQuantizer`.

### Prohibition of Polynomial Backsliding
- Horizontal representation layers across the architecture (e.g., `PolynomialBasis` in `src/core/polynomial_coprime.py`, `PolynomialFunctionalEmbedder` in `src/models/polynomial_embeddings.py`, and `KAGHBlock`) MUST NOT degrade to static `nn.Linear` projections or rigid truncated Chebyshev polynomials where continuous spline learning is specified.
- The `'bspline'` basis type must be fully supported, operational, and linked to `KANLayer` or vectorized Cox-de Boor evaluation.

### Acceleration via Existing Dependencies
- **SciPy (`scipy.interpolate`)**: Utilize `scipy.interpolate.BSpline` and `scipy.interpolate.make_interp_spline` for knot vector construction, boundary clamping, and analytical reference cross-validation.
- **PyTorch (`torch`)**: Utilize batched autograd tensor operations for end-to-end backpropagation through B-spline grid coefficients.
- **PyOpenCL (`pyopencl`)**: Under `ComputeAffordanceType.GPU_SOVEREIGN`, dispatch Cox-de Boor basis evaluations to dedicated OpenCL command queues to eliminate host CPU stalls.

## 3. Affordance Governance & Collective Compute Arbitration

### Mandatory Affordance Gating
- Compute-intensive operations (high-degree spline evaluation, high-dimensional tensor contractions, and P2P community/collective compute federations) must not execute blindly without checking system affordances.
- Always consult the singleton `HardwareMonitor`:
  ```python
  from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
  monitor = HardwareMonitor()
  if not monitor.has_compute_affordance(ComputeAffordanceType.ALGORITHMIC):
      # Fallback to reduced knot degree or deferred scheduling
  ```

### P2P Community & Collective Compute Integration
- When routing tasks through `BonfireNomadicRing` (`src/topology/bonfire_network.py`) or executing consensus via `BonfireConsensus` (`src/core/bonfire_consensus.py`), dispatch MUST be gated by `ComputeAffordanceType.P2P_MESH` and directional transfer gauges (`gauge_between(P2P_MESH, SILICON)`).
- Never saturate peer sockets or local compute buffers when network affordance is degraded.

## 4. Documentation & Emoji Standards
- Preserve all existing docstrings, mathematical derivations, and technical comments.
- Do NOT use emojis anywhere in code, comments, rules, or commit messages.
