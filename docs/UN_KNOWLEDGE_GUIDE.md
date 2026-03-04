# Developer Guide: Unknowledge Flux & Mischief

This document provides operational instructions for utilizing the **Unknowledge Substrate** within the Gyroidic Flux Reasoner.

## 1. Nostalgic Leak Configuration

Use the `NostalgicLeakFunctional` to preserve subcultural archetypes from ergodic erasure.

```python
from src.core.unknowledge_flux import NostalgicLeakFunctional

# Define a leak toward a specific internet archetype soliton
leak = NostalgicLeakFunctional(
    fossil_dim=64,
    alpha=10.0,  # Tightness of the visibility mask (The Apple)
    device='cuda'
)

# μ_l (Archetype coefficients) define the "flavor" of the leak
# o (Obstruction point) defines the location of the concealed center
```

## 2. Mischief Band Monitoring

The `EntropicMischiefProbe` tracks the "Good Bug" energy. Monitor these metrics in your logs to ensure the system isn't being lobotomized.

- **High $H_{mischief}$**: Indicates a healthy, playful manifold that is discovering hidden revelations.
- **Zero $H_{meta}$**: A danger sign! The system has become a sterile, crystalline machine. Increase the `play_ratio` in the sampler.

## 3. DAQUF Operator Usage

Evolution from `DAQF` to `DAQUF` allows for **Diegetic Amortization**.

```python
from src.core.daqf_operator import DAQUFOperator

daquf = DAQUFOperator(num_fossils=1024, fossil_dim=64)

# Apply during training loop
results = daquf.apply_daquf(
    failures=is_ruptured,
    flux_scores=speculative_flux,
    results={
        'energy_gaps': current_gaps,
        'mischief_scores': current_mischief
    }
)

# Check persistence
# Fossil persists if it represents a stable "unknowledge soliton"
is_persistent = results['persistence']
```

## 4. The Non-Dual Trainer

The `StructuralAdaptor` (in `trainer.py`) manages the interplay. 
- Ensure `is_good_bug` triggers are tuned to your manifold curvature.
- Monitor `daquf_persistence` vs `pressure` to find the non-dual equilibrium.

## 5. The Unknowledge Domain ($\mathcal{U}$)

The `UnknowledgeDomain` (in `unknowledge_domain.py`) protects functionally creative or "dream-like" topological cycles from being crushed by standard reconstruction constraints. Rather than evaluating states by their reduction of standard Loss, it measures the degree to which Mischief ($H_{mischief}$) allows a cycle to survive tension safely. Hyper-ring Closure topologies matching 'survivable_soliton' are aggressively shielded using the Computable Flux ($V_m$) score.

---
> **"We stop pretending to be algebra. We start being an ecology of unknowledge."**
