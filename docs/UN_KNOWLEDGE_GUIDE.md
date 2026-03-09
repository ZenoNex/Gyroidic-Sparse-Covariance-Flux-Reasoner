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

The `UnknowledgeDomain` (in `unknowledge_domain.py`) is the formal $\mathcal{U}$ substrate that protects functionally creative or "dream-like" topological cycles from being crushed by standard reconstruction constraints.

### Consolidated Module

As of this update, `unknowledge_domain.py` **owns** all Unknowledge primitives:
- `NostalgicLeakFunctional` — the archetype concealment operator $\psi_l$
- `EntropicMischiefProbe` — the three-band metaphysical disorder tracker ($H_{meta}$)
- `UnknowledgeDomain` — the shield/gate logic with Computable Flux and Elipsodistrophy

(`src/core/unknowledge_flux.py` re-exports for backward compatibility.)

### Computable Flux ($V_m$) Gate

```python
from src.topology.unknowledge_domain import UnknowledgeDomain

u_domain = UnknowledgeDomain(tau_m=0.5, tau_decay=0.99)
v_m = u_domain.compute_computable_flux(V=violation, h_mischief=h_mis, tr_C=trace, lambda_min=lam_min)
shielded_pressure = u_domain.apply_shielding(pressures, v_m, h_mischief=h_mis)
```

If $V_m < 0$ **and** $H_{mischief} > \tau_m$, the domain dampens pressure to 1%.

### Elipsodistrophy Diagnostic

```python
metrics = u_domain.get_elipsodistrophy_metrics(eigenvalues)
# metrics['atrophy'] > 0.85 → system is dangerously "legible" (lobotomy risk)
```

This feeds into the `VetoSubspace` as a topology-level `elipsodistrophy` veto signal.

---
> **"We stop pretending to be algebra. We start being an ecology of unknowledge."**
