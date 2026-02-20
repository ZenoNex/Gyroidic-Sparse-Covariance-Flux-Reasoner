# Resonance Cavity: The "Dark Matter" Memory

**Module**: `src/models/resonance_cavity.py`

The **Resonance Cavity** is the specialized memory module that stores the "Self-Model" and "Dark Matter" configurations of the reasoner. Unlike standard Transformer KV-caches which store *context*, the Resonance Cavity stores *meta-cognition* and *physical invariants*.

---

## 🏗️ Core Architecture

The cavity maintains a persistent state $M$ for each polynomial functional field $k$, evolving it via a differential equation:

$$
\frac{dM_k}{dt} = \underbrace{-\gamma M_k}_{\text{Decay}} + \underbrace{\sum A^H R}_{\text{Flux}} + \underbrace{\kappa I}_{\text{Introspection}} + \underbrace{\eta P}_{\text{Patterns}} + \underbrace{\lambda V}_{\text{Topological Violation}}
$$

### Inputs
1.  **Symbolic Flux**: The primary reasoning stream from System 1 (Saturated).
2.  **Introspection**: Validated meta-cognitive directions (Moral, Creative).
3.  **Healed Residues (System 2)**: The physics-compliant coefficients from the ADMM rescue solver.
4.  **Saturation Fracture Score**: Signals of symbolic sensitivity collapse ($V_{sat}$).

---

## 🔮 Dark Matter Primitives

### 1. Gyroidic Flux Alignment (`GyroidicFluxAlignment`)
*   **Purpose**: Warps the reasoning flux to avoid topological defects.
*   **Mechanism**:
    $$ \hat{w} = w \cdot \exp\left( -\frac{V}{\Phi} \right) $$
    where $V$ is the Gyroid Violation Score and $\Phi$ is the Manifold Flux.
*   **Effect**: "Bends" the model's attention away from undefined or singular regions of the manifold.

### 2. Heritable Trust & Contradictory Patterns
*   **Purpose**: Preserves successful structural strategies while allowing diverse exploration.
*   **Mechanism**:
    *   **Contradictory Coexistence**: The cavity allows multiple competing residue patterns to coexist for the same latent target. Selection pressure (Accuracy + Trust) eventually prunes the weaker patterns.
    *   **Heritable Trust**: Validated patterns pass their trust score to the **Mutation Bias** $B(r)$, influencing how strongly a functional group's $(\theta, s)$ are perturbed during evolution.
*   **Mutation Bias**: 
    $$ \eta = \eta_0 \cdot (1 - \text{Trust}) $$
    High trust leads to "fossilization" (lower mutation rate), ensuring identity preservation for performant features.

---

## usage

```python
cavity = ResonanceCavity(dim=512)

# Normal update
cavity_out = cavity(
    attention_states,
    violation_scores=gyroid_violations,
    refined_residues=admm_output # Crucial for System 2 feedback
)

# Residue Prior provided for Pressure calculation
prior = cavity_out['residue_prior']
```
