# Implementation Integrity Rules

Always adhere strictly to these architectural integrity rules when modifying or creating new logic.

## 1. No Hardcoded Primes (The Hardcoded Prime Heresy)
**FORBIDDEN**: Hardcoding prime arrays, harmonics, or indices (e.g. `primes = [2, 3, 5, 7]`).
**REQUIRED**: Use mathematical configuration objects (e.g. `PolynomialCoprimeConfig`) to generate coefficients dynamically.

## 2. No Placeholders (The Placeholder Lie)
**FORBIDDEN**: Using `pass`, `TODO`, or `torch.randn` as stubs for unwritten logic.
**REQUIRED**: Provide fully functional, mathematically sound logic natively initializing the correct configuration (e.g., using `PolynomialCoprimeConfig.get_coefficients_tensor()`).

## 3. No Teleological Contamination
**FORBIDDEN**: Do not optimize trust scalars using gradient descent (`trust_scalars.requires_grad_(True)`). Do not directly minimize target loss against an objective goal.
**REQUIRED**: Use Evolutionary Trust Updates. Modify scalars through survival thresholds and evolution rates. Use Energy-Based Contrastive Learning where correct states are pushed down and incorrect states are pushed up.

## 4. Constraint Probes
**REQUIRED**: System 2 constraint probes must be Non-Teleological. They must check local feasibility (e.g. `Gyroid violation`) rather than a global optimization objective.
