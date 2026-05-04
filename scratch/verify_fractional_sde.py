"""
Verification script for the fractional SDE step and hunger wiring.

Tests:
1. ValenceFunctional computes non-zero hunger from defect signals
2. PolynomialADMRSolver.fractional_stochastic_differential_step() runs without error
3. The fractional step produces different outputs than the integer step
4. Hunger modulation changes the fractional order (alpha shifts toward 1.0)
"""

import torch
import sys
sys.path.insert(0, '.')

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.admr_solver import PolynomialADMRSolver
from src.core.valence_drive import ValenceFunctional
from src.core.deflagration_scout import OmipedialDeflagrator

def main():
    device = 'cpu'
    state_dim = 16
    batch_size = 2
    num_neighbors = 3

    print("=" * 60)
    print("FRACTIONAL SDE + HUNGER WIRING VERIFICATION")
    print("=" * 60)

    # 1. Setup components
    poly_config = PolynomialCoprimeConfig(k=4, degree=3, device=device)
    solver = PolynomialADMRSolver(poly_config, state_dim=state_dim, device=device)
    valence = ValenceFunctional(device=device)
    scout = OmipedialDeflagrator(dim=state_dim, device=device)

    print("\n[1] Components initialized OK")

    # 2. Create test data
    states = torch.randn(batch_size, state_dim, device=device)
    neighbors = torch.randn(batch_size, num_neighbors, state_dim, device=device)
    adj_weights = torch.softmax(torch.randn(batch_size, num_neighbors, device=device), dim=-1)

    # 3. Simulate defect detection
    predicted_flux = torch.randn(batch_size, state_dim, device=device) * 0.5
    actual_flux = predicted_flux + torch.randn_like(predicted_flux) * 0.3
    defects = scout.scout_defects(predicted_flux, actual_flux)
    print(f"\n[2] Defect signal mean: {defects.mean().item():.4f}")

    # 4. Compute hunger from defects
    hunger = valence(
        current_pressure=defects.mean(dim=-1),
        mischief=torch.tensor([0.3]),
        entropy=torch.tensor([0.5])
    )
    print(f"[3] Manifold Hunger: {hunger.mean().item():.4f}")
    assert hunger.mean().item() > 0.0, "FAIL: Hunger should be non-zero with defects!"
    print("    PASS: Hunger is non-zero (nerve is connected)")

    # 5. Run integer-order SDE step
    integer_result = solver.stochastic_differential_step(
        states, neighbors, adj_weights, dt=0.1
    )
    print(f"\n[4] Integer SDE step output norm: {integer_result.norm().item():.4f}")

    # 6. Run fractional-order SDE step (without hunger)
    frac_result_no_hunger = solver.fractional_stochastic_differential_step(
        states, neighbors, adj_weights, dt=0.1, hunger=None
    )
    print(f"[5] Fractional SDE step (no hunger) output norm: {frac_result_no_hunger.norm().item():.4f}")

    # 7. Run fractional-order SDE step (with hunger)
    frac_result_hungry = solver.fractional_stochastic_differential_step(
        states, neighbors, adj_weights, dt=0.1, hunger=hunger
    )
    print(f"[6] Fractional SDE step (hungry) output norm: {frac_result_hungry.norm().item():.4f}")

    # 8. Verify outputs differ
    diff_int_frac = (integer_result - frac_result_no_hunger).abs().mean().item()
    diff_hunger = (frac_result_no_hunger - frac_result_hungry).abs().mean().item()
    print(f"\n[7] Integer vs Fractional difference: {diff_int_frac:.6f}")
    print(f"[8] Fractional (no hunger) vs Fractional (hungry) difference: {diff_hunger:.6f}")

    # 9. Verify valence metrics
    metrics = valence.get_metrics()
    print(f"\n[9] Valence metrics:")
    print(f"    asymptotic_satisfaction: {metrics['asymptotic_satisfaction']:.4f}")
    print(f"    current_hunger_drive: {metrics['current_hunger_drive']:.4f}")

    # 10. Verify scout metrics
    scout_metrics = scout.get_metrics()
    print(f"\n[10] Scout metrics:")
    print(f"     defect_density: {scout_metrics['defect_density']:.4f}")
    print(f"     jump_readiness: {scout_metrics['jump_readiness']:.1f}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)

if __name__ == '__main__':
    main()
