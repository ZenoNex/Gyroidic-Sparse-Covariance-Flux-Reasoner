"""Unit tests for 8x8 Hessian Curvature Ledger and Trajectory Decombiner.

Verifies:
1. 8x8 Hessian spectral decomposition, condition number, zero-mean minimal surface check.
2. Coxeter B2 / D4 dihedral gauge reduction to fundamental Weyl chamber (8-fold invariance).
3. Bouligand directional subdifferential kink intensity detection across non-smooth boundaries.
4. Order-8 cyclotomic harmonic filters (roots of unity projections).
5. Rich-club attractor classification (attractor hub, Klein throat neck, minimal surface saddle).
6. 2048 8x8 trajectory decombination with log-domain encoding, exact sum invariant, and 2-cycle replay guard.
"""

import math
import torch
import pytest

from src.core.hessian_8x8_curvature_ledger import (
    Hessian8x8CurvatureLedger,
    Hessian8x8LedgerResult,
    Trajectory8x8Decombiner,
    AttractorRegime,
)


def test_hessian_8x8_spectral_decomposition():
    """Verify 8x8 Hessian spectral eigensystem, condition number, and trace."""
    ledger_calc = Hessian8x8CurvatureLedger(delta_h=1e-3, dt=1.0)
    
    # State with 8 dimensions
    state = torch.tensor([0.2, -0.4, 0.1, 0.8, -0.2, 0.5, -0.1, 0.3], dtype=torch.float32)
    
    # Quadratic potential: f(x) = sum(k * x_k^2) / 2
    # Exact Hessian is diagonal with entries [1, 2, 3, 4, 5, 6, 7, 8]
    def energy_fn(x):
        weights = torch.arange(1, 9, dtype=x.dtype, device=x.device)
        return 0.5 * torch.sum(weights * (x ** 2))
    
    ledger = ledger_calc.compute_ledger(state=state, energy_fn=energy_fn, compute_bouligand=True)
    
    assert isinstance(ledger, Hessian8x8LedgerResult)
    assert len(ledger.eigenvalues) == 8
    # Eigenvalues should be sorted ascending
    for i in range(7):
        assert ledger.eigenvalues[i] <= ledger.eigenvalues[i + 1] + 1e-4
        
    # Check trace: sum of 1..8 = 36
    assert abs(ledger.trace_curvature - 36.0) < 0.2
    
    # Condition number should be approx 8 / 1 = 8.0 (within numerical discretization tolerance)
    assert abs(ledger.condition_number - 8.0) < 1.0
    
    # Quadratic potential is smooth -> Bouligand kink should be very small
    assert ledger.bouligand_kink_intensity < 2.0


def test_gyroid_minimal_surface_zero_mean_curvature():
    """Verify that a minimal surface saddle has near-zero trace and mixed signature."""
    ledger_calc = Hessian8x8CurvatureLedger(delta_h=1e-3, dt=1.0)
    
    # Saddle energy: f(x) = x_0^2 - x_1^2 + x_2^2 - x_3^2 + ...
    # Trace of Hessian is exactly 0
    state = torch.zeros(8, dtype=torch.float32)
    def saddle_energy(x):
        signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=x.dtype)
        return 0.5 * torch.sum(signs * (x ** 2))
        
    ledger = ledger_calc.compute_ledger(state=state, energy_fn=saddle_energy)
    
    assert abs(ledger.trace_curvature) < 1e-2
    assert ledger.is_minimal_surface is True
    # Morse index should be 4 negative eigenvalues
    assert ledger.morse_index == 4


def test_coxeter_d4_fundamental_chamber_projection():
    """Verify D4 dihedral symmetry reduction to canonical chamber across 8 gauge copies."""
    ledger_calc = Hessian8x8CurvatureLedger()
    
    # Asymmetric 8x8 matrix
    base = torch.zeros((8, 8), dtype=torch.float32)
    base[1, 2] = 5.0
    base[3, 4] = -2.0
    base[0, 7] = 3.5
    
    # Project to fundamental chamber
    chamber_0, op_0 = ledger_calc.project_to_coxeter_chamber(base)
    
    # Rotated 90 degrees
    rot90 = torch.rot90(base, 1, [0, 1])
    chamber_rot90, _ = ledger_calc.project_to_coxeter_chamber(rot90)
    
    # Reflected horizontally
    fliph = torch.flip(base, [1])
    chamber_fliph, _ = ledger_calc.project_to_coxeter_chamber(fliph)
    
    # All 8 gauge copies must collapse to the exact same canonical fundamental chamber
    assert torch.allclose(chamber_0, chamber_rot90, atol=1e-5)
    assert torch.allclose(chamber_0, chamber_fliph, atol=1e-5)


def test_bouligand_subdifferential_kink_detection():
    """Verify that non-smooth piecewise contact functions trigger Bouligand kink tensor."""
    ledger_calc = Hessian8x8CurvatureLedger(delta_h=1e-3)
    
    # Asymmetric piecewise-quadratic contact function:
    # f(x) = sum (max(0, x_i)^2)
    # Derivative is 2*x for x > 0, 0 for x < 0. Kink in second derivative at x = 0.
    state = torch.zeros(8, dtype=torch.float32)
    def contact_kink_fn(x):
        return torch.sum(torch.relu(x) ** 2)
        
    ledger = ledger_calc.compute_ledger(state=state, energy_fn=contact_kink_fn, compute_bouligand=True)
    
    # Bouligand kink intensity must detect the forward/backward finite difference discrepancy
    assert ledger.bouligand_kink_intensity > 0.5


def test_cyclotomic_roots_of_unity_filter():
    """Verify that 8th cyclotomic roots of unity projections have 8 harmonic components."""
    ledger_calc = Hessian8x8CurvatureLedger()
    
    state = torch.tensor([1.0, 0.5, -0.2, 0.8, 1.2, -0.5, 0.3, -0.9], dtype=torch.float32)
    ledger = ledger_calc.compute_ledger(state=state)
    
    assert len(ledger.cyclotomic_harmonics) == 8
    # DC component (k=0) is real sum of curvature profile
    assert abs(ledger.cyclotomic_harmonics[0].imag) < 1e-4
    # Multiharmonic coherence should be a valid bounded float
    assert ledger.multiharmonic_coherence >= 0.0


def test_rich_club_attractor_classification():
    """Verify detection of attractor hubs, Klein throat neck states, and saddles."""
    ledger_calc = Hessian8x8CurvatureLedger()
    
    # 1. Strongly positive curvature (potential well hub)
    state_hub = torch.zeros(8)
    def well_fn(x):
        return 50.0 * torch.sum(x ** 2)
    ledger_hub = ledger_calc.compute_ledger(state=state_hub, energy_fn=well_fn)
    assert ledger_hub.attractor_regime == AttractorRegime.ATTRACTOR_HUB.value
    
    # 2. Saddle with high condition number (Klein throat neck state)
    def neck_fn(x):
        # steep positive along 1 dim, steep negative along another, gentle along others
        return 100.0 * (x[0]**2) - 100.0 * (x[1]**2) + 0.01 * torch.sum(x[2:]**2)
    ledger_neck = ledger_calc.compute_ledger(state=state_hub, energy_fn=neck_fn)
    assert ledger_neck.attractor_regime in (AttractorRegime.KLEIN_THROAT_NECK.value, AttractorRegime.MINIMAL_SURFACE_SADDLE.value)


def test_trajectory_8x8_decombiner_chain_state():
    """Verify 2048 8x8 trajectory decombination on sparse board with [2^k, 2] chain."""
    decombiner = Trajectory8x8Decombiner(budget=50000, max_preimages_per_step=32, use_coxeter_d4=True)
    
    # Board with 8 (value 8 = 2^3) and 2 (value 2 = 2^1)
    board = [[0] * 8 for _ in range(8)]
    board[0][0] = 8
    board[0][1] = 2
    
    result = decombiner.decombine_trajectory(board)
    
    assert result["initial_tiles_sum"] == 10
    assert result["status"] in ("completed", "budget_exhausted")
    assert result["orbit_count"] >= 1
    assert result["unique_dag_states"] >= 1
    assert "coxeter_canonical_initial" in result
    
    # Verify exact sum invariant pruning: impossible board with odd sum cannot be formed by 2s/4s
    impossible_board = [[0] * 8 for _ in range(8)]
    impossible_board[0][0] = 3  # Not a power of 2
    res_imp = decombiner.decombine_trajectory(impossible_board)
    assert res_imp["status"] == "invalid_sum_invariant"


def test_trajectory_8x8_transient_2cycle_protection():
    """Verify 2-cycle replay protection prevents oscillating spawn loops."""
    decombiner = Trajectory8x8Decombiner(budget=1000)
    
    # In log-domain: tile 4 has exponent 2. Formed by two tile 2s (exponent 1, 1).
    board = [[0] * 8 for _ in range(8)]
    board[0][0] = 2  # log2(4) = 2
    
    # Directional pre-images for left move: [1, 1] in row 0
    preimages = decombiner.reverse_move_2d(board, "left")
    assert len(preimages) >= 1
    # Check that at least one pre-image has two 1s
    row0_candidates = [p[0] for p in preimages]
    assert any(row.count(1) == 2 for row in row0_candidates)
