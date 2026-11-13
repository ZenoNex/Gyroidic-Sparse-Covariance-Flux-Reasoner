import os
os.environ["Sovereign_Disable_Background"] = "1"

from unittest.mock import patch
import torch

from src.topology.modular_homology_fft import CyclotomicTDACompressor
from src.core.leontief_governor import LeontiefGovernor
from src.tda.chebyshev_filtration import MinimaxPolynomialApproximation
from src.core.invariants import compute_landau_homoenergetic_compression_audit
from src.core.spectral_coherence_repair import BezoutCoefficientRefresh
from src.surrogates.calm_predictor import CALM
from src.surrogates.kagh_networks import KAGHBlock
from src.core.hardware_monitor import HardwareMonitor
from src.core.orchestrator import UniversalOrchestrator


def test_cyclotomic_rref_mod_p_triangle_and_disconnected():
    tda = CyclotomicTDACompressor(p=5, ring_size=16)

    # Triangle graph: 3 vertices, 3 edges, 0 faces -> rank(d1)=2, beta_0=1, beta_1=1
    d1_triangle = torch.tensor([
        [-1,  0,  1],
        [ 1, -1,  0],
        [ 0,  1, -1],
    ], dtype=torch.long)
    rref, rank_d1 = tda.rref_mod_p(d1_triangle)
    assert rank_d1 == 2
    assert rref.shape == (3, 3)
    assert tda.exact_betti_mod_p(boundary_k=None, boundary_k_plus_1=d1_triangle, dim_ck=3) == 1
    assert tda.exact_betti_mod_p(boundary_k=d1_triangle, boundary_k_plus_1=None, dim_ck=3) == 1

    # Disconnected pair of edges on 4 vertices: 4 vertices, 2 edges -> rank(d1)=2, beta_0=2, beta_1=0
    d1_disconn = torch.tensor([
        [-1,  0],
        [ 1,  0],
        [ 0, -1],
        [ 0,  1],
    ], dtype=torch.long)
    assert tda.exact_betti_mod_p(boundary_k=None, boundary_k_plus_1=d1_disconn, dim_ck=4) == 2
    assert tda.exact_betti_mod_p(boundary_k=d1_disconn, boundary_k_plus_1=None, dim_ck=2) == 0


def test_leontief_governor_pivoted_solve_and_fallback():
    lg = LeontiefGovernor(state_dim=4)
    A = torch.eye(4).unsqueeze(0) * 0.25
    L, rho, is_prod = lg.compute_leontief_inverse(A)
    assert is_prod
    assert abs(rho - 0.25) < 1e-4
    assert torch.allclose(L, torch.eye(4) * (1.0 / 0.75), atol=1e-4)

    # Non-productive (rho >= safety margin) triggers Neumann fallback cleanly
    A_high = torch.eye(4).unsqueeze(0) * 1.1
    L_fb, rho_high, is_prod_high = lg.compute_leontief_inverse(A_high)
    assert not is_prod_high
    assert torch.isfinite(L_fb).all()


def test_chebyshev_and_landau_and_bezout_pivoted_solves():
    cheb = MinimaxPolynomialApproximation(degree=4)
    cheb.fit_remez_proxy(lambda x: x ** 2)
    assert torch.isfinite(cheb.coefficients).all()

    hc = compute_landau_homoenergetic_compression_audit(torch.eye(4) * 0.1, time_t=0.5)
    assert abs(hc["det_i_plus_tl0"] - (1.05 ** 4)) < 1e-4
    assert not hc["is_hydrodynamic_implosion_risk"]

    bezout = BezoutCoefficientRefresh(num_functionals=3, poly_degree=4, device="cpu")
    residues = torch.linspace(-1.0, 1.0, 4 * 3 * 4).reshape(4, 3, 4)
    mat = bezout.refresh_bezout_coefficients(residues)
    assert mat.shape == (3, 3)
    assert torch.isfinite(mat).all()


def test_calm_and_kagh_feeler_gating():
    hm = HardwareMonitor.get_instance()

    calm = CALM(dim=16, hidden_dim=32)
    hist = torch.linspace(-1.0, 1.0, 2 * 4 * 16).reshape(2, 4, 16)
    targets = torch.zeros(2, 16)
    with patch.object(
        hm,
        "probe_compute_feelers",
        return_value={"t_rfc_stall_detected": True, "memory_pressure_detected": False, "thermal_spike_detected": False},
    ) as mock_calm:
        adapted_calm = calm.adapt(hist, targets, steps=5, lr=0.05)
        assert mock_calm.called
        assert isinstance(adapted_calm, CALM)

    kagh = KAGHBlock(n_in=16, n_out=16, width=16, depth=2)
    x = torch.linspace(-1.0, 1.0, 32).reshape(2, 16)
    y = torch.linspace(-0.5, 0.5, 32).reshape(2, 16)
    with patch.object(
        hm,
        "probe_compute_feelers",
        return_value={"t_rfc_stall_detected": False, "memory_pressure_detected": True, "thermal_spike_detected": False},
    ) as mock_kagh:
        adapted_kagh = kagh.adapt_online(x, y, steps=5, lr=0.05)
        assert mock_kagh.called
        assert isinstance(adapted_kagh, KAGHBlock)


def test_orchestrator_rehybridized_subsystems_and_step():
    orch = UniversalOrchestrator(dim=32, device="cpu")
    expected_attrs = [
        "kagh_block", "admr_solver", "meta_polytope", "quantum_reasoning",
        "voynich_linguist", "modality_embedder", "gyroid_reasoner_cls",
        "bio_governor", "neuromodulatory_bus", "soliton_healer", "sic_fa_admm",
        "constraint_probe", "codes_driver", "local_qp", "continuous_coprimality",
        "voynich_exemption_token_cls", "ntm_operator", "poly_coeff_functional",
        "einstein_tessellation", "cyclotomic_tda", "triadic_reciprocity",
        "chebyshev_filtration", "huffman_codec", "euler_tour_lca", "gdpo_norm",
    ]
    missing = [a for a in expected_attrs if getattr(orch, a, None) is None]
    assert not missing, f"Missing orchestrator subsystems: {missing}"

    state_in = torch.linspace(-0.5, 0.5, 32).unsqueeze(0)
    res = orch.step(state_in)
    assert "state" in res
    assert torch.isfinite(res["state"]).all()
    assert orch.telemetry_state.get("kagh_block_active") is True
    assert orch.telemetry_state.get("admr_solver_active") is True
    assert orch.telemetry_state.get("bio_governor_reflex") is not None

