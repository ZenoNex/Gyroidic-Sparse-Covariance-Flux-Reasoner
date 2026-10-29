import os
import sys
import math
import py_compile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _ensure_clean_files():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    modified_py = [
        'hybrid_backend.py',
        'pickleguiencodeviwer.py',
        'src/ui/diegetic_backend.py',
        'src/ui/voxelboxter_client.py',
        'src/core/invariants.py',
        'src/core/modular_virtualization.py',
        'src/core/fgrt_rnn_cells.py',
        'src/core/false_negative_subsystem.py',
        'src/core/garden_statistical_attractors.py',
        'src/core/veto_subspace.py',
        'src/data/sovereign_ingestor.py',
        'src/data/pressure_ingestor.py',
        'src/data/economic_news_linker.py',
        'src/data/universal_topology_converter.py',
        'src/topology/embedding_graph.py',
        'src/topology/gyroid_covariance.py',
        'src/governance/bio_archetypal_governor.py',
    ]
    for rel in modified_py:
        full_p = os.path.join(root, rel)
        py_compile.compile(full_p, doraise=True)
    print(f"[PASS] py_compile succeeded on all {len(modified_py)} modified Python modules.")

_ensure_clean_files()

import torch


def test_branch1_collatz_diophantine_and_quadratic_surd():
    from src.core.invariants import (
        PERCOLATION_2D_SQUARE,
        QUADRATIC_SURD_PI_BOUND,
        QUADRATIC_SURD_PI_GAP,
        quadratic_surd_polytope_sandwich,
        evaluate_percolation_regime,
    )
    from src.core.modular_virtualization import ModularVirtualizationLayer
    from src.core.fgrt_rnn_cells import LazarusSuperpositionRNNCell

    assert abs(PERCOLATION_2D_SQUARE - 0.592746) < 1e-6
    assert abs(QUADRATIC_SURD_PI_BOUND - (math.sqrt(2.0) + math.sqrt(3.0))) < 1e-9
    assert QUADRATIC_SURD_PI_GAP > 0.0

    r = torch.tensor([1.0, 2.5, 4.0])
    sandwich = quadratic_surd_polytope_sandwich(r)
    assert sandwich["inscribed_4_polytope_bound"] < sandwich["circumscribed_6_polytope_bound"]

    perc_sub = evaluate_percolation_regime(0.45)
    assert perc_sub["regime"] == "MELIPONINI_ISOLATED_CERUMEN"
    perc_sup = evaluate_percolation_regime(0.65)
    assert perc_sup["regime"] == "APIS_SPANNING_HONEYCOMB"

    mvl = ModularVirtualizationLayer(dim=16)
    x = torch.randn(2, 16) * 10.0
    collatz_audit = mvl.audit_collatz_carry_and_diophantine_near_miss(x)
    assert "odd_parity_fraction" in collatz_audit
    assert "diophantine_near_miss_2l_3k" in collatz_audit

    cell = LazarusSuperpositionRNNCell(input_dim=16, hidden_dim=16)
    dioph = cell.audit_diophantine_frequency_near_miss(step=3)
    assert "baker_logarithmic_gap" in dioph
    assert "incommensurate_margin" in dioph


def test_branch2_differential_telemetry_and_wobble_veto():
    from src.data.sovereign_ingestor import SovereignIngestor
    from src.data.pressure_ingestor import PressureIngestor
    from src.data.economic_news_linker import EconomicAgentLinker

    target = torch.tensor([10.0, 10.5, 11.0, 10.8])
    bg = torch.tensor([7.0, 7.1, 7.0, 7.2])
    si = SovereignIngestor()
    norm_res = si.normalize_differential_telemetry(target, bg, structural_wobble_threshold=0.05)
    assert "differential_signal_mean" in norm_res
    assert "wobble_to_noise_veto" in norm_res

    pi = PressureIngestor()
    assert "sentinel2_cogs" in pi.sources
    assert "eia_physical_inventory" in pi.sources
    p_res = pi.normalize_against_background_manifold(target, bg, structural_wobble_limit=0.5)
    assert "epistemic_refusal_nan" in p_res

    linker = EconomicAgentLinker()
    arb = linker.evaluate_physical_telemetry_arbitrage(
        raw_proxy_series=[-7.5, -7.4, -7.6],
        ambient_ground_series=[-7.0, -7.0, -7.1],
        weekly_physical_wobble=1.5,
        measurement_uncertainty=20.0
    )
    assert arb["wobble_veto_triggered"] is True
    assert arb["target_ontology"] == "TOPOLOGICAL_REFUSAL_NAN"


def test_branch3_prime_stride_ufsc_sieve():
    from src.data.universal_topology_converter import UniversalTopologyConverter
    from src.core.false_negative_subsystem import VoynichExemptionToken

    converter = UniversalTopologyConverter()
    row = torch.sin(torch.linspace(0, 4 * math.pi, 197))
    stream = row.repeat(6)
    sieve_res = converter.prime_stride_ufsc_sieve(stream)
    assert sieve_res["optimal_prime_stride"] == 197
    assert sieve_res["max_spatial_coherence"] > 0.9

    token = VoynichExemptionToken.issue_from_ufsc_prime_stride(sieve_res, state=row)
    assert isinstance(token, VoynichExemptionToken)


def test_branch4_percolation_and_isolated_node_reconnection():
    from src.core.garden_statistical_attractors import MemoryPalace
    from src.topology.embedding_graph import GyroidicGraphManager, KnowledgeFossilNode
    from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe

    palace = MemoryPalace(feature_dim=16)
    for i in range(4):
        vec = torch.zeros(16)
        vec[i] = 1.0
        palace.plant_room(f"room_{i}", vec)
    recon = palace.reconnect_isolated_substrate_rooms(min_affinity=0.99)
    assert recon["reconnected_count"] >= 1
    assert abs(recon["percolation_threshold_pc"] - 0.592746) < 1e-6

    gm = GyroidicGraphManager(dim=16)
    for i in range(5):
        vec = torch.zeros(16)
        vec[i % 16] = 1.0 if i < 4 else -10.0
        gm.nodes.append(KnowledgeFossilNode(f"node_{i}", vec, f"text_{i}", {"chiral_score": 0.5}))
    edges = gm.get_adjacency_list()
    bridge_edges = [e for e in edges if e.get("type") == "PERCOLATION_BRIDGE"]
    assert len(bridge_edges) >= 1

    probe = SparseGyroidCovarianceProbe(hidden_dim=16, window_size=4)
    sig = probe.compute_spectral_signature(torch.eye(16))
    assert "percolation_threshold_pc" in sig


def test_branch5_threat_recalibration_and_fixed_reflexes():
    from src.governance.bio_archetypal_governor import BioArchetypalGovernor
    from src.core.veto_subspace import VetoSubspace

    gov = BioArchetypalGovernor(state_dim=16)
    recal = gov.evaluate_threat_recalibration_and_fixed_reflexes(
        structural_entropy=1.2,
        external_pressure=0.4,
        surrounding_pas_h=0.75,
        valence_hunger=0.3,
        jax_rigidity=1.05
    )
    assert recal["is_high_entropy_inconvenience"] is True
    assert recal["is_structural_danger"] is False
    assert len(recal["reflex_scores"]) == 8

    veto = VetoSubspace()
    v_recal = veto.recalibrate_threat_vs_inconvenience(
        instability_severity=0.6,
        pas_h=0.8,
        chiral_score=0.7,
        wobble_sigma=0.01,
        wobble_threshold=0.05
    )
    assert v_recal["status"] == "modulated"
    assert v_recal["is_high_entropy_inconvenience"] is True


def test_hybrid_backend_process_text_and_ui_contracts():
    os.environ["Sovereign_Disable_Background"] = "1"
    from hybrid_backend import HybridAI

    ai = HybridAI(lightweight=True)
    result = ai.process_text("Test Voynich RNS CNN Resonance Cavity and distortion fix")
    assert "error" not in result, f"Unexpected failure: {result}"
    assert "response" in result
    assert "metrics" in result
    assert "voynich_honesty" in result["metrics"]
    assert "rns_collatz_audit" in result["metrics"]
    assert "moment_transport_energy" in result["metrics"]
    assert "thermal_guard" in result["metrics"]

    term_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'ui', 'diegetic_terminal.html')
    with open(term_path, 'r', encoding='utf-8') as f:
        html = f.read()
    assert "jei-tooltip-box" in html
    assert "src/ui/diegetic_backend.py" in html
    assert "user-select: text" in html


if __name__ == "__main__":
    test_branch1_collatz_diophantine_and_quadratic_surd()
    print("[PASS] Branch 1: Collatz Carry-Cascade, Diophantine Near-Miss & Quadratic Surd Bounds")
    test_branch2_differential_telemetry_and_wobble_veto()
    print("[PASS] Branch 2: Differential Telemetry Calibration & Wobble-to-Noise Epistemic Veto")
    test_branch3_prime_stride_ufsc_sieve()
    print("[PASS] Branch 3: Prime-Stride UFSC Sieve (p=197) & VoynichExemptionToken")
    test_branch4_percolation_and_isolated_node_reconnection()
    print("[PASS] Branch 4: Site Percolation Criticality (p_c=0.592746) & Isolated Substrate Reconnection")
    test_branch5_threat_recalibration_and_fixed_reflexes()
    print("[PASS] Branch 5: Threat Recalibration & 8 Fixed Survival Reflexes")
    test_hybrid_backend_process_text_and_ui_contracts()
    print("[PASS] HybridAI.process_text (No distortion error + Voynich/RNS/ICNN/Cavity/Audience) & UI Contracts")
    print("ALL VERIFICATION TESTS PASSED.")
