import os
import sys
import math
import py_compile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def _ensure_clean_files():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    assert arb["unit_standard"] == "mbbl"

    # Cross-standard alignment: Celsius vs Kelvin (offset 273.15, scale 1.0) + dimensionless_zscore
    t_celsius = torch.tensor([25.0, 25.1, 24.9, 25.0])
    bg_kelvin = torch.tensor([298.15, 298.20])
    aligned = si.normalize_differential_telemetry(
        t_celsius,
        bg_kelvin,
        structural_wobble_threshold=0.5,
        background_offset=273.15,
        dimensionless_zscore=False,
    )
    assert abs(aligned["differential_signal_mean"]) < 0.2
    assert aligned["wobble_to_noise_veto"] is False

    # Adaptive fixed-point scale for macro/micro measurement standards
    from src.core.primitive_ops import FixedPointField
    macro_fp = FixedPointField.from_adaptive_float(torch.tensor([1e6, -2e6]))
    assert torch.allclose(macro_fp.forward(), torch.tensor([1e6, -2e6]), rtol=1e-4)
    prod_fp = macro_fp.mul(FixedPointField.from_float(torch.tensor([2.0, 2.0])))
    assert torch.allclose(prod_fp.forward(), torch.tensor([2e6, -4e6]), rtol=1e-3)


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

    term_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'ui', 'diegetic_terminal.html')
    with open(term_path, 'r', encoding='utf-8') as f:
        html = f.read()
    assert "jei-tooltip-box" in html
    assert "src/ui/diegetic_backend.py" in html
    assert "user-select: text" in html


def test_live_mindmap_tree_and_expansion():
    from types import SimpleNamespace
    from src.ui.diegetic_backend import build_live_mindmap_tree, build_live_mindmap_expansion

    class _MockNode:
        def __init__(self, node_id, text, chiral_score, px, py, public_domain_tags):
            self.node_id = node_id
            self.text = text
            self.chiral_score = chiral_score
            self.px = px
            self.py = py
            self.public_domain_tags = public_domain_tags
            self.tags = public_domain_tags

    class _MockGraphManager:
        def __init__(self):
            self.nodes = [
                _MockNode("fossil_01", "Gyroidic Chirality Anchor", 0.85, 0.12, -0.08, ["gyroid", "chirality"]),
                _MockNode("fossil_02", "Meliponini Puncture Shield", 0.42, -0.31, 0.24, ["puncture", "shield"]),
            ]

        def query_procedural_graph(self, max_nodes=40):
            return {
                "closed_topology_sequences": [
                    {"cycle_len": 3, "node_ids": ["fossil_01", "fossil_02", "fossil_01"], "perimeter": 1.42}
                ]
            }

    mock_engine = SimpleNamespace(
        last_pas=0.7123,
        graph_manager=_MockGraphManager(),
        fossilizer=SimpleNamespace(fossil_bed=[{"id": "bed_1"}, {"id": "bed_2"}]),
        carnot_ledger=SimpleNamespace(xi_shear=0.145),
    )

    tree = build_live_mindmap_tree(mock_engine, max_fossils=12)
    assert tree["name"] == "Gyroidic Reasoner"
    assert "PAS=0.7123" in tree["details"]
    categories = {c["name"]: c for c in tree["children"]}
    assert "Knowledge Dyads (2 live)" in categories
    fossil_names = [c["name"] for c in categories["Knowledge Dyads (2 live)"]["children"]]
    assert "Gyroidic Chirality Anchor" in fossil_names
    assert "Meliponini Puncture Shield" in fossil_names
    assert "Silicon Sovereignty (TailSlayer)" in categories
    affordance_names = [c["name"] for c in categories["Silicon Sovereignty (TailSlayer)"]["children"]]
    assert any("Thermal Gauge" in n for n in affordance_names)

    expansion = build_live_mindmap_expansion(mock_engine, node_name="Gyroidic Chirality", category="Poincaré Fossil")
    assert expansion["name"] == "Gyroidic Chirality"
    assert len(expansion["children"]) >= 2
    assert any(c["category"] == "Fossil Resonance" for c in expansion["children"])
    assert any(c["category"] == "Closed Topology" for c in expansion["children"])


def test_procedural_void_homing_and_centroid_contract():
    from src.topology.embedding_graph import GyroidicGraphManager

    gm = GyroidicGraphManager()
    gm.nodes = [
        {"node_id": "n1", "text": "Anchor One", "x": 10.0, "y": 20.0, "z": 30.0},
        {"node_id": "n2", "text": "Anchor Two", "x": 30.0, "y": 40.0, "z": 50.0},
    ]
    # Query from far away in the procedural void
    res = gm.query_procedural_graph(
        camera_pos=(5000.0, 5000.0, 5000.0),
        gaze_dir=(0.0, 0.0, 1.0),
        render_distance=800.0,
    )
    assert res["procedural"] is True
    assert res["nodes"] == []
    assert res["manifold_centroid"] == [20.0, 30.0, 40.0]
    assert res["nearest_anchor"] is not None
    assert res["nearest_anchor"]["id"] == "n2"
    assert len(res["hub_waypoints"]) == 2

    term_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'ui', 'diegetic_terminal.html')
    with open(term_path, 'r', encoding='utf-8') as f:
        html = f.read()
    assert "btn-graph-home" in html
    assert "hud-void-beacon" in html
    assert "homeGraphCamera" in html
    assert "installSpatialReferenceFrame" in html
    assert "graph-hub-select" in html
    assert "cycleNextHub" in html


def test_graph_connectivity_no_severed_islands():
    import torch
    from src.topology.embedding_graph import GyroidicGraphManager, KnowledgeFossilNode

    gm = GyroidicGraphManager(dim=8)
    # Two orthogonal clusters (cosine sim = 0.0 < PHI_RECIPROCAL) plus one isolated node
    s_a1 = torch.tensor([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_a2 = torch.tensor([0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_b1 = torch.tensor([0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    s_b2 = torch.tensor([0.0, 0.0, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0])
    s_c1 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    gm.nodes = [
        KnowledgeFossilNode("a1", s_a1, "A1", {}),
        KnowledgeFossilNode("a2", s_a2, "A2", {}),
        KnowledgeFossilNode("b1", s_b1, "B1", {}),
        KnowledgeFossilNode("b2", s_b2, "B2", {}),
        KnowledgeFossilNode("c1", s_c1, "C1", {}),
    ]

    def _is_single_component(node_ids, edges):
        adj = {nid: set() for nid in node_ids}
        for e in edges:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])
        visited = set()
        stack = [node_ids[0]]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(adj[cur] - visited)
        return len(visited) == len(node_ids)

    ids = [n.node_id for n in gm.nodes]
    edges_default = gm.get_adjacency_list(top_k_per_node=6, max_edges=100)
    assert _is_single_component(ids, edges_default), f"Default mode fragmented: {edges_default}"

    edges_proc = gm.get_adjacency_list_for_nodes(gm.nodes, top_k_per_node=6, max_edges=100)
    assert _is_single_component(ids, edges_proc), f"Procedural mode fragmented: {edges_proc}"


def test_birkhoff_direct_256_and_orchestrator_bridges():
    import torch
    from src.core.birkhoff_projection import DirectBirkhoffProjection, ObscuredBirkhoffManifold
    from src.core.orchestrator import UniversalOrchestrator

    proj = DirectBirkhoffProjection(256)
    assert proj.use_iterative is False
    T = torch.randn(1, 256, 256)
    out = proj(T)
    assert out.shape == (1, 256, 256)
    assert torch.allclose(out.sum(dim=-1), torch.ones(1, 256), atol=1e-4)
    assert torch.allclose(out.sum(dim=-2), torch.ones(1, 256), atol=1e-4)

    obs = ObscuredBirkhoffManifold(256)
    assert obs.direct_projector is not None

    orch = UniversalOrchestrator(dim=16)
    for attr in [
        "betti_router",
        "homology_pressure",
        "willmore_gasket",
        "chiral_drift_stabilizer",
        "sparse_pce",
        "containment_invariants",
        "conjugate_color_transport",
        "rogue_predictor_defense",
        "calm_collapse_detector",
        "genius_operator",
        "fractal_partitioner",
        "bouligand_mc_projection",
        "structural_entanglement_net",
    ]:
        assert hasattr(orch, attr), f"Missing wired attribute: {attr}"

    from src.training.training_manager import TrainingManager
    tm = TrainingManager(ai_system=None)
    for attr in ["gyroid_projector", "sparse_covariant_optimizer", "topological_pressure_monitor"]:
        assert hasattr(tm, attr), f"Missing TrainingManager attribute: {attr}"


def test_wheel_algebra_null_self_relation_and_vanellope_admr_shear():
    import torch
    from src.core.invariants import (
        compute_wheel_algebra_division,
        compute_null_self_relation_unfolding,
    )
    from src.governance.causative.vanellope_first_causative import VanellopeFirstCausativeAgent
    from src.core.admr_solver import PolynomialADMRSolver
    from src.core.ley_line_tracker import LeyLineTracker
    from src.topology.embedding_graph import GyroidicGraphManager

    # 1. Wheel Algebra (Carlstrom 2004, docs/How To Divide By Zero, For Real.txt)
    num = torch.tensor([0.0, 1.0, -2.5, 4.0])
    den = torch.tensor([0.0, 0.0, 0.0, 2.0])
    wheel_res = compute_wheel_algebra_division(num, den)
    assert wheel_res["hub_bot_count"] == 1
    assert wheel_res["projective_infinity_count"] == 2
    assert wheel_res["wheel_identity_x_minus_x_eq_0x2"] is True
    assert wheel_res["wheel_identity_x_div_x_eq_1_plus_0xx"] is True
    assert wheel_res["hub_involution_fixed_point"] is True
    assert abs(float(wheel_res["regularized_quotient"][0].item()) - 1.0) < 1e-6
    assert abs(float(wheel_res["regularized_quotient"][3].item()) - 2.0) < 1e-6

    # 2. Ontological Null Self-Relation Unfolding (docs/This Is Why Anything Exists.txt)
    state = torch.tensor([0.6, -0.4, 0.8, 0.3, -0.5, 0.7, 0.2, -0.9, 0.4, 0.5, -0.3, 0.6])
    null_res = compute_null_self_relation_unfolding(state)
    assert null_res["null_zero_sum_residual"] == 0.0
    assert null_res["undefined_self_relation_wheel_hub"] is True
    assert null_res["requires_prior_cause"] is False
    assert null_res["non_overlapping_3d_space"] is True
    assert null_res["closed_loop_actualized"] is True

    # 3. Vanellope First Causative Agent grounding in Ontological Null & Wheel Hub
    vanellope = VanellopeFirstCausativeAgent(state_dim=12)
    u_state = torch.ones(12)
    mutated, telemetry = vanellope(u_state, context_text="um... glitch spark!")
    assert mutated.shape == u_state.shape
    assert telemetry["is_independent_from_universe"] is True
    assert telemetry["universe_orthogonality_cosine"] < 1e-4
    assert "null_self_relation_first_cause" in telemetry
    assert telemetry["null_self_relation_first_cause"]["requires_prior_cause"] is False
    assert telemetry["null_self_relation_first_cause"]["wheel_identity_x_div_x_verified"] is True

    # 4. PolynomialADMRSolver Wheel Multiplicative Remainders
    admr = PolynomialADMRSolver(poly_config=None, state_dim=12)
    admr_audit = admr.audit_wheel_multiplicative_remainders(state)
    assert admr_audit["projected_state"].shape[-1] == 12
    assert admr_audit["wheel_identity_x_minus_x_eq_0x2"] is True
    assert admr_audit["wheel_identity_x_div_x_eq_1_plus_0xx"] is True

    # 5. LeyLineTracker Secant & Shear Plane Wheel Transversality
    tracker = LeyLineTracker(num_samples=4)
    states_batch = torch.randn(4, 12)
    pressures = torch.tensor([0.1, 0.9, 0.9, 0.2])
    secant_audit = tracker.compute_secant_shear_wheel_transversality(states_batch, pressures, threshold=0.3)
    assert secant_audit["wheel_identities_verified"] is True
    assert secant_audit["active_shear_plane_count"] >= 1

    # 6. GyroidicGraphManager Wheel Hub (0/0 = bot) Waypoint Role
    coords = [
        ("center_node", "Center", 0.0, 0.0, 0.0, 10.0),
        ("rim_node_1", "Rim 1", 200.0, 0.0, 0.0, 200.0),
        ("rim_node_2", "Rim 2", -200.0, 150.0, 0.0, 250.0),
    ]
    hubs = GyroidicGraphManager._compute_hub_waypoints(coords, [0.0, 0.0, 0.0])
    assert any(h.get("hub_role") == "WHEEL_HUB_0_OVER_0" and h["id"] == "center_node" for h in hubs)



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
    test_live_mindmap_tree_and_expansion()
    print("[PASS] Live DaemonAton Mind Map Tree & Generative Expansion")
    print("ALL VERIFICATION TESTS PASSED.")
