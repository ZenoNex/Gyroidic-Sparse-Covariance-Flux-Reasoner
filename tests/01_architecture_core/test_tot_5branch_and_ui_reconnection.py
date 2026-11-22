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

    # 7. Gaussian Mollifier Projection (Dirac Effect, MATHEMATICAL_DETAILS.md 45.2, SYSTEM_ARCHITECTURE.md 9.5)
    from src.core.invariants import DIRAC_SPECTRUM_BETA_COH, compute_gaussian_mollifier_projection
    from src.core.video_dyad_parser import VideoDyadParser
    from src.core.gluing_operator import GluingOperator
    from src.core.meta_polytope_matrioshka import MetaPolytopeMatrioshka, BoundaryState
    from src.core.knowledge_dyad_fossilizer import ResidueFusion

    spike_signal = torch.tensor([[0.1, -0.1, 0.05, 8.5, -0.05, 0.1, 0.0, -7.2]])
    moll_res = compute_gaussian_mollifier_projection(spike_signal, scale=1.5)
    assert moll_res["projected_state"].shape == spike_signal.shape
    assert moll_res["unit_mass_preserved"] is True
    assert abs(moll_res["beta_coh"] - DIRAC_SPECTRUM_BETA_COH) < 1e-6
    assert moll_res["spike_count"] >= 2

    vparser = VideoDyadParser(chunk_size=16, max_chunks=4, device="cpu")
    rot_sig = vparser._apply_topological_rotation(spike_signal.repeat(4, 1), scale=1.5)
    assert rot_sig.shape == (4, 8)
    assert vparser.last_mollifier_diagnostics.get("unit_mass_preserved") is True

    gluer = GluingOperator(dim=8)
    glued_out = gluer(spike_signal)
    assert glued_out.shape == spike_signal.shape
    assert gluer.last_mollifier_diagnostics.get("unit_mass_preserved") is True

    matrioshka = MetaPolytopeMatrioshka(max_depth=3, base_dim=8)
    b_state = BoundaryState.from_crossing(
        state_direction=torch.ones(8),
        facet_normal=torch.ones(8),
        alpha=0,
        level=-1,
        max_level=3,
    )
    proj_dir = matrioshka.project_direction(spike_signal, spike_signal.clone(), b_state)
    assert proj_dir.shape == spike_signal.shape

    rfusion = ResidueFusion(feature_dim=16)
    fused_res = rfusion(torch.randn(1, 96), torch.randn(1, 16))
    assert fused_res.shape == (1, 16)
    assert rfusion.last_mollifier_diagnostics.get("unit_mass_preserved") is True

    # 8. Hardware Storage Profile, SQLite WAL Fossil Index, Birkhoff n>32 Backward, & Zeitgeist Pad Order
    from src.core.hardware_monitor import HardwareMonitor
    from src.core.birkhoff_projection import BouligandBirkhoffManifold
    from src.core.zeitgeist_router import ZeitgeistRouter

    hm = HardwareMonitor.get_instance()
    prof = hm.detect_storage_and_hardware_profile()
    assert "profile_name" in prof and prof["io_workers"] >= 1
    assert prof["sqlite_mmap_bytes"] >= 0

    b_manifold = BouligandBirkhoffManifold(n=36)
    t_in = torch.randn(1, 36, 36, requires_grad=True)
    t_out = b_manifold(t_in)
    t_out.sum().backward()
    assert t_in.grad is not None and t_in.grad.shape == (1, 36, 36)

    zr = ZeitgeistRouter(dim=16)
    z_mode, z_state, z_diag, z_out = zr(torch.randn(1, 10))
    assert z_out.shape[-1] == 16

    # 9. CUDA + PyOpenCL Sovereignty Acceleration & Disconnected Class Wiring
    from src.core.device_utils import get_compute_acceleration_summary
    from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
    from src.core.noncommutativity_curvature import NonCommutativityCurvature
    from src.core.polychoron_quantization import Polychoron600Quantizer
    from src.core.orchestrator import UniversalOrchestrator

    accel = get_compute_acceleration_summary()
    assert "cuda_available" in accel and "opencl_platforms" in accel

    se = SiliconSovereigntyEngine()
    A_m = torch.randn(16, 16)
    B_m = torch.randn(16, 16)
    comm, kappa = se.compute_noncommutativity_commutator(A_m, B_m)
    assert comm.shape == (16, 16) and kappa.shape == (16, 16)
    assert torch.allclose(kappa, -kappa.T, atol=1e-5)

    nc = NonCommutativityCurvature(dim=16)
    nc_res = nc.compute_curvature(A_m, B_m)
    assert nc_res["kappa"].shape == (16, 16)

    q600 = Polychoron600Quantizer()
    q_out = q600(torch.randn(20, 4))
    assert q_out.shape == (20, 4)

    orch = UniversalOrchestrator(dim=16, device="cpu")
    assert len(orch.fgrt_rnn_bank) == 7
    assert orch.fractional_tag_stacker is not None
    assert orch.defect_attractor is not None

    # 10. Doc-Aligned Remaining Classes & PyOpenCL Hooks
    from src.core.archetype_engines import ArchetypalSynthesisEngine, DeformationFirewallOperator
    from src.core.leontief_governor import LeontiefGovernor
    from src.optimization.muon_hybrid import zeropower_via_newtonschulz5
    from src.core.jspace_pca_mapper import JSpacePCAMapper
    from src.core.sparse_higher_order_tensors import SparseHigherOrderTensorDynamics
    from src.core.non_ergodic_entropy import AdaptiveFractalPartitioner

    ase = ArchetypalSynthesisEngine(state_dim=16)
    assert isinstance(ase.deformation_firewall, DeformationFirewallOperator)

    lg = LeontiefGovernor(state_dim=12, neumann_terms=6)
    A_cons = torch.eye(12) * 0.25
    L_neumann = lg._neumann_series(A_cons, torch.eye(12))
    assert L_neumann.shape == (12, 12) and torch.isfinite(L_neumann).all()

    ns_out = zeropower_via_newtonschulz5(torch.randn(12, 16), steps=3)
    assert ns_out.shape == (12, 16) and torch.isfinite(ns_out).all()

    jmap = JSpacePCAMapper(n_components=4)
    j_proj = jmap.project_hyper_ring_invariants(torch.randn(4, 16), target_dim=3)
    assert j_proj.shape == (4, 3)

    shot = SparseHigherOrderTensorDynamics(max_order=3, num_shells=3, base_dim=16)
    with torch.no_grad():
        shot_out = shot(torch.randn(4, 16))
    assert set(shot_out.keys()) == {1, 2, 3} and shot_out[3].shape == (4, 16)

    afp = AdaptiveFractalPartitioner(min_block=2, max_block=4)
    blocks = afp.partition(torch.randn(6, 12))
    assert len(blocks) >= 1 and blocks[0][0] == 0 and blocks[-1][1] == 12


def test_codec_crt_bridge_and_predictive_arithmetic_compression():
    from src.core.huffman_entropy import HuffmanEntropyCodec
    from src.codec.gyroidic_codec import GyroidicCodec, CodecConfig
    from src.core.orchestrator import UniversalOrchestrator

    # 1. Context-Adaptive Predictive Arithmetic Coding (3Blue1Brown Compression-is-Intelligence)
    skewed_stream = [0] * 40 + [1] * 8 + [2] * 4 + [0] * 30 + [3] * 2
    packed, meta = HuffmanEntropyCodec.encode_predictive_arithmetic(
        skewed_stream, alphabet=[0, 1, 2, 3], online_adapt=True
    )
    decoded = HuffmanEntropyCodec.decode_predictive_arithmetic(packed, meta)
    assert decoded == skewed_stream
    assert meta["online_adaptation_gain_bits"] > 0.0
    assert meta["bits_per_symbol"] < 2.0  # Strictly beats uniform 2 bits/symbol

    # 2. GyroidicCodec + residue stream compression
    codec = GyroidicCodec(CodecConfig(K=4, n=4, gyroid_resolution=8, device="cpu"))
    enc = codec.encode("gyroidic predictive compression test", image=torch.randn(16))
    comp = codec.compress_residue_stream(enc, quant_levels=16, use_predictive_arithmetic=True)
    assert comp["roundtrip_exact"] is True
    assert comp["huffman_bits"] > 0 and comp["arithmetic_bits"] > 0

    # 3. UniversalOrchestrator CodecCRTBridge + Conformal Log-Polar + ImageExtension reconnection
    orch = UniversalOrchestrator(dim=16, device="cpu")
    assert orch.codec_crt_bridge is not None
    bridge_report = orch.audit_codec_and_compression_bridge(torch.randn(16), text="orchestrator_bridge")
    assert bridge_report.get("codec_crt_bridge_active") is True
    assert bridge_report.get("conformal_log_polar_norm", 0.0) >= 0.0
    assert bridge_report.get("structural_entanglement_net_norm", 0.0) >= 0.0
    assert bridge_report.get("conjugate_color_transport_active") is True
    assert bridge_report["predictive_compression"]["roundtrip_exact"] is True


def test_tutte_matching_bft_non_dual_economy_and_dmet_qsci():
    from src.core.admr_solver import PolynomialADMRSolver
    from src.core.polynomial_coprime import PolynomialCoprimeConfig
    from src.core.non_dual_coin import CerumenPotWallet, ChernSimonsValidator, EconomicAbortException
    from src.core.orchestrator import UniversalOrchestrator

    # 1. Tutte's Theorem odd-component matching audit (Dead Logic rejection)
    poly_cfg = PolynomialCoprimeConfig(k=4, degree=4, device="cpu")
    solver = PolynomialADMRSolver(poly_config=poly_cfg, state_dim=8)

    # K_4 (complete graph on 4 vertices) has a perfect matching -> tutte_satisfied=True
    k4 = torch.ones(4, 4) - torch.eye(4)
    res_k4 = solver.audit_tutte_matching(k4, threshold=0.5)
    assert res_k4["tutte_satisfied"] is True
    assert res_k4["dead_logic_detected"] is False

    # Star graph K_{1,3}: removing center vertex 0 leaves 3 odd components > |U|=1 -> Dead Logic
    star = torch.zeros(4, 4)
    star[0, 1:] = 1.0
    star[1:, 0] = 1.0
    res_star = solver.audit_tutte_matching(star, threshold=0.5)
    assert res_star["tutte_satisfied"] is False
    assert res_star["dead_logic_detected"] is True
    assert res_star["max_tutte_deficit"] >= 2

    # 2. Non-Dual Coin: CerumenPotWallet slogdet volume + Mohr-Coulomb EconomicAbortException
    w1 = CerumenPotWallet(dim=8, device="cpu")
    w2 = CerumenPotWallet(dim=8, device="cpu")
    assert w1.get_volume() > 0.0
    with torch.no_grad():
        w2.state.mul_(50.0)  # Extreme parasitic extraction shear
    validator = ChernSimonsValidator(yield_criteria=2.5)
    aborted = False
    try:
        validator.validate_fusion(w1, w2)
    except EconomicAbortException:
        aborted = True
    assert aborted is True

    # 3. UniversalOrchestrator BFT Quorum, Non-Dual Economy, Bezout Supremacy, and QSCI
    orch = UniversalOrchestrator(dim=16, device="cpu")
    assert orch.quantum_reasoning is not None
    cycle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.2]]
    econ = orch.audit_bft_and_non_dual_economy(torch.randn(16), cycle_coords=cycle, num_peers=4, max_byzantine_faults=1)
    assert econ["bft_quorum_satisfied"] is True
    assert econ["sys1_wallet_slogdet_volume"] > 0.0
    assert econ["tripsodic_global_volume"] >= 1000.0
    assert econ.get("carnot_mobius_active") is True
    assert "tutte_matching" in econ
    assert "bezout_condition_number" in econ
    if orch.bonfire_ring is not None:
        assert econ.get("non_euler_shamir_closed") is True


def test_voxelboxter_worldgen_presets_boot_and_mindmap_ui_contracts():
    from src.ui.voxelboxter_simulation import (
        WORLDGEN_PRESETS,
        list_worldgen_presets,
        WorldGenConfig,
        VoxelboxterEngine,
    )
    from src.ui.voxelboxter_client import PatchStateResource, BootStage

    # 1. Selectable World Generation System: all 5 presets + deterministic seeded generation
    presets = list_worldgen_presets()
    expected_presets = {
        "gyroidic_minimal_surface",
        "mangosteen_fl0w_life",
        "non_dual_cerumen_ecology",
        "dmet_qsci_rbm_lattice",
        "conformal_escher_logpolar",
    }
    assert expected_presets.issubset(set(WORLDGEN_PRESETS.keys()))
    assert len(presets) >= 5

    eng = VoxelboxterEngine(device="cpu", lightweight=True)
    for preset_id in sorted(expected_presets):
        cfg = WorldGenConfig.from_dict({"preset": preset_id, "seed": 4242, "world_radius": 4})
        summary = eng.generate_world(cfg)
        assert summary.get("preset") == preset_id
        assert summary.get("seed") == 4242
        assert summary.get("block_count", 0) > 0

    # Deterministic repeatability check on same preset & seed
    s1 = eng.generate_world(WorldGenConfig.from_dict({"preset": "mangosteen_fl0w_life", "seed": 1337, "world_radius": 4}))
    s2 = eng.generate_world(WorldGenConfig.from_dict({"preset": "mangosteen_fl0w_life", "seed": 1337, "world_radius": 4}))
    assert s1.get("block_count") == s2.get("block_count")

    # 2. Race-free Voxelboxter boot_sequence (offline/lightweight)
    patch_res = PatchStateResource(lightweight=True)
    assert patch_res.boot_stage == BootStage.UNINITIALIZED
    boot_report = patch_res.boot_sequence(
        worldgen_config={"preset": "gyroidic_minimal_surface", "seed": 2026, "world_radius": 4},
        probe_backend=False,
    )
    assert boot_report["status"] == "ok"
    assert boot_report["engine_ready"] is True
    assert patch_res.boot_stage in (BootStage.WORLDGEN_COMPLETE, BootStage.RUNNING)
    assert patch_res.mesh_dirty is True

    # 3. Mind Map overlay & Voxelboxter preset selector HTML contracts
    term_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'ui', 'diegetic_terminal.html')
    with open(term_path, 'r', encoding='utf-8') as f:
        html = f.read()
    assert "btn-mindmap-toggle" in html
    assert "toggleMindMapOverlay" in html
    assert "mindmap-overlay" in html
    assert "gyroidic_minimal_surface" in html
    assert "mangosteen_fl0w_life" in html


def test_domain_transversality_on3_amelioration_and_survival_systems(tmp_path):
    """
    Verifies:
      1. O(N^3) ameliorations (VideoDyadParser 128x128 svd_lowrank, AlexanderWildEmbeddingOperator,
         PersistentHomologyComputer Union-Find/Euler fallback, JSpacePCAMapper lowrank SVD).
      2. UniversalTopologyConverter <-> VideoDyadParser reconnection on video bitstreams.
      3. UniversalOrchestrator.audit_domain_transversality_and_survival (RP^4 transversality +
         FossilizedSurvivalLattice on pas_h < 0.2 + TopologicalErosionFBM + SolitonHealer).
    """
    os.environ["Sovereign_Disable_Background"] = "1"
    import torch
    from src.core.video_dyad_parser import VideoDyadParser
    from src.data.universal_topology_converter import UniversalTopologyConverter
    from src.topology.persistence_obstruction import PersistentHomologyComputer
    from src.core.jspace_pca_mapper import JSpacePCAMapper
    from src.core.orchestrator import UniversalOrchestrator

    # 1. VideoDyadParser O(N^3) amelioration on 128x128 sparse_cov (> 64 triggers svd_lowrank)
    vdp = VideoDyadParser(chunk_size=128, max_chunks=8, device="cpu")
    A = torch.randn(128, 16)
    v_metrics = {
        "sparse_covariance": A @ A.T + torch.eye(128) * 0.01,
        "fractal_entropy": torch.tensor(0.7),
        "substream_residue": torch.zeros(6),
        "honest_jitter": torch.tensor(0.1),
        "signal_length": torch.tensor(1024.0),
        "substream_entropy": torch.tensor(0.5),
    }
    sig96 = vdp.extract_96_spectral_signature(v_metrics)
    assert sig96.shape[-1] == 96 and torch.isfinite(sig96).all()

    # 2. UniversalTopologyConverter <-> VideoDyadParser on .mp4 bitstream
    fake_mp4 = tmp_path / "sample_dyad.mp4"
    fake_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42moovmdatavc1mp4a" + bytes(range(256)) * 4)
    utc = UniversalTopologyConverter(target_dim=64)
    utc_res = utc.process_artifact(fake_mp4)
    assert utc_res["format"] == ".mp4"
    assert "video_dyad_atoms" in utc_res and utc_res["spectral_tensor"].shape == (1, 64)

    # 3. PersistentHomologyComputer O(N^3) Union-Find/Euler fallback for > 64 vertices
    ph = PersistentHomologyComputer(max_dimension=1)
    cpx_large = {0: [(i,) for i in range(70)], 1: [(i, i + 1) for i in range(68)], 2: []}
    betti = ph.compute_betti_numbers(cpx_large)
    assert betti[0] == 2 and betti[1] == 0

    # 4. JSpacePCAMapper O(N^3) lowrank SVD path for > 64 dims
    jmap = JSpacePCAMapper(n_components=4)
    U = jmap.fit_transform(torch.randn(70, 70), torch.randn(70, 70))
    assert U.shape[0] == 70 and torch.isfinite(U).all()

    # 5. UniversalOrchestrator Domain Transversality & Topological Survival expansion
    orch = UniversalOrchestrator(dim=16, device="cpu")
    surv_rep = orch.audit_domain_transversality_and_survival(
        torch.randn(16), pas_h=0.1, artifact_path=str(fake_mp4)
    )
    assert surv_rep["direct_embedding_bypassed"] is True
    assert surv_rep["transversal_domains_verified"] is True
    assert surv_rep["survival_lattice_triggered"] is True
    assert surv_rep["utc_has_video_dyad_atoms"] is True
    assert "affordance_gauge_transform" in surv_rep


def test_compute_affordance_budget_and_cross_class_gating():
    """Verify typed Compute Affordance scaling and thermal-runaway degradation across classes."""
    import torch
    from src.core.hardware_monitor import (
        HardwareMonitor,
        ComputeAffordanceType,
        TypedComputeAffordance,
        get_affordance_budget,
        should_offload_gpu_sovereign,
    )
    from src.core.leontief_governor import LeontiefGovernor
    from src.core.hessian_8x8_curvature_ledger import Hessian8x8CurvatureLedger
    from src.topology.persistence_obstruction import ResidueFiltration

    steps, aff = get_affordance_budget(ComputeAffordanceType.ALGORITHMIC, base_steps=12, min_steps=2)
    assert isinstance(aff, TypedComputeAffordance)
    assert aff.affordance_type == ComputeAffordanceType.ALGORITHMIC
    assert 2 <= steps <= 12

    gpu_ok, gpu_aff = should_offload_gpu_sovereign(dim=16, min_dim=8)
    assert isinstance(gpu_aff, TypedComputeAffordance)
    assert gpu_aff.affordance_type == ComputeAffordanceType.GPU_SOVEREIGN
    assert isinstance(gpu_ok, bool)

    # Verify thermal_runaway degrades step budget to min_steps and disables GPU offload cleanly
    hm = HardwareMonitor.get_instance()
    prev_runaway = hm.thermal_runaway
    try:
        hm.thermal_runaway = True
        throttled_steps, _ = get_affordance_budget(ComputeAffordanceType.ALGORITHMIC, base_steps=12, min_steps=2)
        assert throttled_steps == 2
        throttled_gpu, _ = should_offload_gpu_sovereign(dim=32, min_dim=8)
        assert throttled_gpu is False

        # Classes continue to run cleanly under thermal throttling (no crash, bounded work)
        lg = LeontiefGovernor(state_dim=12, neumann_terms=10)
        L_out, rho, _ = lg.compute_leontief_inverse(torch.eye(12).unsqueeze(0) * 0.25)
        assert L_out.shape == (12, 12) and torch.isfinite(L_out).all()

        hl = Hessian8x8CurvatureLedger()
        h_res = hl.compute_boundary_curvature_ledger(torch.randn(16))
        assert h_res.hessian_8x8.shape == (8, 8)

        pts = torch.randn(8, 4)
        rf = ResidueFiltration(pts, pts)
        cpx = rf.build_simplicial_complex(pts, max_dimension=2)
        assert 0 in cpx and 1 in cpx
    finally:
        hm.thermal_runaway = prev_runaway


def test_spliced_prequential_mdl_optimization_and_models():
    """Verify spliced prequential MDL / predictive arithmetic coupling across optimization and models."""
    import torch
    from src.optimization.ricci_flow_optimizer import RicciFlowOptimizer
    from src.optimization.muon_hybrid import MuonRicciHybridOptimizer
    from src.models.resonance_cavity import HeritableTrustVault
    from src.models.modular_embeddings import CRTModularEmbeddingTable
    from src.models.polynomial_embeddings import PolynomialFunctionalEmbedder
    from src.core.polynomial_coprime import PolynomialCoprimeConfig

    # 1. Optimization: RicciFlowOptimizer & MuonRicciHybridOptimizer prequential MDL step governance
    w_ricci = torch.nn.Parameter(torch.randn(8, 8))
    opt_ricci = RicciFlowOptimizer([w_ricci], lr=1e-2, enable_mdl_gate=True, mdl_beta=1.5)
    loss_r = (w_ricci ** 2).sum()
    loss_r.backward()
    opt_ricci.step()
    assert 0.25 <= opt_ricci.last_mdl_metrics["step_scale"] <= 1.5
    assert opt_ricci.last_mdl_metrics["online_codelength_bits"] > 0.0
    assert opt_ricci.last_mdl_metrics["surplus_description_length_bits"] >= 0.0

    w_muon = torch.nn.Parameter(torch.randn(8, 8))
    opt_muon = MuonRicciHybridOptimizer([w_muon], lr=1e-2, enable_mdl_gate=True, mdl_beta=1.0)
    loss_m = (w_muon ** 2).sum()
    loss_m.backward()
    opt_muon.step()
    assert 0.25 <= opt_muon.last_mdl_metrics["step_scale"] <= 1.5
    assert opt_muon.last_mdl_metrics["entropic_temperature_eps"] == 1.0

    # 2. ResonanceCavity HeritableTrustVault prequential MDL survivorship weighting
    vault = HeritableTrustVault(table_size=64, k_dim=8)
    # Row 0 is highly structured/compressible; Row 1 is high-entropy uniform-like
    structured = torch.zeros(1, 8)
    structured[0, 0] = 1.0
    noisy = torch.linspace(-1.0, 1.0, 8).unsqueeze(0)
    residues = torch.cat([structured, noisy], dim=0)
    mdl_surv = vault.compute_prequential_mdl_survivorship(residues)
    assert mdl_surv["mdl_survivorship"].shape == (2,)
    assert mdl_surv["mdl_survivorship"][0].item() > mdl_surv["mdl_survivorship"][1].item()
    vault.update(residues, torch.tensor([1.0, 1.0]), use_mdl_weighting=True)
    assert hasattr(vault, "last_mdl_report") and vault.last_mdl_report["mean_online_codelength_bits"] > 0.0

    # 3. Models: CRTModularEmbeddingTable multi-channel MDL & PolynomialFunctionalEmbedder Hutter MDL
    crt_table = CRTModularEmbeddingTable(vocab_size=128, embed_dim=32, num_primes=3)
    skewed_tokens = torch.tensor([[0, 0, 0, 0, 3, 0, 0, 6, 0, 0, 0, 9]], dtype=torch.long)
    crt_mdl = crt_table.compute_crt_channel_mdl(skewed_tokens)
    assert len(crt_mdl["channel_reports"]) == 3
    assert crt_mdl["crt_data_codelength_bits"] > 0.0
    assert crt_mdl["hutter_crt_total_mdl_bits"] > crt_mdl["crt_decoder_param_bits"]

    poly_cfg = PolynomialCoprimeConfig(k=3, degree=3)
    embedder = PolynomialFunctionalEmbedder(
        text_dim=16, graph_dim=8, num_dim=4, hidden_dim=16,
        poly_config=poly_cfg, use_text=True, use_graph=False, use_num=False,
    )
    out = embedder(text_emb=torch.randn(2, 16))
    poly_mdl = embedder.compute_hutter_prequential_mdl(out["residue_distributions"])
    assert poly_mdl["data_codelength_bits"] > 0.0
    assert poly_mdl["hutter_total_mdl_bits"] == poly_mdl["decoder_param_bits"] + poly_mdl["data_codelength_bits"]
    assert poly_mdl["surplus_description_length_bits"] >= 0.0


def test_train_mimicry_connections_and_diegetic_how_are_you_response():
    os.environ["Sovereign_Disable_Background"] = "1"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # 1. Verify documentation cross-links for train mimicry
    doc_checks = [
        os.path.join(root, 'docs', 'stepping_stairs_tier_0_vestibule', '01_architecture_core', 'MODULE_INVENTORY.md'),
        os.path.join(root, 'docs', 'stepping_stairs_tier_0_vestibule', '01_architecture_core', 'SYSTEM_ARCHITECTURE.md'),
        os.path.join(root, 'docs', 'stepping_stairs_tier_0_vestibule', 'stepping_stairs_tier_1_mezzanine', '04_components_and_mechanics', 'DIEGETIC_INTEGRATION.md'),
        os.path.join(root, 'docs', 'stepping_stairs_tier_0_vestibule', 'stepping_stairs_tier_1_mezzanine', 'stepping_stairs_tier_2_crypt', '05_implementation_and_guides', 'how to make a voynich talk human-walkthrough.md'),
    ]
    for doc_path in doc_checks:
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_txt = f.read()
        assert "_train_mimicry" in doc_txt, f"Missing _train_mimicry reference in {doc_path}"
        assert "_train_mimicry_step" in doc_txt, f"Missing _train_mimicry_step reference in {doc_path}"

    from hybrid_backend import HybridAI
    ai = HybridAI(lightweight=True)
    assert ai.engine is not None, "DiegeticPhysicsEngine must be attached to HybridAI"

    # 2. Verify _train_mimicry_step canonical connections (Voynich seam cuts, TrustVault, Hebbian, JEPA, PROVE DrGRPO, MDL)
    step_loss = ai.engine._train_mimicry_step("Resonance flows across the gyroid manifold with steady coherence.")
    assert step_loss is not None and math.isfinite(step_loss)
    step_tel = getattr(ai.engine, 'last_mimicry_telemetry', {})
    assert step_tel.get('method') == '_train_mimicry_step'
    assert step_tel.get('seam_cuts_count', 0) > 0
    assert step_tel.get('hebbian_updated') is True
    assert step_tel.get('jepa_energy') is not None
    assert step_tel.get('prove_executed') is True, "PROVE DrGRPO must execute without curr_state NameError"
    assert 0.15 <= step_tel.get('trust_damp', 0.0) <= 1.0
    assert step_tel.get('mdl_codelength_bits') is not None

    # 3. Verify inline _train_mimicry canonical connections
    seed_vec = ai.engine._text_to_tensor("Manifold harmonics guide legible vocalization.").to(ai.engine.device)
    ai.engine._train_mimicry(seed_vec, "Manifold harmonics guide legible vocalization.")
    inline_tel = getattr(ai.engine, 'last_mimicry_telemetry', {})
    assert inline_tel.get('method') == '_train_mimicry'
    assert inline_tel.get('seam_cuts_count', 0) > 0
    assert inline_tel.get('hebbian_updated') is True
    assert inline_tel.get('jepa_energy') is not None
    assert 0.15 <= inline_tel.get('trust_damp', 0.0) <= 1.0
    assert inline_tel.get('mdl_codelength_bits') is not None

    # 4. Verify "how are you doing?" flows cleanly without math/Greek symbol soup or pathological garble
    res = ai.process_text("PROMPT: how are you doing?")
    assert "error" not in res, f"Unexpected error: {res}"
    raw_resp = str(res.get("response", "")).strip()
    assert len(raw_resp) > 0, "Empty response for 'how are you doing?'"
    assert all(ord(c) < 128 for c in raw_resp), f"Unprompted non-ASCII math/Greek glyphs in response: {raw_resp!r}"
    # Check larynx-decoded trace cleanly whether routed through [CONVERGED RESPONSE] or [LAZARUS_DREAM]
    import re
    dream_match = re.search(r"audience filter:\s*'([^']+)'", raw_resp)
    if dream_match:
        larynx_body = dream_match.group(1).strip()
    else:
        larynx_body = raw_resp.split("\n", 1)[1].strip() if "\n" in raw_resp and raw_resp.startswith("[") else raw_resp
    assert len(larynx_body) >= 12, f"Larynx body too short: {larynx_body!r}"
    assert set("{}[]\\|^~$#@&*+=<>").isdisjoint(larynx_body), f"Unprompted code/symbol soup in larynx body: {larynx_body!r}"
    alpha_ratio = sum(1 for c in larynx_body if c.isalpha()) / max(1, len(larynx_body))
    assert alpha_ratio >= 0.70, f"Low alphabetic density ({alpha_ratio:.2f}) in larynx body: {larynx_body!r}"
    is_garble, garble_reason = ai.engine.voynich_linguist._is_pathological_garble(larynx_body)
    assert is_garble is False, f"Pathological garble detected ({garble_reason}): {larynx_body!r}"

    # 5. Also verify direct _generate_converged_response path for "how are you doing?"
    conv_raw = ai.engine._generate_converged_response(
        text_input="PROMPT: how are you doing?",
        seed_state=ai.engine._text_to_tensor("how are you doing?").to(ai.engine.device),
        fingerprint=None,
        affordance_gradients={"executability_pressure": 0.05, "conversational_embedding_pressure": 0.4},
    )
    conv_body = conv_raw.split("\n", 1)[1].strip() if "\n" in conv_raw else conv_raw.strip()
    assert len(conv_body) >= 12, f"Converged body too short: {conv_body!r}"
    assert all(ord(c) < 128 for c in conv_body), f"Non-ASCII glyphs in converged body: {conv_body!r}"
    assert set("{}[]\\|^~$#@&*+=<>").isdisjoint(conv_body), f"Symbol soup in converged body: {conv_body!r}"
    conv_alpha = sum(1 for c in conv_body if c.isalpha()) / max(1, len(conv_body))
    assert conv_alpha >= 0.70, f"Low alphabetic density ({conv_alpha:.2f}) in converged body: {conv_body!r}"
    conv_garble, conv_reason = ai.engine.voynich_linguist._is_pathological_garble(conv_body)
    assert conv_garble is False, f"Pathological garble in converged body ({conv_reason}): {conv_body!r}"


def test_silicon_die_patent_filter_and_structural_blueprint_connections(tmp_path):
    """
    Verifies Silicon Die Image Ingestion with Automated Expired-Patent / Free-Content Filtering:
      (a) Active patent / NC-ND / ARR / unverified modern die veto vs expired patent
          (pre-1995 17y grant, post-1995 20y filing, maintenance-fee lapse, SCPA 10y+EOY mask work)
          and CC-BY / CC0 / TinyTapeout OSHW allow.
      (b) Structural blueprint generation (conformal log-polar + CRT residues + sparse covariance
          + AddonRoutine / SiliconDieBlueprintLayer).
      (c) All 15+ (18) wired connection sites across src/ and hybrid_backend.py.
    """
    os.environ["Sovereign_Disable_Background"] = "1"
    from datetime import date
    import json
    from PIL import Image
    from src.data.silicon_die_ingestor import (
        SILICON_DIE_REPOSITORIES,
        SiliconDiePatentFilter,
        SiliconDieImageIngestor,
        build_siliconprawn_urls,
        build_wikimedia_commons_api_url,
        build_patent_lookup_endpoints,
    )
    from src.core.agent_substrate_bridge import AgentSubstrateBridge
    from src.core.structural_blueprints import AddonRoutine
    from src.data.universal_topology_converter import UniversalTopologyConverter
    from src.data.local_dataset_ingestor import LocalDatasetIngestor
    from src.core.knowledge_dyad_fossilizer import KnowledgeDyadFossilizer
    from src.topology.embedding_graph import GyroidicGraphManager
    from src.ui.voxelboxter_simulation import VoxelboxterEngine
    from src.core.diegetic_physics_engine import DiegeticPhysicsEngine as CoreDiegeticPhysicsEngine
    from src.core.orchestrator import UniversalOrchestrator

    today = date(2026, 11, 22)
    pf = SiliconDiePatentFilter()

    # 1. Verify open repository catalog & URL/tile/API builders
    for key in ("siliconprawn", "zeptobars", "wikimedia_commons", "tinytapeout", "righto", "visual6502", "silicon_zoo", "chipdb"):
        assert key in SILICON_DIE_REPOSITORIES
    sp_urls = build_siliconprawn_urls("intel", "4004", layer="top_metal", z=3, x=2, y=1)
    assert sp_urls["config_url"] == "https://siliconprawn.org/map/intel/4004/config.json"
    assert sp_urls["tile_url"] == "https://siliconprawn.org/map/intel/4004/3/2/1.jpg"
    assert "Category%3AIntegrated_circuit_dies" in build_wikimedia_commons_api_url()
    ep_urls = build_patent_lookup_endpoints("US4764846A")
    assert "api.uspto.gov" in ep_urls["uspto_odp_search"]
    assert "ops.epo.org" in ep_urls["epo_ops_inpadoc_legal"]

    # 2. Verify patent ID parsing (US / EP / WO / JP)
    parsed = pf.parse_patent_ids("Covers US 4,764,846 A, EP1234567B1, WO2020123456A1, and JP2019123456A.")
    parsed_ids = {p["patent_id"] for p in parsed}
    assert {"US4764846A", "EP1234567B1", "WO2020123456A1", "JP2019123456A"}.issubset(parsed_ids)

    # 3. STRICT VETO cases: active utility patent, active SCPA mask work, NC/ND license, ARR, bot-wall
    active_patent_meta = {
        "repo_id": "siliconprawn",
        "vendor": "acme",
        "chipname": "modern_npu_2022",
        "license": "CC-BY-4.0",
        "fab_or_release_year": 2022,
        "patents": [{"patent_id": "US11999999B2", "filing_date": "2020-05-01", "grant_date": "2022-03-01"}],
    }
    rep_active = pf.verify_admissibility(active_patent_meta, as_of=today)
    assert rep_active.admissible is False
    assert any("active_or_unverified_patent" in r for r in rep_active.veto_reasons)

    nc_nd_meta = {
        "repo_id": "visual6502",
        "vendor": "mos",
        "chipname": "6502",
        "license": "CC-BY-NC-ND-3.0",
        "fab_or_release_year": 1975,
    }
    rep_nc = pf.verify_admissibility(nc_nd_meta, as_of=today)
    assert rep_nc.admissible is False
    assert any("license_veto" in r for r in rep_nc.veto_reasons)

    bot_wall_meta = {
        "repo_id": "siliconprawn",
        "vendor": "intel",
        "chipname": "4004",
        "license": "CC0-1.0",
        "fab_or_release_year": 1971,
        "description": "Access is temporarily restricted. Please verify you are a human.",
    }
    rep_wall = pf.verify_admissibility(bot_wall_meta, as_of=today)
    assert rep_wall.admissible is False
    assert "restriction_or_bot_wall_detected" in rep_wall.veto_reasons

    # 4. ALLOW cases: expired pre-1995 patent + CC-BY-4.0, maintenance-fee-lapsed patent, and TinyTapeout OSHW
    expired_die_meta = {
        "repo_id": "siliconprawn",
        "vendor": "intel",
        "chipname": "4004",
        "license": "CC-BY-4.0",
        "fab_or_release_year": 1971,
        "first_commercial_exploitation_date": "1971-11-15",
        "patents": [
            {"patent_id": "US3821715A", "filing_date": "1972-03-06", "grant_date": "1974-06-28"},
            {"patent_id": "US9000001B2", "filing_date": "2012-01-01", "grant_date": "2015-01-01", "unpaid_maintenance_fee_year": 3.5},
        ],
    }
    rep_ok = pf.verify_admissibility(expired_die_meta, as_of=today)
    assert rep_ok.admissible is True, f"Unexpected veto: {rep_ok.veto_reasons}"
    assert "silicon_die:intel_4004" in rep_ok.public_domain_tags
    assert "patent_expired:us3821715a" in rep_ok.public_domain_tags
    assert "patent_expired:us9000001b2" in rep_ok.public_domain_tags
    assert "cc-by:cc-by-4.0" in rep_ok.public_domain_tags
    assert all(AgentSubstrateBridge.is_public_domain_or_structural_tag(t) for t in rep_ok.public_domain_tags)

    oshw_meta = {
        "repo_id": "tinytapeout",
        "vendor": "tinytapeout",
        "chipname": "tt06_alu",
        "license": "Apache-2.0",
        "fab_or_release_year": 2024,
    }
    rep_oshw = pf.verify_admissibility(oshw_meta, as_of=today)
    assert rep_oshw.admissible is True, f"OSHW TinyTapeout should pass: {rep_oshw.veto_reasons}"

    # 5. Prepare deterministic synthetic silicon die photomicrograph on disk
    die_img_path = tmp_path / "intel_4004_die.png"
    die_meta_path = tmp_path / "intel_4004_die.json"
    grid = torch.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
    rgb = torch.stack([grid, torch.sin(grid * 6.28).abs(), 1.0 - grid], dim=-1)
    Image.fromarray((rgb.numpy() * 255).astype("uint8")).save(die_img_path)
    die_meta_path.write_text(json.dumps(expired_die_meta), encoding="utf-8")

    # 6. Full SiliconDieImageIngestor pipeline (active patent veto vs expired blueprint + fossilizer + graph)
    fossilizer = KnowledgeDyadFossilizer(storage_dir=str(tmp_path / "fossils"), feature_dim=64)
    graph_mgr = GyroidicGraphManager(data_dir=str(tmp_path / "graph"), dim=64)
    ingestor = SiliconDieImageIngestor(
        device="cpu",
        fossilizer=fossilizer,
        graph_manager=graph_mgr,
        lightweight=True,
    )

    vetoed_res = ingestor.ingest_die_image(die_img_path, active_patent_meta, as_of=today)
    assert vetoed_res["vetoed"] is True and vetoed_res["admissible"] is False
    assert vetoed_res["blueprint"] is None

    ok_res = ingestor.ingest_die_image(
        die_img_path,
        expired_die_meta,
        as_of=today,
        compile_blueprint=True,
        fossilize=True,
        register_graph=True,
    )
    assert ok_res["admissible"] is True and ok_res["vetoed"] is False
    bp = ok_res["blueprint"]
    assert bp["fingerprint_96"].shape == (96,)
    assert bp["crt_residues"].ndim >= 2
    assert bp["sparse_covariance_3x3"].shape == (3, 3)
    assert ok_res["structural_graph"] is not None and len(ok_res["structural_graph"].blocks) == 64
    assert ok_res["graph_node"] is not None and len(graph_mgr.nodes) == 1
    expected_subsystems = {
        "silicon_die_patent_filter",
        "conformal_log_polar",
        "conjugate_color_transport",
        "texture_dsp",
        "vision_utilities",
        "gyroidic_codec",
        "image_processor",
        "structural_entanglement_net",
        "canonical_projector",
        "agent_substrate_bridge",
        "structural_blueprints",
        "embedding_graph",
    }
    assert expected_subsystems.issubset(set(ok_res["connected_subsystems"])), (
        f"Missing subsystems: {expected_subsystems - set(ok_res['connected_subsystems'])}"
    )

    # 7. Connection Site: UniversalTopologyConverter.process_artifact with die_metadata gate
    utc = UniversalTopologyConverter(target_dim=64)
    utc_vetoed = utc.process_artifact(die_img_path, die_metadata=active_patent_meta)
    assert utc_vetoed.get("vetoed") is True and utc_vetoed.get("admissible") is False
    utc_ok = utc.process_artifact(die_img_path, die_metadata=expired_die_meta)
    assert utc_ok.get("vetoed") is not True
    assert "silicon_die:intel_4004" in utc_ok.get("public_domain_tags", [])

    # 8. Connection Site: LocalDatasetIngestor.silicon_die_generator
    ldi = LocalDatasetIngestor(datasets_root=str(tmp_path), device="cpu")
    die_convs = list(ldi.silicon_die_generator(die_dir=tmp_path, limit=5, as_of=today))
    assert len(die_convs) == 1
    assert "silicon_die:intel_4004" in die_convs[0].context["public_domain_tags"]

    # 9. Connection Site: VoxelboxterEngine.mount_silicon_die_blueprint & export_world_state
    vb = VoxelboxterEngine()
    vb_mount = vb.mount_silicon_die_blueprint(bp["die_id"], bp)
    assert vb_mount["block_count"] == 64
    assert "silicon_die:intel_4004" in vb.get_public_domain_tags()
    assert "silicon_die:intel_4004" in vb.export_world_state()["public_domain_tags"]

    # 10. Connection Site: Core DiegeticPhysicsEngine.mount_silicon_die_octree (PointerlessOctree)
    core_phys = CoreDiegeticPhysicsEngine(state_dim=64, device="cpu")
    oct_res = core_phys.mount_silicon_die_octree(bp)
    assert oct_res["mounted_nodes"] == 64

    # 11. Connection Site: UniversalOrchestrator.ingest_silicon_die_blueprint
    orch = UniversalOrchestrator(dim=64, device="cpu")
    orch_veto = orch.ingest_silicon_die_blueprint(die_img_path, active_patent_meta, as_of=today)
    assert orch_veto["vetoed"] is True
    orch_ok = orch.ingest_silicon_die_blueprint(die_img_path, expired_die_meta, as_of=today)
    assert orch_ok["admissible"] is True and orch_ok["blueprint"]["die_id"] == "intel_4004"

    # 12. Connection Site: HybridAI + UI DiegeticPhysicsEngine (lightweight=True)
    from hybrid_backend import HybridAI
    ai = HybridAI(lightweight=True)
    assert getattr(ai, "silicon_die_ingestor", None) is not None
    assert getattr(ai.engine, "silicon_die_ingestor", None) is not None
    hyb_veto = ai.ingest_silicon_die(die_img_path, active_patent_meta, as_of=today)
    assert hyb_veto["vetoed"] is True
    hyb_ok = ai.ingest_silicon_die(die_img_path, expired_die_meta, as_of=today)
    assert hyb_ok["admissible"] is True and hyb_ok["blueprint"]["die_id"] == "intel_4004"


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
