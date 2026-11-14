import os
import ssl
import torch

os.environ["Sovereign_Disable_Background"] = "1"


def test_universal_orchestrator_19_memory_systems_unified():
    from src.core.orchestrator import UniversalOrchestrator

    orch = UniversalOrchestrator(dim=16)
    telemetry = orch.get_unified_memory_telemetry()

    assert telemetry["total_systems"] == 19
    assert telemetry["meliponini_isolation_enforced"] is True
    assert telemetry["shared_memory_palace_unified"] is True
    assert telemetry["trust_vault_bound"] is True
    for sys_name, is_active in telemetry["systems"].items():
        assert is_active is True, f"Memory system {sys_name} is not active on UniversalOrchestrator"


def test_diegetic_engine_19_memory_systems_unified():
    from src.ui.diegetic_backend import DiegeticPhysicsEngine

    engine = DiegeticPhysicsEngine(dim=32, k=5, lightweight=True)
    telemetry = engine.get_unified_memory_telemetry()

    assert telemetry["total_systems"] == 19
    assert telemetry["meliponini_isolation_enforced"] is True
    assert telemetry["shared_memory_palace_unified"] is True
    assert telemetry["trust_vault_bound"] is True
    for sys_name, is_active in telemetry["systems"].items():
        assert is_active is True, f"Memory system {sys_name} is not active on DiegeticPhysicsEngine"


def test_voxelboxter_memory_systems_and_visitor_chisel_hook():
    from src.ui.voxelboxter_simulation import VoxelboxterEngine, Block
    from src.ui.voxelboxter_client import PatchStateResource, switch_game_mode, EngineMode, Role

    sim = VoxelboxterEngine()
    assert sim.terrain_octree is not None
    assert sim.chisels_bits is not None
    assert sim.multi_mine is not None
    assert sim.purgatory is not None
    assert sim.luggage_storage is not None
    assert sim.memory_palace is not None
    assert sim.resonance_cavity is not None
    assert sim.trust_vault is not None

    state = PatchStateResource()
    assert state.chisels_bits is not None
    assert state.terrain_octree is not None
    assert state.purgatory is not None
    assert state.luggage_storage is not None

    # Test Role.VISITOR transition to PLAY mode hooks ChiselsAndBitsMicroOctree & PurgatoryBufferMap
    state.permissions.roles[state.local_peer_id] = Role.VISITOR
    state.graph.blocks[(0, 0, 0)] = Block(local_cell=(0, 0, 0), material_id=2, rotation=(0, 0, 0, 1), health=100.0)
    switch_game_mode(state, EngineMode.PLAY)
    assert state.mode == EngineMode.PLAY
    assert state.local_inventory.block_masses.get(2, 0) >= 1
    assert f"visitor_graph_{state.local_peer_id}" in state.purgatory.suspended_stranded_tensors


def test_anti_sycophancy_robots_tls_and_researchgate_wall():
    from src.data.textbook_filter import TextbookFilter
    from src.data.knowledge_ingestor import TLSResilientAdapter, ArXivSovereignIngestor
    from src.core.investor_news_ingestor import InvestorNewsIngestor
    from src.core.knowledge_dyad_fossilizer import DyadFossilizer

    # 1. Verify ResearchGate restriction text is rejected by TextbookFilter
    rg_wall_text = (
        "Researchgate\n"
        "Access is temporarily restricted\n"
        "We detected unusual activity from your device or network.\n"
        "Reasons may include:\n"
        " Rapid taps or clicks\n"
        " JavaScript disabled or not working\n"
        " Automated (bot) activity on your network (IP 172.56.115.146)\n"
        " Use of developer or inspection tools\n"
        "Need help? Submit feedback.\n"
        "ID: 42a01088-ce76-071d-d15a-6e09d4672537"
    )
    assert TextbookFilter.is_restriction_or_bot_wall(rg_wall_text) is True
    tf = TextbookFilter(state_dim=32)
    report = tf.assess(rg_wall_text, source="web_https://www.researchgate.net/publication/123")
    assert report.is_admissible is False
    assert "restriction_wall_detected" in report.flags

    # 2. Verify TLSResilientAdapter sets OP_IGNORE_UNEXPECTED_EOF and OP_LEGACY_SERVER_CONNECT
    adapter = TLSResilientAdapter()
    ssl_ctx = adapter._build_ssl_context()
    expected_op = getattr(ssl, "OP_IGNORE_UNEXPECTED_EOF", 0)
    if expected_op:
        assert (ssl_ctx.options & expected_op) == expected_op

    # 3. Verify non-sycophantic transformative distillation policy in ArXivSovereignIngestor & InvestorNewsIngestor
    fossilizer = DyadFossilizer(storage_dir="data/encodings", feature_dim=32)
    ingestor = ArXivSovereignIngestor(fossilizer=fossilizer, engine_dim=32)
    courtesy = ingestor.evaluate_sovereign_robots_courtesy(
        "https://example.org/persuasion-its-components-principles-and-techniques.html"
    )
    assert courtesy["allow_transformative_distillation"] is True

    inv = InvestorNewsIngestor(state_dim=16)
    assert inv.is_allowed("https://example.org/restricted-feed", transformative_distillation=True) is True


def test_unicorn_synthesis_psychotopological_substrate_sing_and_talk_in_voxelboxter():
    from src.ui.voxelboxter_simulation import VoxelboxterEngine
    from src.ui.voxelboxter_client import PatchStateResource
    from src.core.orchestrator import UniversalOrchestrator

    sim = VoxelboxterEngine(device="cpu")
    assert sim.larynx is not None
    assert sim.chern_simons_gasket is not None
    assert sim.voynich_linguist is not None
    assert sim.texture_dsp is not None
    assert sim.audience_projector is not None
    assert sim.archetype_governor is not None
    assert sim.bio_governor is not None
    assert sim.skeleton_harness is not None

    # Test 1: Singing mode with K=64 Chebyshev harmonic envelope
    harmonics = torch.sin(torch.linspace(0.0, 6.28, 64))
    sing_res = sim.vocalize_psychotopological_substrate(
        prompt_text="Isotremoptememally",
        mode="sing",
        audio_harmonics_k64=harmonics,
        max_steps=16,
        pas_h=0.82,
    )
    assert sing_res["mode"] == "sing"
    assert isinstance(sing_res["utterance"], str) and len(sing_res["utterance"]) > 0
    assert sing_res["voynich_sub_pairs_count"] == 36  # 2n for 18-char string 'Isotremoptememally'
    assert sing_res["morton_subcarrier_keys_count"] > 0
    assert "compute_affordance" in sing_res
    assert "restless_500ms_guard" in sing_res["compute_affordance"]
    assert "affordance_reset_gibbs" in sing_res
    assert "bio_reflex_telemetry" in sing_res

    # Test 2: Talking mode via PatchStateResource & UniversalOrchestrator
    client_state = PatchStateResource()
    talk_res = client_state.vocalize_psychotopological_substrate(
        prompt_text="CerumenPot",
        mode="talk",
        max_steps=12,
        pas_h=0.75,
    )
    assert talk_res["mode"] == "talk"
    assert talk_res["voynich_sub_pairs_count"] == 20  # 2n for 10-char string 'CerumenPot'

    orch = UniversalOrchestrator(dim=16)
    orch_res = orch.vocalize_psychotopological_substrate(
        prompt_text="Murmuration",
        mode="sing",
        max_steps=10,
    )
    assert orch_res["voynich_sub_pairs_count"] == 22


def test_19_arxiv_pdfs_mathematical_invariants():
    from src.core.invariants import (
        compute_affordance_reset_gibbs_occupation,
        compute_euler_heisenberg_cubic_screening,
        compute_feshbach_zeno_boundary_decay,
        compute_alabama_apportionment_paradox_prob,
        compute_novikov_optimal_training_allocation,
        compute_landau_homoenergetic_compression_audit,
        compute_zolotarev_vandermonde_reciprocity_sign,
        compute_gibbs_gnz_variational_beta,
        compute_rmgrit_bevy_runahead_schedule,
        compute_prove_self_play_drgrpo_advantage,
    )

    # 1. 2611.02220v1: Affordance-Reset Gibbs Occupation
    occ = compute_affordance_reset_gibbs_occupation(
        exit_barriers=[-1.2, -0.3, 0.5],
        reward_potentials=[0.8, 0.4, 0.1],
        temperature=0.5,
        beta_reward=1.0,
    )
    assert abs(sum(occ["occupation_probs"]) - 1.0) < 1e-5
    assert occ["dominant_index"] == 0

    # 2. 2611.05240v1: Euler-Heisenberg Cubic Screening
    field = torch.tensor([10.0, -10.0, 5.0, -5.0])
    eh = compute_euler_heisenberg_cubic_screening(field, alpha_eh=0.1)
    assert eh["screening_ratio"] < 1.0  # Strong field cubic saturation screens blow-up

    # 3. 2611.05268v1: Feshbach-Lowdin Quantum Zeno Decay
    z_low = compute_feshbach_zeno_boundary_decay(absorber_gamma=1.0)
    z_high = compute_feshbach_zeno_boundary_decay(absorber_gamma=20.0)
    assert z_high["zeno_decay_rate_gamma_n"] < z_low["zeno_decay_rate_gamma_n"]
    assert z_high["is_zeno_frozen"] is True

    # 4. 2610.40682v1: McCune's Alabama Paradox Formula
    p3 = compute_alabama_apportionment_paradox_prob(3)
    p1000 = compute_alabama_apportionment_paradox_prob(1000)
    assert abs(p3["alabama_paradox_prob"] - (6.0 / (36.0 * 12.0))) < 1e-6
    assert abs(p1000["alabama_paradox_prob"] - (1.0 / 36.0)) < 1e-3

    # 5. 2611.05224v1: Novikov Optimal Training Allocation & Double-Descent Peak K*(t) = t + 1
    nov = compute_novikov_optimal_training_allocation(
        env_probs=[0.6, 0.3, 0.08, 0.02],
        budget_d=1.5,
        current_step_t=3,
    )
    assert nov["double_descent_peak_k_star"] == 4
    assert nov["is_at_double_descent_peak"] is True
    assert nov["recommended_safe_k"] == 5

    # 6. 2611.05236v1: Landau Homoenergetic Compression Implosion Audit
    landau_comp = compute_landau_homoenergetic_compression_audit(
        torch.diag(torch.tensor([-0.5, -0.5, -0.5])), time_t=2.0
    )
    assert landau_comp["is_hydrodynamic_implosion_risk"] is True

    # 7. 2610.35990v1: Zolotarev Vandermonde Reciprocity Sign
    zol = compute_zolotarev_vandermonde_reciprocity_sign(a=2, p=7, q=3)
    assert zol["zolotarev_permutation_sign"] == 1  # 2^3 = 8 = 1 (mod 7)

    # 8. 1701.08105v2: Variational GNZ Inverse Temperature Estimator
    gnz = compute_gibbs_gnz_variational_beta(torch.tensor([0.2, 0.5, 0.9, 1.4]))
    assert gnz["partition_free_gnz"] is True and gnz["estimated_beta_gnz"] > 0.0

    # 9. 2611.03621v1 & 2611.04794v1: R-MGRIT & Bevy Runahead Schedule
    rmg = compute_rmgrit_bevy_runahead_schedule(total_steps_nt=20, truncation_gamma=0.2, request_layers=[2, 1, 1, 3])
    assert rmg["truncated_trajectory_nt"] == 4
    assert rmg["lowest_active_layer_l_min"] == 1
    assert rmg["lowest_layer_compute_group"] == [1, 2]

    # 10. 2611.05158v1: PROVE Self-Play DrGRPO Advantage
    prv = compute_prove_self_play_drgrpo_advantage([0.2, 0.8, 0.5], y1_verdict=1, y2_verdict=0)
    assert abs(sum(prv["drgrpo_advantages"])) < 1e-5


def test_cross_referenced_lowly_consequential_invariants_and_subsystems():
    from src.environment.cleanroom_advanced_extensions import OpenGliderKinematics, GliderDeploymentState
    from src.core.carnot_mobius_ledger import CarnotMobiusLedger
    from src.core.context_aware_quantizer import ContextAwareQuantizer
    from src.core.video_dyad_parser import VideoDyadParser
    from src.models.resonance_cavity import ResonanceCavity
    from src.core.legibility_audit import LegibilityTripwire
    from src.core.structural_monitors import MetaInfraIntraMonitor
    from src.core.chern_simons_gasket import SolitonStabilityHealer
    from src.core.hessian_8x8_curvature_ledger import Trajectory8x8Decombiner

    # 1. OpenGliderKinematics WIGE ground cushion & Venturi flow choking (2611.04571v1)
    glider = OpenGliderKinematics()
    glider.state = GliderDeploymentState.DEPLOYED
    glider.deployment_progress = 1.0
    _, tel_cushion = glider.update_flight_kinematics(
        velocity=(10.0, -0.5, 0.0),
        orientation_pitch_yaw_roll=(2.0, 0.0, 0.0),
        ambient_wind=(0.0, 0.0, 0.0),
        altitude_agl=0.8,
        chord_length=2.0,
    )
    assert tel_cushion["wige_aerodynamics"]["regime"] == "WIGE_GROUND_CUSHION"
    assert tel_cushion["wige_aerodynamics"]["lift_multiplier"] > 1.0
    _, tel_choke = glider.update_flight_kinematics(
        velocity=(10.0, -0.5, 0.0),
        orientation_pitch_yaw_roll=(2.0, 0.0, 0.0),
        ambient_wind=(0.0, 0.0, 0.0),
        altitude_agl=0.10,
        chord_length=2.0,
    )
    assert tel_choke["wige_aerodynamics"]["regime"] == "VENTURI_CHOKED"
    assert tel_choke["wige_aerodynamics"]["is_venturi_choked"] is True

    # 2. CarnotMobiusLedger Landau Homoenergetic Compression Audit (2611.05236v1)
    ledger = CarnotMobiusLedger()
    ledger_out = ledger(
        lambda_vac=torch.tensor(0.2),
        lambda_pump=torch.tensor(1.0),
        admr_residues=torch.tensor([[-0.6, -0.6, -0.6, -0.6]]),
        depth_N=3,
    )
    assert "landau_compression_audit" in ledger_out
    assert ledger_out["landau_compression_audit"]["min_eigenvalue_l0"] < 0.0

    # 3. ContextAwareQuantizer Novikov Double-Descent Audit (2611.05224v1)
    caq = ContextAwareQuantizer(dim=8)
    nov_diag = caq.audit_novikov_double_descent(step_t=3, active_cells_k=4)
    assert nov_diag["is_at_double_descent_peak"] is True
    assert nov_diag["recommended_safe_cells_k"] == 5

    # 4. VideoDyadParser REEL Representation Amortization (2610.38158v1)
    vparser = VideoDyadParser(device="cpu")
    reel_breather = vparser.evaluate_reel_representation_amortization(num_queries_m=5, retain_breather_state=True)
    reel_scalar = vparser.evaluate_reel_representation_amortization(num_queries_m=5, retain_breather_state=False)
    assert reel_breather["expected_answer_flip_rate"] < reel_scalar["expected_answer_flip_rate"]
    assert reel_breather["flip_risk_avoided"] > 0.20

    # 5. ResonanceCavity Weakly Viscous Nonlocal Wave Damping & DN Potential (2611.04789v1)
    cavity = ResonanceCavity(hidden_dim=8, num_modes=4)
    wave_diag = cavity.evaluate_nonlocal_wave_damping_and_dn_potential(
        state_u=torch.sin(torch.linspace(0, 3.14, 16)),
        r_viscous=0.25,
        s_elastic=0.75,
    )
    assert wave_diag["is_weakly_viscous_regime"] is True
    assert wave_diag["entanglement_unique_continuation"] is True

    # 6. LegibilityTripwire Chen-Stein Hypergraph Degree Anomaly (2611.04705v1)
    tripwire = LegibilityTripwire(hidden_dim=8)
    sel = torch.ones(4, 8)
    rej = torch.zeros(4, 8)
    tw_out = tripwire(sel, rej)
    assert "stein_hypergraph_anomaly" in tw_out
    assert "eigenvalue_ratio_tau" in tw_out["stein_hypergraph_anomaly"]

    # 7. MetaInfraIntraMonitor McCune's Alabama Apportionment Paradox (2610.40682v1)
    monitor = MetaInfraIntraMonitor()
    ala = monitor.audit_alabama_apportionment_paradox(house_size_h=99)
    assert ala["mod_3_residue"] == 0
    assert ala["max_paradox_jump_i_max"] == 97

    # 8. SolitonStabilityHealer Q10.5 & Trajectory8x8Decombiner Q12.4
    healer = SolitonStabilityHealer(device="cpu")
    conv_diag = healer.audit_euler_heisenberg_and_zeno_convexity(
        dp_stress=torch.tensor([5.0, 12.0, -8.0]),
        gcve_pressure=10.0,
    )
    assert conv_diag["drucker_prager_globally_convex"] is True
    assert conv_diag["euler_heisenberg_screening"]["screening_ratio"] < 1.0

    decomb = Trajectory8x8Decombiner(budget=64)
    board = [[0] * 8 for _ in range(8)]
    board[0][0] = 4
    board[0][1] = 2
    dec_res = decomb.decombine_trajectory(board, max_depth=2)
    assert dec_res["zolotarev_chiral_parity_sign"] in (-1, 0, 1)


def test_five_highly_consequential_upgrades_and_closed_loop_stream(tmp_path):
    from src.core.zeitgeist_router import ZeitgeistRouter
    from src.models.modular_attention import ModularAttention
    from src.core.polynomial_coprime import PolynomialCoprimeConfig
    from src.core.manifold_time import TwoCopsSchedule
    from src.models.diegetic_heads import ResonanceLarynx
    from src.core.codes_constraint_framework import create_codes_framework
    from src.core.garden_statistical_attractors import GardenOrchestrator, MemoryPalace
    from src.core.knowledge_dyad_fossilizer import DyadFossilizer

    # HC-1: Affordance-Reset Dwell-Time Gibbs Routing in ZeitgeistRouter & ModularAttention
    router = ZeitgeistRouter(dim=16)
    x_in = torch.randn(2, 16)
    mode, zg_state, zg_diag, x_steered = router(x_in, tadc_kwargs={"pas_h": 0.75, "entropy": 0.3})
    assert "dwell_occupation_probs" in zg_diag
    assert abs(sum(zg_diag["dwell_occupation_probs"]) - 1.0) < 1e-4
    assert len(zg_diag["facet_dwell_occupation_probs"]) == router.M

    poly_cfg = PolynomialCoprimeConfig(k=5, degree=4)
    mod_attn = ModularAttention(hidden_dim=16, num_heads=2, poly_config=poly_cfg, num_functionals=5)
    attn_out = mod_attn(
        torch.randn(2, 4, 16),
        trust_scalars=zg_diag["facet_dwell_occupation_probs"],
    )
    assert attn_out.shape == (2, 4, 16)
    assert hasattr(mod_attn, "last_dwell_gibbs_stats")
    assert abs(sum(mod_attn.last_dwell_gibbs_stats["occupation_probs"]) - 1.0) < 1e-4

    # HC-2: Restarted MGRIT + Anderson Acceleration AA(w) + Bevy Runahead in TwoCopsSchedule
    sched = TwoCopsSchedule(macro_steps=4)
    s0 = torch.ones(2, 16)
    s1 = s0 + 0.2 * torch.randn(2, 16)
    s2 = s1 + 0.1 * torch.randn(2, 16)
    s3 = s2 + 0.05 * torch.randn(2, 16)
    r_hist = [s1 - s0, s2 - s1, s3 - s2]
    aa_res = sched.compute_rmgrit_schedule_and_anderson_acceleration(
        state_history=[s0, s1, s2, s3],
        residual_history=r_hist,
        total_micro_steps=5,
        gamma=0.2,
        window_w=2,
        request_layers=[1, 0, 2],
    )
    assert aa_res["anderson_applied"] is True
    assert aa_res["accelerated_state"].shape == (2, 16)
    assert aa_res["rmgrit_schedule"]["lowest_active_layer_l_min"] == 0

    # HC-3: Live PROVE Perturb-Repair-Verify Self-Play DrGRPO Weight Mutation in ResonanceLarynx
    larynx = ResonanceLarynx(hidden_dim=16)
    prove_res = larynx.prove_self_play_drgrpo_step(
        seed_state=torch.randn(2, 16),
        group_size=4,
        lr=0.005,
    )
    assert "drgrpo_advantages" in prove_res
    assert len(prove_res["drgrpo_advantages"]) == 4
    assert abs(sum(prove_res["drgrpo_advantages"])) < 1e-4

    # HC-4: Partition-Free Variational GNZ Local Insertion Energy & Widom-Rowlinson Repulsion
    codes = create_codes_framework(state_dim=16, max_constraints=6)
    _, void_diag = codes.monitor_void_friction(torch.randn(2, 16), f_void=0.3)
    assert void_diag["partition_free_gnz"] is True
    assert void_diag["estimated_beta_gnz"] > 0.0

    garden = GardenOrchestrator(feature_dim=16, num_attractors=4, device="cpu")
    garden_out = garden.evolve_garden(torch.randn(3, 16), dt=0.1)
    assert "estimated_beta_gnz" in garden_out
    assert "widom_rowlinson_insertion_energy" in garden_out

    # HC-5: Dynamic q-Hierarchical Grounded View-Tree V_D(T_Q) Index Migration
    fossilizer = DyadFossilizer(storage_dir=str(tmp_path), feature_dim=16)
    fossilizer._index_ready.wait(timeout=5.0)
    fossilizer.update_fossil_view_weight("dyad_alpha.pt", delta_weight=1.5)
    fossilizer.update_fossil_view_weight("dyad_beta.pt", delta_weight=0.2)
    top_leaf = fossilizer.select_fossil_by_weighted_rank(0.1)
    assert top_leaf == "dyad_alpha.pt"

    mp = MemoryPalace(feature_dim=16)
    mp.plant_room("room_1", torch.randn(16), variance=1.0, residue_tuple=(1, 2, 3), seam_tension=0.8)
    mp.plant_room("room_2", torch.randn(16), variance=1.0, residue_tuple=(4, 5, 6), seam_tension=0.3)
    new_w = mp.update_room_view_weight("room_2", delta_weight=2.0)
    assert new_w > 2.0
    ranked_rooms = mp.query_ranked_rooms_view_tree(k_limit=2)
    assert ranked_rooms[0].room_id == "room_2"


def test_neglecton_rehydration_and_graph_bounds():
    import json
    from src.core.orchestrator import UniversalOrchestrator
    from src.topology.embedding_graph import GyroidicGraphManager

    orch = UniversalOrchestrator(dim=16)
    telemetry = orch.get_unified_memory_telemetry()
    assert "neglecton_telemetry" in telemetry
    neg_telem = telemetry["neglecton_telemetry"]
    assert "node_count" in neg_telem
    assert neg_telem["node_count"] > 0
    assert neg_telem["snapshot_restored"] is True

    # Test explicit rehydration
    rehydrate_res = orch.rehydrate_neglecton(limit=100)
    assert rehydrate_res["status"] == "rehydrated"
    assert rehydrate_res["neglecton_nodes"] > 0
    assert "cayley_cubic_anchor" in rehydrate_res
    assert "folium_anchor" in rehydrate_res
    assert rehydrate_res["synced_outliers_into_memory_palace"] > 0

    # Test graph export bounds and top-k degree pruning
    graph_json_str = orch.graph_manager.export_graph_json()
    graph_data = json.loads(graph_json_str)
    assert "nodes" in graph_data
    assert "links" in graph_data
    assert len(graph_data["nodes"]) > 0
    # Links must be pruned to avoid WebGL hairball freezes
    assert len(graph_data["links"]) <= 3500

    # Test GyroidicGraphManager limit handling does not trigger disk scan if snapshot has nodes
    mgr = GyroidicGraphManager("data/encodings")
    res = mgr.load_fossils(limit=5000, use_snapshot=True)
    assert mgr.loaded_from_snapshot is True
    assert len(mgr.nodes) > 0


def test_arxiv_eprint_multiformat_payload_extraction():
    import io
    import tarfile
    import zipfile
    from src.data.knowledge_ingestor import ArXivSovereignIngestor

    ingestor = ArXivSovereignIngestor(fossilizer=None, engine_dim=16)

    # 1. Plain text / HTML should not crash tarfile and return empty list
    html_payload = b"<!DOCTYPE html><html><body>Error 404</body></html>"
    assert ingestor._extract_images_from_payload(html_payload) == []

    text_payload = b"\\documentclass{article}\\begin{document}Hello world\\end{document}"
    assert ingestor._extract_images_from_payload(text_payload) == []

    # 2. PDF payload with simulated JPEG stream
    # Minimal 1x1 JPEG bytes
    jpeg_bytes = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0"
        b"\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00"
        b"\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01"
        b"\x01\x00\x00?\x00\xbf\x00\xff\xd9"
    )
    pdf_payload = b"%PDF-1.7\nstream\n" + jpeg_bytes + b"\nendstream\n%%EOF"
    pdf_extracted = ingestor._extract_images_from_payload(pdf_payload)
    assert len(pdf_extracted) == 1
    assert pdf_extracted[0] == jpeg_bytes

    # 3. Zip payload containing an image
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("figure1.jpg", jpeg_bytes)
        z.writestr("paper.tex", "hello")
    zip_extracted = ingestor._extract_images_from_payload(zip_buf.getvalue())
    assert len(zip_extracted) == 1
    assert zip_extracted[0] == jpeg_bytes

    # 4. Tar payload containing an image
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        tarinfo = tarfile.TarInfo(name="figure2.jpg")
        tarinfo.size = len(jpeg_bytes)
        tar.addfile(tarinfo, io.BytesIO(jpeg_bytes))
    tar_extracted = ingestor._extract_images_from_payload(tar_buf.getvalue())
    assert len(tar_extracted) == 1
    assert tar_extracted[0] == jpeg_bytes


def test_system1_system2_harmonization_tempolock_ema_and_honest_jitter_rehydration():
    from src.core.manifold_time import TwoCopsSchedule
    from src.core.bulletin_board import BulletinBoard
    from src.core.spectral_coherence_repair import apply_energy_based_stabilization
    from src.core.orchestrator import UniversalOrchestrator

    # 1. TEMPOLOCK Prime Gate Calibration (gcd(step, active_prime) > 1 or kappa > 0.35)
    sched = TwoCopsSchedule(macro_steps=5, use_tempolock=True, alpha=0.15)
    # Step 1: active_prime is 3 (since 2 is skipped when macro_steps > 2), step=1 -> False
    _, sync_1 = sched.step(torch.tensor(0.1), curvature=0.1)
    assert sync_1 is False
    # Step 2: step=2, active_prime=3 -> False
    _, sync_2 = sched.step(torch.tensor(0.1), curvature=0.1)
    assert sync_2 is False
    # Step 3: step=3 shares divisor with active_prime=3 -> True
    _, sync_3 = sched.step(torch.tensor(0.1), curvature=0.1)
    assert sync_3 is True
    # Step 4 with Red Zone curvature (kappa = 0.42 > 0.35) -> True even outside prime divisor
    _, sync_4_red = sched.step(torch.tensor(0.1), curvature=0.42)
    assert sync_4_red is True

    # 2. Bulletin Board EMA (alpha = 0.15 with shock damping)
    bb = BulletinBoard(size=8, alpha=0.15)
    bb.force_register.fill_(1.0)
    bb.post_force(torch.zeros(8))
    # With alpha=0.15 and 0 force norm, new force should be 0.85 * 1.0 = 0.85 (preserving 85% persistence)
    assert torch.allclose(bb.read_force(), torch.full((8,), 0.85), atol=1e-4)

    # 3. Honest Jitter Birkhoff Rehydration on NaN stasis
    orch = UniversalOrchestrator(dim=16)
    stasis_state = torch.full((1, 16), 0.5)
    stasis_state[0, :8] = float("nan")  # Drop half the state into NaN stasis
    rehydrated = apply_energy_based_stabilization(
        stasis_state.clone(),
        birkhoff_projector=orch.birkhoff_projector,
    )
    assert not torch.isnan(rehydrated).any()
    # Rehydrated dimensions must have genuine physical energy (> 0.01), not crushed to 1e-8
    assert torch.norm(rehydrated[0, :8]).item() > 0.01

    # End-to-end Orchestrator step recovery from NaN stasis
    out = orch.step(stasis_state)
    assert not torch.isnan(out["state"]).any()
    assert orch.telemetry_state.get("honest_jitter_rehydrated") is True


def test_p2p_collective_agent_smith_non_memorization_and_structural_algebra(tmp_path):
    """
    Verifies that DyadFossilizer.export_agent_smith, AgentSubstrateBridge, BonfireNomadicRing,
    BonfireNetwork, and UniversalOrchestrator enforce the Idea-Expression Dichotomy and
    Non-Memorization across local and collective P2P distribution:
      1. Verbatim expressive text and raw Euclidean meshes are stripped from soliton_smith.pt.
      2. Only Betti invariants, CRT residue tuples, prime-ladder frequencies, and selectively
         punctured (~33% zeroed) structural weights are retained and broadcasted.
    """
    from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
    from src.core.agent_substrate_bridge import AgentSubstrateBridge
    from src.p2p.bonfire_consensus import BonfireNomadicRing
    from src.core.orchestrator import UniversalOrchestrator

    fossilizer = DyadFossilizer(storage_dir=str(tmp_path), feature_dim=16)
    fossilizer._index_ready.wait(timeout=5.0)

    verbatim_sentence = "It was the best of times, it was the worst of times, in a copyrighted novel."
    raw_mesh_vertices = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    dyad = KnowledgeDyad(
        timestamp="2026-11-06T09:00:00",
        linguistic_description=verbatim_sentence,
        meta_state=torch.ones(16),
        gyroid_residue=torch.linspace(0.1, 1.6, 16),
        metadata={
            "raw_text": verbatim_sentence,
            "vertices": raw_mesh_vertices,
            "polygons": [[0, 1, 2]],
            "tags": ["novel_chapter_one", "It was the best of times"],
            "domain": "topology_test",
        },
    )
    prime_freqs = torch.linspace(2.0, 17.0, 16)
    betti_dict = {0: 1.0, 1: 2.0, 2: 0.0}

    export_path = fossilizer.export_agent_smith(
        dyad=dyad,
        prime_frequencies=prime_freqs,
        betti_numbers=betti_dict,
        filename="soliton_smith_test.pt",
    )
    saved = torch.load(export_path, map_location="cpu")

    # 1. Verbatim text and raw mesh polygons/vertices must be completely absent
    assert saved["description"] != verbatim_sentence
    assert saved["description"].startswith("structural_soliton_crt_")
    assert "worst" not in saved["tags"]
    assert "times," not in saved["tags"]
    assert "raw_text" not in saved["dyad_metadata"]
    assert "vertices" not in saved["dyad_metadata"]
    assert "polygons" not in saved["dyad_metadata"]
    assert saved["dyad_metadata"].get("domain") == "topology_test"

    # 2. Structural algebra attestation and ~33% Meliponini puncture must be present
    attest = saved["structural_algebra_attestation"]
    assert attest["idea_expression_dichotomy_compliant"] is True
    assert attest["verbatim_text_discarded"] is True
    assert attest["raw_euclidean_mesh_discarded"] is True
    assert attest["memorization_regurgitation_risk"] == 0.0
    assert 0.25 <= attest["non_invertible_puncture_ratio"] <= 0.45

    shielded = saved["meta_state_shielded"]
    zero_fraction = float((shielded == 0).sum().item()) / float(shielded.numel())
    assert 0.25 <= zero_fraction <= 0.45

    # 3. Verify P2P BonfireNomadicRing broadcast & reception strips any injected verbatim fields
    class DummyFreenet:
        def __init__(self):
            self.subs = {}
            self.published = []

        def subscribe(self, contract_id, cb):
            self.subs[contract_id] = cb

        def publish(self, contract_id, payload):
            self.published.append((contract_id, payload))
            if contract_id in self.subs:
                self.subs[contract_id](payload)

    dummy_fn = DummyFreenet()
    ring = BonfireNomadicRing(dummy_fn)

    received_payloads = []
    ring.on_agent_smith_received = lambda p: received_payloads.append(
        fossilizer.inject_agent_smith(p, expected_dim=16)
    )

    # Corrupt the saved file on disk with a verbatim field to prove broadcast_agent_smith sanitizes it
    corrupted = dict(saved)
    corrupted["description"] = verbatim_sentence
    corrupted["raw_mesh"] = raw_mesh_vertices
    corrupted_path = str(tmp_path / "corrupted_smith.pt")
    torch.save(corrupted, corrupted_path)

    pkt = ring.broadcast_agent_smith("peer_alpha", corrupted_path)
    assert pkt is not None
    assert pkt["structural_algebra_attestation"]["verbatim_text_discarded"] is True
    assert len(received_payloads) == 1
    rehydrated = received_payloads[0]
    assert rehydrated["description"].startswith("structural_soliton_crt_")
    assert "raw_mesh" not in rehydrated
    assert "meta_state_aligned" in rehydrated
    assert rehydrated["meta_state_aligned"].shape[-1] == 16

    # 4. Verify UniversalOrchestrator.export_and_broadcast_agent_smith end-to-end
    orch = UniversalOrchestrator(dim=16)
    orch.dyad_fossilizer = fossilizer
    orch.bonfire_ring = ring
    export_info = orch.export_and_broadcast_agent_smith(
        state=torch.randn(1, 16),
        betti_numbers={0: 1.0, 1: 3.0},
        description=verbatim_sentence,
        filename="orch_smith.pt",
    )
    assert export_info["structural_algebra_attestation"]["idea_expression_dichotomy_compliant"] is True
    assert export_info["bonfire_network_header"]["verbatim_text_discarded"] is True








