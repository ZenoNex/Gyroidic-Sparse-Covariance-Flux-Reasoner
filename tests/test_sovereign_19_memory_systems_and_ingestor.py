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


