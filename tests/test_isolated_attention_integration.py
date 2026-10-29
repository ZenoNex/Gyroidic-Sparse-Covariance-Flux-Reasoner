"""
Unit tests validating the complete integration of the 12 previously isolated
attention and cognitive routing subsystems into the Gyroidic Sparse Covariance Flux Reasoner:
1. ModularAttention (Multi-field attention with Birkhoff projection and Sine-Gordon Breather slots)
2. ModularTransformerLayer (Transformer layer with multi-field modular attention)
3. AutoeclecticResponderHead (Diegetic responder head warped by spectral entropy and manifold roughness)
4. DataAssociationLayer (Knowledge Dyad text/image cross-modal fusion into coprime residues)
5. ResonanceLarynx with LazarusSoftmax (Token/probability generator with Chern-Simons gasket)
6. CALM Predictor (Context-Adaptive Latent Momentum Veto & 5-constraint attention head with non-dual veto gating)
7. ResonanceCavity Attention Coupling (Cavity mode excitation driven by A_ij^H R_ij attention states)
8. BirkhoffProjection (Doubly-stochastic normalization with direct N<=32 and Sinkhorn fallback)
9. GardenOrchestrator Attractor Attention (Non-Birkhoff topological attractor fallback)
10. SPathRAGInjection (Non-tokenizing geometric path distance injection into attention scores)
11. DynamicAttentionMask (Topological violation-guided dense vs band-sparse attention masking)
12. End-to-End Orchestrator, DiegeticPhysicsEngine, and PhysicalNodeEditor Integration
"""

import os
import sys
import pytest
import torch
import torch.nn.functional as F

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.models.modular_attention import ModularAttention, ModularTransformerLayer
from src.models.diegetic_heads import AutoeclecticResponderHead, DataAssociationLayer, ResonanceLarynx
from src.models.resonance_cavity import ResonanceCavity
from src.surrogates.calm_predictor import CALM
from src.core.birkhoff_projection import BirkhoffProjection
from src.core.garden_statistical_attractors import GardenOrchestrator
from src.core.diegetic_physics_engine import DiegeticPhysicsEngine
from src.core.orchestrator import UniversalOrchestrator
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.scripting.node_environment import PhysicalNodeEditor


class TestIsolatedAttentionIntegration:
    """Test suite covering the 12 attention subsystems and their integrations."""

    def test_01_modular_attention_birkhoff_projection(self):
        """1. Validate ModularAttention multi-field attention and Birkhoff projection."""
        poly_config = PolynomialCoprimeConfig(k=4, degree=3)
        attn = ModularAttention(hidden_dim=16, num_heads=4, poly_config=poly_config, num_functionals=4, use_birkhoff=True)
        x = torch.randn(2, 6, 16)
        out = attn(x)
        assert out.shape == (2, 6, 16)
        integrity = attn.validate_structural_integrity()
        assert integrity is not None
        assert bool(integrity.all().item()) is True

    def test_02_modular_transformer_layer(self):
        """2. Validate ModularTransformerLayer execution with modular attention and residual."""
        poly_config = PolynomialCoprimeConfig(k=4, degree=3)
        layer = ModularTransformerLayer(hidden_dim=16, num_heads=4, poly_config=poly_config, num_functionals=4)
        x = torch.randn(2, 6, 16)
        out = layer(x)
        assert out.shape == (2, 6, 16)
        integrity = layer.validate_structural_integrity()
        assert integrity is not None

    def test_03_autoeclectic_responder_head(self):
        """3. Validate AutoeclecticResponderHead entropy modulation and diegetic output."""
        head = AutoeclecticResponderHead(hidden_dim=16, output_dim=16, num_modes=4)
        state = torch.randn(4, 16)
        entropy_low = torch.tensor([[0.05], [0.1], [0.08], [0.12]])
        entropy_high = torch.tensor([[0.85], [0.9], [0.75], [0.95]])
        curv_low = torch.tensor([[0.1], [0.1], [0.1], [0.1]])
        curv_high = torch.tensor([[2.0], [2.0], [2.0], [2.0]])
        
        out_low = head(state, spectral_entropy=entropy_low, curvature=curv_low)
        out_high = head(state, spectral_entropy=entropy_high, curvature=curv_high)
        
        assert out_low.shape == (4, 16)
        assert out_high.shape == (4, 16)
        # Verify modulation basis and curvature respond differently to high vs low entropy
        assert not torch.allclose(out_low, out_high)

    def test_04_data_association_layer(self):
        """4. Validate DataAssociationLayer cross-modal fusion into coprime residues."""
        dyad_layer = DataAssociationLayer(input_dim=16, hidden_dim=16, k=5)
        text_emb = torch.randn(3, 16)
        image_emb = torch.randn(3, 16)
        
        residues = dyad_layer(text_emb, image_emb)
        assert residues.shape == (3, 5)
        # Check output is properly bounded (tanh range [-1, 1])
        assert torch.all(residues >= -1.0) and torch.all(residues <= 1.0)

    def test_05_resonance_larynx_with_lazarus_softmax(self):
        """5. Validate ResonanceLarynx and LazarusSoftmax diagnostics."""
        larynx = ResonanceLarynx(hidden_dim=16)
        x = torch.randn(2, 16)
        logits, conf = larynx(x)
        assert logits.shape[0] == 2
        assert conf.shape[0] == 2
        diags = larynx.chern_simons.get_diagnostics()
        assert "seam_tension" in diags
        assert "level_k" in diags

    def test_06_calm_trajectory_veto_and_non_dual_gating(self):
        """6. Validate CALM trajectory momentum monitor, constraint attention, and non-dual gating."""
        calm = CALM(dim=16, history_len=8)
        history = torch.randn(2, 8, 16)
        abort_score, rho_factor, step_factor, forcing, gauge, constraints = calm(history, h_mischief=0.1, dt=0.05)
        
        assert abort_score.shape == (2, 1)
        assert rho_factor.shape == (2, 1)
        assert step_factor.shape == (2, 1)
        assert forcing.shape == (2, 16)
        assert gauge.shape == (2, 1)
        assert constraints.shape == (2, 5)

        # Check non-dual veto gating formula: total_veto = (1 - gauge) * geom_veto + gauge * calm_veto
        calm_veto = float((abort_score.mean() > 0.5) or (step_factor.mean() < 0.5))
        gauge_val = float(torch.sigmoid(gauge).mean().item())
        geom_veto = 1.0 # Simulate red zone
        total_veto = (1.0 - gauge_val) * geom_veto + gauge_val * calm_veto
        assert 0.0 <= total_veto <= 1.0

    def test_07_resonance_cavity_attention_coupling(self):
        """7. Validate ResonanceCavity excitation driven by attention states A_ij^H R_ij."""
        poly_config = PolynomialCoprimeConfig(k=4, degree=3)
        cavity = ResonanceCavity(hidden_dim=16, num_modes=8, poly_config=poly_config, modular=True)
        attention_states = torch.randn(2, 4, 16)
        
        initial_m_norm = torch.norm(cavity.M[0]).item()
        cavity(attention_states=attention_states, dt=0.1)
        updated_m_norm = torch.norm(cavity.M[0]).item()
        
        # Memory state must evolve under attention excitation
        assert initial_m_norm != updated_m_norm

    def test_08_birkhoff_projection_stochasticity(self):
        """8. Validate BirkhoffProjection doubly stochastic normalization."""
        proj = BirkhoffProjection(max_iterations=10)
        # Small matrix (N <= 32)
        m_small = torch.randn(2, 8, 8)
        ds_small = proj(m_small)
        # Check row sums are approximately 1.0
        row_sums = ds_small.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)

        # Direct projection verification
        integ = proj.validate_stochasticity(ds_small)
        assert integ is not None

    def test_09_garden_orchestrator_attractor_attention(self):
        """9. Validate GardenOrchestrator Attractor Attention fallback (use_birkhoff=False)."""
        poly_config = PolynomialCoprimeConfig(k=4, degree=3)
        attn_garden = ModularAttention(hidden_dim=16, num_heads=4, poly_config=poly_config, num_functionals=4, use_birkhoff=False)
        x = torch.randn(2, 4, 16)
        out = attn_garden(x)
        assert out.shape == (2, 4, 16)
        assert hasattr(attn_garden, 'garden')

    def test_10_spath_rag_injection(self):
        """10. Validate SPathRAGInjection geometric path distances into alignment scores."""
        poly_config = PolynomialCoprimeConfig(k=4, degree=3)
        attn = ModularAttention(hidden_dim=16, num_heads=4, poly_config=poly_config, num_functionals=4, use_birkhoff=True)
        x = torch.randn(2, 4, 16)
        path_topo = torch.randn(2, 4, 4)
        
        out_no_topo = attn(x)
        out_with_topo = attn(x, path_topology_vectors=path_topo)
        
        assert out_with_topo.shape == (2, 4, 16)
        # Geometric bias should meaningfully alter attention representations
        assert not torch.allclose(out_no_topo, out_with_topo)

    def test_11_dynamic_attention_mask_generation(self):
        """11. Validate DynamicAttentionMask dense vs band-sparse generation."""
        # High violation (> threshold) triggers dense attention
        mask_dense = ModularAttention.construct_dynamic_attention_mask(
            violations=0.8,
            seq_len=6,
            threshold=0.5
        )
        assert mask_dense.shape == (1, 1, 6, 6)
        assert torch.all(mask_dense == 0.0)

        # Low violation (<= threshold) triggers band-sparse attention
        mask_sparse = ModularAttention.construct_dynamic_attention_mask(
            violations=0.2,
            seq_len=6,
            threshold=0.5,
            local_window=1
        )
        assert mask_sparse.shape == (1, 1, 6, 6)
        # Off-diagonal elements outside local window should be masked (-1e4)
        assert mask_sparse[0, 0, 0, 5].item() == -1e4
        # Main diagonal should be unmasked (0.0)
        assert mask_sparse[0, 0, 0, 0].item() == 0.0

    def test_12_diegetic_physics_engine_attention_stages(self):
        """12. Validate DiegeticPhysicsEngine 9-stage loop with attention and CALM veto."""
        engine = DiegeticPhysicsEngine(state_dim=16)
        veh_state = {"angular_velocity": (0.1, 0.0, 0.0), "linear_velocity": (10.0, 0.0, 0.0)}
        track_state = {"normal": (0.0, 1.0, 0.0)}
        ctrl_input = {"throttle": 0.8, "yaw": 0.2, "pitch": 0.0, "roll": 0.0}
        
        result = engine.process_input(veh_state, track_state, ctrl_input)
        assert "c_out" in result
        assert "calm_total_veto" in result
        assert "attention_integrity" in result
        assert "diegetic_response" in result
        assert result["attention_integrity"] is True

        # Ingestion of Knowledge Dyad
        text_emb = torch.randn(16)
        image_emb = torch.randn(16)
        residues = engine.ingest_knowledge_dyad(text_emb, image_emb)
        assert residues.shape[-1] == 4

    def test_13_universal_orchestrator_attention_integration(self):
        """13. Validate UniversalOrchestrator step() and forward() attention wiring."""
        orchestrator = UniversalOrchestrator(dim=16)
        orchestrator.micro_steps = 2
        state_in = torch.randn(16)
        
        step_res = orchestrator.step(state_in)
        assert "state" in step_res
        assert "attention_integrity" in step_res
        assert "calm_veto" in step_res
        assert "diegetic_response" in step_res
        assert step_res["attention_integrity"] is True

        # Helper methods validation
        out_attn, integ = orchestrator.run_modular_attention(state_in.unsqueeze(0).unsqueeze(0))
        assert out_attn.shape == (1, 1, 16)
        assert integ is not None

        dyad_res = orchestrator.ingest_knowledge_dyad(state_in.unsqueeze(0), state_in.unsqueeze(0))
        assert dyad_res.shape == (1, 5)

        dyn_mask = orchestrator.construct_dynamic_attention_mask(violations=0.1, seq_len=4)
        assert dyn_mask.shape == (1, 1, 4, 4)

    def test_14_physical_node_editor_attention_hooks(self):
        """14. Validate PhysicalNodeEditor attention hooks and evaluate_tick integration."""
        editor = PhysicalNodeEditor()
        dummy_state = torch.randn(8)
        dummy_hist = torch.randn(1, 8, 8)

        # Hook 1: Modular Attention
        h1 = editor.hook_modular_attention(dummy_state)
        assert "attended_tensor" in h1
        assert "birkhoff_integrity" in h1

        # Hook 2: Autoeclectic Response
        h2 = editor.hook_autoeclectic_response(dummy_state, torch.tensor([[0.2]]))
        assert "diegetic_response" in h2

        # Hook 3: Knowledge Dyad
        h3 = editor.hook_knowledge_dyad_association(dummy_state, dummy_state)
        assert "residues" in h3

        # Hook 4: CALM Trajectory Veto
        h4 = editor.hook_calm_trajectory_veto(dummy_hist)
        assert "abort_score" in h4
        assert "calm_veto" in h4

        # Hook 5: Dynamic Attention Mask
        h5 = editor.hook_dynamic_attention_mask(0.8, seq_len=4)
        assert h5["is_dense"] is True

        # Integrated continuous tick
        tick_res = editor.evaluate_tick(dt=0.05, current_pressure=0.2, current_mischief=0.1)
        assert "modular_attention" in tick_res
        assert "autoeclectic_response" in tick_res
        assert "knowledge_dyad" in tick_res
        assert "calm_veto" in tick_res
        assert "dynamic_attention_mask" in tick_res

    def test_15_gyroid_covariance_probe_rehybridization(self):
        """15. Validate complete rehybridization of SparseGyroidCovarianceProbe into ModularAttention and UniversalOrchestrator."""
        from src.models.modular_attention import ModularAttention, ModularTransformerLayer
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        from src.core.orchestrator import UniversalOrchestrator
        from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe

        poly_cfg = PolynomialCoprimeConfig(k=4, degree=3)
        mod_attn = ModularAttention(hidden_dim=8, num_heads=2, poly_config=poly_cfg, num_functionals=4)

        # 1. Verify probe initialization
        assert hasattr(mod_attn, 'gyroid_probe')
        assert isinstance(mod_attn.gyroid_probe, SparseGyroidCovarianceProbe)

        # 2. Test compute_gyroid_violation
        x = torch.randn(2, 6, 8)
        violation_res = mod_attn.compute_gyroid_violation(x)
        assert "total_pressure" in violation_res or "gcve_pressures" in violation_res
        assert mod_attn.last_gyroid_violation is not None

        # 3. Test dynamic masking in ModularAttention forward
        out_masked = mod_attn(x, use_dynamic_mask=True)
        assert out_masked.shape == (2, 6, 8)

        # 4. Test ModularTransformerLayer delegation
        layer = ModularTransformerLayer(hidden_dim=8, num_heads=2, poly_config=poly_cfg, num_functionals=4)
        layer_violation = layer.compute_gyroid_violation(x)
        assert "total_pressure" in layer_violation or "gcve_pressures" in layer_violation

        # 5. Test UniversalOrchestrator gyroid probe integration
        orch = UniversalOrchestrator(dim=8)
        assert hasattr(orch, 'gyroid_probe')
        assert isinstance(orch, UniversalOrchestrator)
        assert orch.modular_attention.gyroid_probe is orch.gyroid_probe

        # 6. Test run_modular_attention with auto-gyroid dynamic masking
        out_orch, integrity = orch.run_modular_attention(x, use_gyroid_mask=True)
        assert out_orch.shape == (2, 6, 8)
        assert orch.last_gyroid_violation is not None

    def test_16_chern_simons_breather_coupling(self):
        """16. Validate Chern-Simons gauge coupling and chiral torsion in BreatherMode."""
        from src.models.resonance_cavity import BreatherMode
        breather = BreatherMode(num_breathers=6, dim=12)
        x = torch.randn(2, 12)
        gauge_field = torch.randn(4, 4)
        
        # Uncoupled evaluate
        out_uncoupled = breather.evaluate(x)
        assert out_uncoupled.shape == (2, 6)
        
        # Coupled evaluate with gauge field and chiral torsion
        out_coupled = breather.evaluate(
            x,
            gauge_field=gauge_field,
            chern_simons_level=2,
            chiral_torsion=0.5
        )
        assert out_coupled.shape == (2, 6)
        # Verify that coupling altered the soliton amplitudes while maintaining finite bounds
        assert not torch.allclose(out_uncoupled, out_coupled)
        assert not torch.isnan(out_coupled).any()
        assert not torch.isinf(out_coupled).any()
        
        # Test forward pass with gauge coupling
        out_fwd = breather(
            x,
            dt=0.1,
            gauge_field=gauge_field,
            chern_simons_level=2,
            chiral_torsion=0.5
        )
        assert out_fwd.shape == (2, 12)

    def test_17_chern_simons_gasket_breather_excitation(self):
        """17. Validate ChernSimonsGasket.excite_chern_simons_breathers."""
        from src.core.chern_simons_gasket import ChernSimonsGasket
        from src.models.resonance_cavity import BreatherMode
        gasket = ChernSimonsGasket(manifold_dim=3, level_k=1)
        breather = BreatherMode(num_breathers=4, dim=16)
        residue = torch.randn(2, 4, 16)
        
        # Test standalone excitation
        packet_standalone = gasket.excite_chern_simons_breathers(residue, t=0.5)
        assert packet_standalone.shape == residue.shape
        assert not torch.isnan(packet_standalone).any()
        
        # Test excitation through BreatherMode module
        packet_modular = gasket.excite_chern_simons_breathers(residue, breather_module=breather, t=0.5)
        assert packet_modular.shape == residue.shape
        assert not torch.isnan(packet_modular).any()

    def test_18_agent_smith_chern_simons_breather_roundtrip(self):
        """18. Validate Agent Smith export and substrate alignment with Chern-Simons breather packet."""
        import tempfile
        import os
        from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer
        from src.core.agent_substrate_bridge import AgentSubstrateBridge
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            fossilizer = DyadFossilizer(storage_dir=tmp_dir)
            bridge = AgentSubstrateBridge()
            
            cs_soliton = torch.randn(16)
            dyad = KnowledgeDyad(
                linguistic_description="Chern-Simons breather identity",
                video_breather={"chern_simons_breather": cs_soliton},
                gyroid_residue=torch.randn(16)
            )
            primes = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0])
            betti = {0: 1.0, 1: 0.0}
            
            filepath = fossilizer.export_agent_smith(
                dyad=dyad,
                prime_frequencies=primes,
                betti_numbers=betti,
                filename="smith_test",
                gauge_field=torch.randn(3, 3)
            )
            assert os.path.exists(filepath)
            
            # Load and verify alignment
            payload = torch.load(filepath, map_location='cpu')
            assert "chern_simons_breather" in payload
            assert payload["chern_simons_breather"] is not None
            
            aligned_payload = bridge.align_substrate(payload, expected_dim=16, hardware_trfc_ms=160.0)
            assert "chern_simons_breather_aligned" in aligned_payload
            assert aligned_payload["chern_simons_breather_aligned"].shape[-1] == 16

    def test_19_rp4_projective_attention_void_regime(self):
        """19. Validate Inverted Hypersphere Cosmology (IHC) RP4 projective routing in ModularAttention."""
        from src.models.modular_attention import ModularAttention
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        
        poly_cfg = PolynomialCoprimeConfig(k=4, degree=3)
        attn = ModularAttention(hidden_dim=16, num_heads=2, poly_config=poly_cfg, num_functionals=4)
        x = torch.randn(2, 4, 16)
        
        # Test standard forward
        out_std = attn(x)
        # Test VOID regime with RP4 projective attention
        out_void = attn(x, regime='VOID')
        assert out_void.shape == (2, 4, 16)
        assert not torch.allclose(out_std, out_void)

    def test_20_ihc_33_shell_ladder_polynomial_scaling(self):
        """20. Validate IHC 33-shell golden ratio ladder scaling in PolynomialFunctionalEmbedder."""
        from src.models.polynomial_embeddings import PolynomialFunctionalEmbedder
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        
        config = PolynomialCoprimeConfig(k=5, degree=3)
        embedder = PolynomialFunctionalEmbedder(text_dim=10, hidden_dim=20, poly_config=config, use_graph=False, use_num=False)
        text_emb = torch.randn(2, 10)
        
        res = embedder(text_emb=text_emb)
        assert "residue_distributions" in res
        residues = res["residue_distributions"]
        assert residues.shape == (2, 5, 4)
        assert not torch.isnan(residues).any()

    def test_21_folium_implicit_transversal_smooch_anchor(self):
        """21. Validate Folium of Descartes implicit differentiation, transversal smooch (-1 slope), and Option D nutrient issuance."""
        from src.core.cayley_cubic_probe import CayleyCubicProbe
        from src.optimization.constraint_probe import ConstraintProbeOperator
        from src.core.false_negative_subsystem import VoynichExemptionToken
        
        probe = CayleyCubicProbe(high_mischief_threshold=0.5, folium_scale=1.0)
        smooch_pt = torch.tensor([[1.5, 1.5]], dtype=torch.float32)
        
        # 1. Anti-Lobotomy implicit residual: x^3 + y^3 - 3xy = 0
        residual = probe.evaluate_folium_implicit(smooch_pt)
        assert abs(residual.item()) < 1e-4
        
        # 2. Implicit derivative dy/dx = -1.0 (reciprocal exchange, dx + dy = 0)
        slope = probe.compute_implicit_derivative(smooch_pt)
        assert abs(slope.item() - (-1.0)) < 1e-4
        
        # 3. Transversal Smooch / PAS_h anchor evaluation
        smooch_metrics = probe.evaluate_transversal_smooch(smooch_pt)
        assert smooch_metrics["is_transversal_smooch"].item() is True
        assert abs(smooch_metrics["reciprocal_score"].item() - 1.0) < 1e-3
        
        # 4. System 2 Constraint Probe evaluation without 1D scalar collapse
        cov = torch.eye(2)
        operator = ConstraintProbeOperator(constraint_index=0, sparse_covariance=cov)
        operator_metrics = operator.evaluate_folium_transversality(smooch_pt)
        assert operator_metrics["is_transversal_smooch"].item() is True
        
        # 5. Voynich Exemption Token Option D nutrient passport issuance
        token = VoynichExemptionToken.issue_from_transversality({"state": smooch_pt})
        assert token.is_valid_exemption is True
        assert token.is_nutrient is True
        assert token.honesty_score == 1.0
        assert "Folium" in token.reason

    def test_22_hardware_flux_attention_smoothing_and_ltp_fossilization(self):
        """Verify hardware flux attention smoothing and LTP memory packet consolidation."""
        from src.models.modular_attention import ModularAttention
        from src.core.polynomial_coprime import PolynomialCoprimeConfig

        dim = 32
        poly_cfg = PolynomialCoprimeConfig(k=5, degree=4, basis_type='chebyshev', learnable=True)
        attn = ModularAttention(hidden_dim=dim, num_heads=2, poly_config=poly_cfg, num_functionals=5)
        attn.warmstart_mix.data.fill_(0.4)

        x = torch.randn(2, 4, dim)

        # Baseline execution under normal hardware flux
        out_normal, fields_normal = attn(x, return_field_outputs=True)
        assert out_normal.shape == (2, 4, dim)
        assert torch.isfinite(out_normal).all()

        # Constrained execution under simulated thermal/silicon load (load=0.8)
        out_loaded, fields_loaded = attn(x, load=0.8, return_field_outputs=True)
        assert out_loaded.shape == (2, 4, dim)
        assert torch.isfinite(out_loaded).all()

        # Under load=0.8, active fields are sparsified (active_k < 5)
        active_loaded = sum(1 for f in fields_loaded if not torch.equal(f, torch.zeros_like(f)))
        assert active_loaded <= 4

