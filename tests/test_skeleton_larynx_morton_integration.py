import pytest
import torch
import math
from src.core.adaptive_skeleton_harness import (
    AdaptiveSkeletonHarness,
    InhibitionStabilizedTopologicalNetwork,
    AmbulatoryClass,
    EnemySubtype
)
from src.core.archetype_engines import BardoRouter
from src.models.diegetic_heads import ResonanceLarynx
from src.models.modular_attention import ModularAttention
from src.core.polynomial_coprime import PolynomialCoprimeConfig


def test_inhibition_stabilized_network_dynamics():
    """Verify ISN maintains E/I stability and defect diffusion across iterations."""
    isn = InhibitionStabilizedTopologicalNetwork(num_archetypes=7, state_dim=32)
    
    # Run multiple steps under external stimulus
    stim = torch.randn(7).abs() * 0.5
    for _ in range(10):
        metrics = isn.step(external_stimulus=stim, gyroid_potential=0.2, dt=0.05)
        
    assert "excitation" in metrics
    assert "inhibition" in metrics
    assert "ei_ratio" in metrics
    assert (metrics["excitation"] > 0.0).all()
    assert (metrics["inhibition"] > 0.0).all()
    # Check that network did not blow up to infinity
    assert (metrics["excitation"] < 10.0).all()
    assert (metrics["inhibition"] < 10.0).all()


def test_skeleton_harness_larynx_attention_morton_modulation():
    """
    Verify PAL/NTSC color subcarrier analogy:
    Luminance = 3D skeleton geometry along Morton Z-order curve
    Chrominance = ResonanceLarynx (U) + ModularAttention (V) modulated on omega_micro
    """
    state_dim = 32
    harness = AdaptiveSkeletonHarness(state_dim=state_dim, device="cpu")
    
    # Bind larynx and attention
    larynx = ResonanceLarynx(hidden_dim=state_dim)
    poly_cfg = PolynomialCoprimeConfig(moduli=[2, 3, 5, 17, 41])
    attn = ModularAttention(hidden_dim=state_dim, num_heads=4, poly_config=poly_cfg, num_functionals=3)
    
    harness.bind_larynx(larynx)
    harness.bind_modular_attention(attn)
    
    # Base skeleton kinematic tensor
    skeleton = torch.randn(state_dim)
    
    # Modulate onto Morton stream
    stream_packet = harness.modulate_larynx_attention_morton_stream(
        skeleton_tensor=skeleton,
        carrier_freq=137.0,
        subcarrier_gain=0.05
    )
    
    assert "morton_keys" in stream_packet
    assert len(stream_packet["morton_keys"]) > 0
    assert "composite_coords" in stream_packet
    assert "luminance_skeleton" in stream_packet
    assert "u_larynx" in stream_packet
    assert "v_attention" in stream_packet
    assert stream_packet["pal_alternation"] is True
    
    # Demodulate packet
    demod = harness.demodulate_morton_stream(stream_packet)
    assert "recovered_luminance" in demod
    assert "recovered_u_larynx" in demod
    assert "recovered_v_attention" in demod
    
    # Backward compatibility: baseband luminance is preserved with high fidelity (> 0.95)
    fidelity = demod["demodulation_fidelity"]
    assert fidelity > 0.95, f"Demodulation fidelity too low: {fidelity}"


def test_build_superposed_rig_with_isn_and_subcarrier():
    """Verify build_superposed_rig integrates ISN and optional Morton subcarrier."""
    state_dim = 32
    harness = AdaptiveSkeletonHarness(state_dim=state_dim, device="cpu")
    
    weights = {"archetype_sovereign": 1.0}
    rig = harness.build_superposed_rig(
        weights=weights,
        ambulatory_class=AmbulatoryClass.BIPED,
        enemy_subtype=EnemySubtype.EXISTENTIAL_SENTIENT,
        modulate_morton_subcarrier=True
    )
    
    assert rig is not None
    assert rig.shape[-1] == state_dim
    assert "isn_telemetry" in harness.last_rig_telemetry
    assert "subcarrier_stream" in harness.last_rig_telemetry
    assert harness.last_rig_telemetry["subcarrier_stream"] is not None


def test_compile_to_voxelboxter_construct_with_subcarrier():
    """Verify compile_to_voxelboxter_construct includes subcarrier and ISN telemetry."""
    state_dim = 32
    harness = AdaptiveSkeletonHarness(state_dim=state_dim, device="cpu")
    
    construct = harness.compile_to_voxelboxter_construct(
        ambulatory_class=AmbulatoryClass.BIPED,
        enemy_subtype=EnemySubtype.NONE,
        modulate_morton_subcarrier=True
    )
    
    assert "graph" in construct
    assert "rigid_body" in construct
    assert "morton_codes" in construct
    assert "isn_telemetry" in construct
    assert "subcarrier_stream" in construct
    assert construct["subcarrier_stream"] is not None


def test_bardo_router_nondual_isn_soft_routing():
    """Verify BardoRouter supports soft continuous superposition rather than only hard argmax snapping."""
    state_dim = 32
    router = BardoRouter(state_dim=state_dim, num_archetypes=6)
    isn = InhibitionStabilizedTopologicalNetwork(num_archetypes=6, state_dim=state_dim)
    
    state = torch.randn(state_dim)
    
    # Default: hard snap
    hard_out = router(state, use_isn_soft_routing=False)
    
    # Soft non-dual routing via ISN balance
    soft_out = router(state, use_isn_soft_routing=True, isn_network=isn)
    
    assert hard_out.shape == (state_dim,)
    assert soft_out.shape == (state_dim,)
    # Soft output is a continuous superposition and will generally differ from hard snapping
    assert not torch.allclose(hard_out, soft_out)
