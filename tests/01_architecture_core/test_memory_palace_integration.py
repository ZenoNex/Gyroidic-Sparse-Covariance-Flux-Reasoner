#!/usr/bin/env python3
"""
Unit tests for Memory Palace (MEMPALACE) integration and
non-invasive immunological memory hybridization.
"""

import pytest
import torch
from src.core.garden_statistical_attractors import (
    MemoryPalace,
    MemoryPalaceRoom,
    CerumenPotIsolation,
    _slerp
)
from src.core.orchestrator import UniversalOrchestrator


def test_slerp_interpolation():
    """Verify that SLERP preserves spherical geometry on high-dimensional vectors."""
    v0 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0])
    
    mid = _slerp(v0, v1, t=0.5)
    # Norm should be preserved (~1.0)
    assert abs(torch.norm(mid).item() - 1.0) < 1e-4
    # Dot product with v0 and v1 should be equal (~cos(45 deg) = 0.7071)
    dot0 = torch.dot(mid, v0).item()
    dot1 = torch.dot(mid, v1).item()
    assert abs(dot0 - dot1) < 1e-4
    assert abs(dot0 - 0.7071) < 1e-3


def test_memory_palace_room_planting_and_isolation():
    """Verify Meliponini Cerumen Barrier: isolated S^2 chambers with detached tensors."""
    palace = MemoryPalace(feature_dim=64)
    
    center = torch.randn(64, requires_grad=True)
    room = palace.plant_room(
        room_id="room_grief_boundary",
        center=center,
        variance=1.2,
        residue_tuple=(3, 7, 11),
        seam_tension=0.85,
        metadata={"category": "emotional_boundary"}
    )
    
    assert room.room_id == "room_grief_boundary"
    assert "room_grief_boundary" in palace.rooms
    # Tensor center must be detached (zero autograd gradient sharing)
    assert not room.center.requires_grad
    assert room.variance == 1.2
    assert room.residue_tuple == (3, 7, 11)


def test_room_breeding_slerp():
    """Verify GANBREEDER tag-based room breeding via great-circle SLERP."""
    palace = MemoryPalace(feature_dim=32)
    
    c1 = torch.zeros(32)
    c1[0] = 1.0
    c2 = torch.zeros(32)
    c2[1] = 1.0
    
    palace.plant_room("room_a", center=c1, residue_tuple=(2, 3), seam_tension=0.5)
    palace.plant_room("room_b", center=c2, residue_tuple=(5, 7), seam_tension=0.9)
    
    bred = palace.breed_rooms(
        new_room_id="room_hybrid",
        base_room_id="room_a",
        donor_weights={"room_b": 1.0}
    )
    
    assert bred.room_id == "room_hybrid"
    assert "room_hybrid" in palace.rooms
    # Bred room should inherit composite residues
    assert 2 in bred.residue_tuple and 5 in bred.residue_tuple
    assert bred.metadata["breeding_method"] == "SLERP_great_circle"


def test_outlier_seed_planting():
    """Verify outlier seed planting at extremal-kappa transversal points."""
    palace = MemoryPalace(feature_dim=32)
    
    c_outlier = torch.randn(32)
    seed = palace.plant_outlier_seed(
        seed_id="outlier_singularity_1",
        center=c_outlier,
        seam_tension=3.2,
        residue_tuple=(13, 17)
    )
    
    assert seed.is_outlier_seed is True
    assert seed.variance == 0.5
    assert seed.seam_tension == 3.2
    
    diag = palace.get_diagnostics()
    assert diag["outlier_seeds"] == 1
    assert diag["total_rooms"] == 1


def test_selective_puncture_event():
    r"""Verify discrete Selective Puncture Operator (S^2 -> S^2 \ {pt})."""
    palace = MemoryPalace(feature_dim=16)
    
    target_center = torch.ones(16)
    palace.plant_room("chamber_target", center=target_center, variance=1.0)
    
    # 1. High pull candidate (close to center)
    high_energy_query = target_center + 0.05 * torch.randn(16)
    puncture_ok, room, pull = palace.selective_puncture(high_energy_query, resonance_threshold=0.5)
    assert puncture_ok is True
    assert room is not None
    assert room.room_id == "chamber_target"
    assert room.sealed is False
    assert pull > 0.5
    
    # Verify selective puncture dimension masking (prevents diffusion toxins)
    punctured_state = room.get_punctured_state(high_energy_query)
    # Check that punctured state has some zeroed dimensions according to (i + sum(r)) % 3 == 0
    assert torch.any(punctured_state == 0.0)
    
    # 2. Far candidate (weak resonance) -> Puncture refused (diffusion toxin blocked)
    far_query = target_center + 10.0 * torch.randn(16)
    puncture_refused, room_none, far_pull = palace.selective_puncture(far_query, resonance_threshold=0.5)
    assert puncture_refused is False
    assert room_none is None
    assert far_pull < 0.5


def test_immunological_memory_query_and_orchestrator_sovereign_exemption():
    """
    Verify non-invasive immunological memory read in UniversalOrchestrator:
    Candidate matching inoculated scar receives Sovereign Exemption without lobotomy.
    """
    dim = 32
    orchestrator = UniversalOrchestrator(dim=dim)
    
    assert hasattr(orchestrator, 'memory_palace')
    assert orchestrator.memory_palace is not None
    
    # Plant an inoculated paradox scar in the Memory Palace
    scar_center = torch.randn(dim)
    orchestrator.plant_memory_palace_room(
        room_id="inoculated_russell_paradox",
        center=scar_center,
        variance=1.5,
        residue_tuple=(11, 13),
        seam_tension=1.8,
        metadata={"antibody": "russell_paradox_v1"}
    )
    
    # Query candidate that aligns with the inoculated paradox
    candidate = scar_center + 0.1 * torch.randn(dim)
    
    subversion, toxicity = orchestrator.resolve_containment_pressure(candidate)
    
    # Should receive Sovereign Exemption: toxicity suppressed to 0.01, subversion boosted
    assert toxicity.item() == pytest.approx(0.01, abs=1e-3)
    assert subversion.item() > 0.0
    
    # ZERO LOBOTOMY RISK: Ensure the room center and variance were NOT modified by the query
    stored_room = orchestrator.memory_palace.rooms["inoculated_russell_paradox"]
    assert torch.allclose(stored_room.center, scar_center, atol=1e-5)
    assert stored_room.variance == 1.5


def test_room_breeding_with_superposed_tag_stacker():
    """Verify breeding rooms drawing donors from SuperposedTagStacker."""
    from src.core.superposed_tag_stacker import SuperposedTagStacker
    
    dim = 32
    palace = MemoryPalace(feature_dim=dim)
    base_center = torch.zeros(dim)
    base_center[0] = 1.0
    palace.plant_room("room_base", center=base_center, residue_tuple=(3, 7))
    
    stacker = SuperposedTagStacker(state_dim=dim)
    donor_vec = torch.zeros(dim)
    donor_vec[1] = 1.0
    # Add textbook-quality context
    stacker.add_tag("quantum_soliton", donor_vec, "Quantum soliton mechanics in a textbook topology proof.")
    
    bred = palace.breed_rooms(
        new_room_id="room_from_tag",
        base_room_id="room_base",
        donor_weights={"quantum_soliton": 0.8},
        tag_stacker=stacker
    )
    
    assert bred.room_id == "room_from_tag"
    assert "room_from_tag" in palace.rooms
    # Ensure bred center is on sphere surface (SLERP radius matches target radius)
    expected_norm = torch.norm(base_center + 0.8 * (donor_vec - base_center)).item()
    assert torch.norm(bred.center).item() == pytest.approx(expected_norm, rel=1e-3)


def test_navigate_streamline_slerp():
    """Verify following the SLERP streamline path toward a room."""
    dim = 16
    palace = MemoryPalace(feature_dim=dim)
    
    target_center = torch.zeros(dim)
    target_center[2] = 1.0
    palace.plant_room("room_dest", center=target_center)
    
    start = torch.zeros(dim)
    start[0] = 1.0
    
    path = palace.navigate_streamline(start, "room_dest", steps=4)
    assert len(path) == 5
    # First step is start
    assert torch.allclose(path[0], start, atol=1e-4)
    # Last step is target
    assert torch.allclose(path[-1], target_center, atol=1e-4)


def test_selective_puncture_heritable_trust_vault_bridge():
    """Verify selective puncture synchronizes unsealed high-survivorship residue into HeritableTrustVault."""
    from src.models.resonance_cavity import HeritableTrustVault
    
    dim = 16
    palace = MemoryPalace(feature_dim=dim)
    vault = HeritableTrustVault(table_size=32, k_dim=3)
    
    target_center = torch.ones(dim)
    palace.plant_room("chamber_trust", center=target_center, residue_tuple=(5, 7, 13))
    
    query = target_center + 0.02 * torch.randn(dim)
    puncture_ok, room, pull = palace.selective_puncture(
        query_state=query,
        room_id="chamber_trust",
        resonance_threshold=0.5,
        trust_vault=vault
    )
    
    assert puncture_ok is True
    assert room is not None
    assert room.sealed is False
    
    # Query vault for trust score of residue tuple
    res_tensor = torch.tensor([[5.0, 7.0, 13.0]])
    trust_score = vault.query(res_tensor)
    assert trust_score.item() > 0.0


def test_garden_orchestrator_basin_sync_to_memory_palace():
    """Verify GardenOrchestrator fossilized basins synchronize into MemoryPalace chambers."""
    from src.core.garden_statistical_attractors import GardenOrchestrator
    
    dim = 16
    garden = GardenOrchestrator(num_attractors=4, feature_dim=dim)
    
    # Artificial high trust scalar on attractor 0
    garden.influence_attractors.trust_scalars[0] = 2.0
    
    planted = garden.sync_fossilized_basins_to_memory_palace()
    assert "garden_basin_0" in planted
    assert "garden_basin_0" in garden.memory_palace.rooms
    assert garden.memory_palace.rooms["garden_basin_0"].metadata["source"] == "garden_fossilized_basin"


def test_meliponini_crt_residue_tuple_generation_and_masking():
    """Verify default CRT prime moduli (61, 67, 71) and 33% dimension masking on unsealed states."""
    palace = MemoryPalace(feature_dim=32)
    center = torch.ones(32) * 123.456
    
    # Plant room without explicit residue tuple to verify Meliponini generation
    room = palace.plant_room("room_crt_test", center=center)
    
    assert len(room.residue_tuple) == 3
    # Check that moduli are within [0, 60], [0, 66], [0, 70]
    assert 0 <= room.residue_tuple[0] < 61
    assert 0 <= room.residue_tuple[1] < 67
    assert 0 <= room.residue_tuple[2] < 71
    
    # Selective puncture unsealing
    puncture_ok, punctured_room, pull = palace.selective_puncture(
        query_state=center,
        room_id="room_crt_test",
        resonance_threshold=0.5
    )
    assert puncture_ok is True
    # Masked state should have ~33% dimensions zeroed
    punctured_state = punctured_room.get_punctured_state(center)
    zero_count = (punctured_state == 0.0).sum().item()
    assert zero_count > 0, "Selective puncture must zero out ~33% of dimensions to prevent diffusion toxins"


def test_breather_mode_soliton_packet_storage():
    """Verify BreatherMode soliton integration and O(K) VRAM packet serialization."""
    from src.models.resonance_cavity import BreatherMode
    
    palace = MemoryPalace(feature_dim=16)
    breather = BreatherMode(num_breathers=4, dim=16)
    
    room = palace.plant_room(
        room_id="room_breather",
        center=torch.randn(16),
        breather_soliton=breather
    )
    
    assert room.breather_soliton is not None
    assert "breather_soliton" in room.metadata
    assert room.metadata["breather_soliton"]["num_modes"] == 4
    
    # Generate packet through palace helper
    soliton_t = palace.generate_breather_soliton(room.room_id, t=0.5)
    assert soliton_t is not None
    assert isinstance(soliton_t, torch.Tensor)
    assert soliton_t.shape[-1] == 4


def test_universal_orchestrator_subsystem_bridges():
    """Verify UniversalOrchestrator regimes, paradox hardening, dyadic leakage, and feature preservation."""
    orchestrator = UniversalOrchestrator(dim=16)
    
    # Test setting RNN regime
    orchestrator.set_rnn_regime("chiral")
    assert orchestrator.rnn_core.regime == "chiral"
    
    # Test dyadic leakage tracking
    task_states = torch.randn(1, 8, 16)
    gating = torch.ones(1, 8)
    leaked_states, report = orchestrator.track_dyadic_leakage(task_states=task_states, gating=gating)
    assert leaked_states.shape == task_states.shape
    assert isinstance(report, dict)
    
    # Test step execution with System 1 and System 2 through forward()
    state = torch.randn(1, 16)
    out_state, regime, routing, stacked_target = orchestrator.forward(
        state=state,
        pressure_grad=torch.randn(1, 16),
        pas_h=0.7, # triggers system 2 sync
        coherence=torch.tensor([0.9]),
        atrophy=0.9
    )
    assert out_state.shape == state.shape
    assert isinstance(regime, str)


def test_knowledge_dyad_fossilizer_export_and_tunnel():
    """Verify KnowledgeDyadFossilizer exports into MemoryPalace and tunnels stranded RP4 states."""
    import tempfile
    from src.core.knowledge_dyad_fossilizer import KnowledgeDyadFossilizer, KnowledgeDyad
    from src.core.archetype_engines import RP4ProjectiveRouter
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        fossilizer = KnowledgeDyadFossilizer(storage_dir=tmp_dir)
        palace = MemoryPalace(feature_dim=16)
        
        # Create a mock KnowledgeDyad
        dyad = KnowledgeDyad(
            timestamp=12345678.0,
            linguistic_description="paradoxical_quantum_knot",
            meta_state=torch.randn(1, 16)
        )
        prime_freqs = torch.tensor([2.0, 3.0, 5.0, 7.0])
        betti = {0: 1.0, 1: 2.0}
        
        # Test export_agent_smith with memory_palace planting
        filepath = fossilizer.export_agent_smith(
            dyad=dyad,
            prime_frequencies=prime_freqs,
            betti_numbers=betti,
            filename="smith_test.pt",
            memory_palace=palace
        )
        assert filepath.endswith(".pt")
        # Ensure room got planted in palace
        assert len(palace.rooms) == 1
        planted_room = next(iter(palace.rooms.values()))
        assert planted_room.room_id.startswith("smith_")
        assert planted_room.soliton_payload["type"] == "soliton_smith"
        
        # Test tunnel_and_plant_stranded_dyad via RP4ProjectiveRouter
        router = RP4ProjectiveRouter(state_dim=16)
        tunneled_id = fossilizer.tunnel_and_plant_stranded_dyad(
            dyad=dyad,
            router=router,
            void_friction=0.95, # > 0.8 triggers puncture
            memory_palace=palace
        )
        assert tunneled_id is not None
        assert tunneled_id in palace.rooms
        assert palace.rooms[tunneled_id].is_outlier_seed is True


