import pytest
import torch
import math

from src.core.adaptive_skeleton_harness import (
    AdaptiveSkeletonHarness,
    AmbulatoryClass,
    EnemySubtype
)
from src.core.invariants import get_prime_ladder
from src.ui.voxelboxter_simulation import StructuralGraph, RigidBody


def test_harness_initialization_and_existing_components():
    state_dim = 16
    harness = AdaptiveSkeletonHarness(state_dim=state_dim, device="cpu")
    
    assert harness.state_dim == state_dim
    assert harness.modular_rns is not None
    assert harness.fossilizer is not None
    assert harness.chern_simons is not None
    assert harness.substrate_bridge is not None
    assert harness.resonance_ladder is not None
    assert harness.chern_simons.gauge_field.shape == (state_dim, state_dim)


def test_compute_affordance_gating():
    harness = AdaptiveSkeletonHarness(state_dim=16, device="cpu")
    affordance = harness.evaluate_compute_affordance()
    
    assert "local_affordance" in affordance
    assert "p2p_affordance" in affordance
    assert "combined_budget" in affordance
    assert "can_deep_retrieve" in affordance
    assert "can_structural_compile" in affordance
    assert 0.0 <= affordance["local_affordance"] <= 1.0


def test_agent_smith_fossil_retrieval_no_hardcoded_primes():
    harness = AdaptiveSkeletonHarness(state_dim=16, device="cpu")
    
    # In-situ retrieval when no file matches
    fossil_data = harness.retrieve_agent_smith_fossil(
        archetype_tag="nonexistent_test_archetype",
        demand_dim=16,
        search_dirs=[]
    )
    
    assert "meta_state" in fossil_data
    assert "gauge_field" in fossil_data
    assert "prime_frequencies" in fossil_data
    assert "pot_identity_crt" in fossil_data
    
    # Verify no hardcoded (3, 5, 7) fallback
    pot_id = fossil_data["pot_identity_crt"]
    assert pot_id is not None
    assert isinstance(pot_id, (tuple, list))
    assert pot_id != (3, 5, 7)
    
    # Verify prime frequencies are derived dynamically via get_prime_ladder
    primes = get_prime_ladder(16, device=torch.device("cpu"))
    assert torch.allclose(fossil_data["prime_frequencies"].cpu(), primes)


def test_multidimensional_gauge_functional_and_morton_encoding():
    harness = AdaptiveSkeletonHarness(state_dim=16, device="cpu")
    coord = torch.randn(16)
    
    transformed, morton_keys = harness.apply_multidimensional_gauge_functional(coord)
    
    assert transformed.shape == coord.shape
    assert len(morton_keys) > 0
    # Verify all Morton keys are valid 30-bit integers
    for code in morton_keys:
        assert isinstance(code, int)
        assert 0 <= code <= 0x3FFFFFFF


def test_modular_algebra_harmonics_dynamic_primes():
    harness = AdaptiveSkeletonHarness(state_dim=16, device="cpu")
    coord = torch.ones(16)
    
    # 1. Automatic dynamic CRT residue derivation
    modulated_auto = harness.apply_modular_algebra_harmonics(coord)
    assert modulated_auto.shape == coord.shape
    assert not torch.allclose(modulated_auto, coord)
    
    # 2. Specific dynamic residue tuple
    custom_residues = (11, 13, 17)
    modulated_custom = harness.apply_modular_algebra_harmonics(coord, crt_identity=custom_residues)
    assert modulated_custom.shape == coord.shape
    assert not torch.allclose(modulated_custom, coord)


def test_structural_algebra_functional_compilation():
    harness = AdaptiveSkeletonHarness(state_dim=16, device="cpu")
    skeleton = torch.randn(16)
    morton_keys = [12345, 67890, 11111]
    
    graph = harness.apply_structural_algebra(skeleton, morton_keys)
    assert isinstance(graph, StructuralGraph)
    assert len(graph.blocks) > 0


def test_compile_to_voxelboxter_construct_wager_2_split():
    harness = AdaptiveSkeletonHarness(state_dim=16, device="cpu")
    construct = harness.compile_to_voxelboxter_construct(
        ambulatory_class=AmbulatoryClass.BIPED,
        enemy_subtype=EnemySubtype.DEMIURGIC_TITAN,
        difficulty_scale=1.2,
        bone_ratio=0.8
    )
    
    assert "graph" in construct
    assert "rigid_body" in construct
    assert "skeleton" in construct
    assert "morton_codes" in construct
    assert "fossil_split" in construct
    assert isinstance(construct["graph"], StructuralGraph)
    assert isinstance(construct["rigid_body"], RigidBody)
    
    # Verify Wager #2 (80/20 Fossilization Split)
    fossil_split = construct["fossil_split"]
    assert "bone_ratio" in fossil_split
    assert math.isclose(fossil_split["bone_ratio"], 0.8, abs_tol=0.2)
    assert construct["bone_blocks"] > 0
    assert construct["flesh_blocks"] >= 0


def test_pull_one_shot_memory_systems():
    harness = AdaptiveSkeletonHarness(state_dim=16, device="cpu")
    vec = torch.randn(16)
    
    # Test Dyad Fossilization
    fossil_vec = harness.pull_one_shot_memory("fossilization", vec)
    assert fossil_vec.shape == vec.shape
    
    # Test Chiral Residue Cache (Honest Jitter)
    chiral_vec = harness.pull_one_shot_memory("chiral_cache", vec)
    assert chiral_vec.shape == vec.shape
    
    # Test Manifold Clock Breathing
    clock_vec = harness.pull_one_shot_memory("manifold_clock", vec)
    assert clock_vec.shape == vec.shape
    
    # Test Breather Soliton (Dynamic primes)
    soliton_vec = harness.pull_one_shot_memory("breather_soliton", vec)
    assert soliton_vec.shape == vec.shape


def test_build_enemy_rig_full_integration():
    harness = AdaptiveSkeletonHarness(state_dim=16, device="cpu")
    
    enemy = harness.build_enemy_rig(
        subtype=EnemySubtype.ABSTRACTED_GLITCH,
        ambulatory_class=AmbulatoryClass.BIPED,
        difficulty_scale=1.5
    )
    
    assert enemy["subtype"] == EnemySubtype.ABSTRACTED_GLITCH
    assert enemy["ambulatory_class"] == AmbulatoryClass.BIPED
    assert enemy["max_health"] == 250.0 * 1.5
    assert enemy["damage_type"] == "phase_noise"
    assert enemy["skeleton"] is not None
    assert "morton_codes" in enemy
    assert "fossil_telemetry" in enemy
    assert "compute_affordance" in enemy
    if enemy["compute_affordance"]["can_structural_compile"]:
        assert enemy["voxel_construct"] is not None
