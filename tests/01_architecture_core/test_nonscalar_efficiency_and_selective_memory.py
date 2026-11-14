"""
Unit and integration tests for non-scalar serving efficiency (DeepSeek style)
and Latent 3D Gaussian selective memory updates (L3GWorld style).
"""

import os
import shutil
import tempfile
import time
import pytest
import torch

from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
from src.core.garden_statistical_attractors import MemoryPalace
from src.core.invariants import (
    compute_meliponini_effective_packing_fraction,
    evaluate_percolation_regime,
    PHI_RCP,
    PHI_APIS_MAX,
    PERCOLATION_2D_SQUARE
)


class TestDyadFossilizerInMemoryCache:
    """Tests DeepSeek-style in-memory caching and bounded retrieval in DyadFossilizer."""

    @pytest.fixture
    def temp_storage(self):
        temp_dir = tempfile.mkdtemp(prefix="test_fossil_cache_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_population_and_retrieval_speedup(self, temp_storage):
        dim = 16
        fossilizer = DyadFossilizer(storage_dir=temp_storage, feature_dim=dim, memory_cache_capacity=64)

        # Fossilize 20 synthetic dyads
        for i in range(20):
            tensor = torch.randn(dim)
            dyad = KnowledgeDyad(
                linguistic_description=f"query test concept {i}",
                metadata={"index": i}
            )
            fossilizer.fossilize(
                dyad=dyad,
                text_embedding=tensor
            )

        # Cache should hold all 20 entries in memory
        assert len(fossilizer._fossil_memory_cache) == 20

        # Retrieval within limit should serve from cache directly
        start_time = time.perf_counter()
        fossils = fossilizer.recover_fossils(limit=10)
        duration = time.perf_counter() - start_time

        assert len(fossils) == 10
        # In-memory retrieval of 10 items should take well under 50ms
        assert duration < 0.05

        # Confirm contents have valid structure
        for f in fossils:
            assert "residue_vector" in f
            assert "prompt_hash" in f
            assert f["residue_vector"].shape[-1] == dim

    def test_bounded_retrieval_limit(self, temp_storage):
        dim = 8
        fossilizer = DyadFossilizer(storage_dir=temp_storage, feature_dim=dim, memory_cache_capacity=32)

        for i in range(25):
            dyad = KnowledgeDyad(
                linguistic_description=f"bounded limit item {i}"
            )
            fossilizer.fossilize(
                dyad=dyad,
                text_embedding=torch.randn(dim)
            )

        # Speculative coprime gate default limit=16 check
        limited_fossils = fossilizer.recover_fossils(limit=16)
        assert len(limited_fossils) == 16


class TestMemoryPalaceSelectiveUpdate:
    """Tests L3GWorld-style Spawning vs Refinement in MemoryPalace."""

    def test_spawning_novel_region(self):
        palace = MemoryPalace(feature_dim=16)
        c1 = torch.randn(16)
        c1 = c1 / torch.norm(c1)

        room1, mode1 = palace.selective_update(c1, room_id_prefix="assoc_test", similarity_threshold=0.85)
        assert mode1 == "spawned"
        assert len(palace.rooms) == 1
        assert room1.room_id in palace.rooms

    def test_refinement_similar_region(self):
        palace = MemoryPalace(feature_dim=16)
        c1 = torch.randn(16)
        c1 = c1 / torch.norm(c1)

        # First update spawns room
        room1, mode1 = palace.selective_update(c1, room_id_prefix="assoc_test", similarity_threshold=0.85)
        assert mode1 == "spawned"
        initial_room_id = room1.room_id

        # Perturb center slightly (cosine similarity > 0.85)
        perturbation = torch.randn(16) * 0.05
        c2 = c1 + perturbation
        c2 = c2 / torch.norm(c2)

        # Second update should refine rather than spawn a duplicate room
        room2, mode2 = palace.selective_update(c2, room_id_prefix="assoc_test", similarity_threshold=0.85, residual_lr=0.2)
        assert mode2 == "refined"
        assert room2.room_id == initial_room_id
        # Total room count must remain 1 (no accumulation bloat)
        assert len(palace.rooms) == 1

        # Third update with orthogonal vector should spawn a new room
        c3 = torch.randn(16)
        # Gram-Schmidt to make c3 orthogonal to c1
        c3 = c3 - torch.dot(c3, c1) * c1
        c3 = c3 / torch.norm(c3)
        room3, mode3 = palace.selective_update(c3, room_id_prefix="assoc_test", similarity_threshold=0.85)
        assert mode3 == "spawned"
        assert len(palace.rooms) == 2

    def test_bounded_capacity_pruning(self):
        max_rooms = 5
        palace = MemoryPalace(feature_dim=16)

        # Spawn 8 distinct orthogonal or semi-random rooms
        for i in range(8):
            vec = torch.randn(16)
            vec = vec / torch.norm(vec)
            # Use very high similarity threshold to force spawning and max_rooms limit
            palace.selective_update(vec, room_id_prefix=f"room_{i}", similarity_threshold=0.999, max_rooms=max_rooms)

        # Capacity pruning must keep room count <= max_rooms
        assert len(palace.rooms) <= max_rooms


class TestMeliponiniPackingFraction:
    """Tests continuous packing fraction equation of state and percolation decoupling."""

    def test_atrophy_limits_and_monotonicity(self):
        # Atrophy = 0 (perfect coherence) -> close to Kepler sphere packing phi_max
        res_zero = compute_meliponini_effective_packing_fraction(0.0)
        assert res_zero["regime"] == "APIS_HEXAGONAL"
        assert res_zero["phi_effective"] > 0.70
        assert res_zero["phi_effective"] <= PHI_APIS_MAX

        # Atrophy = 1 (complete atrophy) -> close to Random Close Packing phi_RCP
        res_one = compute_meliponini_effective_packing_fraction(1.0)
        assert res_one["regime"] == "MELIPONINI_CERUMEN"
        assert res_one["phi_effective"] < 0.68
        assert res_one["phi_effective"] >= PHI_RCP

        # Monotonically decreasing
        phi_prev = 1.0
        for a_step in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            res = compute_meliponini_effective_packing_fraction(a_step)
            assert res["phi_effective"] <= phi_prev + 1e-6
            phi_prev = res["phi_effective"]

    def test_percolation_decoupling(self):
        res = compute_meliponini_effective_packing_fraction(0.5)
        assert res["decoupled_from_percolation"] is True
        assert res["percolation_site_pc"] == PERCOLATION_2D_SQUARE

        # Discrete percolation check
        perc_below = evaluate_percolation_regime(0.50)
        assert perc_below["is_spanning_cluster"] is False
        assert perc_below["regime"] == "MELIPONINI_ISOLATED_CERUMEN"

        perc_above = evaluate_percolation_regime(0.65)
        assert perc_above["is_spanning_cluster"] is True
        assert perc_above["regime"] == "APIS_SPANNING_HONEYCOMB"
