"""
Unit and integration tests for Canonical Huffman Entropy Coding and
L3GWorld-inspired Latent 3D Gaussian Memory in Voxelboxter Non-Dual Ecology.
"""

import os
import shutil
import tempfile
import pytest
import torch

from src.core.huffman_entropy import HuffmanEntropyCodec
from src.ui.voxelboxter_simulation import PointerlessOctree
from src.environment.cleanroom_mechanics import RedNet16BundledCable, CompositeVoxelConduit
from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad


class TestHuffmanEntropyCodec:
    """Tests the canonical Huffman entropy codec for theoretical correctness and compression."""

    def test_huffman_roundtrip_basic(self):
        # Skewed symbol stream (air and rock common, rare ores infrequent)
        symbols = [0] * 50 + [1] * 30 + [2] * 15 + [3] * 4 + [4] * 1
        packed_bytes, metadata = HuffmanEntropyCodec.encode_symbols(symbols)

        assert len(packed_bytes) > 0
        assert metadata["total_symbols"] == 100
        assert metadata["shannon_entropy"] > 0.0

        # Shannon source coding theorem bound: H(X) <= L_avg < H(X) + 1
        h = metadata["shannon_entropy"]
        l_avg = metadata["avg_code_length"]
        assert h <= l_avg + 1e-4
        assert l_avg < h + 1.0

        # Decoding roundtrip
        decoded = HuffmanEntropyCodec.decode_symbols(packed_bytes, metadata)
        assert decoded == symbols

    def test_huffman_rle_codec(self):
        # Long runs of repetitive symbols (e.g. cavern voids)
        data = [0] * 100 + [1] * 50 + [0] * 80 + [2] * 20
        packed_bytes, meta = HuffmanEntropyCodec.encode_rle_huffman(data)

        assert len(packed_bytes) > 0
        assert meta["raw_length"] == 250

        # Reconstructed stream must match perfectly
        reconstructed = HuffmanEntropyCodec.decode_rle_huffman(packed_bytes, meta)
        assert reconstructed == data


class TestVoxelboxterL3GWorldAndHuffman:
    """Tests L3GWorld selective latent updates and Huffman octree compression in PointerlessOctree."""

    def test_octree_huffman_export_and_load(self):
        octree = PointerlessOctree(max_depth=6)

        # Place several structured voxels
        for i in range(10):
            octree.set_voxel(i, i, i, material=1, complexity=0.5)
        for i in range(5):
            octree.set_voxel(i + 20, 5, 5, material=2, complexity=0.8)

        original_count = octree.total_voxels
        assert original_count == 15

        # Export via Huffman codec
        packed_bytes, metadata = octree.export_octree_huffman()
        assert len(packed_bytes) > 0
        assert metadata["voxel_count"] == 15

        # Restore into clean octree
        new_octree = PointerlessOctree(max_depth=6)
        loaded_count = new_octree.load_octree_huffman(packed_bytes, metadata)

        assert loaded_count == original_count
        assert new_octree.total_voxels == original_count
        # Verify Morton keys match
        for k, v in octree.morton_grid.items():
            assert k in new_octree.morton_grid
            assert new_octree.morton_grid[k] == v

    def test_selective_latent_update_spawning_vs_refinement(self):
        octree = PointerlessOctree(max_depth=6)
        coords = (10, 10, 10)
        latent1 = torch.randn(16)
        latent1 = latent1 / torch.norm(latent1)

        # 1. First visit -> Spawning
        res1 = octree.selective_latent_update(coords, latent1, material_id=1, opacity=1.0)
        assert res1["mode"] == "spawned"
        assert res1["total_nodes"] == 1
        assert res1["latent_lifted"] is True

        # 2. Second visit with similar latent -> Refinement (no duplicate node created)
        perturb = torch.randn(16) * 0.05
        latent2 = latent1 + perturb
        latent2 = latent2 / torch.norm(latent2)

        res2 = octree.selective_latent_update(coords, latent2, material_id=1, similarity_threshold=0.85, residual_lr=0.2)
        assert res2["mode"] == "refined"
        assert res2["similarity"] > 0.85
        # Total nodes must remain 1
        assert octree.total_voxels == 1

        # 3. Third visit to distinct coordinates -> Spawning
        res3 = octree.selective_latent_update((20, 20, 20), latent1, material_id=2)
        assert res3["mode"] == "spawned"
        assert octree.total_voxels == 2

    def test_hierarchical_memory_pruning(self):
        octree = PointerlessOctree(max_depth=6)
        max_extent = 8

        # Spawn 15 distinct voxels
        for i in range(15):
            feat = torch.randn(8)
            octree.selective_latent_update((i, 0, 0), feat, max_scene_extent=max_extent)

        # Extent bounding must prune oldest/excess voxels
        assert octree.total_voxels <= max_extent


class TestCleanroomConduitHuffmanTelemetry:
    """Tests Huffman telemetry compression on Cleanroom RedNet cables and composite conduits."""

    def test_rednet16_telemetry_compression(self):
        cable = RedNet16BundledCable()
        # Set realistic sparse signals (most subnets idle 0)
        cable.set_signal("white", 255)
        cable.set_signal("red", 128)
        cable.set_signal("blue", 15)

        packed_bytes, meta = cable.pack_huffman_telemetry()
        assert len(packed_bytes) > 0
        assert meta["total_symbols"] == 16

        # Decode into receiving cable
        rx_cable = RedNet16BundledCable()
        rx_cable.unpack_huffman_telemetry(packed_bytes, meta)

        assert rx_cable.get_signal("white") == 255
        assert rx_cable.get_signal("red") == 128
        assert rx_cable.get_signal("blue") == 15
        assert rx_cable.get_signal("green") == 0

    def test_composite_conduit_huffman_compression(self):
        conduit = CompositeVoxelConduit()
        conduit.has_power_conduit = True
        conduit.has_fluid_conduit = True
        conduit.power_fe_stored = 150.0
        conduit.fluid_mb_stored = 80.0

        packed_bytes, meta = conduit.pack_conduit_state_huffman()
        assert len(packed_bytes) > 0

        rx_conduit = CompositeVoxelConduit()
        rx_conduit.unpack_conduit_state_huffman(packed_bytes, meta)

        assert rx_conduit.has_power_conduit is True
        assert rx_conduit.has_fluid_conduit is True
        assert rx_conduit.has_gas_conduit is False
        assert rx_conduit.power_fe_stored == 150.0
        assert rx_conduit.fluid_mb_stored == 80.0


class TestDyadFossilizerAgentSmithHuffman:
    """Tests Agent Smith discrete syntax Huffman compression."""

    @pytest.fixture
    def temp_storage(self):
        temp_dir = tempfile.mkdtemp(prefix="test_fossil_huff_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_agent_smith_huffman_compression(self, temp_storage):
        fossilizer = DyadFossilizer(storage_dir=temp_storage, feature_dim=16)

        symbols = [61, 67, 71, 0, 100, 1, 0]
        packed, meta = fossilizer.compress_payload_huffman(symbols)
        assert len(packed) > 0
        assert meta["total_symbols"] == len(symbols)

        restored = fossilizer.decompress_payload_huffman(packed, meta)
        assert restored == symbols
