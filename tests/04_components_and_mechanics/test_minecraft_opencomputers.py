"""
Unit tests for OpenComputers and ComputerCraft world data ingestion in MinecraftIngestionPipeline.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
import torch

from src.data.minecraft_ingestor import (
    MinecraftIngestionPipeline,
    CodecConfig,
    PolynomialCoprimeConfig,
    get_block_registry_code,
)


@pytest.fixture
def temp_world_dir():
    test_dir = tempfile.mkdtemp(prefix="mc_test_world_")
    world_path = Path(test_dir)
    try:
        yield world_path
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_block_registry_codes_opencomputers():
    """Verify that OpenComputers and ComputerCraft block names receive distinct harmonic subspace codes."""
    code_air = get_block_registry_code("minecraft:air")
    code_stone = get_block_registry_code("minecraft:stone")
    code_oc_case = get_block_registry_code("opencomputers:case1")
    code_oc_screen = get_block_registry_code("opencomputers:screen2")
    code_cc_computer = get_block_registry_code("computercraft:computer_advanced")

    assert 0 <= code_air < 1024
    assert 0 <= code_stone < 1024
    # OpenComputers blocks are mapped to the [512, 1023] subspace
    assert 512 <= code_oc_case < 1024
    assert 512 <= code_oc_screen < 1024
    # ComputerCraft blocks are mapped to the [256, 511] subspace
    assert 256 <= code_cc_computer < 512


def test_opencomputers_tile_entity_extraction():
    """Verify parsing of OpenComputers tile entities containing EEPROM, screen buffer, and items."""
    config = CodecConfig(K=5, n=4)
    poly_config = PolynomialCoprimeConfig(k=5)
    pipeline = MinecraftIngestionPipeline(config, poly_config)

    mock_te = {
        "id": "opencomputers:case2",
        "oc:data": {
            "node": {"address": "b47a13d2-349c-4822-8703-a417551fa123"},
            "oc:label": "Mainframe Node 01",
            "oc:eeprom": {
                "oc:label": "Lua BIOS v1.0",
                "oc:eeprom": list(b'local component = require("component")\nprint("BIOS Init")'),
            },
            "oc:buffer": {
                "lines": [
                    "OpenOS 1.7.5 (booted)",
                    "Ready for command:",
                ]
            },
            "Items": [
                {
                    "id": "opencomputers:storage",
                    "tag": {
                        "oc:data": {
                            "oc:fs.label": "System Drive",
                            "node": {"address": "hdd-uuid-999"},
                        }
                    }
                }
            ]
        }
    }

    mock_chunk_nbt = {
        "Level": {
            "TileEntities": [mock_te]
        }
    }

    texts = pipeline._extract_nbt_texts(mock_chunk_nbt)
    assert len(texts) > 0

    all_text = "\n".join(texts)
    assert "Mainframe Node 01" in all_text
    assert "b47a13d2-349c-4822-8703-a417551fa123" in all_text
    assert 'local component = require("component")' in all_text
    assert "OpenOS 1.7.5 (booted)" in all_text
    assert "System Drive" in all_text
    assert "hdd-uuid-999" in all_text


def test_opencomputers_inworld_filesystem_extraction(temp_world_dir):
    """Verify recursive scanning and extraction of OpenComputers and ComputerCraft filesystems."""
    # Create OpenComputers directory structure
    uuid_dir = temp_world_dir / "opencomputers" / "e9b21a30-1111-2222-3333-444455556666"
    usr_bin = uuid_dir / "usr" / "bin"
    usr_bin.mkdir(parents=True)

    init_script = uuid_dir / "init.lua"
    init_script.write_text("local gpu = component.gpu\ngpu.set(1, 1, 'Booting System...')\n", encoding="utf-8")

    custom_prog = usr_bin / "quarry.lua"
    custom_prog.write_text("local robot = require('robot')\nwhile robot.detect() do robot.swing() end\n", encoding="utf-8")

    # Create ComputerCraft directory structure
    cc_dir = temp_world_dir / "computer" / "42"
    cc_dir.mkdir(parents=True)
    cc_startup = cc_dir / "startup.lua"
    cc_startup.write_text("print('ComputerCraft 42 online')\nredstone.setOutput('top', true)\n", encoding="utf-8")

    config = CodecConfig(K=5, n=4)
    poly_config = PolynomialCoprimeConfig(k=5)
    pipeline = MinecraftIngestionPipeline(config, poly_config)

    scripts = pipeline.extract_inworld_computer_scripts(temp_world_dir)
    assert len(scripts) == 3

    filenames = {s["filename"] for s in scripts}
    assert "init.lua" in filenames
    assert "quarry.lua" in filenames
    assert "startup.lua" in filenames

    oc_scripts = [s for s in scripts if s["source_mod"] == "opencomputers"]
    assert len(oc_scripts) == 2
    for s in oc_scripts:
        assert s["component_uuid"] == "e9b21a30-1111-2222-3333-444455556666"

    cc_scripts = [s for s in scripts if s["source_mod"] == "computercraft"]
    assert len(cc_scripts) == 1
    assert cc_scripts[0]["computer_id"] == "42"


def test_ingest_minecraft_world_with_opencomputers(temp_world_dir):
    """Verify end-to-end world ingestion produces valid non-commutative dyads from OpenComputers world."""
    # Write OpenComputers in-world code
    oc_dir = temp_world_dir / "opencomputers" / "miner_bot_uuid"
    oc_dir.mkdir(parents=True)
    (oc_dir / "init.lua").write_text("print('OpenOS Kernel Initialized')\nlocal s = component.proxy(component.list('screen')())", encoding="utf-8")
    (oc_dir / "miner.lua").write_text("local nav = require('navigation')\nnav.findWaypoints(64)\n", encoding="utf-8")

    config = CodecConfig(K=5, n=4)
    poly_config = PolynomialCoprimeConfig(k=5)
    pipeline = MinecraftIngestionPipeline(config, poly_config)

    results = pipeline.ingest_minecraft_world(temp_world_dir)

    assert results["opencomputers_scripts_extracted"] == 2
    assert results["combined_residue"] is not None
    assert results["combined_residue"].shape == (5, 4, 4)
    # The residue matrices should be non-zero and Lie-group regularized
    assert not torch.allclose(results["combined_residue"], torch.zeros_like(results["combined_residue"]))
    assert results["commutativity_gap"] >= 0.0
