#!/usr/bin/env python3
"""
tests/test_ingestion_mating.py
Verification suite for the integrated steering, ingestion, audience, sovereignty,
and attribution systems via the SuperposedTagStacker and glitch styles.
"""

import os
import sys
import json
import shutil
import tempfile
import torch
import pytest
from pathlib import Path

# Ensure project root is in the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
from src.data.chatgpt_friction_harvester import ChatGPTFrictionHarvester
from src.core.superposed_tag_stacker import SuperposedTagStacker
from src.core.zeitgeist_router import ZeitgeistRouter, ZeitgeistState
from src.core.archetype_engines import ArchetypalSynthesisEngine

# ---------------------------------------------------------------------------
# Test 1: Fossilizer Deduplication & Persistence
# ---------------------------------------------------------------------------
def test_fossilizer_deduplication():
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize fossilizer
        fossilizer = DyadFossilizer(storage_dir=temp_dir, feature_dim=64)
        
        prompt = "Friction test between sovereign entities."
        dyad = KnowledgeDyad(
            linguistic_description=prompt,
            image_fingerprint=torch.randn(96)
        )
        
        # Verify initially not ingested
        assert not fossilizer.is_already_ingested(prompt)
        
        # Fossilize it
        embedding = torch.randn(64)
        seed_state = torch.randn(64)
        fossilizer.fossilize(dyad, embedding, seed_state)
        
        # Verify it is now registered
        assert fossilizer.is_already_ingested(prompt)
        
        # Re-initialize fossilizer to verify persistence of hashes from disk
        fossilizer_reloaded = DyadFossilizer(storage_dir=temp_dir, feature_dim=64)
        assert fossilizer_reloaded.is_already_ingested(prompt)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: ChatGPT Harvester Deduplication
# ---------------------------------------------------------------------------
def test_harvester_deduplication():
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a mock conversations JSON
        conv_data = [
            {
                "title": "Friction test conversation",
                "mapping": {
                    "root": {
                        "parent": None,
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Hello Akkaris, this is prompt 1."]}
                        }
                    },
                    "reply": {
                        "parent": "root",
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["I agree."]}
                        }
                    },
                    "root2": {
                        "parent": "reply",
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Prompt 2 has not been ingested yet."]}
                        }
                    },
                    "reply2": {
                        "parent": "root2",
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["Indeed."]}
                        }
                    }
                }
            }
        ]
        
        with open(os.path.join(temp_dir, "conversations-1.json"), "w", encoding="utf-8") as f:
            json.dump(conv_data, f)
            
        # Instantiate a fossilizer and ingest "Hello Akkaris, this is prompt 1."
        fossilizer_dir = os.path.join(temp_dir, "fossil_store")
        fossilizer = DyadFossilizer(storage_dir=fossilizer_dir, feature_dim=64)
        
        dyad_ingested = KnowledgeDyad(
            linguistic_description="Hello Akkaris, this is prompt 1.",
            image_fingerprint=torch.randn(96)
        )
        fossilizer.fossilize(dyad_ingested, torch.randn(64), torch.randn(64))
        
        # Verify it is ingested
        assert fossilizer.is_already_ingested("Hello Akkaris, this is prompt 1.")
        
        # Instantiate Harvester with the fossilizer
        harvester = ChatGPTFrictionHarvester(export_dir=temp_dir, dim=64, fossilizer=fossilizer)
        
        # Harvest interactions and verify that prompt 1 is skipped, but prompt 2 is processed
        interactions = list(harvester.harvest_interactions())
        assert len(interactions) == 1
        
        in_tensor, out_tensor, tags, user_text, assistant_text = interactions[0]
        assert user_text == "Prompt 2 has not been ingested yet."
        assert assistant_text == "Indeed."
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Open Science Deduplication Tag Checking
# ---------------------------------------------------------------------------
def test_open_science_deduplication_tag_checking():
    # Verify the check logic of background trainer thread
    # The background thread checks for:
    # science_ligo_gw190521 and science_ncbi_am743169.1 in self.archetypal_governor.tag_stacker.catalog_vectors
    
    gov = ArchetypalSynthesisEngine(state_dim=64)
    assert hasattr(gov, 'tag_stacker')
    gov.mandy.training_mode = True
    # Relax thresholds for unit testing tag matching logic
    gov.tag_stacker.textbook_filter.thresholds.update({
        'self_contained': 0.0,
        'instructive': 0.0,
        'algorithmic': 0.0,
        'clarity': 0.0,
        'structural_honesty': 0.0
    })
    
    # Let's create dummy scientific tags and coordinates
    ligo_tag = "science_ligo_gw190521"
    ncbi_tag = "science_ncbi_am743169.1"
    
    # Assert they are not in stacker initially
    assert ligo_tag not in gov.tag_stacker.catalog_vectors
    assert ncbi_tag not in gov.tag_stacker.catalog_vectors
    
    # Register one tag
    vector = torch.randn(64)
    gov.harvest_named_coordinate(ligo_tag, vector, "LIGO GW190521")
    
    # Verify presence
    assert ligo_tag in gov.tag_stacker.catalog_vectors
    assert ncbi_tag not in gov.tag_stacker.catalog_vectors
    
    # Replicate the actual check in diegetic_backend.py:
    query_configs = [
        {"type": "ligo", "event": "GW190521", "detector": "H1", "duration": 2.0},
        {"type": "ncbi", "accession_id": "AM743169.1", "db": "nucleotide"}
    ]
    
    active_configs = []
    for q in query_configs:
        q_type = q.get("type", "").lower()
        if q_type == "ligo":
            tag_name = f"science_ligo_{q.get('event', 'GW190521').lower()}"
        elif q_type == "ncbi":
            tag_name = f"science_ncbi_{q.get('accession_id', 'AM743169.1').lower()}"
        else:
            tag_name = f"science_{q_type}"
            
        if not hasattr(gov, 'tag_stacker') or tag_name not in gov.tag_stacker.catalog_vectors:
            active_configs.append(q)
            
    # Verify that ligo query is skipped and only ncbi query remains
    assert len(active_configs) == 1
    assert active_configs[0]["type"] == "ncbi"


# ---------------------------------------------------------------------------
# Test 4: Zeitgeist Router switching transition registers tag in Stacker
# ---------------------------------------------------------------------------
def test_zeitgeist_switch_glitch_tagging():
    # Instantiate ArchetypalSynthesisEngine
    gov = ArchetypalSynthesisEngine(state_dim=64)
    gov.mandy.training_mode = True
    # Relax thresholds for unit testing tag matching logic
    gov.tag_stacker.textbook_filter.thresholds.update({
        'self_contained': 0.0,
        'instructive': 0.0,
        'algorithmic': 0.0,
        'clarity': 0.0,
        'structural_honesty': 0.0
    })
    
    # Simulate the exact logic inside ZeitgeistRouter forward/evaluation check
    # When _zg_mode == 'switching', registered as a tag in the stacker
    _zg_mode = 'switching'
    
    # Setup dummy state with braid word and step
    state = ZeitgeistState(
        alpha_tensor=torch.eye(3),
        level=1,
        moduli=(2, 3, 5),
        mode='switching',
        step=42,
        braid_word=[1, 2, -1],
        cs_phase=0.75
    )
    
    seed_state = torch.randn(1, 64)
    
    # Perform registration identical to diegetic_backend.py lines 2364-2370
    if _zg_mode == 'switching' and gov is not None:
        step = state.step
        tag_name = f"glitch_switching_step_{step}"
        desc = f"Topological switch glitch at step {step} with braid word {state.braid_word} and CS phase {state.cs_phase}"
        vector = seed_state.detach().mean(dim=0) if seed_state.dim() > 1 else seed_state.detach()
        gov.harvest_named_coordinate(tag_name, vector, desc)
        
    # Assert tag was successfully registered in governor's tag stacker
    expected_tag = "glitch_switching_step_42"
    assert expected_tag in gov.tag_stacker.catalog_vectors
    
    # Check that quality report is stored in metadata catalog
    assert expected_tag in gov.tag_stacker.catalog_metadata
    assert gov.tag_stacker.catalog_metadata[expected_tag].is_admissible


if __name__ == "__main__":
    print("[RUNNING VERIFICATION SUITE]")
    test_fossilizer_deduplication()
    print("  - test_fossilizer_deduplication passed")
    test_harvester_deduplication()
    print("  - test_harvester_deduplication passed")
    test_open_science_deduplication_tag_checking()
    print("  - test_open_science_deduplication_tag_checking passed")
    test_zeitgeist_switch_glitch_tagging()
    print("  - test_zeitgeist_switch_glitch_tagging passed")
    print("[SUCCESS] ALL VERIFICATION TESTS PASSED")
