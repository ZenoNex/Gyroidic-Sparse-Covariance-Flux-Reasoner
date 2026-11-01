import os
import time
import pytest
import torch
from unittest.mock import MagicMock, patch

from src.core.hardware_monitor import HardwareMonitor, budget_cpu_threads, get_cpu_gpu_layout
from src.core.voynich_architecture import VoynichLinguist


def test_budget_cpu_threads():
    """Verify that budget_cpu_threads constrains PyTorch thread allocation."""
    assigned = budget_cpu_threads(max_threads=2)
    assert assigned == 2
    assert torch.get_num_threads() == 2


def test_voynich_pathological_garble_detection():
    """Verify VoynichLinguist accurately diagnoses vowel starvation and consonant clustering."""
    # Vowel starved gibberish
    is_garbled, reason = VoynichLinguist._is_pathological_garble("bcdfghjklmnpqrstvwxyz")
    assert is_garbled is True
    assert reason in ("VOWEL_STARVATION", "CONSONANT_CLUSTERING")

    # High consonant clustering with enough vowels to not trigger vowel starvation
    is_garbled, reason = VoynichLinguist._is_pathological_garble("strengthproblem")
    assert is_garbled is True
    assert reason == "CONSONANT_CLUSTERING"

    # Nominal word
    is_garbled, reason = VoynichLinguist._is_pathological_garble("resonator")
    assert is_garbled is False
    assert reason == "NOMINAL"


def test_voynich_linguist_forward_four_tuple_unpack():
    """Verify VoynichLinguist.forward returns exactly a 4-tuple and unpacks safely."""
    vl = VoynichLinguist(latent_dim=64)
    dummy_thought = torch.randn(1, 64)
    output = vl(dummy_thought)

    assert isinstance(output, tuple)
    assert len(output) == 4
    residues, symbol_val, pas_s, token = output
    assert residues is not None
    assert symbol_val is not None
    assert pas_s is not None
    assert token is not None


def test_gibberish_generation_throttles_length_and_tracks_progress(capsys):
    """
    Verify that when an honest gibberish prompt is passed to _generate_converged_response,
    the length is gated to <= 80, progress updates are emitted, and execution terminates promptly.
    """
    from src.ui.diegetic_backend import DiegeticPhysicsEngine

    # Use lightweight configuration to satisfy test safety rules
    os.environ["Sovereign_Disable_Background"] = "1"
    engine = DiegeticPhysicsEngine(lightweight=True)

    gibberish_input = "asdfghjklzxcvbnmqwrytp"
    seed_state = torch.randn(1, engine.dim)
    affordance_gradients = {
        'executability_pressure': 0.1,
        'runtime_expandability': 0.1,
        'conversational_embedding_pressure': 0.1,
        'knowledge_seeking': 0.1
    }

    start_time = time.time()
    response = engine._generate_converged_response(
        text_input=gibberish_input,
        seed_state=seed_state,
        fingerprint=None,
        affordance_gradients=affordance_gradients
    )
    elapsed = time.time() - start_time

    # Must complete fast (under 10s on CPU)
    assert elapsed < 10.0
    assert "[CONVERGED RESPONSE]" in response

    # Progress telemetry must be completed
    progress = engine.get_generation_progress()
    assert progress["is_generating"] is False
    assert progress["total"] <= 80
    assert progress["percent"] == 100.0

    # Verify stdout contained progress reports
    captured = capsys.readouterr()
    assert "[LARYNX] Honest gibberish / pathological garble detected" in captured.out
    assert "[LARYNX GENERATION]" in captured.out


def test_api_progress_and_hardware_telemetry():
    """Verify that get_cpu_gpu_layout recognizes CPU and OpenCL GPU configurations."""
    layout = get_cpu_gpu_layout()
    assert "cpu" in layout
    assert "gpu" in layout
    assert layout["cpu"]["physical_cores"] >= 1
    assert isinstance(layout["gpu"]["devices"], list)


def test_hybrid_and_diegetic_handlers_expose_progress_route():
    """Verify that both HybridHandler and DiegeticRequestHandler route /api/progress without 404."""
    import json
    from hybrid_backend import HybridHandler
    from src.ui.diegetic_backend import RequestHandler as DiegeticRequestHandler

    # Test HybridHandler routing
    mock_hybrid = MagicMock(spec=HybridHandler)
    mock_hybrid.path = "/api/progress"
    sent_data = []
    mock_hybrid._send_json = lambda d: sent_data.append(d)
    mock_hybrid._send_error_json = lambda err, code=500: sent_data.append({"error": err, "code": code})
    
    HybridHandler.do_GET(mock_hybrid)
    assert len(sent_data) == 1
    assert sent_data[0].get("status") == "ok"
    assert "progress" in sent_data[0]

    # Test DiegeticRequestHandler routing
    mock_diegetic = MagicMock(spec=DiegeticRequestHandler)
    mock_diegetic.path = "/api/progress"
    d_sent_data = []
    mock_diegetic._send_json = lambda d: d_sent_data.append(d)
    mock_diegetic._send_error_json = lambda err, code=500: d_sent_data.append({"error": err, "code": code})
    
    DiegeticRequestHandler.do_GET(mock_diegetic)
    assert len(d_sent_data) == 1
    assert d_sent_data[0].get("status") == "ok"
    assert "progress" in d_sent_data[0]

