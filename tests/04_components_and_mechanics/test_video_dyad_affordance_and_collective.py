import os
import base64
import torch
import pytest

os.environ["Sovereign_Disable_Background"] = "1"

def test_video_dyad_parser_affordance_integration():
    """Verify VideoDyadParser respects local and collective affordances and computes 96-dim signature."""
    from src.core.video_dyad_parser import VideoDyadParser
    from src.p2p.bonfire_consensus import BonfireNomadicRing
    from src.p2p.freenet_ws_client import FreenetClient

    dummy_client = FreenetClient()
    ring = BonfireNomadicRing(dummy_client)
    # Simulate active peer allocations
    ring.peer_allocations = {"peer_1": 0.75, "peer_2": 0.85}

    parser = VideoDyadParser(chunk_size=512, max_chunks=32, device="cpu", p2p_ring=ring)
    assert parser.p2p_ring is ring

    dummy_bytes = b"ftypmp42\x00\x00\x00\x00moov\x00\x00\x00\x10mdat" + bytes(range(256)) * 8
    dummy_b64 = base64.b64encode(dummy_bytes).decode("ascii")

    parsed = parser.parse_video_b64(dummy_b64, extract_audio=False)

    assert "sparse_covariance" in parsed
    assert "fractal_entropy" in parsed
    assert "substream_residue" in parsed
    assert "local_affordance" in parsed
    assert "collective_affordance" in parsed

    col_aff = parsed["collective_affordance"]
    assert col_aff["active_peers"] == 2
    assert col_aff["kelly_consensus"] > 0.0

    # Test 96-dimensional unified spectral signature extraction
    sig_96 = parser.extract_96_spectral_signature(parsed)
    assert sig_96.shape == (96,), f"Expected 96-dim spectral signature, got {sig_96.shape}"


def test_orchestrator_video_dyad_collective_ingest():
    """Verify UniversalOrchestrator ingests video dyad with full affordance diagnostics."""
    from src.core.orchestrator import UniversalOrchestrator

    orch = UniversalOrchestrator(dim=32, device="cpu")
    # Verify video_dyad_parser is bound to bonfire_ring
    if orch.bonfire_ring is not None:
        assert orch.video_dyad_parser.p2p_ring is orch.bonfire_ring

    dummy_bytes = b"ftypmp42\x00\x00\x00\x00moov\x00\x00\x00\x10mdat\x01\x02\x03\x04\x05\x06"
    dummy_b64 = base64.b64encode(dummy_bytes).decode("ascii")

    residues = orch.ingest_video_dyad(
        video_b64=dummy_b64,
        description="Quantum Field Video Test",
        extract_audio=False,
        fossilize=False
    )

    assert residues.shape == (1, 5)
    last_dyad = orch.last_dyad
    assert last_dyad is not None
    assert "Quantum Field Video Test" in last_dyad.linguistic_description
    assert last_dyad.metadata["source"] == "video"
    assert "local_silicon_gauge" in last_dyad.metadata
    assert "p2p_mesh_gauge" in last_dyad.metadata
    assert "kelly_consensus" in last_dyad.metadata

    # Check video_breather diagnostics
    vb = last_dyad.video_breather
    assert vb is not None
    assert "sparse_covariance" in vb
    assert "local_affordance" in vb
    assert "collective_affordance" in vb

    # Check unified_spectral_signature is populated
    assert last_dyad.unified_spectral_signature is not None
    assert last_dyad.unified_spectral_signature.shape == (96,)

