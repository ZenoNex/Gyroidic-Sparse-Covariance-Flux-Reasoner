import os
import tempfile
import torch
import numpy as np
import pytest

os.environ["Sovereign_Disable_Background"] = "1"

from src.core.video_dyad_parser import VideoDyadParser
from src.core.orchestrator import UniversalOrchestrator


def create_dummy_mp4_with_atoms(file_path: str, size_bytes: int = 1024 * 1024 * 2):
    """
    Creates a binary file that has realistic MP4 atom signatures
    embedded at standard container locations (head and tail).
    """
    with open(file_path, "wb") as f:
        # Header with ftyp, moov, avc1, mp4a
        header = b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00isommp42"
        header += b"\x00\x00\x01\x00moov\x00\x00\x00\x50trak\x00\x00\x00\x30mdia"
        header += b"\x00\x00\x00\x20minf\x00\x00\x00\x10stbl\x00\x00\x00\x08avc1"
        header += b"\x00\x00\x00\x08mp4a\x00\x00\x00\x08stco"
        f.write(header)
        
        # Intermediate payload padding
        padding_size = max(0, size_bytes - len(header) - 100)
        chunk = b"A" * 65536
        written = 0
        while written < padding_size:
            to_write = min(len(chunk), padding_size - written)
            f.write(chunk[:to_write])
            written += to_write
            
        # Tail with mdat and meta
        tail = b"\x00\x00\x00\x20mdat" + (b"\x12" * 24) + b"\x00\x00\x00\x10meta\x00\x00\x00\x00"
        f.write(tail)


def test_video_dyad_parser_file_pointer():
    """Verify that [FILE_POINTER:<path>] parses streamed disk files without loading into RAM."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        file_size = 2 * 1024 * 1024  # 2MB
        create_dummy_mp4_with_atoms(tmp_path, size_bytes=file_size)

        parser = VideoDyadParser(device=torch.device("cpu"))
        
        # Test 1: [FILE_POINTER:<path>] format
        ptr_string = f"[FILE_POINTER:{tmp_path}]"
        parsed = parser.parse_video_b64(ptr_string, extract_audio=False)

        assert "sparse_covariance" in parsed
        assert parsed["sparse_covariance"].ndim == 2
        assert parsed["sparse_covariance"].shape[0] == parsed["sparse_covariance"].shape[1]
        assert "substream_residue" in parsed
        assert parsed["substream_residue"].shape == (6,)
        # Verify atom detection worked on streamed head/tail
        assert parsed["substream_residue"].sum().item() > 0.0
        assert parsed["signal_length"].item() == pytest.approx(float(os.path.getsize(tmp_path)), rel=1e-3)

        # Test 2: Extract 96-dim spectral signature
        sig_96 = parser.extract_96_spectral_signature(parsed)
        assert sig_96.shape == (96,)
        assert torch.isfinite(sig_96).all()

        # Test 3: Raw filepath format directly
        parsed_direct = parser.parse_video_b64(tmp_path, extract_audio=False)
        assert parsed_direct["sparse_covariance"].ndim == 2
        assert parsed_direct["sparse_covariance"].shape[0] == parsed_direct["sparse_covariance"].shape[1]
        assert parsed_direct["substream_residue"].shape == (6,)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_video_dyad_parser_safe_fallbacks():
    """Verify that empty pointers, placeholder strings, and invalid base64 payloads never crash."""
    parser = VideoDyadParser(device=torch.device("cpu"))

    # Empty string
    empty_res = parser.parse_video_b64("", extract_audio=False)
    assert empty_res["sparse_covariance"].shape == (32, 32)
    assert empty_res["signal_length"].item() == 0.0

    # Literal placeholder "[FILE_POINTER]"
    placeholder_res = parser.parse_video_b64("[FILE_POINTER]", extract_audio=False)
    assert placeholder_res["sparse_covariance"].shape == (32, 32)
    assert placeholder_res["signal_length"].item() == 0.0

    # Corrupt / invalid base64 string
    corrupt_res = parser.parse_video_b64("NOT_VALID_BASE64_%%%@@@!!!", extract_audio=False)
    assert corrupt_res["sparse_covariance"].shape == (32, 32)
    assert corrupt_res["signal_length"].item() == 0.0


def test_orchestrator_ingest_video_dyad_file_pointer():
    """Verify UniversalOrchestrator ingests streamed file pointer into KnowledgeDyad."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_dummy_mp4_with_atoms(tmp_path, size_bytes=1024 * 1024)
        
        # Test orchestrator in lightweight mode
        orchestrator = UniversalOrchestrator(dim=32, device="cpu")
        ptr_string = f"[FILE_POINTER:{tmp_path}]"
        
        out = orchestrator.ingest_video_dyad(
            video_b64=ptr_string,
            description="Large Payload Stream Test Video",
            extract_audio=False
        )
        
        assert out is not None
        assert torch.is_tensor(out)
        assert orchestrator.video_breather is not None
        assert "sparse_covariance" in orchestrator.video_breather
        assert "substream_residue" in orchestrator.video_breather
        assert orchestrator.video_breather["signal_length"] > 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
