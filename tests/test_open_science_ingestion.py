#!/usr/bin/env python3
"""
tests/test_open_science_ingestion.py
Comprehensive test suite for the OpenScienceIngestor and its integration with the
DatasetIngestionSystem, with strict timeout signals and clean exits.

Run with:
    $env:PYTHONPATH="."; .venv\\Scripts\\python.exe -u tests\\test_open_science_ingestion.py
"""

import sys
import os
import time
import json
import shutil
import threading
import traceback
from pathlib import Path

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.open_science_ingestor import OpenScienceIngestor
from dataset_ingestion_system import DatasetIngestionSystem, DatasetConfig

# ---------------------------------------------------------------------------
# Timeout harness
# ---------------------------------------------------------------------------
TIMEOUT_SECONDS = 30


def run_with_timeout(fn, timeout=TIMEOUT_SECONDS):
    """
    Run fn() in a daemon thread. Returns (passed, message).
    If the thread does not finish within `timeout` seconds it is declared
    a timeout failure (the daemon thread is left to expire naturally).
    """
    result = {"passed": False, "msg": "timeout"}

    def _target():
        try:
            fn()
            result["passed"] = True
            result["msg"] = "ok"
        except AssertionError as exc:
            result["msg"] = f"AssertionError: {exc}"
        except Exception as exc:
            result["msg"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    return result["passed"], result["msg"]


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def _test_ligo_direct():
    """Direct fetch of LIGO strain data (using fallback simulation if offline/missing deps)."""
    cache_dir = Path("datasets/test_cache_ligo")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    ingestor = OpenScienceIngestor(cache_dir=str(cache_dir))
    data = ingestor.fetch_ligo_strain(event_name="GW190521", detector="H1", duration_sec=2.0, sample_rate=4096)
    
    assert isinstance(data, dict), "LIGO data should be a dictionary"
    assert "strain" in data, "LIGO data should contain 'strain' key"
    assert "detector" in data, "LIGO data should contain 'detector' key"
    assert data["detector"] == "H1", "Detector mismatch"
    assert len(data["strain"]) > 0, "Strain data should not be empty"
    assert "simulated" in data, "LIGO data should have 'simulated' status indicator"
    
    # Verify cache writing works
    assert len(list(cache_dir.glob("ligo_*.json"))) == 1, "Cache file should be created"
    
    # Cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)


def _test_sdss_direct():
    """Direct fetch of SDSS catalog (using fallback simulation if offline/missing deps)."""
    cache_dir = Path("datasets/test_cache_sdss")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    ingestor = OpenScienceIngestor(cache_dir=str(cache_dir))
    data = ingestor.fetch_sdss_catalog(catalog_id="J/A+A/540/A106", row_limit=5)
    
    assert isinstance(data, dict), "SDSS catalog should be a dictionary"
    assert "chunk_paths" in data, "SDSS catalog should contain 'chunk_paths' key"
    assert "columns" in data, "SDSS catalog should contain 'columns' key"
    assert "row_count" in data, "SDSS catalog should contain 'row_count' key"
    assert data["row_count"] == 5, f"Expected 5 rows, got {data['row_count']}"
    assert "simulated" in data, "SDSS catalog should contain 'simulated' status indicator"
    
    # Verify cache writing works
    assert len(list(cache_dir.glob("sdss_*.json"))) == 1, "Cache file should be created"
    
    # Cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)


def _test_ncbi_direct():
    """Direct fetch of NCBI sequence (using fallback simulation if offline/missing deps)."""
    cache_dir = Path("datasets/test_cache_ncbi")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    ingestor = OpenScienceIngestor(cache_dir=str(cache_dir))
    data = ingestor.fetch_ncbi_sequence(accession_id="AM743169.1", db="nucleotide")
    
    assert isinstance(data, dict), "NCBI sequence should be a dictionary"
    assert "chunk_paths" in data, "NCBI sequence should contain 'chunk_paths' key"
    assert "accession" in data, "NCBI sequence should contain 'accession' key"
    assert data["accession"] == "AM743169.1", "Accession mismatch"
    assert data["length"] > 0, "Sequence string should not be empty"
    assert "simulated" in data, "NCBI data should contain 'simulated' status indicator"
    
    # Verify cache writing works
    assert len(list(cache_dir.glob("ncbi_*.json"))) == 1, "Cache file should be created"
    
    # Cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)


def _test_openneuro_direct():
    """Direct fetch of OpenNeuro fMRI scan (using fallback simulation if offline/missing deps)."""
    cache_dir = Path("datasets/test_cache_fmri")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    ingestor = OpenScienceIngestor(cache_dir=str(cache_dir))
    data = ingestor.fetch_openneuro_fmri(dataset_id="ds003445", subject_id="sub-01", run_id="run-1")
    
    assert isinstance(data, dict), "OpenNeuro data should be a dictionary"
    assert "fmri_correlation" in data, "OpenNeuro data should contain 'fmri_correlation' key"
    assert "dataset_id" in data, "OpenNeuro data should contain 'dataset_id' key"
    assert len(data["fmri_correlation"]) == 90, "fMRI correlation matrix must represent 90 regions"
    assert len(data["fmri_correlation"][0]) == 90, "fMRI correlation matrix columns must represent 90 regions"
    assert "simulated" in data, "OpenNeuro data should contain 'simulated' status indicator"
    
    # Verify cache writing works
    assert len(list(cache_dir.glob("fmri_*.json"))) == 1, "Cache file should be created"
    
    # Cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)


def _test_query_aggregation():
    """Verify flexible query aggregation and standard sample formatting."""
    cache_dir = Path("datasets/test_cache_agg")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    ingestor = OpenScienceIngestor(cache_dir=str(cache_dir))
    
    configs = [
        {"type": "ligo", "event": "GW190521", "detector": "H1", "duration": 2.0},
        {"type": "sdss", "catalog_id": "J/A+A/540/A106", "row_limit": 5},
        {"type": "ncbi", "accession_id": "AM743169.1", "db": "nucleotide"},
        {"type": "openneuro", "dataset_id": "ds003445", "subject_id": "sub-01"}
    ]
    
    samples = ingestor.query_and_aggregate(configs)
    
    assert isinstance(samples, list), "Aggregated samples should be a list"
    assert len(samples) == 4, f"Expected 4 samples, got {len(samples)}"
    
    for sample in samples:
        assert "text" in sample, "Sample must contain 'text' field"
        assert "source" in sample, "Sample must contain 'source' field"
        assert "metadata" in sample, "Sample must contain 'metadata' field"
        assert isinstance(sample["metadata"], dict), "Metadata must be a dictionary"
        assert "type" in sample["metadata"], "Metadata must contain 'type' field"
        
    # Cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)


def _test_dataset_ingestion_system_integration():
    """Verify integration of Open Science sources into DatasetIngestionSystem."""
    class MockEngine:
        def process_input(self, *args, **kwargs):
            return {"response": "mocked", "backend": "mocked"}

    system = DatasetIngestionSystem(engine=MockEngine())
    
    # Setup test config
    config = DatasetConfig(
        name="test_open_science_dataset",
        source_type="open_science",
        source_path="ligo,sdss,ncbi,openneuro",
        max_samples=10,
        preprocessing="text"
    )
    
    # Clear main cache to avoid loading old schema JSONs
    shutil.rmtree(Path("datasets/open_science_cache"), ignore_errors=True)
    
    # Clean previous test directory if exists
    dataset_dir = Path("datasets") / config.safe_name
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
        
    # Ingest using standard pipeline
    success = system.add_dataset_source(config)
    assert success, "DatasetIngetionSystem failed to ingest open science source"
    
    # Confirm directory and manifest were created
    assert dataset_dir.exists(), "Dataset output directory was not created"
    manifest_file = dataset_dir / "manifest.json"
    assert manifest_file.exists(), "Manifest file was not created"
    
    with open(manifest_file, "r") as f:
        manifest = json.load(f)
        
    assert manifest["num_samples"] > 0, "No samples ingested"
    assert manifest["config"]["source_type"] == "open_science", "Manifest config source_type mismatch"
    
    # Check if chunk files exist
    chunks_dir = dataset_dir / "chunks"
    assert chunks_dir.exists(), "Chunks directory was not created"
    assert len(list(chunks_dir.glob("chunk_*.pt"))) > 0, "No chunk files written"
    
    # Clean up test directories
    shutil.rmtree(dataset_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TESTS = [
    ("LIGO Strain Fetch Direct", _test_ligo_direct, 10),
    ("SDSS Catalog Fetch Direct", _test_sdss_direct, 10),
    ("NCBI Sequence Fetch Direct", _test_ncbi_direct, 10),
    ("OpenNeuro fMRI Fetch Direct", _test_openneuro_direct, 10),
    ("Query Aggregation Format", _test_query_aggregation, 15),
    ("DatasetIngestionSystem Integration", _test_dataset_ingestion_system_integration, 25),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("[TEST SUITE] Open Science Ingestor & Engine Integration")
    print("=" * 70)

    passed = 0
    failed = 0
    timed_out = 0

    for name, fn, timeout in TESTS:
        print(f"\nRunning: {name} (timeout={timeout}s)")
        t0 = time.time()
        ok, msg = run_with_timeout(fn, timeout=timeout)
        elapsed = time.time() - t0

        if ok:
            print(f"  [OK] {name} ({elapsed:.2f}s)")
            passed += 1
        elif msg == "timeout":
            print(f"  [TIMEOUT] {name} (>{timeout}s)")
            timed_out += 1
        else:
            print(f"  [FAIL] {name} ({elapsed:.2f}s)")
            print(f"         {msg.splitlines()[0]}")
            failed += 1

    total = passed + failed + timed_out
    print("\n" + "=" * 70)
    print(f"[SUMMARY] {passed}/{total} passed | {failed} failed | {timed_out} timed-out")
    if failed == 0 and timed_out == 0:
        print("[SUCCESS] All open science ingestion tests passed.")
    else:
        print("[WARN] Some tests did not pass. Review output above.")

    return failed == 0 and timed_out == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
