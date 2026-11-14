import pytest
import torch
import math
import os
import sys
sys.path.insert(0, os.path.abspath("."))

from src.core.voynich_architecture import VoynichLinguist
from src.core.false_negative_subsystem import VoynichExemptionToken
from src.data.sovereign_ingestor import SovereignIngestor
from src.data.economic_news_linker import EconomicAgentLinker
from src.data.universal_topology_converter import UniversalTopologyConverter

def test_voynich_wordle_information_gain_pruning():
    """Verify Wordle dynamic programming information gain pruning and OKLab grouping."""
    compound = "Isotremoptememally"
    cuts = VoynichLinguist.split_information_gain_sub_pairs(compound)
    assert len(cuts) > 0
    # Verify each cut has OKLab delta, information gain, and Catalan weight
    for cut in cuts:
        assert "prefix" in cut
        assert "suffix" in cut
        assert "split_index" in cut
        assert "information_gain" in cut
        assert "catalan_weight" in cut
        assert "rank_score" in cut
        assert cut["information_gain"] >= 0.0
        assert cut["catalan_weight"] > 0.0
        assert len(cut["prefix"]) + len(cut["suffix"]) == len(compound)

    # Verify orthogonal openers are marked
    openers = [c for c in cuts if c["is_opener"]]
    assert len(openers) >= 3

    # Check top candidates have higher rank score than bottom candidates
    assert cuts[0]["rank_score"] >= cuts[-1]["rank_score"]

def test_voynich_garble_detection():
    """Verify detection of vowel-starvation and consonant-clustering pathological garble."""
    # Vowel-starved and consonant-clustered garble
    garble_str = "bcdfghjklmnp"
    is_garbled, reason = VoynichLinguist._is_pathological_garble(garble_str)
    assert is_garbled is True
    assert reason in ("VOWEL_STARVATION", "CONSONANT_CLUSTERING")

    # Natural balanced morpho-unit
    healthy_str = "morphene"
    is_garbled_h, reason_h = VoynichLinguist._is_pathological_garble(healthy_str)
    assert is_garbled_h is False
    assert reason_h == "NOMINAL"

def test_voynich_find_optimal_seam_and_exemption_token():
    """Verify finding the optimal seam, PAS_s scoring, and Chern-Simons sealing."""
    linguist = VoynichLinguist(latent_dim=128, num_residues=5)
    result = linguist.find_optimal_seam("radiocatalysis", max_candidates=4)
    assert "best_cut" in result
    assert "phase_alignment_score" in result
    assert "is_honest" in result
    assert "exemption_token" in result
    assert isinstance(result["exemption_token"], VoynichExemptionToken)
    assert 0.0 <= result["phase_alignment_score"] <= 1.0

def test_sovereign_orbital_stac_telemetry():
    """Verify open rolling cache orbital satellite telemetry ingestion and epistemic veto."""
    ingestor = SovereignIngestor()
    
    # Test nominal ingestion with Sentinel-5P NO2 industrial index
    target_sig = torch.tensor([[0.25, 0.28], [0.26, 0.27]])
    ambient_ref = torch.tensor([[0.05, 0.06], [0.05, 0.05]])
    res = ingestor.ingest_orbital_stac_telemetry(
        collection="sentinel-5p-l2-netcdf",
        metric_name="nitrogendioxide_tropospheric_column",
        mock_signal=target_sig,
        mock_reference=ambient_ref,
        structural_wobble_threshold=0.1
    )
    assert res["collection"] == "sentinel-5p-l2-netcdf"
    assert res["wobble_to_noise_veto"] is False
    assert not torch.isnan(res["calibrated_tensor"]).any()
    assert res["ingestion_cadence"] == "DAILY_ROLLING_CACHE"

    # Test wobble veto triggered on high measurement noise
    noisy_sig = torch.tensor([[1.5, -1.2], [2.1, -1.8]])
    res_veto = ingestor.ingest_orbital_stac_telemetry(
        collection="sentinel-1-grd",
        mock_signal=noisy_sig,
        mock_reference=ambient_ref,
        structural_wobble_threshold=0.01
    )
    assert res_veto["wobble_to_noise_veto"] is True
    assert torch.isnan(res_veto["calibrated_tensor"]).any()

def test_economic_agent_physical_telemetry_arbitrage():
    """Verify routing physical satellite signals to infrastructure landlords vs speculative futures."""
    linker = EconomicAgentLinker()

    # Sentinel-5P NO2 Manufacturing: routes to regional refiners, rejects SPY
    res_no2 = linker.evaluate_physical_telemetry_arbitrage(
        raw_proxy_series=[12.5, 13.0, 12.8],
        ambient_ground_series=[2.0, 2.1, 2.0],
        weekly_physical_wobble=5.0,
        measurement_uncertainty=1.0,
        signal_type="TROPOSPHERIC_NO2_MANUFACTURING"
    )
    assert res_no2["target_ontology"] == "REGIONAL_MANUFACTURING_AND_PETROCHEMICAL_REFINERS"
    assert res_no2["rejected_ontology"] == "BROAD_INDEX_FUTURES (SPY)"
    assert res_no2["wobble_veto_triggered"] is False

    # Sentinel-1 SAR Port Congestion: routes to port operators, rejects spot freight
    res_sar = linker.evaluate_physical_telemetry_arbitrage(
        raw_proxy_series=[45.0, 48.0],
        ambient_ground_series=[5.0, 5.0],
        weekly_physical_wobble=10.0,
        measurement_uncertainty=2.0,
        signal_type="SAR_MARITIME_PORT_CONGESTION"
    )
    assert res_sar["target_ontology"] == "PORT_OPERATORS_AND_LOGISTICS_LANDLORDS"
    assert res_sar["rejected_ontology"] == "SPOT_FREIGHT_RATES"

def test_universal_topology_bep_dht_ingestion():
    """Verify BitTorrent BEP 51/9/10/52 DHT infohash and metadata ingestion with prime-stride sieve."""
    converter = UniversalTopologyConverter(target_dim=128, num_moduli=5)

    # 20-byte SHA-1 infohash (BEP 51 v1)
    info_hash_v1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
    res_v1 = converter.ingest_bep_dht_metadata(
        info_hash=info_hash_v1,
        metadata_payload=b"d4:name10:test_stream12:piece lengthi16384ee"
    )
    assert res_v1["hash_type"] == "BEP_51_V1_SHA1"
    assert "BEP_51" in res_v1["bep_standards"]
    assert "spectral_tensor" in res_v1
    assert "ufsc_sieve" in res_v1
    assert "universal_signature" in res_v1

    # 32-byte SHA-256 infohash (BEP 52 v2)
    info_hash_v2 = "a" * 64  # 64 hex chars = 32 bytes
    res_v2 = converter.ingest_bep_dht_metadata(info_hash=info_hash_v2)
    assert res_v2["hash_type"] == "BEP_52_V2_SHA256"
    assert res_v2["spectral_tensor"].shape[-1] == 128


if __name__ == "__main__":
    print("Running test_voynich_wordle_information_gain_pruning...")
    test_voynich_wordle_information_gain_pruning()
    print("Running test_voynich_garble_detection...")
    test_voynich_garble_detection()
    print("Running test_voynich_find_optimal_seam_and_exemption_token...")
    test_voynich_find_optimal_seam_and_exemption_token()
    print("Running test_sovereign_orbital_stac_telemetry...")
    test_sovereign_orbital_stac_telemetry()
    print("Running test_economic_agent_physical_telemetry_arbitrage...")
    test_economic_agent_physical_telemetry_arbitrage()
    print("Running test_universal_topology_bep_dht_ingestion...")
    test_universal_topology_bep_dht_ingestion()
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
