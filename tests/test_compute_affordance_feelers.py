# -*- coding: utf-8 -*-
"""
tests/test_compute_affordance_feelers.py

Validates the extensive local compute affordance feelers across:
1. HardwareMonitor.probe_compute_feelers across 8-regimes, memory pressure, and thermal velocity
2. UniversalOrchestrator feeler exposure and Ricci flow learning rate throttling
3. GyroidicGraphManager procedural spatial filtering adaptation based on feeler telemetry
"""

import os
import pytest
import torch

os.environ["Sovereign_Disable_Background"] = "1"


def test_hardware_monitor_extensive_feelers():
    """Verifies that HardwareMonitor.probe_compute_feelers provides multi-dimensional telemetry."""
    from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType

    hw = HardwareMonitor.get_instance()
    feelers = hw.probe_compute_feelers()

    # 1. Nanosecond jitter & t_RFC feeler
    assert "t_rfc_feeler_ns" in feelers
    assert "t_rfc_stall_detected" in feelers
    assert feelers["t_rfc_feeler_ns"] >= 1

    # 2. All 8 regime gauges present
    expected_gauges = [
        "local_silicon_gauge",
        "local_thermal_gauge",
        "local_gpu_sovereign_gauge",
        "topological_gauge",
        "community_p2p_mesh_gauge",
        "diegetic_gauge",
        "algorithmic_gauge",
        "compute_batch_gauge",
    ]
    for gauge in expected_gauges:
        assert gauge in feelers
        assert 0.0 <= feelers[gauge] <= 1.5

    # 3. Physical RAM and Memory pressure feelers
    assert "ram_available_gb" in feelers
    assert "ram_percent" in feelers
    assert "memory_pressure_detected" in feelers
    assert isinstance(feelers["memory_pressure_detected"], bool)

    # 4. Thermodynamic velocity and spike detection
    assert "thermal_velocity_w_per_s" in feelers
    assert "thermal_spike_detected" in feelers
    assert isinstance(feelers["thermal_spike_detected"], bool)

    # 5. Contextual adaptive recommendations
    assert "recommended_covariance_dim" in feelers
    assert feelers["recommended_covariance_dim"] in (64, 128, 256)
    assert "recommended_procedural_render_dist" in feelers
    assert "recommended_procedural_max_nodes" in feelers
    assert "recommended_mandelbulb_power" in feelers
    assert feelers["recommended_mandelbulb_power"] in (4, 6, 8)


def test_orchestrator_feeler_telemetry_and_ricci_adaptation():
    """Verifies that UniversalOrchestrator exposes feelers and adapts Ricci flow parameters."""
    from src.core.orchestrator import UniversalOrchestrator

    orchestrator = UniversalOrchestrator(dim=64)

    # Probe directly
    feelers = orchestrator.probe_compute_feelers()
    assert "t_rfc_feeler_ns" in feelers
    assert "algorithmic_gauge" in feelers

    # Check telemetry exposure
    telemetry = orchestrator.get_compute_affordance_telemetry()
    assert "feelers" in telemetry
    assert "gauges" in telemetry
    assert telemetry["feelers"]["t_rfc_feeler_ns"] >= 1

    # Adapt parameters with Ricci flow
    param_tensor = torch.nn.Parameter(torch.randn(16, 16, requires_grad=True))
    mock_loss = (param_tensor ** 2).sum()

    adapted_val = orchestrator.adapt_parameters_ricci_flow(loss=mock_loss, parameters=[param_tensor], lr=1e-3)
    assert adapted_val is not None


def test_procedural_graph_feeler_dynamic_throttling():
    """Verifies that GyroidicGraphManager embeds compute feelers and adapts procedural limits."""
    from src.topology.embedding_graph import GyroidicGraphManager
    from src.core.hardware_monitor import HardwareMonitor

    hw = HardwareMonitor.get_instance()
    manager = GyroidicGraphManager(dim=64)

    for i in range(15):
        manager.nodes.append({
            "id": f"node_{i}",
            "label": f"Fossil Concept {i}",
            "x": float(i * 20),
            "y": float(i * 15),
            "z": float(i * 5),
            "type": "concept"
        })

    result = manager.query_procedural_graph(
        camera_pos=(0.0, 0.0, 500.0),
        gaze_dir=(0.0, 0.0, -1.0),
        render_distance=800.0,
        max_nodes=100
    )

    assert result.get("procedural") is True
    stats = result.get("procedural_stats", {})
    assert "compute_feelers" in stats
    assert "feeler_throttle_active" in stats
    assert "t_rfc_feeler_ns" in stats["compute_feelers"]

