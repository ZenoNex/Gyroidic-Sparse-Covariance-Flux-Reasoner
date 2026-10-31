import pytest
import os
import sys
from unittest.mock import MagicMock, patch

from src.core.hardware_monitor import (
    HardwareMonitor,
    audit_windows_task_scheduling,
    govern_task_priority,
    get_scheduler_telemetry,
)


def test_audit_windows_task_scheduling():
    """Verify that Windows task scheduling and process priority auditing returns structured telemetry."""
    telemetry = audit_windows_task_scheduling()
    assert isinstance(telemetry, dict)
    assert "platform" in telemetry
    assert "pid" in telemetry
    assert "priority_class_raw" in telemetry
    assert "priority_class_name" in telemetry
    assert isinstance(telemetry["cpu_affinity"], list)
    assert telemetry["thread_count"] >= 1
    assert "scheduler_admissibility" in telemetry
    assert 0.0 <= telemetry["scheduler_admissibility"] <= 1.0
    assert telemetry["status"] in ("nominal", "throttled_background", "high_thread_contention")


def test_govern_task_priority():
    """Verify task priority governance structure and safe fallback."""
    report = govern_task_priority(interactive=True)
    assert isinstance(report, dict)
    assert "previous_priority" in report
    assert "new_priority" in report
    # Priority value should be a valid integer / enum
    assert report["previous_priority"] is not None


def test_get_scheduler_telemetry():
    """Verify consolidated scheduler telemetry."""
    telem = get_scheduler_telemetry()
    assert telem["status"] == "ok"
    assert "task_scheduling" in telem
    assert "thermal_guard_active" in telem
    assert "recommended_batch_concurrency" in telem
    assert telem["recommended_batch_concurrency"] >= 1


def test_diegetic_terminal_3d_graph_controls_markup():
    """Verify diegetic_terminal.html includes navigation controls, contextmenu suppression, and zoom buttons."""
    html_path = os.path.join("src", "ui", "diegetic_terminal.html")
    assert os.path.exists(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Toolbar controls
    assert "btn-toggle-graph-mode" in html
    assert "MODE: ROTATE" in html
    assert "btn-graph-fit" in html
    assert "FIT VIEW" in html
    assert "btn-graph-zoom-in" in html
    assert "btn-graph-zoom-out" in html
    assert "btn-graph-force-refresh" in html
    assert "RESCAN DISK" in html

    # JavaScript navigation handlers
    assert "toggleGraphNavigationMode" in html
    assert "zoomGraph" in html
    assert "fitGraphView" in html
    assert "container.oncontextmenu" in html
    assert "graph-status-banner" in html
    assert "COGNITIVE BUS DISCONNECT" not in html


def test_diegetic_backend_graph_caching_and_scheduler_endpoint():
    """Verify diegetic_backend.py implements cached /graph and /api/scheduler endpoints."""
    backend_path = os.path.join("src", "ui", "diegetic_backend.py")
    assert os.path.exists(backend_path)

    with open(backend_path, "r", encoding="utf-8") as f:
        code = f.read()

    assert "base_path.startswith('/graph')" in code
    assert "force_reload = query.get('force'" in code
    assert "graph_data[\"cached\"] = not force_reload" in code
    assert "base_path == '/api/scheduler'" in code
    assert "get_scheduler_telemetry()" in code


def test_orchestrator_scheduler_governance_binding():
    """Verify UniversalOrchestrator binds hardware monitor scheduler telemetry."""
    os.environ["Sovereign_Disable_Background"] = "1"
    from src.core.orchestrator import UniversalOrchestrator

    orch = UniversalOrchestrator(dim=8)
    assert hasattr(orch, "audit_windows_task_scheduling")
    assert hasattr(orch, "govern_task_priority")
    assert hasattr(orch, "get_scheduler_telemetry")

    sched = orch.get_scheduler_telemetry()
    assert sched["status"] == "ok"

