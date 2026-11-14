# -*- coding: utf-8 -*-
"""
tests/test_upstream_downstream_bridges.py

Verifies the upstream, downstream, and parallel track connections:
1. UniversalOrchestrator.compile_action (ChiralGraphCompiler action layer)
2. UniversalOrchestrator.create_pas_handshake & query_external_safe (GlobalPASAPI / GLYPHLOCK)
3. UniversalOrchestrator.audit_arithmetic_soliton_trajectory (Lychrel 196 + Collatz + Surd sandwich)
4. DiegeticPhysicsEngine parallel track initialization (lightweight=True)
5. GyroidicGraphManager.query_procedural_graph dynamic spatial filtering
"""

import pytest
import torch
import os

# Enforce background task disabling for lightweight test execution
os.environ["Sovereign_Disable_Background"] = "1"


def test_orchestrator_compile_action():
    """Verifies UniversalOrchestrator can compile reasoning trajectories into executable forms."""
    from src.core.orchestrator import UniversalOrchestrator

    orchestrator = UniversalOrchestrator(dim=64)
    assert orchestrator.chiral_compiler is not None

    # Test Python compilation
    artifact_py = orchestrator.compile_action(target="python", module_name="test_pipeline")
    assert artifact_py is not None
    assert artifact_py.target == "python"
    assert "TestPipeline" in artifact_py.code or "run_test_pipeline" in artifact_py.code
    assert artifact_py.non_commutative_hash != ""

    # Test Shell compilation
    artifact_sh = orchestrator.compile_action(target="shell", module_name="test_sh_pipeline")
    assert artifact_sh is not None
    assert artifact_sh.target == "shell"
    assert "$ErrorActionPreference" in artifact_sh.code or "CHIRAL_SHELL" in artifact_sh.code

    # Test UI compilation
    artifact_ui = orchestrator.compile_action(target="ui", module_name="test_ui_widget")
    assert artifact_ui is not None
    assert artifact_ui.target == "ui"
    assert "<title>Chiral Flow Inspector</title>" in artifact_ui.code or "flow-container" in artifact_ui.code


def test_orchestrator_global_pas_api_handshake_and_query():
    """Verifies UniversalOrchestrator Global PAS API cryptographic handshake and safe querying."""
    from src.core.orchestrator import UniversalOrchestrator

    orchestrator = UniversalOrchestrator(dim=64)
    assert orchestrator.global_pas_api is not None

    # Generate cryptographic handshake token
    token = orchestrator.create_pas_handshake(client_id="test_client_alpha", field_id="TEST_FIELD")
    assert token is not None
    assert token.client_id == "test_client_alpha"
    assert token.field_id == "TEST_FIELD"
    assert token.signature != ""

    # Query engine safely via token
    query_payload = {"operation": "ping"}
    result = orchestrator.query_external_safe(token=token, payload=query_payload)
    assert result is not None
    assert result.get("status", "").lower() == "ok"
    assert "coherence_metrics" in result


def test_orchestrator_arithmetic_soliton_audit():
    """Verifies Lychrel 196 soliton audit, Collatz carry, and quadratic surd sandwich."""
    from src.core.orchestrator import UniversalOrchestrator

    orchestrator = UniversalOrchestrator(dim=64)
    assert orchestrator.lychrel_engine is not None

    # Seed in palindromic basin (n = 89)
    report_89 = orchestrator.audit_arithmetic_soliton_trajectory(seed=89, max_steps=30)
    assert report_89["is_palindrome"] is True
    assert report_89["is_196_soliton"] is False
    assert "collatz_audit" in report_89
    assert "quadratic_surd_bound" in report_89

    # Seed at Lychrel 196 barrier
    report_196 = orchestrator.audit_arithmetic_soliton_trajectory(seed=196, max_steps=15)
    assert report_196["is_196_soliton"] is True
    assert report_196["exemption_token_issued"] is True
    assert report_196["chiral_asymmetry_gap"] > 0.0


def test_diegetic_physics_engine_parallel_wiring():
    """Verifies DiegeticPhysicsEngine parallel subsystem instantiation under lightweight mode."""
    from src.ui.diegetic_backend import DiegeticPhysicsEngine

    engine = DiegeticPhysicsEngine(dim=64, k=4, lightweight=True)
    assert hasattr(engine, "global_pas_api")
    assert hasattr(engine, "chiral_compiler")
    assert hasattr(engine, "lychrel_engine")


def test_procedural_graph_manager_spatial_filtering():
    """Verifies GyroidicGraphManager procedural load with cone FOV, shear planes, and occlusion."""
    from src.topology.embedding_graph import GyroidicGraphManager

    manager = GyroidicGraphManager(dim=64)
    # Populate mock nodes
    for i in range(20):
        manager.nodes.append({
            "id": f"node_{i}",
            "label": f"Concept {i}",
            "x": float(i * 30),
            "y": float(i * 10),
            "z": float(i * 5),
            "type": "concept",
            "pas_h": 0.88,
            "betti_1": 1
        })

    result = manager.query_procedural_graph(
        camera_pos=(0.0, 0.0, 500.0),
        gaze_dir=(0.0, 0.0, -1.0),
        fov_deg=65.0,
        render_distance=800.0,
        enable_shear_corridor=True,
        enable_occlusion_culling=True,
        max_nodes=100
    )

    assert "nodes" in result
    assert "links" in result
    assert result.get("procedural") is True
    assert "procedural_stats" in result
    assert "culled_distance" in result["procedural_stats"]

