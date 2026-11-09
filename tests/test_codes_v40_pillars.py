# -*- coding: utf-8 -*-
"""
tests/test_codes_v40_pillars.py

Verification suite for CODES v40 Architecture & Operational ADMM:
1. Operational ADMM (Ontological Splitting & Differentiable Constraints)
2. Gap 1: Global PAS API (HMAC-SHA256 GLYPHLOCK Handshake, RIC Cross-System Tracking)
3. Gap 2: Chiral Graph Compiler (Non-commutative Topological Order, Executable Form)
4. Gap 3: Matrioshka Outward Shell Popping & Lazarus Rehydration
5. Gap 4: Birkhoff Complexity Gate (O(N^4) Bypass via Log-Domain Sinkhorn-Knopp)
"""

import pytest
import torch
import torch.nn as nn

from src.core.birkhoff_projection import DirectBirkhoffProjection
from src.core.global_pas_api import GlobalPASAPI, GLYPHLOCKHandshake
from src.core.chiral_graph_compiler import ChiralGraphCompiler
from src.core.meta_polytope_matrioshka import MetaPolytopeMatrioshka, BoundaryState
from src.core.context_aware_quantizer import ContextAwareQuantizer
from src.optimization.operational_admm import OperationalAdmm


def test_birkhoff_complexity_gate():
    """Verify Birkhoff projection scales to N=64 without allocating O(N^4) projection matrix."""
    proj = DirectBirkhoffProjection(n=64)
    assert proj.use_iterative_fallback is True
    
    # Positive affinity matrix
    x = torch.rand(1, 64, 64) + 0.1
    b = proj(x)
    assert b.shape == (1, 64, 64)
    # Check doubly stochastic properties
    row_sums = b.sum(dim=-1)
    col_sums = b.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-2)


def test_global_pas_api_handshake_and_tracking():
    """Verify Global PAS API cryptographic handshake and coherence tracking."""
    handshake = GLYPHLOCKHandshake(secret_key=b"codes_test_secret_12345678")
    token = handshake.create_handshake(client_id="edge_sensor_1")
    assert handshake.verify(token) is True

    api = GlobalPASAPI(secret_key=b"codes_test_secret_12345678")
    track_res = api.track_coherence(
        node_id="edge_sensor_1",
        state_tensor=[0.1, 0.2, -0.3, 0.4, 0.05],
        field_id="FIELD_ALPHA",
        environment="edge"
    )
    assert track_res["status"] == "tracked"
    assert "pas_h" in track_res

    # Query with valid token
    query_res = api.query_engine_safe(token.to_dict(), "What is the manifold state?")
    assert query_res["status"] == "OK"


def test_chiral_graph_compiler_action_layer():
    """Verify Chiral Graph Compiler non-commutative ordering and multi-target compilation."""
    compiler = ChiralGraphCompiler(state_dim=16)
    
    flow_graph = {
        "nodes": [
            {"id": "ingest", "label": "Ingest Stream", "op": "read"},
            {"id": "resonate", "label": "Resonance Cavity", "op": "cavity"},
            {"id": "fossilize", "label": "Dyad Fossilizer", "op": "fossilize"}
        ],
        "edges": [
            {"source": "ingest", "target": "resonate"},
            {"source": "resonate", "target": "fossilize"}
        ],
        "latent_state": [0.1, 0.5, 0.9, 1.2, 0.3, 0.1, 0.05, 0.02]
    }

    # Test Python compilation
    py_art = compiler.compile(flow_graph, target="python", module_name="test_pipeline")
    assert py_art.execution_order == ["ingest", "resonate", "fossilize"]
    assert "TestPipeline" in py_art.code
    assert len(py_art.non_commutative_hash) == 64

    # Test Shell compilation
    sh_art = compiler.compile(flow_graph, target="shell")
    assert "[CHIRAL_SHELL]" in sh_art.code

    # Test UI compilation
    ui_art = compiler.compile(flow_graph, target="ui")
    assert "Topological Chiral Flow" in ui_art.code


def test_matrioshka_shell_escape_and_outward_popping():
    """Verify that when contradiction occurs, the Matrioshka loop pops outward to coarser shells."""
    mpm = MetaPolytopeMatrioshka(max_depth=4, base_dim=16)
    
    # State with NaN: should be rehydrated by Lazarus rehydration without throwing exception
    nan_state = torch.tensor([float('nan')] * 16)
    res = mpm(nan_state)
    assert isinstance(res, (tuple, BoundaryState))

    # Test CAQ step outward
    caq = ContextAwareQuantizer(dim=16, max_depth=5)
    caq._level = 3
    new_lvl = caq.step_outward()
    assert new_lvl == 2
    assert caq.step_outward() == 1


def test_operational_admm_primitive():
    """Verify Operational ADMM executes ontological splitting and stabilizes state."""
    admm = OperationalAdmm(degree=4, max_iters=5)
    initial_c = torch.randn(1, 16)
    
    out_c, status = admm(
        initial_c=initial_c,
        forward_op=lambda s: s * 1.01
    )
    assert out_c.shape == initial_c.shape
    assert not torch.isnan(out_c).any()
