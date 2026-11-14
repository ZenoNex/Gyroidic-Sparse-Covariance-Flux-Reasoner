import pytest
import torch
import numpy as np
from src.core.lychrel_soliton import (
    AdelicDigitReversalOperator,
    Lychrel196SolitonEngine,
    compute_lychrel_chiral_gap
)
from src.core.number_theoretic_stabilizer import NumberTheoreticStabilizer
from src.data.universal_topology_converter import UniversalTopologyConverter
from src.topology.embedding_graph import GyroidicGraphManager, KnowledgeFossilNode


def test_adelic_digit_reversal_operator():
    """Verify adelic digit reversal and palindromic symmetry metrics."""
    op = AdelicDigitReversalOperator(base=10)
    
    # Palindrome test
    assert op.is_palindrome(12321) is True
    assert op.is_palindrome(12345) is False
    assert op.reverse(1234) == 4321
    
    # Reverse and add step
    step_val, rev_val, is_pal = op.reverse_and_add_step(89)
    assert rev_val == 98
    assert step_val == 89 + 98 == 187
    assert is_pal is False
    
    # Chiral gap for palindrome should be 0.0
    gap_zero = compute_lychrel_chiral_gap(12321)
    assert gap_zero == 0.0
    
    # Non-palindrome should have positive gap
    gap_pos = compute_lychrel_chiral_gap(196)
    assert gap_pos > 0.0


def test_lychrel_convergence_seed_89():
    """Seed 89 should contract to a palindrome in exactly 24 steps."""
    engine = Lychrel196SolitonEngine(base=10, max_steps=50, rns_moduli=[197, 211, 223])
    report = engine.evaluate_trajectory(89)
    
    assert report.seed == 89
    assert report.is_converged is True
    assert report.steps_taken == 24
    assert report.terminal_value == 8813200023188
    assert report.chiral_gap == 0.0
    assert report.exemption_token is None


def test_lychrel_divergence_and_exemption_seed_196():
    """Seed 196 should diverge with persistent non-zero curvature and issue VoynichExemptionToken."""
    engine = Lychrel196SolitonEngine(base=10, max_steps=35, rns_moduli=[197, 211, 223])
    report = engine.evaluate_trajectory(196)
    
    assert report.seed == 196
    assert report.is_converged is False
    assert report.steps_taken == 35
    assert report.chiral_gap > 0.0
    assert report.exemption_token is not None
    assert report.exemption_token.reason == "LYCHREL_196_SOLITON"
    
    # Test honest jitter harvesting
    dim = 32
    jitter = engine.harvest_honest_jitter(report, dim=dim)
    assert isinstance(jitter, torch.Tensor)
    assert jitter.shape == (dim,)
    assert not torch.isnan(jitter).any()
    assert torch.norm(jitter).item() > 0.0


def test_number_theoretic_stabilizer_lychrel_rns():
    """Test RNS multi-modulus evolution without integer overflow."""
    stabilizer = NumberTheoreticStabilizer(state_dim=64)
    
    state = torch.tensor([196.0, 89.0, 12321.0, 456.0])
    stabilized_state, rns_out = stabilizer.stabilize_lychrel_rns(state, base=10)
    
    assert isinstance(stabilized_state, torch.Tensor)
    assert stabilized_state.shape == state.shape
    assert isinstance(rns_out, dict)
    assert "chiral_curvature" in rns_out
    assert "rns_residues" in rns_out
    assert "seed" in rns_out
    assert "terminated" in rns_out


def test_universal_topology_converter_lychrel_197():
    """Test folding the 196 carry cascade into a prime-stride p=197 raster."""
    converter = UniversalTopologyConverter()
    report = converter.classify_lychrel_196_soliton(seed=196, steps=30)
    
    assert isinstance(report, dict)
    assert report["seed"] == 196
    assert "is_lychrel_soliton" in report
    assert "ufsc_197_coherence" in report
    assert "sieve_diagnostics" in report
    assert "optimal_stride" in report
    assert report["voynich_token_issued"] is True


def test_embedding_graph_procedural_query():
    """Test procedural load mode with cone, occlusion, and shear plane corridors."""
    graph_mgr = GyroidicGraphManager(dim=4)
    
    # Seed nodes at various 3D positions using KnowledgeFossilNode
    # Node 1: near origin, directly in front of camera
    n1 = KnowledgeFossilNode(
        node_id="node_near_front",
        state=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        text="Near Front",
        metrics={"x": 0.0, "y": 0.0, "z": 200.0}
    )
    # Node 2: far away along camera gaze (beyond render_dist 800)
    n2 = KnowledgeFossilNode(
        node_id="node_far_front",
        state=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        text="Far Front",
        metrics={"x": 0.0, "y": 0.0, "z": -1500.0}
    )
    # Node 3: behind camera
    n3 = KnowledgeFossilNode(
        node_id="node_behind",
        state=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        text="Behind",
        metrics={"x": 0.0, "y": 0.0, "z": 1200.0}
    )
    # Node 4: along shear corridor (x ~ 600, y ~ -600, z ~ 0 -> x+y+z=0, dist ~ 848 < 2.5*800)
    n4 = KnowledgeFossilNode(
        node_id="node_shear_plane",
        state=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        text="Shear Fault",
        metrics={"x": 600.0, "y": -600.0, "z": 0.0}
    )
    
    graph_mgr.nodes = [n1, n2, n3, n4]

    # Query with camera at (0, 0, 500) looking towards (0, 0, -1)
    result = graph_mgr.query_procedural_graph(
        camera_pos=(0.0, 0.0, 500.0),
        gaze_dir=(0.0, 0.0, -1.0),
        fov_deg=65.0,
        render_distance=800.0,
        enable_shear_corridor=True,
        enable_occlusion_culling=False,
        max_nodes=500
    )

    node_ids = {n["id"] for n in result["nodes"]}
    
    # node_near_front must be included (in front, dist=300 < 800, within cone)
    assert "node_near_front" in node_ids
    
    # node_far_front must be culled (dist=2000 > 800, and not on shear plane)
    assert "node_far_front" not in node_ids
    
    # node_behind must be culled (behind gaze direction)
    assert "node_behind" not in node_ids
    
    # node_shear_plane must be preserved due to shear corridor transversality
    assert "node_shear_plane" in node_ids
    
    assert result["procedural"] is True
    assert result["total_nodes"] == 4
