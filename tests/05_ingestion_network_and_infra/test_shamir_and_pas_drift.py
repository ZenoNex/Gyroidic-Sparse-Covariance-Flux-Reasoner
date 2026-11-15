import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

os.environ["Sovereign_Disable_Background"] = "1"

from src.p2p import BonfireNomadicRing, ShamirThresholdSharing
from src.core.global_pas_api import GLYPHLOCKHandshake


def test_shamir_threshold_int_and_bytes_roundtrip():
    shamir = ShamirThresholdSharing()

    secret_int = 0xDEADBEEF123456789ABCDEF0123456789
    shares = shamir.split_int(secret_int, threshold=3, num_shares=5)
    assert len(shares) == 5

    # Any 3 of 5 shares reconstruct the exact secret via Lagrange interpolation at x=0
    assert shamir.reconstruct_int([shares[0], shares[2], shares[4]]) == secret_int
    assert shamir.reconstruct_int([shares[1], shares[3], shares[4]]) == secret_int

    # Fewer than threshold shares do not reconstruct the secret
    assert shamir.reconstruct_int([shares[0], shares[1]]) != secret_int

    # Multi-block (>64 bytes) secret bytes roundtrip
    secret_bytes = b"SOVEREIGN_NOMADIC_SALT_" * 4 + b"END"
    byte_shares = shamir.split_bytes(secret_bytes, threshold=3, num_shares=5)
    assert shamir.reconstruct_bytes([byte_shares[0], byte_shares[2], byte_shares[4]]) == secret_bytes

    with pytest.raises(ValueError):
        shamir.reconstruct_bytes(byte_shares[:2])


def test_bonfire_nomadic_ring_isolated_traffic_point_recovery():
    mock_freenet = MagicMock()
    ring = BonfireNomadicRing(freenet_client=mock_freenet)

    traffic_points = ["isolated_room_a", "portable_flashdrive_vault", "freenet_peer_1", "freenet_peer_2"]
    secret_digest = b"OPTION_D_NOMADIC_VAULT_KEY_MATERIAL_V1"

    packets = ring.distribute_threshold_shares(
        local_peer_id="node_local",
        secret_id="vault_key_01",
        secret_bytes=secret_digest,
        threshold=2,
        traffic_points=traffic_points,
    )
    assert len(packets) == 4
    assert mock_freenet.publish.call_count == 4

    # Simulate 2 traffic points unreachable; ingest only packets 1 and 3 via handler
    ring._handle_shamir_share_update(packets[1])
    ring._handle_shamir_share_update(packets[3])

    recovered = ring.reconstruct_from_traffic_points("vault_key_01")
    assert recovered == secret_digest


def test_global_pas_real_drift_and_z3_modulation():
    handshake = GLYPHLOCKHandshake()

    # Coherent harmonic state (constant/aligned phases -> high PAS_h)
    base_state = torch.ones(64) * 2.0
    tok1 = handshake.create_handshake("client_a", state_tensor=base_state)
    assert tok1.pas_zeta > 0.0
    assert tok1.pas_zeta <= 0.05

    # Small drift step -> still within zeta_drift
    small_drift_state = base_state + 0.01 * torch.randn(64)
    tok2 = handshake.create_handshake("client_a", state_tensor=small_drift_state)
    assert tok2.pas_zeta > 0.0

    # Large drift jump -> must exceed zeta_drift (0.05) and fail aura_out / is_valid
    large_drift_state = -base_state * 3.0 + torch.randn(64) * 2.0
    tok3 = handshake.create_handshake("client_a", state_tensor=large_drift_state, prev_state_tensor=base_state)
    assert tok3.pas_zeta > 0.05
    assert tok3.aura_out is False
    assert handshake.verify(tok3) is False

    # GlobalPASAPI.track_coherence must also track non-zero PAS_zeta and JSON-serializable records
    from src.core.global_pas_api import GlobalPASAPI
    api = GlobalPASAPI()
    rec1 = api.track_coherence("edge_1", base_state)
    assert rec1["pas_zeta"] > 0.0
    rec2 = api.track_coherence("edge_1", large_drift_state)
    assert rec2["pas_zeta"] > 0.05
    assert rec2["is_coherent"] is False


def test_non_euler_shamir_and_procedural_closed_topology_sequences():
    import tempfile
    from src.p2p import NonEulerShamirTopologySharing
    from src.topology.embedding_graph import GyroidicGraphManager, KnowledgeFossilNode

    # Reject collapsed/trivial Euler point loop
    non_euler = NonEulerShamirTopologySharing()
    with pytest.raises(ValueError):
        non_euler.split_closed_topology_sequence([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    # Build a 3-node procedural fossil cycle (non-trivial closed triangle)
    mgr = GyroidicGraphManager(data_dir=tempfile.gettempdir(), dim=16)
    shared = torch.ones(16)
    coords = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (10.0, 20.0, 0.0)]
    for i, (x, y, z) in enumerate(coords):
        node = KnowledgeFossilNode(
            node_id=f"cyc_{i}",
            state=shared + 0.01 * i,
            text=f"Cycle node {i}",
            metrics={"chiral_score": 0.1, "spectral_entropy": 0.5, "tags": ["cycle_test"]},
        )
        node.x, node.y, node.z = x, y, z
        mgr.nodes.append(node)

    proc = mgr.query_procedural_graph(camera_pos=(10.0, 5.0, 150.0), gaze_dir=(0.0, 0.0, -1.0), fov_deg=90.0)
    assert proc["procedural_stats"]["closed_topology_count"] >= 1
    assert len(proc["closed_topology_sequences"]) >= 1
    assert proc["closed_topology_sequences"][0]["closed"] is True

    # Export closed-topology sequence via BonfireNomadicRing Non-Euler Shamir (2, 3) shares
    mock_freenet = MagicMock()
    ring = BonfireNomadicRing(freenet_client=mock_freenet)
    traffic_points = ["isolated_room_x", "portable_vault_y", "freenet_peer_z"]
    exported = mgr.export_closed_topology_shares(
        bonfire_ring=ring,
        local_peer_id="node_local",
        traffic_points=traffic_points,
        threshold=2,
        camera_pos=(10.0, 5.0, 150.0),
        gaze_dir=(0.0, 0.0, -1.0),
        fov_deg=90.0,
    )
    assert len(exported) >= 1
    assert exported[0]["structural_algebra_attestation"]["idea_expression_dichotomy_compliant"] is True
    assert exported[0]["structural_algebra_attestation"]["non_invertible_puncture_ratio"] >= 0.3
    topo_id = exported[0]["topology_id"]
    pkts = exported[0]["packets"]
    # Simulate 1 traffic point offline; reconstruct from packets 0 and 2
    ring._handle_shamir_share_update(pkts[0])
    ring._handle_shamir_share_update(pkts[2])
    rec = ring.reconstruct_closed_topology_from_traffic_points(topo_id)
    assert rec["closure"]["is_nontrivial"] is True
    assert rec["selective_puncture_applied"] is True
    assert rec["non_invertible_puncture_ratio"] >= 0.3
    assert len(rec["coords"]) == 3

