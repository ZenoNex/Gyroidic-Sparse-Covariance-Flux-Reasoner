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

