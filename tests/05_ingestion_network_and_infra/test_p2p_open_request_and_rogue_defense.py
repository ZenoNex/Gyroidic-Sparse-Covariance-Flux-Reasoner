import unittest
import time
from src.p2p.freenet_ws_client import FreenetClient, RoguePredictorDefense
from src.core.federated_router import FederatedNetworkMonitor


class TestP2POpenRequestAndRogueDefense(unittest.TestCase):
    """
    Validates the Rogue ML Predictor Defense and OpenRequest architecture
    across the P2P Freenet / Locutus decentralized overlay network.
    """

    def setUp(self):
        self.defense = RoguePredictorDefense(jitter_min_ms=5.0, jitter_max_ms=25.0)
        self.client = FreenetClient(host="127.0.0.1", port=3000)

    def test_honest_timing_jitter_generation(self):
        """Verifies that honest timing jitter falls within bounds and increments shielded count."""
        jitter_ms = self.defense.generate_defense_jitter()
        self.assertGreaterEqual(jitter_ms, 5.0)
        self.assertLessEqual(jitter_ms, 25.0)
        self.assertEqual(self.defense.requests_shielded, 1)
        self.assertGreater(self.defense.total_jitter_applied_ms, 0.0)

    def test_zk_state_commitment(self):
        """Verifies deterministic Blake2s zero-knowledge state commitment."""
        state_a = {"weight": 0.42, "betti_1": 2, "soliton": "stable"}
        state_b = {"soliton": "stable", "weight": 0.42, "betti_1": 2}  # Permuted keys
        
        comm_a = self.defense.compute_zk_commitment(state_a)
        comm_b = self.defense.compute_zk_commitment(state_b)
        
        self.assertEqual(comm_a, comm_b)
        self.assertEqual(len(comm_a), 64)  # 256-bit hex hash

    def test_rogue_predictor_synthetic_timing_detection(self):
        """Tests that automated synthetic timing regularity triggers rogue predictor slashing."""
        peer_id = "peer_synthetic_probe_99"
        
        # Simulate 12 requests in the past with exact 0.01s intervals ending just before now
        now = time.time()
        self.defense.peer_history[peer_id] = [now - 0.12 + (i * 0.01) for i in range(12)]
        
        is_rogue, reason = self.defense.record_and_audit_peer(peer_id, "contract_zeta", 10.0)
        self.assertTrue(is_rogue)
        self.assertEqual(reason, "SYNTHETIC_TIMING_REGULARITY")
        self.assertTrue(self.defense.is_peer_slashed(peer_id))
        
        # Verify RP4 projective boundary assignment
        slash_rec = self.defense.slashed_peers[peer_id]
        self.assertEqual(slash_rec["status"], "ISOLATED")
        self.assertEqual(slash_rec["rp4_boundary_projection"], [0.0, 0.0, 0.0, 0.0, 1.0])

    def test_open_request_dispatch_and_telemetry(self):
        """Verifies freenet-core OpenRequest envelope packaging and security telemetry."""
        req_id = self.client.send_open_request(
            request_type="put",
            contract_key="contract_topological_state_alpha",
            state={"betti_0": 1, "betti_1": 3},
            apply_jitter=True
        )
        self.assertTrue(req_id.startswith("req_"))
        self.assertIn(req_id, self.client.cached_open_requests)
        
        cached = self.client.cached_open_requests[req_id]
        self.assertEqual(cached["type"], "open_request")
        self.assertEqual(cached["request"]["type"], "put")
        self.assertIn("zk_commitment", cached["security"])
        self.assertGreater(cached["security"]["jitter_applied_ms"], 0.0)

        # Telemetry
        telem = self.client.get_security_telemetry()
        self.assertTrue(telem["active_defense"])
        self.assertGreaterEqual(telem["total_open_requests_shielded"], 1)
        self.assertEqual(telem["cached_open_requests"], 1)

    def test_federated_router_security_telemetry(self):
        """Verifies FederatedNetworkMonitor captures rogue predictor defense telemetry."""
        monitor = FederatedNetworkMonitor(freenet_ws=self.client)
        
        # Trigger an open request on client
        self.client.get_contract("contract_zeta_omega")
        
        telem = monitor.get_telemetry()
        self.assertIn("rogue_predictor_defense", telem)
        self.assertTrue(telem["rogue_predictor_defense"])
        self.assertIn("open_requests_shielded", telem)
        self.assertGreaterEqual(telem["open_requests_shielded"], 1)
        self.assertIn("open_request_jitter_ms", telem)


if __name__ == "__main__":
    unittest.main()
