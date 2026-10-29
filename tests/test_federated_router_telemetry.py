import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.federated_router import FederatedNetworkMonitor
from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType

class TestFederatedRouterCommunityCompute(unittest.TestCase):
    def test_federated_network_monitor_telemetry(self):
        monitor = FederatedNetworkMonitor()
        telemetry = monitor.get_telemetry()
        
        self.assertIn("freenet_p2p_status", telemetry)
        self.assertIn("openrouter_status", telemetry)
        self.assertIn("bonfire_status", telemetry)
        self.assertIn("bonfire_peer_count", telemetry)
        self.assertIn("egalitarian_kelly_consensus", telemetry)
        self.assertIn("community_compute_affordance", telemetry)
        self.assertIn("p2p_mesh_gauge", telemetry)
        
        # Test with mocked bonfire ring
        class DummyBonfireRing:
            def __init__(self):
                self.peer_allocations = {"NodeA": 0.45, "NodeB": 0.55}
                
        ring = DummyBonfireRing()
        monitor_with_ring = FederatedNetworkMonitor(bonfire_ring=ring)
        telem_ring = monitor_with_ring.get_telemetry()
        
        self.assertEqual(telem_ring["bonfire_status"], "ONLINE")
        self.assertEqual(telem_ring["bonfire_peer_count"], 2)
        self.assertAlmostEqual(telem_ring["egalitarian_kelly_consensus"], 0.5, places=3)
        self.assertGreaterEqual(telem_ring["community_compute_affordance"], 0.0)

if __name__ == "__main__":
    unittest.main()

