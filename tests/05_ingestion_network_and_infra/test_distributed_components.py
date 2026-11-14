import unittest
import torch
import os
import json
import time
from src.core.investor_news_ingestor import InvestorNewsIngestor
from src.topology.bonfire_network import BonfireNetwork

class TestDistributedComponents(unittest.TestCase):
    def setUp(self):
        self.ingestor = InvestorNewsIngestor()
        self.network = BonfireNetwork(node_id="test_node", local_url="http://localhost:9999")

    def tearDown(self):
        self.network.close()

    def test_robots_txt_leniency(self):
        # Local or common URLs should return allowed by default
        self.assertTrue(self.ingestor.is_allowed("http://localhost:8000/news"))
        self.assertTrue(self.ingestor.is_allowed("https://finance.yahoo.com/news/rssindex"))

    def test_rss_feed_regex_parsing(self):
        # Test clean RSS parsing of simulated feed content
        simulated_rss = """
        <rss>
            <channel>
                <item>
                    <title><![CDATA[Alpha Market Surge]]></title>
                    <link>http://example.com/item1</link>
                    <description>Market gains 5% in early trading.</description>
                </item>
                <item>
                    <title>Beta Contraction</title>
                    <link>http://example.com/item2</link>
                    <description><![CDATA[Topological assets decline.]]></description>
                </item>
            </channel>
        </rss>
        """
        # Patch urllib.request.urlopen to return simulated RSS feed
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = simulated_rss.encode('utf-8')
        
        with patch('urllib.request.urlopen', return_value=mock_response):
            items = self.ingestor.fetch_rss_feed("http://example.com/rss")
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["title"], "Alpha Market Surge")
            self.assertEqual(items[0]["description"], "Market gains 5% in early trading.")
            self.assertEqual(items[1]["title"], "Beta Contraction")
            self.assertEqual(items[1]["description"], "Topological assets decline.")

    def test_bonfire_peer_list_management(self):
        initial_peer_count = len(self.network.peers)
        new_peer = "http://localhost:9005"
        
        # Test add
        self.network.add_peer(new_peer)
        self.assertIn(new_peer, self.network.peers)
        
        # Test remove
        self.network.remove_peer(new_peer)
        self.assertNotIn(new_peer, self.network.peers)

    def test_bonfire_nonblocking_synchronize(self):
        # Verify synchronization returns cached results immediately (O(1))
        start_time = time.time()
        signatures = self.network.synchronize_invariants("sig_local_test")
        duration = time.time() - start_time
        
        # Should be instantaneous (sub-millisecond) rather than waiting on HTTP timeouts
        self.assertLess(duration, 0.05)
        self.assertEqual(self.network.local_signature, "sig_local_test")

    def test_bonfire_nonblocking_kelly_consensus(self):
        # Verify Kelly consensus returns consensus stats instantly from cache
        start_time = time.time()
        consensus = self.network.share_kelly_consensus(local_kelly=0.75, local_p_success=0.8)
        duration = time.time() - start_time
        
        self.assertLess(duration, 0.05)
        self.assertIn("consensus_kelly", consensus)
        self.assertIn("consensus_p_success", consensus)
        self.assertEqual(self.network.local_kelly, 0.75)
        self.assertEqual(self.network.local_p_success, 0.8)

    def test_bonfire_admr_fallback(self):
        # Test that distributing ADMR to an empty registry safely returns None instantly
        self.network.healthy_peers = []
        dummy_state = torch.randn(2, 64)
        
        start_time = time.time()
        result = self.network.distribute_admr_to_synthetic_endpoint(dummy_state)
        duration = time.time() - start_time
        
        self.assertIsNone(result)
        self.assertLess(duration, 0.05)

    def test_hardware_fingerprint_and_creator_auth(self):
        from src.safety.hardware_fingerprint import (
            get_stable_hardware_fingerprint,
            discover_public_ip,
            verify_sole_creator,
            is_creator_initialized
        )
        import src.safety.hardware_fingerprint as hf
        from unittest.mock import patch
        
        fingerprint = get_stable_hardware_fingerprint()
        self.assertEqual(len(fingerprint), 64)
        
        ip = discover_public_ip()
        self.assertTrue(isinstance(ip, str))
        
        temp_sig_file = os.path.join("data", "creator_sig_temp_test.json")
        if os.path.exists(temp_sig_file):
            os.remove(temp_sig_file)
            
        with patch.object(hf, 'SIG_FILE', temp_sig_file):
            self.assertFalse(is_creator_initialized())
            
            res_reg = verify_sole_creator("super_secure_creator_passcode_1337")
            self.assertTrue(res_reg)
            self.assertTrue(is_creator_initialized())
            
            self.assertTrue(verify_sole_creator("super_secure_creator_passcode_1337"))
            self.assertFalse(verify_sole_creator("incorrect_passcode"))
            
            if os.path.exists(temp_sig_file):
                os.remove(temp_sig_file)

if __name__ == "__main__":
    unittest.main()
