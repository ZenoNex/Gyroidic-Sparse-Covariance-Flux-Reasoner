import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.topology.embedding_graph import GyroidicGraphManager, KnowledgeFossilNode

class TestGraphVisualization(unittest.TestCase):
    def setUp(self):
        self.manager = GyroidicGraphManager()
        
    def test_metric_extraction(self):
        """Verify that KnowledgeFossilNode extracts metrics correctly."""
        metrics = {
            'chiral_score': 0.8,
            'spectral_entropy': 2.5,
            'matrioshka_level': 3,
            'quantum_superposition': True,
            'repair_diagnostics': {'spectral_repair': True},
            'coprime_lock': False,
            'love_diagnostics': {'protected': True}
        }
        
        node = KnowledgeFossilNode("test_node", torch.zeros(64), "test_text", metrics)
        
        self.assertEqual(node.matrioshka_level, 3)
        self.assertTrue(node.quantum_superposition)
        self.assertTrue(node.repair_active)
        self.assertTrue(node.love_invariant_protected)
        self.assertFalse(node.coprime_lock)
        
    def test_label_dynamic_sizing(self):
        """Verify that high importance nodes get detailed labels, low get truncated."""
        # High importance: High Chirality + Low Degree (Novelty)
        metrics_high = {'chiral_score': 0.9, 'spectral_entropy': 1.0}
        long_text = "This is a very long text that represents a complex thought " * 10
        node_high = KnowledgeFossilNode("node_high", torch.zeros(64), long_text, metrics_high)
        
        # Low importance: Low Chirality
        metrics_low = {'chiral_score': 0.1, 'spectral_entropy': 1.0}
        node_low = KnowledgeFossilNode("node_low", torch.zeros(64), long_text, metrics_low)
        
        self.manager.nodes = [node_high, node_low]
        mermaid = self.manager.generate_mermaid_text()
        
        # High importance -> Should contain <br/> wraps and represent full thought
        self.assertIn("<br/>", mermaid.split('node_high')[1]) 
        # Low importance -> Should be truncated with ...
        self.assertIn("...", mermaid.split('node_low')[1])

    def test_mermaid_symbols_repaired(self):
        """Verify 'Repaired' nodes get the Wrench icon and Orange style."""
        metrics = {'repair_diagnostics': {'fixed': True}, 'spectral_entropy': 1.0}
        node = KnowledgeFossilNode("node_repaired", torch.zeros(64), "fixed text", metrics)
        self.manager.nodes = [node]
        
        mermaid = self.manager.generate_mermaid_text()
        
        self.assertIn("🔧", mermaid)
        self.assertIn("fill:#ff990022", mermaid) # Orange tint
        
    def test_style_priority(self):
        """Verify Style Priority: Quantum > High Chirality."""
        metrics = {'quantum_superposition': True, 'chiral_score': 0.9} # Both high
        node = KnowledgeFossilNode("node_conflict", torch.zeros(64), "conflict", metrics)
        self.manager.nodes = [node]
        
        mermaid = self.manager.generate_mermaid_text()
        
        # Should be PURPLE (Quantum), not PINK (Chiral)
        self.assertIn("fill:#9900ff22", mermaid) 
        self.assertNotIn("fill:#ff00f222", mermaid)

    def test_mermaid_symbols_matrioshka(self):
        """Verify 'Matrioshka' nodes get the Doll icon."""
        metrics = {'matrioshka_level': 4, 'spectral_entropy': 1.0}
        node = KnowledgeFossilNode("node_matrioshka", torch.zeros(64), "deep state", metrics)
        self.manager.nodes = [node]
        
        mermaid = self.manager.generate_mermaid_text()
        
        self.assertIn("🪆4", mermaid)

    def test_mermaid_symbols_locked(self):
        """Verify 'Locked' nodes get the Lock icon and Green style."""
        metrics = {'coprime_lock': True, 'spectral_entropy': 1.0}
        node = KnowledgeFossilNode("node_locked", torch.zeros(64), "fossilized", metrics)
        self.manager.nodes = [node]
        
        mermaid = self.manager.generate_mermaid_text()
        
        self.assertIn("🔐", mermaid)
        self.assertIn("fill:#00ff9922", mermaid) # Green tint

if __name__ == '__main__':
    unittest.main()
