import torch
import unittest
from src.core.quantum_tda import QuantumBettiApproximator
from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
from src.core.archetype_engines import ArchetypalSynthesisEngine
from src.core.five_gate_pipeline import FiveGatePipeline, KnowledgeState

class TestUnicornSynthesis(unittest.TestCase):
    def setUp(self):
        self.state_dim = 16
        self.batch_size = 4
        self.dummy_state = torch.randn(self.batch_size, self.state_dim)

    def test_quantum_tda_minimax(self):
        """Phase 3: Verify the Quantum-Inspired Laplacian Polynomial approximation."""
        approximator = QuantumBettiApproximator()
        
        # Test 1: Disconnected graph (beta_0 should be close to 2)
        adj_matrix = torch.zeros(10, 10)
        adj_matrix[0,1] = adj_matrix[1,0] = 1
        adj_matrix[1,2] = adj_matrix[2,1] = 1
        adj_matrix[5,6] = adj_matrix[6,5] = 1
        
        results = approximator.estimate_betti_numbers(adj_matrix, max_dim=1)
        self.assertTrue(0 in results)
        # Should detect components; exact eig is hit for small N < 150
        # For N >= 150, our polynomial trace estimator triggers. Let's force it.
        
        large_adj = torch.zeros(200, 200)
        large_adj[0,1] = large_adj[1,0] = 1
        results_large = approximator.estimate_betti_numbers(large_adj, max_dim=1)
        # Betti_0 estimate shouldn't fail (be float/tensor)
        self.assertGreater(results_large[0], 0.0)

    def test_five_gate_pipeline(self):
        """Phase 3: Verify Tri-State Confabulation Logic."""
        pipeline = FiveGatePipeline(state_dim=self.state_dim)
        
        # Scenario A: Known State
        res_a = pipeline.process_pipeline(
            self.dummy_state[0], internal_certainty=0.9, current_pas_h=0.8, target_mischief=0.1
        )
        self.assertEqual(res_a["knowledge_state"], KnowledgeState.KNOWN)
        self.assertFalse(res_a["search_attempted"])

        # Scenario B: Search Needed (Low certainty, malformed query = no search, force Confab or Search Needed)
        # We manually overwrite well_posed projector for test isolation
        pipeline.search_gate.well_posed_projector.weight.data.fill_(10.0) 
        pipeline.search_gate.well_posed_projector.bias.data.fill_(10.0)
        
        res_b = pipeline.process_pipeline(
            self.dummy_state[0], internal_certainty=0.2, current_pas_h=0.1, target_mischief=0.2
        )
        self.assertTrue(res_b["search_attempted"])
        self.assertEqual(res_b["knowledge_state"], KnowledgeState.SEARCH_NEEDED)

        # Scenario C: Honest Confabulation (Search attempted, failed, high mischief)
        def mock_retrieve(q): return None, False
        res_c = pipeline.process_pipeline(
            self.dummy_state[0], internal_certainty=0.2, current_pas_h=0.1, target_mischief=0.9,
            diegetic_retrieval_fn=mock_retrieve
        )
        self.assertEqual(res_c["knowledge_state"], KnowledgeState.CONFABULATED)

    def test_archetypal_synthesis_engine(self):
        """Phase 2: Verify Grand Governor of Interpretation."""
        engine = ArchetypalSynthesisEngine(state_dim=self.state_dim)
        stranded = torch.randn(2, self.state_dim)
        
        # Scenario 1: Extreme Abstraction (Ego Death via Ra)
        # Ra = (Es(Tm + d)) / Li = (0.9 * (0.9 + 0.9)) / 0.1 = 16.2 >> 1.0
        res_death = engine.run_archetypes(
            self.dummy_state, stranded, current_mischief=0.5, phase_alignment=0.5,
            love_strengths=torch.tensor([1.0]), void_frictions=torch.tensor([0.0, 0.0]),
            global_dt=1.0, env_luminosity=1.0, volitional_scalar=0.0,
            system_entropy=0.9, memory_trauma=0.9, dissonance=0.9, lucidity_idx=0.1,
            raw_unquantized_state=self.dummy_state
        )
        self.assertTrue(res_death["system_collapsed"])

        # Scenario 2: Volitional Conjuring (High Willpower)
        # Ra = 0
        res_conjure = engine.run_archetypes(
            self.dummy_state, stranded, current_mischief=0.5, phase_alignment=0.5,
            love_strengths=torch.tensor([1.0]), void_frictions=torch.tensor([0.0, 0.0]),
            global_dt=1.0, env_luminosity=1.0, volitional_scalar=0.95, # Conjure!
            system_entropy=0.0, memory_trauma=0.0, dissonance=0.0, lucidity_idx=1.0,
            raw_unquantized_state=self.dummy_state
        )
        self.assertFalse(res_conjure["system_collapsed"])

    def test_silicon_sovereignty(self):
        """Phase 1: Dual-command queues and LSB stochastic routing."""
        try:
            engine = SiliconSovereigntyEngine()
        except RuntimeError as e:
            # Skip test if OpenCL platform is missing on the environment
            self.skipTest(str(e))
            
        poly_residues = torch.randint(0, 10, (self.batch_size, 5))
        rounded = engine.apply_stochastic_rounding(poly_residues, scale=1.0)
        
        # Should remain integer layout
        self.assertEqual(rounded.dtype, torch.int64) # PyOpenCL binding returns int64
        
        # Repunit parity shift validation
        targets = torch.zeros_like(poly_residues)
        valid_mask = engine.filter_dead_logic(poly_residues.numpy(), targets.numpy())
        self.assertEqual(valid_mask.shape, poly_residues.shape)

if __name__ == '__main__':
    unittest.main()
