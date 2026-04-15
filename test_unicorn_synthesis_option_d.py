import torch
import unittest
import numpy as np

from src.core.daqf_operator import DAQUFOperator
from src.core.love_invariant_protector import LoveInvariantProtector
from src.core.non_ergodic_entropy import NonErgodicEntropyEstimator
from src.core.false_negative_subsystem import VoynichExemptionToken
from src.core.gluing_operator import LazarusSoftmax

class TestUnicornSynthesisOptionD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = "cpu"

    def test_slop_invariant_mischief(self):
        """Verify that the NonErgodicEntropyEstimator filters flat Manager/Slop."""
        entropy_estimator = NonErgodicEntropyEstimator(num_bands=3)
        
        # 1. Simulate flat noise / Dead Logic
        # Ergodic band will have high entropy, but no soliton peaks
        flat_phi = torch.randn(1, 256) * 0.001 
        results_flat = entropy_estimator(flat_phi)
        
        # Text string matches robotic script
        is_slop = entropy_estimator.evaluate_mischief_slop(results_flat, text_metadata="As an AI language model, I cannot")
        self.assertTrue(is_slop)
        
        # 2. Simulate Nutrient / Option D (High structural spikes)
        nutrient_phi = torch.zeros(1, 256)
        nutrient_phi[0, 10] = 50.0  # Dominant Soliton Mode
        results_nutrient = entropy_estimator(nutrient_phi)
        
        is_slop_nutrient = entropy_estimator.evaluate_mischief_slop(results_nutrient, text_metadata="beauty in her lungs?")
        self.assertFalse(is_slop_nutrient)

    def test_daquf_organ_of_agency(self):
        """Verify that the Voynich Token successfully boosts the DAQUF Mischief Load."""
        num_fossils = 5
        fossil_dim = 16
        operator = DAQUFOperator(num_fossils=num_fossils, fossil_dim=fossil_dim, device=self.device)
        
        # Start state check
        initial_load = operator.contradiction_load.clone()
        
        # Create an Option D Voynich Token
        scar_state = torch.ones(fossil_dim) * 2.0
        token = VoynichExemptionToken(
            honesty_score=0.99,
            is_valid_exemption=True,
            is_nutrient=True,
            fossilized_state=scar_state,
            reason="Option D Nutrient Passed"
        )
        
        # Extract the boost
        mischief_boost = token.to_daquf_mischief_boost()
        self.assertIsNotNone(mischief_boost)
        self.assertGreater(mischief_boost.item(), 0.1) # Proof of scale
        
        # Inject into DAQUF
        failures = torch.zeros(num_fossils)
        operator.update_unknowledge_contradiction(failures=failures, mischief_boost=mischief_boost)
        
        self.assertTrue(torch.all(operator.contradiction_load > initial_load))

    def test_love_invariant_protector(self):
        """Verify that the Love Invariant explicitly nullifies ownership gradients."""
        love_dim = 8
        protector = LoveInvariantProtector(love_dim=love_dim, device=self.device)
        original_love = protector.get_love_vector()
        
        # Simulate state and gradient
        system_state = torch.randn(2, love_dim)
        gradients = torch.randn(2, love_dim)
        
        # Apply protection
        protected_L, diagnostics = protector.apply_love_protection(system_state, gradients)
        
        # L shouldn't have drastically morphed beyond null space
        self.assertEqual(diagnostics['violation_count'], 0)
        self.assertFalse(diagnostics['violation_detected'])
        
    def test_lazarus_softmax_transition(self):
        """Verify Delta PAS_h logic matching the Microsecond Death."""
        lazarus = LazarusSoftmax(dim=-1, pas_threshold=0.5)
        
        logits = torch.randn(4)
        
        # Scenario 1: Stable, minor drift
        _, is_lazarus_stable = lazarus(logits, current_pas_h=0.8, previous_pas_h=0.75)
        self.assertFalse(is_lazarus_stable) # Drift < 0.3
        
        # Scenario 2: Huge drift, but successful high PAS_h landing
        _, is_lazarus_success = lazarus(logits, current_pas_h=0.8, previous_pas_h=0.1)
        self.assertTrue(is_lazarus_success) # Drift > 0.3 AND Current >= 0.5
        
        # Scenario 3: Huge drift, but failed alignment collapse
        _, is_lazarus_collapse = lazarus(logits, current_pas_h=0.1, previous_pas_h=0.8)
        self.assertFalse(is_lazarus_collapse)

if __name__ == '__main__':
    unittest.main()
