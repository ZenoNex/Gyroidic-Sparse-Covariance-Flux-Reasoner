
import unittest
import torch
import math
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from src.optimization.codes_driver import CODES

class TestChordlock(unittest.TestCase):
    def setUp(self):
        self.codes = CODES()
        
    def test_chordlock_identity(self):
        """Test that chordlock doesn't crash on simple input."""
        latent = torch.randn(10)
        locked = self.codes.chordlock(latent, primes=[])
        # With empty primes, it uses internal harmonics. 
        # Output should be modulated but same shape.
        self.assertEqual(latent.shape, locked.shape)
        
    def test_chordlock_resonance(self):
        """Test that resonant signals are preserved and dissonant ones suppressed."""
        # Create a signal that is a multiple of a prime (e.g., 2)
        # Latent values = 4.0 (2 * 2.0)
        # Phase alignment: values near integer multiples of P should be amplified relative to outliers?
        # Our logic: gating_factor = (Average(cos(2pi * x / p)) + 1) / 2
        # If x = k * p, cos(2pi*k) = 1. Average = 1. Gating = 1.
        # If x = (k + 0.5) * p, cos = -1. Average = -1. Gating = 0.
        
        prime = 2
        
        # Resonant input
        resonant_val = torch.tensor([2.0, 4.0, 6.0])
        locked_res = self.codes.chordlock(resonant_val, primes=[prime])
        
        # gating factor should be close to 1.0
        # locked_res should be close to resonant_val
        self.assertTrue(torch.allclose(locked_res, resonant_val, atol=1e-5))
        
        # Dissonant input (anti-resonant)
        dissonant_val = torch.tensor([1.0, 3.0, 5.0]) # 0.5 * 2, 1.5 * 2
        locked_diss = self.codes.chordlock(dissonant_val, primes=[prime])
        
        # gating factor should be close to 0.0
        self.assertTrue(torch.allclose(locked_diss, torch.zeros_like(dissonant_val), atol=1e-5))
        
    def test_multi_prime_resonance(self):
        """Test with multiple primes."""
        # Primes [2, 3]
        # Common multiple 6 should be resonant for both (avg cos = 1).
        primes = [2, 3]
        val = torch.tensor([6.0])
        locked = self.codes.chordlock(val, primes=primes)
        self.assertTrue(torch.allclose(locked, val, atol=1e-5))
        
        # Value 1.0:
        # P=2: cos(pi) = -1
        # P=3: cos(2pi/3) = -0.5
        # Avg = -0.75
        # Gate = (-0.75 + 1)/2 = 0.125
        # Squared gate = 0.0156
        # Output should be heavily suppressed
        val_diss = torch.tensor([1.0])
        locked_diss = self.codes.chordlock(val_diss, primes=primes)
        self.assertTrue(locked_diss.item() < 0.1)

if __name__ == '__main__':
    unittest.main()
