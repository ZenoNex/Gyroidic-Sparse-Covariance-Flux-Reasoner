import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.cayley_cubic_probe import CayleyCubicProbe, CayleyAdversarialDecomposer
from src.core.topological_ingestion_validator import TopologicalIngestionValidator


class TestCayleyAdversarialHardening(unittest.TestCase):
    def setUp(self):
        self.decomposer = CayleyAdversarialDecomposer()
        self.validator = TopologicalIngestionValidator(
            poly_config=None,
            state_dim=64,
            pas_h_tolerance=0.05
        )

    def test_gibberifier_invisible_unicode_decomposition(self):
        # Sample mimicking Gibberifier output with vertical/alphabetic/arabic presentation forms & zero-width
        gibberified = "H\ufe12\ufeb2\ufe82e\ufb12\ufe18\ufe12l\ufe18\ufe12\ufb12l\ufe18o, \u200bworld\u200c!"
        res = self.decomposer.decompose(gibberified)

        # 1. Clean semantic text must be recovered exactly
        self.assertEqual(res["clean_text"], "Hello, world!")

        # 2. Adversarial flag must be triggered
        self.assertTrue(res["is_adversarial"])
        self.assertGreater(res["perturbation_density"], 0.10)
        self.assertGreater(len(res["perturbation_chars"]), 5)

        # 3. Cayley vector must satisfy Cayley cubic surface constraint
        # x^2 + y^2 + z^2 - xyz = 4 at A_1 nodal singularities (+-2, +-2, +-2)
        cayley_v = res["cayley_vector"]
        self.assertEqual(cayley_v.shape, torch.Size([3]))
        x, y, z = cayley_v.tolist()
        surface_val = x**2 + y**2 + z**2 - x * y * z
        self.assertAlmostEqual(surface_val, 4.0, places=4)

        # 4. Cayley scar energy must be non-zero
        self.assertGreater(res["cayley_scar_energy"], 0.0)

    def test_homoglyph_confusable_normalization(self):
        # Homoglyphs using fullwidth characters and compatibility forms
        homoglyph_text = "\uff28\uff45\uff4c\uff4c\uff4f" # Fullwidth "Hello"
        res = self.decomposer.decompose(homoglyph_text)
        self.assertEqual(res["clean_text"], "Hello")

    def test_clean_text_no_false_positives(self):
        normal_text = "Standard mathematical reasoning on Riemannian manifolds."
        res = self.decomposer.decompose(normal_text)
        self.assertEqual(res["clean_text"], normal_text)
        self.assertFalse(res["is_adversarial"])
        self.assertEqual(len(res["perturbation_chars"]), 0)
        self.assertEqual(res["cayley_scar_energy"], 0.0)

    def test_topological_ingestion_validator_on_gibberish(self):
        gibberified = "H\ufe12\ufeb2\ufe82e\ufb12\ufe18\ufe12l\ufe18\ufe12\ufb12l\ufe18o \u200bquantum\u200c resonance!"
        report = self.validator.validate_text(gibberified)

        # Ingestion must not crash or flag sterile Lazarus void on recoverable words
        self.assertTrue(report.get("is_adversarial_gibberish", False))
        self.assertEqual(report.get("clean_text"), "Hello quantum resonance!")
        self.assertGreater(report.get("cayley_scar_energy", 0.0), 0.0)
        self.assertEqual(report.get("cayley_vector").shape, torch.Size([3]))


if __name__ == "__main__":
    unittest.main()

