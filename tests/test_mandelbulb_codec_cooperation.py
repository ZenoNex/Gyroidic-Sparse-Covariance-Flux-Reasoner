import unittest
import torch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.augmentation.mandelbulb_gyroidic_augmenter import (
    MandelbulbAugmenter,
    MandelbulbGyroidicAugmenter,
    AugmentationConfig
)
from src.codec.gyroidic_codec import GyroidicCodec, CodecConfig, EncodingResult
from src.core.orchestrator import UniversalOrchestrator


class TestMandelbulbCodecCooperation(unittest.TestCase):
    """Test suite verifying cooperative integration coverage between MandelbulbAugmenter and GyroidicCodec."""

    def test_mandelbulb_encode_to_codec_bridge(self):
        """Verify MandelbulbAugmenter can directly encode through GyroidicCodec without competing."""
        config = AugmentationConfig(mandelbulb_power=8, max_iterations=20)
        augmenter = MandelbulbGyroidicAugmenter(config)

        # Synthetic feature tensor [batch, dim]
        X = torch.randn(4, 16)
        result = augmenter.encode_to_codec(X, text="Topological Soil State")

        self.assertIsInstance(result, EncodingResult)
        self.assertIsNotNone(result.encoded)
        self.assertIsNotNone(result.residue)
        self.assertGreater(result.commutativity_gap, 0.0)
        self.assertIn("entanglement_ratio", result.diagnostics)

    def test_codec_mandelbulb_prefold(self):
        """Verify GyroidicCodec can natively fold inputs via MandelbulbEmbedder."""
        codec_cfg = CodecConfig(K=3, n=4, use_mandelbulb_prefold=True, mandelbulb_power=8)
        codec = GyroidicCodec(codec_cfg)

        spectral_sig = torch.randn(24)
        result = codec.encode("Multimodal Entanglement", unified_spectral_signature=spectral_sig)

        self.assertIsInstance(result, EncodingResult)
        self.assertEqual(result.encoded.shape, torch.Size([4, 4]))
        self.assertEqual(result.text_residues.shape, torch.Size([3, 4, 4]))
        self.assertEqual(result.image_residues.shape, torch.Size([3, 4, 4]))

    def test_codec_augment_and_encode(self):
        """Verify GyroidicCodec.augment_and_encode performs joint augmentation and encoding."""
        codec = GyroidicCodec(CodecConfig(K=3, n=4))
        image_tensor = torch.randn(16, 16)

        result = codec.augment_and_encode("Joint Manifold", image=image_tensor, augmentation_factor=1)
        self.assertIsInstance(result, EncodingResult)
        self.assertGreaterEqual(result.diagnostics.get("entanglement_ratio", 0.0), 0.0)

    def test_universal_orchestrator_cooperative_dyad_ingestion(self):
        """Verify UniversalOrchestrator combines data association with cooperative Mandelbulb and Codec passes."""
        orchestrator = UniversalOrchestrator(dim=16)

        self.assertIsNotNone(orchestrator.gyroidic_codec)
        self.assertIsNotNone(orchestrator.mandelbulb_augmenter)

        text_emb = torch.randn(1, 16)
        image_emb = torch.randn(1, 16)

        # Standard ingestion returns the expected residue tensor for backward compatibility
        residues = orchestrator.ingest_knowledge_dyad(text_emb, image_emb, use_mandelbulb_prefold=True)
        self.assertEqual(residues.shape, torch.Size([1, 5]))
        self.assertIsNotNone(orchestrator.last_dyad_codec_result)

        # Diagnostics method returns the full diagnostic dictionary
        diag = orchestrator.ingest_knowledge_dyad_diagnostics(text_emb, image_emb, text_raw="Sovereign Dyad")
        self.assertIn("residues", diag)
        self.assertIn("codec_result", diag)
        self.assertIn("entanglement_ratio", diag)
        self.assertTrue(diag["has_mandelbulb_prefold"])


if __name__ == '__main__':
    unittest.main()
