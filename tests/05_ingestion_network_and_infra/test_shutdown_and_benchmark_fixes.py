import os
import sys
import unittest
import torch

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.polynomial_crt import PolynomialCRT
from src.benchmarks.reactbench_evaluator import ReactBenchEvaluator
from src.core.knowledge_dyad_fossilizer import DyadFossilizer
from src.data.knowledge_ingestor import ArXivSovereignIngestor

class TestShutdownAndBenchmarkFixes(unittest.TestCase):
    def test_reactbench_and_crt_shape_resilience_k4_and_k5(self):
        """Verify that ReactBenchEvaluator and PolynomialCRT handle both k=4, k=5 and arbitrary k without non-singleton mismatch."""
        branch_a = [0.2, 0.4, 0.6]
        branch_b = [0.22, 0.45, 0.55]

        # Test k=4
        rb_k4 = ReactBenchEvaluator(k=4, degree=2)
        success_4, p_val_4 = rb_k4.evaluate_branch_convergence(branch_a, branch_b)
        self.assertTrue(success_4)
        self.assertIsInstance(p_val_4, float)

        # Test k=5 (the engine runtime default)
        rb_k5 = ReactBenchEvaluator(k=5, degree=2)
        success_5, p_val_5 = rb_k5.evaluate_branch_convergence(branch_a, branch_b)
        self.assertTrue(success_5)
        self.assertIsInstance(p_val_5, float)

        # Direct PolynomialCRT test where input tensor has different K than poly_config
        cfg_k5 = PolynomialCoprimeConfig(k=5, degree=2)
        crt = PolynomialCRT(poly_config=cfg_k5)

        # Pass residues with 4 channels
        residues_4 = torch.rand(1, 4, 3)
        recon_4 = crt.forward(residues_4)
        self.assertEqual(recon_4.shape, (1, 3))

        # Pass residues with 6 channels
        residues_6 = torch.rand(1, 6, 3)
        recon_6 = crt.forward(residues_6)
        self.assertEqual(recon_6.shape, (1, 3))

    def test_arxiv_sovereign_ingestor_stop(self):
        """Verify that ArXivSovereignIngestor terminates gracefully when stop() is called."""
        fossilizer = DyadFossilizer(storage_dir="test_storage")
        ingestor = ArXivSovereignIngestor(fossilizer=fossilizer, engine_dim=64)
        self.assertFalse(ingestor._stop_event.is_set())
        ingestor.stop()
        self.assertTrue(ingestor._stop_event.is_set())
        # With stop set, query generation and searxng searches return empty/False immediately
        self.assertEqual(ingestor._generate_larynx_query(), "")
        self.assertFalse(ingestor.ingest_searxng_by_query("test"))

    def test_benchmark_dashboard_html_exists(self):
        """Verify benchmark_dashboard.html file exists in expected paths for HTTP server."""
        p = os.path.join("src", "ui", "benchmark_dashboard.html")
        self.assertTrue(os.path.exists(p))

if __name__ == "__main__":
    unittest.main()
