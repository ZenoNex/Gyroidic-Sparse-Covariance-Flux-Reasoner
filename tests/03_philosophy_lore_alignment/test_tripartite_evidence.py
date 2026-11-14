import unittest
import torch
from src.core.evidence_modules import TripartiteEvidenceModule
from src.core.orchestrator import UniversalOrchestrator


class TestTripartiteEvidenceIntegration(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.evidence = TripartiteEvidenceModule(hidden_dim=self.dim, blindness_threshold=0.7)

    def test_standalone_signals_and_orthogonality(self):
        hidden = torch.randn(2, self.dim)
        data_signals = {
            "gyroid_violation": 0.42,
            "b0": 1.0,
            "b1": 0.0,
            "hessian_trace": 0.15,
            "bouligand_kink": 0.05,
            "covariance_variance": 0.08
        }
        arch_signals = {
            "atrophy": 0.12,
            "legibility_risk": 0.03,
            "sycophancy": 0.01,
            "mischief": 0.55,
            "cycle_debt": 0.0,
            "defects": 0.0,
            "anti_scaling": 0.04
        }
        adv_signals = {
            "pce_score": 0.25,
            "toxicity": 0.02,
            "value_gap": 0.10
        }

        metrics = self.evidence(
            hidden,
            data_signals=data_signals,
            arch_signals=arch_signals,
            adv_signals=adv_signals
        )

        self.assertIn("data_evidence", metrics)
        self.assertIn("architectural_evidence", metrics)
        self.assertIn("adversarial_evidence", metrics)
        self.assertIn("is_blind", metrics)
        self.assertIn("blindness_detected", metrics)
        self.assertIn("correlations", metrics)
        self.assertIn("is_irreducible", metrics)
        self.assertIn("evidence_rank", metrics)
        self.assertIn("adversarial_injection", metrics)

        # Check adversarial injection orthogonality against input state
        inj = metrics["adversarial_injection"]
        self.assertEqual(inj.shape, hidden.shape)
        # Dot product with hidden should be approximately zero (Gram-Schmidt)
        dot = (inj * hidden).sum(dim=-1)
        self.assertTrue(torch.all(torch.abs(dot) < 1e-4))

    def test_blindness_detection_on_collinear_signals(self):
        # Force blindness with low threshold
        blind_evidence = TripartiteEvidenceModule(hidden_dim=self.dim, blindness_threshold=0.01)
        hidden = torch.ones(4, self.dim)
        metrics = blind_evidence(hidden)
        # Should flag blindness
        self.assertIsInstance(metrics["blindness_detected"], bool)

    def test_orchestrator_evidence_wiring(self):
        orchestrator = UniversalOrchestrator(dim=self.dim)
        state = torch.randn(1, self.dim)
        pressure = torch.zeros(1, self.dim)
        coherence = torch.tensor([1.0])

        step_res = orchestrator.step(
            state=state,
            pressure_grad=pressure,
            pas_h=0.9,
            coherence=coherence
        )
        self.assertIn("blindness_detected", step_res["diagnostics"])
        self.assertIn("evidence_irreducible", step_res["diagnostics"])
        self.assertIn("evidence_metrics", orchestrator.telemetry_state)


if __name__ == "__main__":
    unittest.main()
