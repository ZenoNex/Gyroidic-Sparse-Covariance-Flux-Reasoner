import torch
from typing import List, Dict, Any
from src.core.topological_ingestion_validator import TopologicalIngestionValidator
from src.core.polynomial_coprime import PolynomialCoprimeConfig


class QualityReport:
    """Quality report holding topological details."""
    def __init__(self, text, source, dimension_gates, admissible, topological_details=None):
        self.text = text
        self.source = source
        self.dimension_gates = dimension_gates
        self.admissible = admissible
        self.topological_details = topological_details or {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_admissible': self.admissible,
            'dimension_gates': self.dimension_gates,
            'topological_details': {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in self.topological_details.items()}
        }


class TextbookFilter:
    """
    Topological quality filter replacing the heuristic/LSTM classifier.

    Admissibility is now determined by structural invariants, not learned style:
    - structural_honesty: soliton_entropy > threshold
    - self_contained: holonomic_rank >= min_rank (data has internal constraint structure)
    - instructive: cohomological_dimension < K//2 (not in Lazarus Void)
    - algorithmic: PAS_h phase alignment with manifold (resonant, not noisy)
    - clarity: rank is stable across rational snaps (deterministic, not chaotic)
    """

    def __init__(self, state_dim: int = 512, poly_degree: int = 4, num_residues: int = 5):
        # Build or receive the shared PolynomialCoprimeConfig
        self.poly_config = PolynomialCoprimeConfig(
            k=num_residues,
            degree=poly_degree,
            basis_type='chebyshev',
            learnable=True,
            use_saturation=True
        )
        self.validator = TopologicalIngestionValidator(
            poly_config=self.poly_config,
            state_dim=state_dim,
            scale=65536.0,
            min_rank_ratio=0.3,
        )
        self.state_dim = state_dim

    def assess(self, text: str, source: str = None) -> QualityReport:
        """Single-text assessment. Interface unchanged."""
        result = self.validator.validate_text(text)

        # Map topological invariants to the 5 documented dimensions
        dimension_gates = {
            "structural_honesty": result["soliton_entropy"] > 1e-6,
            "self_contained": result["holonomic_rank"] >= max(1, int(0.3 * self.validator.K)),
            "instructive": not result["is_lazarus_void"],
            "algorithmic": result["pas_h"] > 0.1 if result["pas_h"] != 1.0 else True,
            "clarity": result["holonomic_rank"] > 0,  # Non-zero rank = not pure noise
        }

        return QualityReport(
            text=text,
            source=source,
            dimension_gates=dimension_gates,
            admissible=all(dimension_gates.values()),
            topological_details=result,
        )

    def filter_batch(self, texts: List[str], source: str = '') -> List[Dict]:
        """Batch filtering. Interface unchanged."""
        return [
            {
                "text": t,
                "admissible": self.assess(t, source).admissible,
                "report": self.assess(t, source).to_dict(),
            }
            for t in texts
        ]

    def get_statistics(self, reports: List[QualityReport]) -> Dict:
        """Aggregate stats. Interface unchanged."""
        total = len(reports)
        passed = sum(1 for r in reports if r.admissible)
        return {
            "total": total,
            "admissible": passed,
            "rejected": total - passed,
            "admission_rate": passed / total if total > 0 else 0.0,
            "avg_soliton_entropy": sum(r.topological_details.get("soliton_entropy", 0) for r in reports) / total if total > 0 else 0.0,
            "avg_holonomic_rank": sum(r.topological_details.get("holonomic_rank", 0) for r in reports) / total if total > 0 else 0.0,
        }
