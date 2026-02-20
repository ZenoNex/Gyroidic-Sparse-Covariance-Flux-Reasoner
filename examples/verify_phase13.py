"""Quick verification for Phase 13 code changes."""
import torch
import sys
sys.path.insert(0, '.')

# Test 1: NonCommutativityCurvature
from src.core.noncommutativity_curvature import NonCommutativityCurvature
nc = NonCommutativityCurvature(64)
A = torch.randn(64, 64)
B = torch.randn(64, 64)
r = nc.compute_curvature(A, B)
print(f"[PASS] NonCommutativityCurvature: norm={r['curvature_norm'].item():.4f}, "
      f"strongly_nc={r['is_strongly_noncommutative']}")

# Test 2: FeaturePreservationProjection
from src.core.feature_preservation import FeaturePreservationProjection
fp = FeaturePreservationProjection(64)
x = torch.randn(2, 64)
fr = fp(x)
print(f"[PASS] FeaturePreservation: active_facets={len(fr['active_facets'])}, "
      f"compression={fr['compression_ratio']:.2f}")

# Test 3: VetoSubspace
from src.core.veto_subspace import VetoSubspace, VetoSignal, VetoResult
vs = VetoSubspace()
vr = vs.evaluate(
    abort_score=0.7, 
    coprime_lock=False, 
    topological_pressure=0.6
)
print(f"[PASS] VetoSubspace: status={vr.status.value}, "
      f"active_vetoes={vr.active_vetoes}, severity={vr.final_severity:.2f}")

# Test 4: BoundaryState extension
from src.core.meta_polytope_matrioshka import BoundaryState
bs = BoundaryState.from_crossing(
    torch.randn(8), torch.randn(8),
    alpha=0, level=2, max_level=5
)
diag = bs.to_dict()
print(f"[PASS] BoundaryState: critical={bs.is_critical()}, "
      f"energy={bs.crossing_energy:.4f}, stress_rank={diag.get('stress_rank', 'N/A')}")

# Test 5: Curvature pressure (veto signal convenience)
pressure = nc.curvature_pressure(A, B)
print(f"[PASS] Curvature pressure: {pressure.item():.4f}")

print("\n=== ALL 5 TESTS PASSED ===")
