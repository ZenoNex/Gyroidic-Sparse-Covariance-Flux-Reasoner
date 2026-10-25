import pytest
import asyncio
from src.benchmarks.topobench_evaluator import TopoBenchEvaluator
from src.benchmarks.reactbench_evaluator import ReactBenchEvaluator
from src.benchmarks.narrative_yield_evaluator import NarrativeYieldEvaluator
from src.benchmarks.ai_reasoner_evaluator import AIReasonerEvaluator
from src.benchmarks.chordlock_evaluator import ChordlockEvaluator
from src.benchmarks.memory_palace_evaluator import MemoryPalaceEvaluator
from src.benchmarks.topological_singularity_evaluator import TopologicalSingularityEvaluator
from src.benchmarks.run_all_benchmarks import BenchmarkSuiteRunner, run_all_benchmarks, run_all_benchmarks_async

def test_topobench_evaluator():
    evaluator = TopoBenchEvaluator(manifold_dim=3, level_k=1, device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 7
    assert results["total_count"] == 7
    assert results["tests"]["loop_closure"]["pass"] is True
    assert results["tests"]["manifold_contiguity"]["pass"] is True
    assert results["tests"]["geometric_shield"]["pass"] is True
    assert results["tests"]["betti_stability"]["pass"] is True
    assert results["tests"]["gyroid_flow"]["pass"] is True
    assert results["tests"]["folium_transversality"]["pass"] is True
    assert results["tests"]["birkhoff_polytope_integrity"]["pass"] is True

    # Async activation verification
    async_res = asyncio.run(evaluator.async_run_suite())
    assert async_res["success"] is True
    assert async_res["passed_count"] == 7

def test_reactbench_evaluator():
    evaluator = ReactBenchEvaluator(k=4, degree=2, device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 7
    assert results["total_count"] == 7
    assert results["tests"]["branch_convergence"]["pass"] is True
    assert results["tests"]["cyclic_catalysis"]["pass"] is True
    assert results["tests"]["tristate_gate"]["pass"] is True
    assert results["tests"]["continuous_coprimality"]["pass"] is True
    assert results["tests"]["bezout_refresh"]["pass"] is True
    assert results["tests"]["muon_polar_decomposition"]["pass"] is True
    assert results["tests"]["carnot_mobius_backpressure"]["pass"] is True

    # Async activation verification
    async_res = asyncio.run(evaluator.async_run_suite())
    assert async_res["success"] is True
    assert async_res["passed_count"] == 7

def test_narrative_yield_evaluator():
    evaluator = NarrativeYieldEvaluator(device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 5
    assert results["total_count"] == 5
    assert results["tests"]["narrative_collapse"]["pass"] is True
    assert results["tests"]["legibility_audit"]["pass"] is True
    assert results["tests"]["daqf_amortization"]["pass"] is True
    assert results["tests"]["relational_friction"]["pass"] is True
    assert results["tests"]["homological_collapse_alarm"]["pass"] is True

def test_ai_reasoner_evaluator():
    evaluator = AIReasonerEvaluator(device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 5
    assert results["total_count"] == 5
    assert results["tests"]["reasoner_forward_coherence"]["pass"] is True
    assert results["tests"]["hessian_8x8_ledger"]["pass"] is True
    assert results["tests"]["bouligand_kinks"]["pass"] is True
    assert results["tests"]["cyclotomic_harmonics"]["pass"] is True
    assert results["tests"]["trajectory_decombining"]["pass"] is True

def test_chordlock_evaluator():
    evaluator = ChordlockEvaluator(device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 5
    assert results["total_count"] == 5
    assert results["tests"]["chordlock_identity"]["pass"] is True
    assert results["tests"]["coherent_prime_resonance"]["pass"] is True
    assert results["tests"]["multi_prime_lattice"]["pass"] is True
    assert results["tests"]["hardware_hedged_read"]["pass"] is True
    assert results["tests"]["chiral_unfolding"]["pass"] is True

def test_memory_palace_evaluator():
    evaluator = MemoryPalaceEvaluator(device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 5
    assert results["total_count"] == 5
    assert results["tests"]["spatial_isolation"]["pass"] is True
    assert results["tests"]["slerp_breeding"]["pass"] is True
    assert results["tests"]["outlier_resilience"]["pass"] is True
    assert results["tests"]["selective_puncture"]["pass"] is True
    assert results["tests"]["necrotic_well_crt"]["pass"] is True

def test_topological_singularity_evaluator():
    evaluator = TopologicalSingularityEvaluator(device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 5
    assert results["total_count"] == 5
    assert results["tests"]["layer1_octree_multiplexing"]["pass"] is True
    assert results["tests"]["layer2_void_friction_bypass"]["pass"] is True
    assert results["tests"]["layer3_mechanical_shear"]["pass"] is True
    assert results["tests"]["layer4_necrotic_stagnation"]["pass"] is True
    assert results["tests"]["layer5_pushforward_luminance"]["pass"] is True

def test_unified_benchmark_runner():
    report = run_all_benchmarks(device="cpu")
    assert report["success"] is True
    assert report["total_passed"] == report["total_tests"]
    assert report["total_passed"] == 39
    assert report["total_tests"] == 39
    assert len(report["suites"]) == 7

    # Async unified benchmark runner verification
    async_report = asyncio.run(run_all_benchmarks_async(device="cpu"))
    assert async_report["success"] is True
    assert async_report["total_passed"] == 39

