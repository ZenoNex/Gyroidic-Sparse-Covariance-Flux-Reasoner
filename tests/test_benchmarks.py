import pytest
from src.benchmarks.topobench_evaluator import TopoBenchEvaluator
from src.benchmarks.reactbench_evaluator import ReactBenchEvaluator
from src.benchmarks.narrative_yield_evaluator import NarrativeYieldEvaluator
from src.benchmarks.run_all_benchmarks import BenchmarkSuiteRunner, run_all_benchmarks

def test_topobench_evaluator():
    evaluator = TopoBenchEvaluator(manifold_dim=3, level_k=1, device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 5
    assert results["total_count"] == 5
    assert results["tests"]["loop_closure"]["pass"] is True
    assert results["tests"]["manifold_contiguity"]["pass"] is True
    assert results["tests"]["geometric_shield"]["pass"] is True
    assert results["tests"]["betti_stability"]["pass"] is True
    assert results["tests"]["gyroid_flow"]["pass"] is True

def test_reactbench_evaluator():
    evaluator = ReactBenchEvaluator(k=4, degree=2, device="cpu")
    results = evaluator.run_suite()
    assert results["success"] is True
    assert results["passed_count"] == 5
    assert results["total_count"] == 5
    assert results["tests"]["branch_convergence"]["pass"] is True
    assert results["tests"]["cyclic_catalysis"]["pass"] is True
    assert results["tests"]["tristate_gate"]["pass"] is True
    assert results["tests"]["continuous_coprimality"]["pass"] is True
    assert results["tests"]["bezout_refresh"]["pass"] is True

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

def test_unified_benchmark_runner():
    report = run_all_benchmarks(device="cpu")
    assert report["success"] is True
    assert report["total_passed"] == 15
    assert report["total_tests"] == 15
    assert len(report["suites"]) == 3
