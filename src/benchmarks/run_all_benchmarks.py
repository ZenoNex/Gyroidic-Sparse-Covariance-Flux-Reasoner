import sys
import os
import time
import logging
from typing import Dict, Any

# Guarantee project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.benchmarks.topobench_evaluator import TopoBenchEvaluator
from src.benchmarks.reactbench_evaluator import ReactBenchEvaluator
from src.benchmarks.narrative_yield_evaluator import NarrativeYieldEvaluator
from src.benchmarks.ai_reasoner_evaluator import AIReasonerEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BenchmarkSuiteRunner")

class BenchmarkSuiteRunner:
    """
    Unified Orchestrator for all Gyroidic Flux Reasoner benchmarks:
    - TopoBench: Holonomy, Contiguity, Geometric Shield, Homology, Gyroid Flow
    - ReactBench: Convergence, Cyclic Catalysis, Tri-State, Coprimality, Bezout Refresh
    - NarrativeYield: Collapse Detection, Legibility Audit, DAQUF Amortization, Relational Friction, Homological Protection
    - AIReasoner: Forward Inference Coherence, 8x8 Hessian Ledger, Bouligand Kinks, Cyclotomics, Trajectory Decombining
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.topobench = TopoBenchEvaluator(device=device)
        self.reactbench = ReactBenchEvaluator(device=device)
        self.narrative = NarrativeYieldEvaluator(device=device)
        self.aireasoner = AIReasonerEvaluator(device=device)

    def run_all(self) -> Dict[str, Any]:
        """
        Executes all benchmark suites, prints consolidated tabular summary,
        and returns aggregated report dictionary.
        """
        start_time = time.time()
        logger.info("Initiating Comprehensive Benchmark Evaluation across all suites...")
        
        suite_reports = {}
        
        # 1. Execute TopoBench
        logger.info(">>> Executing Suite 1/4: TopoBenchEvaluator")
        topo_res = self.topobench.run_suite()
        suite_reports['TopoBenchEvaluator'] = topo_res
        
        # 2. Execute ReactBench
        logger.info(">>> Executing Suite 2/4: ReactBenchEvaluator")
        react_res = self.reactbench.run_suite()
        suite_reports['ReactBenchEvaluator'] = react_res
        
        # 3. Execute NarrativeYield
        logger.info(">>> Executing Suite 3/4: NarrativeYieldEvaluator")
        narr_res = self.narrative.run_suite()
        suite_reports['NarrativeYieldEvaluator'] = narr_res
        
        # 4. Execute AIReasoner
        logger.info(">>> Executing Suite 4/4: AIReasonerEvaluator")
        ai_res = self.aireasoner.run_suite()
        suite_reports['AIReasonerEvaluator'] = ai_res
        
        elapsed = time.time() - start_time
        
        # Calculate summary metrics
        total_passed = sum(s['passed_count'] for s in suite_reports.values())
        total_tests = sum(s['total_count'] for s in suite_reports.values())
        all_passed = (total_passed == total_tests) and all(s['success'] for s in suite_reports.values())
        
        # Render ASCII Summary Table
        summary_lines = [
            "",
            "=" * 80,
            "GYROIDIC SPARSE COVARIANCE FLUX REASONER - BENCHMARK EXECUTION SUMMARY",
            "=" * 80,
            f"{'Suite Name':<28} {'Passed':<10} {'Total':<10} {'Status':<10}",
            "-" * 80
        ]
        
        for name, rep in suite_reports.items():
            status_str = "PASS" if rep['success'] else "FAIL"
            summary_lines.append(f"{name:<28} {rep['passed_count']:<10} {rep['total_count']:<10} {status_str:<10}")
            
        summary_lines.append("-" * 80)
        pass_pct = (total_passed / total_tests * 100.0) if total_tests > 0 else 0.0
        overall_status = "PASS" if all_passed else "FAIL"
        summary_lines.append(f"{'TOTAL':<28} {total_passed:<10} {total_tests:<10} {overall_status:<10} ({pass_pct:.1f}%)")
        summary_lines.append(f"Execution Duration: {elapsed:.2f} seconds")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        for line in summary_lines:
            print(line)
            
        final_summary = {
            'timestamp': time.time(),
            'duration_seconds': elapsed,
            'total_passed': total_passed,
            'total_tests': total_tests,
            'success': all_passed,
            'suites': suite_reports
        }
        return final_summary

def run_all_benchmarks(device: str = 'cpu') -> Dict[str, Any]:
    runner = BenchmarkSuiteRunner(device=device)
    return runner.run_all()

if __name__ == "__main__":
    results = run_all_benchmarks()
    sys.exit(0 if results['success'] else 1)
