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
from src.benchmarks.chordlock_evaluator import ChordlockEvaluator
from src.benchmarks.memory_palace_evaluator import MemoryPalaceEvaluator
from src.benchmarks.topological_singularity_evaluator import TopologicalSingularityEvaluator

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
    - Chordlock: Harmonic Closure, Coherent Prime Resonance, Lattice Scaling, Hedged Read Path, Chiral Unfolding
    - MemoryPalace: High-D Spatial Isolation, SLERP Soliton Breeding, Outlier Resilience, Puncture Invariance, CRT Compression
    - TopologicalSingularity: Octree Multiplexing, Coprime Bypass, Mechanical Shear, Necrotic Stagnation, Push-Forward Luminance
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.topobench = TopoBenchEvaluator(device=device)
        self.reactbench = ReactBenchEvaluator(device=device)
        self.narrative = NarrativeYieldEvaluator(device=device)
        self.aireasoner = AIReasonerEvaluator(device=device)
        self.chordlock = ChordlockEvaluator(device=device)
        self.memory_palace = MemoryPalaceEvaluator(device=device)
        self.topological_singularity = TopologicalSingularityEvaluator(device=device)

    def run_all(self) -> Dict[str, Any]:
        """
        Executes all benchmark suites, prints consolidated tabular summary,
        and returns aggregated report dictionary.
        """
        start_time = time.time()
        logger.info("Initiating Comprehensive Benchmark Evaluation across all 7 suites...")
        
        suite_reports = {}
        
        # 1. Execute TopoBench
        logger.info(">>> Executing Suite 1/7: TopoBenchEvaluator")
        topo_res = self.topobench.run_suite()
        suite_reports['TopoBenchEvaluator'] = topo_res
        
        # 2. Execute ReactBench
        logger.info(">>> Executing Suite 2/7: ReactBenchEvaluator")
        react_res = self.reactbench.run_suite()
        suite_reports['ReactBenchEvaluator'] = react_res
        
        # 3. Execute NarrativeYield
        logger.info(">>> Executing Suite 3/7: NarrativeYieldEvaluator")
        narr_res = self.narrative.run_suite()
        suite_reports['NarrativeYieldEvaluator'] = narr_res
        
        # 4. Execute AIReasoner
        logger.info(">>> Executing Suite 4/7: AIReasonerEvaluator")
        ai_res = self.aireasoner.run_suite()
        suite_reports['AIReasonerEvaluator'] = ai_res

        # 5. Execute Chordlock
        logger.info(">>> Executing Suite 5/7: ChordlockEvaluator")
        chord_res = self.chordlock.run_suite()
        suite_reports['ChordlockEvaluator'] = chord_res

        # 6. Execute MemoryPalace
        logger.info(">>> Executing Suite 6/7: MemoryPalaceEvaluator")
        mem_res = self.memory_palace.run_suite()
        suite_reports['MemoryPalaceEvaluator'] = mem_res

        # 7. Execute TopologicalSingularity
        logger.info(">>> Executing Suite 7/7: TopologicalSingularityEvaluator")
        sing_res = self.topological_singularity.run_suite()
        suite_reports['TopologicalSingularityEvaluator'] = sing_res
        
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

    async def async_run_all(self) -> Dict[str, Any]:
        """Asynchronously executes all 7 benchmark suites in a background thread."""
        import asyncio
        return await asyncio.to_thread(self.run_all)

def run_all_benchmarks(device: str = 'cpu') -> Dict[str, Any]:
    runner = BenchmarkSuiteRunner(device=device)
    return runner.run_all()

async def run_all_benchmarks_async(device: str = 'cpu') -> Dict[str, Any]:
    runner = BenchmarkSuiteRunner(device=device)
    return await runner.async_run_all()

if __name__ == "__main__":
    results = run_all_benchmarks()
    sys.exit(0 if results['success'] else 1)
