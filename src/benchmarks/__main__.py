import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.benchmarks.run_all_benchmarks import run_all_benchmarks

if __name__ == "__main__":
    report = run_all_benchmarks()
    sys.exit(0 if report["success"] else 1)
