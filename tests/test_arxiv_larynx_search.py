import os
import sys
import torch

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.ui.diegetic_backend import DiegeticPhysicsEngine
from src.data.knowledge_ingestor import ArXivSovereignIngestor

def test_arxiv_larynx_search():
    print("Initializing DiegeticPhysicsEngine for ArXiv Larynx Search verification...")
    
    # Initialize engine in cpu/fallback mode for testing
    try:
        engine = DiegeticPhysicsEngine(dim=256, device='cpu')
        print("PASS: DiegeticPhysicsEngine initialized successfully.")
    except Exception as e:
        print(f"FAIL: Engine initialization failed: {e}")
        return False
        
    # Verify that the ArXiv Ingestor is initialized on the engine
    if not hasattr(engine, 'arxiv_ingestor') or engine.arxiv_ingestor is None:
        print("FAIL: engine.arxiv_ingestor is missing or None.")
        return False
        
    arxiv_ingestor = engine.arxiv_ingestor
    print("PASS: ArXivSovereignIngestor instance found on engine.")
    
    # Test 1: Autoregressive larynx query generation
    print("\nTest 1: Generating search query using character-level ResonanceLarynx...")
    try:
        query = arxiv_ingestor._generate_larynx_query()
        print(f"PASS: Successfully generated search query: '{query}'")
        if not isinstance(query, str):
            print(f"FAIL: Generated query is of type {type(query)}, expected str")
            return False
    except Exception as e:
        print(f"FAIL: Query generation raised exception: {e}")
        return False
        
    # Test 2: Query search and ingestion
    print("\nTest 2: Testing query search against ArXiv Search API...")
    try:
        # We can use a simple known test query or the generated one
        test_query = query if len(query) >= 3 else "quantum topology"
        print(f"Querying ArXiv Search API for: '{test_query}'...")
        arxiv_ingestor.ingest_arxiv_by_query(test_query)
        print("PASS: Query search and ingestion run completed.")
    except Exception as e:
        print(f"FAIL: Query ingestion raised exception: {e}")
        return False
        
    print("\nVerification Complete: All tests passed!")
    return True

if __name__ == "__main__":
    success = test_arxiv_larynx_search()
    sys.exit(0 if success else 1)
