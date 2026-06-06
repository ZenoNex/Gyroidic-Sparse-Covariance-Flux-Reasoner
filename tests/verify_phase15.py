
import sys
import os
import torch

# Add src to path
sys.path.append(os.getcwd())

def test_imports():
    print("Testing imports...")
    
    # 1. DyadFossilizer
    try:
        from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
        print("[OK] DyadFossilizer imported")
        fossilizer = DyadFossilizer(storage_dir="tests/tmp_dyads")
        print("[OK] DyadFossilizer instantiated")
    except ImportError as e:
        print(f"[FAIL] DyadFossilizer import failed: {e}")
    except Exception as e:
        print(f"[FAIL] DyadFossilizer instantiation failed: {e}")

    # 2. CollapsePathPoisoner (AdversarialStressTester)
    try:
        from src.core.collapse_poisoner import CollapsePathPoisoner, AdversarialStressTester
        print("[OK] CollapsePathPoisoner imported")
        poisoner = CollapsePathPoisoner(hidden_dim=64)
        print("[OK] CollapsePathPoisoner instantiated")
        alias_poisoner = AdversarialStressTester(hidden_dim=64)
        print("[OK] AdversarialStressTester alias instantiated")
        assert isinstance(alias_poisoner, CollapsePathPoisoner)
        print("[OK] Alias identity verified")
    except ImportError as e:
        print(f"[FAIL] CollapsePathPoisoner import failed: {e}")
    except Exception as e:
        print(f"[FAIL] CollapsePathPoisoner instantiation failed: {e}")
        sys.exit(1)

    # 3. SituationalBatchSampler (LoveInvariant)
    try:
        from src.core.situational_batching import SituationalBatchSampler
        print("[OK] SituationalBatchSampler imported")
        sampler = SituationalBatchSampler(num_samples=100, batch_size=10, device='cpu')
        print("[OK] SituationalBatchSampler instantiated")
        
        # Test method renaming
        indices = [0, 1, 2]
        pressure = torch.tensor([0.1, 0.2, 0.3])
        mischief = torch.tensor([0.0, 0.0, 0.0])
        sampler.update_love_invariant(indices, pressure, mischief)
        print("[OK] update_love_invariant called")
        sampler.update_pusafiliacrimonto(indices, pressure, mischief)
        print("[OK] update_pusafiliacrimonto alias called")
    except ImportError as e:
        print(f"[FAIL] SituationalBatchSampler import failed: {e}")
    except Exception as e:
        print(f"[FAIL] SituationalBatchSampler test failed: {e}")
        sys.exit(1)

    # 4. GyroidCovariance (GCVE)
    try:
        from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe
        print("[OK] SparseGyroidCovarianceProbe imported")
        probe = SparseGyroidCovarianceProbe(hidden_dim=64)
        print("[OK] SparseGyroidCovarianceProbe instantiated")
        
        # Test rename
        C = torch.eye(32)
        val = probe.compute_gcve(C, 0.5)
        print(f"[OK] compute_gcve called (result: {val})")
        
        # Check if compute_gmve is gone
        if hasattr(probe, 'compute_gmve'):
            print("[WARN] compute_gmve still exists")
        else:
            print("[OK] compute_gmve successfully removed/renamed")
            
    except ImportError as e:
        print(f"[FAIL] SparseGyroidCovarianceProbe import failed: {e}")
    except Exception as e:
        print(f"[FAIL] SparseGyroidCovarianceProbe test failed: {e}")
 
    # 5. DiegeticBackend (Imports)
    try:
        from src.ui.diegetic_backend import DiegeticPhysicsEngine
        print("[OK] DiegeticPhysicsEngine imported (Backend modified correctly)")
    except ImportError as e:
        print(f"[FAIL] DiegeticPhysicsEngine import failed: {e}")

if __name__ == "__main__":
    test_imports()
