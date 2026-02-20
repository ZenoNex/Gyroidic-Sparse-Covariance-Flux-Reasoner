
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
        print("✅ DyadFossilizer imported")
        fossilizer = DyadFossilizer(storage_dir="tests/tmp_dyads")
        print("✅ DyadFossilizer instantiated")
    except ImportError as e:
        print(f"❌ DyadFossilizer import failed: {e}")
    except Exception as e:
        print(f"❌ DyadFossilizer instantiation failed: {e}")

    # 2. CollapsePathPoisoner (AdversarialStressTester)
    try:
        from src.core.collapse_poisoner import CollapsePathPoisoner, AdversarialStressTester
        print("✅ CollapsePathPoisoner imported")
        poisoner = CollapsePathPoisoner(hidden_dim=64)
        print("✅ CollapsePathPoisoner instantiated")
        alias_poisoner = AdversarialStressTester(hidden_dim=64)
        print("✅ AdversarialStressTester alias instantiated")
        assert isinstance(alias_poisoner, CollapsePathPoisoner)
        print("✅ Alias identity verified")
    except ImportError as e:
        print(f"❌ CollapsePathPoisoner import failed: {e}")
    except Exception as e:
        print(f"❌ CollpasePathPoisoner instantiation failed: {e}")
        sys.exit(1)

    # 3. SituationalBatchSampler (LoveInvariant)
    try:
        from src.core.situational_batching import SituationalBatchSampler
        print("✅ SituationalBatchSampler imported")
        sampler = SituationalBatchSampler(num_samples=100, batch_size=10, device='cpu')
        print("✅ SituationalBatchSampler instantiated")
        
        # Test method renaming
        indices = [0, 1, 2]
        pressure = torch.tensor([0.1, 0.2, 0.3])
        mischief = torch.tensor([0.0, 0.0, 0.0])
        sampler.update_love_invariant(indices, pressure, mischief)
        print("✅ update_love_invariant called")
        sampler.update_pusafiliacrimonto(indices, pressure, mischief)
        print("✅ update_pusafiliacrimonto alias called")
    except ImportError as e:
        print(f"❌ SituationalBatchSampler import failed: {e}")
    except Exception as e:
        print(f"❌ SituationalBatchSampler test failed: {e}")
        sys.exit(1)

    # 4. GyroidCovariance (GCVE)
    try:
        from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe
        print("✅ SparseGyroidCovarianceProbe imported")
        probe = SparseGyroidCovarianceProbe(hidden_dim=64)
        print("✅ SparseGyroidCovarianceProbe instantiated")
        
        # Test rename
        C = torch.eye(32)
        val = probe.compute_gcve(C, 0.5)
        print(f"✅ compute_gcve called (result: {val})")
        
        # Check if compute_gmve is gone
        if hasattr(probe, 'compute_gmve'):
            print("⚠️ compute_gmve still exists (did you mean to remove it?)")
        else:
            print("✅ compute_gmve successfully removed/renamed")
            
    except ImportError as e:
        print(f"❌ SparseGyroidCovarianceProbe import failed: {e}")
    except Exception as e:
        print(f"❌ SparseGyroidCovarianceProbe test failed: {e}")

    # 5. DiegeticBackend (Imports)
    try:
        # We can't easily instantiate the engine without full environment, but we can check imports
        from src.ui.diegetic_backend import DiegeticPhysicsEngine
        print("✅ DiegeticPhysicsEngine imported (Backend modified correctly)")
    except ImportError as e:
        print(f"❌ DiegeticPhysicsEngine import failed: {e}")

if __name__ == "__main__":
    test_imports()
