
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add root to path
sys.path.append(os.getcwd())

class TestGraphIntegration(unittest.TestCase):
    
    @patch('hybrid_backend.TrainingManager')
    @patch('hybrid_backend.DatasetIngestionSystem')
    @patch('hybrid_backend.NonLobotomyTemporalModel')
    @patch('hybrid_backend.PolynomialADMRSolver')
    @patch('hybrid_backend.LeyLineGeodesicMetric')
    @patch('hybrid_backend.MoebiusFiberBundle')
    @patch.dict(sys.modules, {'src.topology.embedding_graph': MagicMock()})
    def test_graph_integration_logic(self, mock_moebius, mock_leyline, mock_admr, mock_model, mock_dataset, mock_trainer):
        # We need the REAL GyroidicGraphManager, but we had to mock the module above to avoid side effects?
        # Actually, let's just mock the heavy imports in hybrid_backend, but keeps src.topology.embedding_graph real.
        pass

# Redo: import properly without patching sys.modules for the graph manager
# We only want to patch the HEAVY stuff in hybrid_backend

def setup_mocks():
    sys.modules['src.training.training_manager'] = MagicMock()
    sys.modules['examples.enhanced_temporal_training'] = MagicMock()
    sys.modules['src.core.admr_solver'] = MagicMock()
    sys.modules['dataset_ingestion_system'] = MagicMock()
    # But we want real hybrid_backend logic
    
from hybrid_backend import HybridAI
from src.topology.embedding_graph import GyroidicGraphManager

def verify_integration():
    print("🧪 Verifying Graph Integration (Mocked)...")
    
    # Mock heavy components on the class instance to speed up init
    # Actually, HybridAI imports them at top level or inside init. 
    # We can patch them before importing HybridAI if we used untrusted modules, 
    # but here we can just instantiate HybridAI and hope the try/except blocks handle missing heavy stuff 
    # or we mock the modules in sys.modules first.
    
    # Force Mocking
    sys.modules['enhanced_temporal_training'] = MagicMock()
    sys.modules['src.core.admr_solver'] = MagicMock()
    sys.modules['dataset_ingestion_system'] = MagicMock()
    
    # Re-import HybridAI now that modules are mocked
    if 'hybrid_backend' in sys.modules:
        del sys.modules['hybrid_backend']
        
    from hybrid_backend import HybridAI
    
    # Instantiate
    print("⚙️ Initializing HybridAI with mocks...")
    ai = HybridAI(use_spectral_correction=False)
    
    # Verify Graph Manager Init
    if not isinstance(ai.graph_manager, GyroidicGraphManager):
        print("❌ Graph Manager not initialized correctly!")
        return False
        
    print(f"✅ Graph Manager initialized. Node count: {len(ai.graph_manager.nodes)}")
    
    # Test Fossil Save
    print("📝 Generating test fossil...")
    ai._save_fossil("TEST FOSSIL", torch.randn(256), {"test": True})
    
    if len(ai.graph_manager.nodes) == 0:
        print("❌ Fossil not added to graph!")
        return False
    
    print(f"✅ Fossil added. Node count: {len(ai.graph_manager.nodes)}")
    
    # Test Export
    json_out = ai.graph_manager.export_graph_json()
    if "nodes" not in json_out:
        print("❌ JSON export invalid!")
        return False
        
    print("✅ JSON export valid.")
    return True

if __name__ == "__main__":
    try:
        if verify_integration():
            print("🚀 Verification SUCCESS!")
            sys.exit(0)
        else:
            print("💥 Verification FAILED!")
            sys.exit(1)
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
