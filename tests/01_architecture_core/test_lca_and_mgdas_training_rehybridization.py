"""
Unit & Integration Tests for:
1. MGDAS Foundational Multi-Scale Geometric Soil in ConstraintDataset & StructuralAdaptor (src/training/trainer.py)
2. Bitwise Morton O(1) LCA in PointerlessOctree (src/ui/voxelboxter_simulation.py)
3. Euler Tour + Sparse Table (RMQ) O(1) LCA Engine (src/core/euler_tour_lca.py)
4. ChatGPT Conversation Branching Divergence (src/data/chatgpt_friction_harvester.py)
5. Semiotic Hierarchy Arbitration via LexicographicalOrderingDispatcher in UniversalOrchestrator
"""

import math
import torch
import numpy as np
import pytest

from src.training.trainer import ConstraintDataset, StructuralAdaptor, MandelbulbAugmentedDataset
from src.augmentation.mandelbulb_gyroidic_augmenter import MandelbulbGyroidicAugmenter, AugmentationConfig
from src.ui.voxelboxter_simulation import PointerlessOctree
from src.core.euler_tour_lca import EulerTourSparseTableLCA
from src.data.chatgpt_friction_harvester import build_conversation_lca, find_branch_divergence
from src.core.invariant_optimization import SemioticState, LexicographicalOrderingDispatcher
from src.core.orchestrator import UniversalOrchestrator


class TestMGDASFoundationalTraining:
    """Tests confirming MGDAS is active, foundational soil in trainer.py and eliminates initialization sensitivity."""
    
    def test_constraint_dataset_mgdas_prefold_active_by_default(self):
        """Verify that ConstraintDataset enables geometric pre-folding by default."""
        dataset = ConstraintDataset(num_samples=10, text_dim=64, graph_dim=32, num_dim=16)
        assert dataset.use_geometric_prefold is True
        assert dataset.augmenter is not None
        
        sample = dataset[0]
        assert 'text_emb' in sample
        assert 'graph_emb' in sample
        assert 'num_features' in sample
        assert sample['text_emb'].shape == (64,)
        assert sample['graph_emb'].shape == (32,)
        assert sample['num_features'].shape == (16,)
        assert np.isfinite(sample['text_emb']).all()
        assert np.isfinite(sample['graph_emb']).all()
        assert np.isfinite(sample['num_features']).all()
        
    def test_constraint_dataset_toggle_prefold(self):
        """Verify that pre-folding can be explicitly disabled for raw benchmarks."""
        dataset_raw = ConstraintDataset(num_samples=5, text_dim=32, graph_dim=16, num_dim=8, use_geometric_prefold=False)
        assert dataset_raw.use_geometric_prefold is False
        assert dataset_raw.augmenter is None
        sample = dataset_raw[0]
        assert sample['text_emb'].shape == (32,)

    def test_mandelbulb_augmented_dataset_dict_wrapping(self):
        """Verify MandelbulbAugmentedDataset properly handles dictionary multimodal datasets like ConstraintDataset."""
        base_dataset = ConstraintDataset(num_samples=6, text_dim=32, graph_dim=16, num_dim=8, use_geometric_prefold=False)
        wrapped = MandelbulbAugmentedDataset(base_dataset, apply_online=True)
        assert len(wrapped) == 6
        
        item = wrapped[0]
        assert isinstance(item, dict)
        assert 'text_emb' in item
        assert 'anchor' in item
        assert item['text_emb'].shape == (32,)
        assert np.isfinite(item['text_emb']).all()

    def test_structural_adaptor_geometric_binding(self):
        """Verify StructuralAdaptor has native MGDAS augmenters and dataset wrapping."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 10)
            def forward(self, **kwargs):
                return {'gyroid_pressure': torch.tensor(0.05), 'num_cycles': 1}
                
        model = MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        adaptor = StructuralAdaptor(model, optimizer, device='cpu')
        
        assert hasattr(adaptor, 'geometric_augmenter')
        assert hasattr(adaptor, 'wrap_geometric_dataset')
        
        base_dataset = ConstraintDataset(num_samples=4, text_dim=32, graph_dim=16, num_dim=8)
        geo_ds = adaptor.wrap_geometric_dataset(base_dataset)
        assert isinstance(geo_ds, MandelbulbAugmentedDataset)
        assert len(geo_ds) == 4


class TestPointerlessOctreeMortonLCA:
    """Tests confirming bitwise Morton O(1) LCA in PointerlessOctree."""
    
    def test_bitwise_morton_lca_identical_nodes(self):
        octree = PointerlessOctree(max_depth=8)
        code = octree._interleave_bits(10, 20, 30)
        lca_code, lca_depth = octree.compute_morton_lca(code, code)
        assert lca_code == code
        assert lca_depth == 8

    def test_bitwise_morton_lca_siblings(self):
        """Two nodes sharing the same parent octant differ only in the lowest 3 bits."""
        octree = PointerlessOctree(max_depth=8)
        # Coordinates sharing same parent cell (e.g. x=2, y=2, z=2 vs x=3, y=2, z=2)
        code_a = octree._interleave_bits(2, 2, 2)
        code_b = octree._interleave_bits(3, 2, 2)
        
        lca_code, lca_depth = octree.compute_morton_lca(code_a, code_b)
        assert lca_depth == 7  # One level above max_depth=8
        
        # Both codes when shifted down by 3 bits should match
        assert (code_a >> 3) == (code_b >> 3)
        assert (code_a >> 3) << 3 == lca_code

    def test_bitwise_morton_lca_divergent_octants(self):
        """Nodes in opposite corners of the space should have the root (0, 0) as LCA."""
        octree = PointerlessOctree(max_depth=8)
        code_a = octree._interleave_bits(1, 1, 1)
        code_b = octree._interleave_bits(512, 512, 512)
        
        lca_code, lca_depth = octree.compute_morton_lca(code_a, code_b)
        # Highest bit differs at top octant, LCA depth is 0
        assert lca_depth == 0

    def test_find_spatial_lca(self):
        octree = PointerlessOctree(max_depth=8)
        coord_a = (16, 32, 48)
        coord_b = (17, 32, 48)
        lca_code, lca_depth, lca_coords = octree.find_spatial_lca(coord_a, coord_b)
        assert lca_depth == 7
        assert lca_coords[0] == 16
        assert lca_coords[1] == 32
        assert lca_coords[2] == 48

    def test_ancestral_material_retrieval(self):
        octree = PointerlessOctree(max_depth=8)
        code_a = octree._interleave_bits(4, 4, 4)
        code_b = octree._interleave_bits(5, 4, 4)
        lca_code, _ = octree.compute_morton_lca(code_a, code_b)
        octree.morton_grid[lca_code] = 7  # Custom ancestral material
        
        assert octree.get_ancestral_material(code_a, code_b) == 7


class TestEulerTourSparseTableLCA:
    """Tests confirming Euler Tour + Sparse Table (RMQ) O(1) LCA on arbitrary trees and conversational DAGs."""
    
    def test_general_tree_lca_and_distance(self):
        """
        Tree topology:
               1 (root)
              / \
             2   3
            / \   \
           4   5   6
              /
             7
        """
        tree = {
            1: [2, 3],
            2: [4, 5],
            3: [6],
            4: [],
            5: [7],
            6: [],
            7: []
        }
        lca_engine = EulerTourSparseTableLCA(tree, root=1)
        
        # Test basic LCA queries in O(1)
        assert lca_engine.lca(4, 5) == 2
        assert lca_engine.lca(4, 7) == 2
        assert lca_engine.lca(7, 6) == 1
        assert lca_engine.lca(2, 7) == 2
        assert lca_engine.lca(1, 6) == 1
        assert lca_engine.lca(7, 7) == 7
        
        # Test distance queries
        assert lca_engine.distance(4, 5) == 2
        assert lca_engine.distance(4, 7) == 3  # 4->2 (1) + 2->5 (1) + 5->7 (1)
        assert lca_engine.distance(7, 6) == 5  # 7->5->2->1 (3) + 1->3->6 (2)
        
        # Test ancestry checks
        assert lca_engine.is_ancestor(1, 7) is True
        assert lca_engine.is_ancestor(2, 7) is True
        assert lca_engine.is_ancestor(3, 7) is False

    def test_chatgpt_conversation_lca_and_divergence(self):
        """
        Simulate a branching ChatGPT conversation export:
        Turn 1: user prompt (root)
          Turn 2: assistant response A
            Turn 3: user edit A1
              Turn 4: assistant response A1_1
            Turn 5: user alternate edit A2
              Turn 6: assistant response A2_1
        """
        mapping = {
            "root_msg": {"id": "root_msg", "parent": None, "children": ["asst_1"]},
            "asst_1": {"id": "asst_1", "parent": "root_msg", "children": ["user_edit_a", "user_edit_b"]},
            "user_edit_a": {"id": "user_edit_a", "parent": "asst_1", "children": ["asst_a"]},
            "asst_a": {"id": "asst_a", "parent": "user_edit_a", "children": []},
            "user_edit_b": {"id": "user_edit_b", "parent": "asst_1", "children": ["asst_b"]},
            "asst_b": {"id": "asst_b", "parent": "user_edit_b", "children": []},
        }
        
        engine = build_conversation_lca(mapping)
        assert engine.lca("asst_a", "asst_b") == "asst_1"
        assert engine.lca("user_edit_a", "asst_b") == "asst_1"
        assert engine.lca("user_edit_a", "asst_a") == "user_edit_a"
        
        # Test find_branch_divergence
        lca_node, path_a, path_b = find_branch_divergence(mapping, "asst_a", "asst_b")
        assert lca_node == "asst_1"
        assert path_a == ["user_edit_a", "asst_a"]
        assert path_b == ["user_edit_b", "asst_b"]


class TestSemioticHierarchyArbitration:
    """Tests confirming the Semiotic Hierarchy where System 2 Invariant Admissibility dominates System 1."""
    
    def test_lexicographical_ordering_dispatcher(self):
        dispatcher = LexicographicalOrderingDispatcher(eps=1e-5)
        
        # State A: Perfectly admissible (0.01 error), but slower heuristic (speed 10)
        state_a = SemioticState(system_2_admissibility=0.01, system_1_heuristic=10.0, state_tensor=torch.zeros(4))
        # State B: Slight invariant violation (0.10 error), but ultra-fast heuristic (speed 1000)
        state_b = SemioticState(system_2_admissibility=0.10, system_1_heuristic=1000.0, state_tensor=torch.zeros(4))
        
        # System 2 admissibility strictly dominates System 1 heuristic
        assert dispatcher.compare(state_a, state_b) == -1  # A is preferred despite B's 100x speed!
        best = dispatcher.select_best([state_b, state_a])
        assert best is state_a

    def test_tie_breaking_by_system1_when_invariants_equal(self):
        dispatcher = LexicographicalOrderingDispatcher(eps=1e-5)
        
        # Both states satisfy System 2 invariants identically (admissibility = 0.005)
        state_slow = SemioticState(system_2_admissibility=0.005, system_1_heuristic=20.0, state_tensor=torch.zeros(4))
        state_fast = SemioticState(system_2_admissibility=0.005, system_1_heuristic=80.0, state_tensor=torch.zeros(4))
        
        # When primary index is identical, secondary index selects faster proposal
        assert dispatcher.compare(state_fast, state_slow) == -1
        assert dispatcher.select_best([state_slow, state_fast]) is state_fast

    def test_universal_orchestrator_semiotic_arbitration(self):
        orchestrator = UniversalOrchestrator(dim=16)
        assert hasattr(orchestrator, 'semiotic_dispatcher')
        assert hasattr(orchestrator, 'arbitrate_semiotic_state')
        
        state_valid = SemioticState(system_2_admissibility=0.02, system_1_heuristic=50.0, state_tensor=torch.ones(16))
        state_violator = SemioticState(system_2_admissibility=0.99, system_1_heuristic=500.0, state_tensor=torch.zeros(16))
        
        chosen = orchestrator.arbitrate_semiotic_state([state_violator, state_valid])
        assert chosen is state_valid
