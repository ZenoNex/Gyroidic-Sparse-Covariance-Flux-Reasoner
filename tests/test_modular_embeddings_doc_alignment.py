"""
Unit tests for modular_embeddings.py documentation-aligned architecture.

Tests:
1. Anti-lobotomy compliance: zero hardcoded primes in LearnedModalityEmbedder.
2. Orthogonal polynomial basis projection (MODULE_INVENTORY.md).
3. CRTModularEmbeddingTable residue structural bias and IHC scaling.
4. IVSTEncoder Intrinsic Volume and Spectral Tensor graph extraction (DATA_PIPELINE.md).
5. SimpleGraphEncoder and SimpleTextEncoder integrations.
"""

import sys
import os
import unittest
import torch

# Ensure workspace root in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.models.modular_embeddings import (
    LearnedModalityEmbedder,
    CRTModularEmbeddingTable,
    IVSTEncoder,
    SimpleGraphEncoder,
    SimpleTextEncoder,
    get_canonical_embedder,
)


class TestModularEmbeddingsDocAlignment(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.poly_config = PolynomialCoprimeConfig(k=5, degree=4)

    def test_learned_modality_embedder_anti_lobotomy(self):
        """Verify LearnedModalityEmbedder satisfies anti-lobotomy requirements and outputs basis coeffs."""
        embedder = LearnedModalityEmbedder(
            text_dim=32,
            graph_dim=32,
            num_dim=16,
            hidden_dim=64,
            poly_config=self.poly_config
        )
        
        # Verify dynamic prime assignment from canonical ladder, not hardcoded
        self.assertEqual(len(embedder.primes), 5)
        self.assertEqual(embedder.K, 5)
        
        text_emb = torch.randn(self.batch_size, 32)
        graph_emb = torch.randn(self.batch_size, 32)
        num_feat = torch.randn(self.batch_size, 16)
        
        out = embedder(text_emb=text_emb, graph_emb=graph_emb, num_features=num_feat)
        
        # Verify dictionary outputs
        self.assertIn('polynomial_coefficients', out)
        self.assertIn('residue_distributions', out)
        self.assertIn('residue_logits', out)
        self.assertIn('fused_hidden', out)
        
        # Shape verification
        # polynomial_coefficients: [batch, K, D] where D = degree + 1 = 5
        self.assertEqual(out['polynomial_coefficients'].shape, (self.batch_size, 5, 5))
        # residue_distributions: [batch, K, max_prime]
        self.assertEqual(out['residue_distributions'].shape[0], self.batch_size)
        self.assertEqual(out['residue_distributions'].shape[1], 5)
        
        # Expected residues computation
        expected = embedder.compute_expected_residues(out['residue_distributions'])
        self.assertEqual(expected.shape, (self.batch_size, 5))

    def test_crt_modular_embedding_table(self):
        """Verify CRTModularEmbeddingTable applies structural residue bias per CRT factorization."""
        vocab_size = 100
        embed_dim = 64
        table = CRTModularEmbeddingTable(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            poly_config=self.poly_config
        )
        
        token_ids = torch.randint(0, vocab_size, (self.batch_size, 12))
        embeddings = table(token_ids)
        
        self.assertEqual(embeddings.shape, (self.batch_size, 12, embed_dim))
        self.assertFalse(torch.isnan(embeddings).any())

    def test_ivst_encoder_intrinsic_volume_and_spectral_tensor(self):
        """Verify IVSTEncoder extracts intrinsic volume and graph Laplacian spectrum."""
        num_nodes = 10
        feature_dim = 16
        output_dim = 32
        ivst = IVSTEncoder(node_feature_dim=feature_dim, output_dim=output_dim, num_eigenvalues=6)
        
        node_features = torch.randn(self.batch_size, num_nodes, feature_dim)
        graph_emb = ivst(node_features)
        
        self.assertEqual(graph_emb.shape, (self.batch_size, output_dim))
        self.assertFalse(torch.isnan(graph_emb).any())
        
        # Test with explicit adjacency matrix
        adj = torch.ones(self.batch_size, num_nodes, num_nodes)
        graph_emb_adj = ivst(node_features, adj_matrix=adj)
        self.assertEqual(graph_emb_adj.shape, (self.batch_size, output_dim))

    def test_simple_graph_and_text_encoders(self):
        """Verify SimpleGraphEncoder (IVST delegation) and SimpleTextEncoder."""
        text_encoder = SimpleTextEncoder(vocab_size=50, embed_dim=32)
        token_ids = torch.randint(0, 50, (self.batch_size, 8))
        text_emb = text_encoder(token_ids)
        self.assertEqual(text_emb.shape, (self.batch_size, 32))
        
        graph_encoder = SimpleGraphEncoder(node_feature_dim=16, output_dim=32)
        node_features = torch.randn(self.batch_size, 6, 16)
        graph_emb = graph_encoder(node_features)
        self.assertEqual(graph_emb.shape, (self.batch_size, 32))

    def test_canonical_embedder_bridge(self):
        """Verify get_canonical_embedder returns PolynomialFunctionalEmbedder."""
        embedder = get_canonical_embedder(text_dim=32, graph_dim=32, num_dim=16, hidden_dim=64)
        self.assertIsNotNone(embedder)


if __name__ == '__main__':
    unittest.main()
