"""
Modular and CRT residue embeddings for multi-modal inputs.

Implements the canonical architectures documented in:
- docs/stepping_stairs_tier_0_vestibule/01_architecture_core/MODULE_INVENTORY.md:
    * LearnedModalityEmbedder: Adaptive multimodal projector mapping text tokens,
      graph nodes, and sensor floats into orthogonal polynomial basis coefficients.
    * modular_embeddings.py: CRT-modular token embedding table organized by residue
      class under coprime moduli.
    * SimpleTextEncoder: Baseline text encoder projecting token histograms into
      initial semantic vectors prior to polynomial functional projection.
- docs/stepping_stairs_tier_0_vestibule/01_architecture_core/DATA_PIPELINE.md:
    * IVSTEncoder: Intrinsic Volume and Spectral Tensor graph topology encoder.
- docs/.../summaries/ANTI_LOBOTOMY_FIXES.md:
    * Anti-Lobotomy Clause: Strict prohibition of hardcoded primes ([3, 5, 7, 11, 13]).
      Dynamically bound to PolynomialCoprimeConfig and get_prime_ladder.

Author: William Matthew Bryant / Architecture Alignment
Created: January 2026 / Rehybridized 2026
"""

import os
import sys
from typing import Dict, Optional, List, Tuple, Union, Any
import torch
import torch.nn as nn

from src.core.gdpo_normalization import GDPONormalization
from src.core.invariants import PHI, get_ihc_ladder_scale, get_prime_ladder
from src.core.polynomial_coprime import PolynomialCoprimeConfig, PolynomialBasis


class CRTModularEmbeddingTable(nn.Module):
    """
    CRT-modular token embedding table.
    
    Tokens sharing the same residue class under coprime moduli (r_k = t mod p_k)
    share component distributions from base centroids, structurally biasing the
    embedding space to respect the Chinese Remainder Theorem factorization.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 768,
        poly_config: Optional[PolynomialCoprimeConfig] = None,
        primes: Optional[List[int]] = None,
        num_primes: int = 5,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Anti-Lobotomy Clause: No hardcoded primes. Derive from poly_config or canonical ladder
        if primes is None:
            if poly_config is not None:
                ladder = get_prime_ladder(poly_config.k)
            else:
                ladder = get_prime_ladder(num_primes)
            primes = [int(p.item()) for p in ladder]
            
        self.primes = primes
        self.K = len(primes)
        
        # Base independent token embedding table
        self.base_embedding = nn.Embedding(vocab_size, embed_dim, device=device)
        
        # Per-modulus residue class centroids
        # Each residue class r in Z/p_k Z has a learnable bias vector
        self.residue_centroids = nn.ModuleList([
            nn.Embedding(p_k, embed_dim, device=device) for p_k in self.primes
        ])
        
        # Scale factor for residue class injection
        self.residue_scale = nn.Parameter(torch.tensor(0.1, device=device))
        
        # Precompute CRT residue lookups for vocabulary
        indices = torch.arange(vocab_size, device=device)
        for k, p_k in enumerate(self.primes):
            self.register_buffer(f"residue_map_{k}", indices % p_k)
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens with CRT residue structural bias.
        
        Args:
            token_ids: [batch, seq_len] tensor of token indices
            
        Returns:
            embeddings: [batch, seq_len, embed_dim] structurally biased embeddings
        """
        base = self.base_embedding(token_ids)
        
        residue_bias = torch.zeros_like(base)
        for k, p_k in enumerate(self.primes):
            residue_map = getattr(self, f"residue_map_{k}")
            token_residues = residue_map[token_ids]
            centroid = self.residue_centroids[k](token_residues)
            ladder_scale = get_ihc_ladder_scale(k, num_shells=33)
            residue_bias = residue_bias + centroid * ladder_scale
            
        return base + self.residue_scale * residue_bias


class IVSTEncoder(nn.Module):
    """
    Intrinsic Volume and Spectral Tensor (IVST) graph encoder.
    
    Captures non-local topological data across structural topologies by mapping
    node features and connectivity into:
    1. Intrinsic Volume: preserving scale and geometric weight of the subgraph
       (Euler characteristic, node/edge volume, loop density).
    2. Spectral Tensor: graph Laplacian eigenvalues summarizing topological shape
       and Cheeger spectral gap without O(N^2) token comparisons.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 64,
        output_dim: int = 256,
        num_eigenvalues: int = 8,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.output_dim = output_dim
        self.num_eigenvalues = num_eigenvalues
        
        # Node projection
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # Intrinsic volume projection (Euler characteristic, density, volume)
        self.volume_proj = nn.Linear(3, hidden_dim)
        
        # Spectral tensor projection (Laplacian eigenvalues)
        self.spectral_proj = nn.Linear(num_eigenvalues, hidden_dim)
        
        # Fusion to output
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def _compute_laplacian_spectrum(
        self,
        adj: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Computes normalized Laplacian spectrum lambda_1 <= lambda_2 <= ..."""
        device = adj.device
        deg = torch.sum(adj, dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        d_mat = torch.diag_embed(deg_inv_sqrt)
        
        # Normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
        eye = torch.eye(num_nodes, device=device).unsqueeze(0).expand_as(adj)
        l_sym = eye - torch.bmm(torch.bmm(d_mat, adj), d_mat)
        
        # Eigenvalues
        try:
            evals = torch.linalg.eigvalsh(l_sym)
            if evals.shape[-1] >= self.num_eigenvalues:
                return evals[:, :self.num_eigenvalues]
            else:
                pad = torch.zeros(evals.shape[0], self.num_eigenvalues - evals.shape[-1], device=device)
                return torch.cat([evals, pad], dim=-1)
        except Exception:
            return torch.zeros(adj.shape[0], self.num_eigenvalues, device=device)
            
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, node_feature_dim]
            adj_matrix: Optional [batch, num_nodes, num_nodes] adjacency matrix
            
        Returns:
            graph_emb: [batch, output_dim] global graph topological embedding
        """
        batch_size, num_nodes, _ = node_features.shape
        device = node_features.device
        
        # 1. Projected Node representation
        h_nodes = self.node_proj(node_features).mean(dim=1)  # [batch, hidden_dim]
        
        # Construct approximate adjacency if not explicitly provided
        if adj_matrix is None:
            # Pairwise cosine similarity as soft adjacency
            norm_nodes = node_features / (torch.norm(node_features, dim=-1, keepdim=True) + 1e-6)
            adj = torch.bmm(norm_nodes, norm_nodes.transpose(1, 2)).clamp(min=0.0)
        else:
            adj = adj_matrix
            
        # 2. Intrinsic Volume calculation
        # V0: normalized node count
        # V1: edge density
        # V2: loop / triangle density (trace(A^3) / 6)
        v0 = torch.full((batch_size, 1), float(num_nodes) / 100.0, device=device)
        v1 = adj.sum(dim=(-2, -1)).unsqueeze(-1) / max(1.0, float(num_nodes * num_nodes))
        
        a2 = torch.bmm(adj, adj)
        triangles = torch.diagonal(torch.bmm(a2, adj), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 6.0
        v2 = triangles / max(1.0, float(num_nodes * num_nodes * num_nodes))
        
        intrinsic_vols = torch.cat([v0, v1, v2], dim=-1)  # [batch, 3]
        h_volume = self.volume_proj(intrinsic_vols)        # [batch, hidden_dim]
        
        # 3. Spectral Tensor (Graph Laplacian spectrum)
        spectrum = self._compute_laplacian_spectrum(adj, num_nodes)
        h_spectral = self.spectral_proj(spectrum)           # [batch, hidden_dim]
        
        # 4. Fusion
        fused = torch.cat([h_nodes, h_volume, h_spectral], dim=-1)
        return self.out_proj(fused)


class SimpleGraphEncoder(nn.Module):
    """
    Graph encoder delegating to IVST (Intrinsic Volume and Spectral Tensor).
    """
    
    def __init__(self, node_feature_dim: int = 64, output_dim: int = 256):
        super().__init__()
        self.ivst = IVSTEncoder(node_feature_dim=node_feature_dim, output_dim=output_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, node_feature_dim]
            adj_matrix: Optional [batch, num_nodes, num_nodes]
            
        Returns:
            graph_emb: [batch, output_dim] global graph embedding
        """
        return self.ivst(node_features, adj_matrix=adj_matrix)


class SimpleTextEncoder(nn.Module):
    """
    Baseline text encoder projecting character/word histograms into initial
    semantic vectors prior to polynomial functional projection.
    """
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear_norm = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
            
        Returns:
            text_emb: [batch, embed_dim] mean-pooled embeddings
        """
        embeds = self.embedding(token_ids)  # [batch, seq_len, embed_dim]
        pooled = embeds.mean(dim=1)         # [batch, embed_dim]
        return self.linear_norm(pooled)


class LearnedModalityEmbedder(nn.Module):
    """
    Adaptive multimodal projector mapping text tokens, graph nodes, and sensor
    floats into orthogonal polynomial basis coefficients and CRT residue distributions.
    
    Anti-Lobotomy Compliance:
    - Zero hardcoded primes (eliminated [3, 5, 7, 11, 13]).
    - Polynomial coprime basis via PolynomialCoprimeConfig over Birkhoff polytope.
    - Dynamic prime ladder generation via canonical get_prime_ladder.
    - IHC 33-shell golden ratio ladder scaling on residue channels.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        graph_dim: int = 256,
        num_dim: int = 64,
        hidden_dim: int = 512,
        poly_config: Optional[PolynomialCoprimeConfig] = None,
        primes: Optional[List[int]] = None,
        num_primes: int = 5,
        use_text: bool = True,
        use_graph: bool = True,
        use_num: bool = True
    ):
        """
        Args:
            text_dim: Dimension of text embeddings
            graph_dim: Dimension of graph embeddings
            num_dim: Dimension of numerical features
            hidden_dim: Hidden dimension for fusion
            poly_config: Optional PolynomialCoprimeConfig
            primes: Optional list of prime moduli (dynamically derived via get_prime_ladder if None)
            num_primes: Number of primes to generate from resonance ladder if primes is None
            use_text: Whether to use text modality
            use_graph: Whether to use graph modality
            use_num: Whether to use numerical modality
        """
        super().__init__()
        
        # 1. Anti-Lobotomy Prime Resolution: Never hardcode primes
        if primes is None:
            k_primes = poly_config.k if poly_config is not None else num_primes
            primes = [int(p.item()) for p in get_prime_ladder(k_primes)]
            
        self.primes = primes
        self.K = len(primes)
        self.max_prime = max(primes)
        
        # 2. Canonical Polynomial Configuration
        if poly_config is None:
            self.poly_config = PolynomialCoprimeConfig(k=self.K, degree=4)
        else:
            self.poly_config = poly_config
            
        self.D = self.poly_config.degree + 1
        self.use_text = use_text
        self.use_graph = use_graph
        self.use_num = use_num
        
        # Input projections
        input_dim = 0
        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            input_dim += hidden_dim
        if use_graph:
            self.graph_proj = nn.Linear(graph_dim, hidden_dim)
            input_dim += hidden_dim
        if use_num:
            self.num_proj = nn.Linear(num_dim, hidden_dim)
            input_dim += hidden_dim
        
        # Fusion network with GDPO Normalization
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GDPONormalization(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Orthogonal Polynomial Basis Heads (MODULE_INVENTORY specification)
        # Maps fused multimodal state into basis coefficients for each functional
        self.coeff_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.D) for _ in range(self.K)
        ])
        
        # Per-prime residue predictors (CRT residue heads)
        self.residue_heads = nn.ModuleList([
            nn.Linear(hidden_dim, p_k) for p_k in self.primes
        ])
    
    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        graph_emb: Optional[torch.Tensor] = None,
        num_features: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Encode multi-modal inputs into polynomial coefficients and residue distributions.
        
        Args:
            text_emb: [batch, text_dim] text embeddings
            graph_emb: [batch, graph_dim] graph embeddings
            num_features: [batch, num_dim] numerical features
            
        Returns:
            Dictionary containing:
                - 'polynomial_coefficients': [batch, K, D] basis coefficients
                - 'residue_distributions': [batch, K, max(p_k)] padded distributions
                - 'residue_logits': List of [batch, p_k] logits per prime
                - 'fused_hidden': [batch, hidden_dim] fused representation
        """
        batch_size = (
            text_emb.shape[0] if text_emb is not None else
            graph_emb.shape[0] if graph_emb is not None else
            num_features.shape[0]
        )
        
        # Project each modality
        modality_features = []
        
        if self.use_text and text_emb is not None:
            modality_features.append(self.text_proj(text_emb))
        
        if self.use_graph and graph_emb is not None:
            modality_features.append(self.graph_proj(graph_emb))
        
        if self.use_num and num_features is not None:
            modality_features.append(self.num_proj(num_features))
        
        if len(modality_features) == 0:
            raise ValueError("At least one modality must be provided")
        
        fused = torch.cat(modality_features, dim=-1)
        h = self.fusion(fused)  # [batch, hidden_dim]
        
        # 1. Compute Orthogonal Polynomial Basis Coefficients [batch, K, D]
        poly_coeffs_list = [head(h).unsqueeze(1) for head in self.coeff_heads]
        poly_coeffs = torch.cat(poly_coeffs_list, dim=1)  # [batch, K, D]
        
        # 2. Compute per-prime residue distributions
        residue_logits = []
        residue_probs = []
        
        for k, head in enumerate(self.residue_heads):
            logits_k = head(h)  # [batch, p_k]
            probs_k = torch.softmax(logits_k, dim=-1)  # [batch, p_k]
            residue_logits.append(logits_k)
            residue_probs.append(probs_k)
        
        # Pad to max_prime for uniform tensor
        residue_distributions = torch.zeros(
            batch_size, self.K, self.max_prime,
            device=h.device, dtype=h.dtype
        )
        
        for k, probs_k in enumerate(residue_probs):
            p_k = self.primes[k]
            # Inverted Hypersphere Cosmology (IHC): 33-shell golden ratio ladder scaling
            ladder_scale = get_ihc_ladder_scale(k, num_shells=33)
            residue_distributions[:, k, :p_k] = probs_k * ladder_scale
        
        return {
            'polynomial_coefficients': poly_coeffs,
            'residue_distributions': residue_distributions,
            'residue_logits': residue_logits,
            'fused_hidden': h
        }
    
    def compute_expected_residues(
        self,
        residue_distributions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute E[r_k] = sum_i i * r_k[i] for each prime field.
        
        Args:
            residue_distributions: [batch, K, max(p_k)]
            
        Returns:
            expected_residues: [batch, K]
        """
        batch_size = residue_distributions.shape[0]
        expected = torch.zeros(batch_size, self.K, device=residue_distributions.device)
        
        for k, p_k in enumerate(self.primes):
            indices = torch.arange(
                p_k,
                dtype=residue_distributions.dtype,
                device=residue_distributions.device
            )
            expected[:, k] = torch.sum(
                residue_distributions[:, k, :p_k] * indices,
                dim=1
            )
        
        return expected


def get_canonical_embedder(**kwargs):
    """Bridge to the canonical polynomial functional embedder."""
    from src.models.polynomial_embeddings import PolynomialFunctionalEmbedder
    return PolynomialFunctionalEmbedder(**kwargs)
