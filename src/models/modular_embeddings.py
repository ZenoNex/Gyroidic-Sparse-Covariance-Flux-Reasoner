"""
Modular residue embeddings for multi-modal inputs.

DEPRECATED: Use polynomial_embeddings.py instead for the January 2026 hybridized architecture.
This file remains as legacy code for Phase 1-5 compatibility.

Author: William Matthew Bryant
Created: January 2026 (Deprecated)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))



class LearnedModalityEmbedder(nn.Module):
    """
    Multi-modal encoder that projects inputs into per-prime residue distributions.
    
    For each prime p_k, outputs a probability distribution over ℤ/p_k.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        graph_dim: int = 256,
        num_dim: int = 64,
        hidden_dim: int = 512,
        primes: List[int] = [3, 5, 7, 11, 13],
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
            primes: List of prime moduli
            use_text: Whether to use text modality
            use_graph: Whether to use graph modality
            use_num: Whether to use numerical modality
        """
        super().__init__()
        
        self.primes = primes
        self.K = len(primes)
        self.max_prime = max(primes)
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
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Per-prime residue predictors
        # W_k h + b_k → logits over ℤ/p_k
        self.residue_heads = nn.ModuleList([
            nn.Linear(hidden_dim, p_k) for p_k in primes
        ])
    
    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        graph_emb: Optional[torch.Tensor] = None,
        num_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-modal inputs into residue distributions.
        
        Args:
            text_emb: [batch, text_dim] text embeddings
            graph_emb: [batch, graph_dim] graph embeddings
            num_features: [batch, num_dim] numerical features
            
        Returns:
            Dictionary containing:
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
        
        # Fuse modalities
        if len(modality_features) == 0:
            raise ValueError("At least one modality must be provided")
        
        fused = torch.cat(modality_features, dim=-1)
        h = self.fusion(fused)  # [batch, hidden_dim]
        
        # Compute per-prime residue distributions
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
            residue_distributions[:, k, :p_k] = probs_k
        
        return {
            'residue_distributions': residue_distributions,
            'residue_logits': residue_logits,
            'fused_hidden': h
        }
    
    def compute_expected_residues(
        self,
        residue_distributions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute E[r_k] = Σ i · r_k[i] for each prime field.
        
        Args:
            residue_distributions: [batch, K, max(p_k)]
            
        Returns:
            expected_residues: [batch, K]
        """
        batch_size = residue_distributions.shape[0]
        expected = torch.zeros(batch_size, self.K, device=residue_distributions.device)
        
        for k, p_k in enumerate(self.primes):
            indices = torch.arange(p_k, dtype=residue_distributions.dtype, 
                                 device=residue_distributions.device)
            expected[:, k] = torch.sum(
                residue_distributions[:, k, :p_k] * indices,
                dim=1
            )
        
        return expected


class SimpleTextEncoder(nn.Module):
    """Simple bag-of-words text encoder for demonstration."""
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
            
        Returns:
            text_emb: [batch, embed_dim] mean-pooled embeddings
        """
        embeds = self.embedding(token_ids)  # [batch, seq_len, embed_dim]
        return embeds.mean(dim=1)  # [batch, embed_dim]


class SimpleGraphEncoder(nn.Module):
    """Simple graph encoder using adjacency matrix."""
    
    def __init__(self, node_feature_dim: int = 64, output_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, node_feature_dim]
            
        Returns:
            graph_emb: [batch, output_dim] global graph embedding
        """
        node_embeds = self.encoder(node_features)  # [batch, num_nodes, output_dim]
        return node_embeds.mean(dim=1)  # [batch, output_dim]
