"""
Polynomial functional embeddings for multi-modal inputs.

Projects text, graph, and numerical inputs into polynomial coefficient distributions
for co-prime polynomial functionals.

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.primitive_ops import FixedPointField, LearnedPrimitivePerturbation

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))



class PolynomialFunctionalEmbedder(nn.Module):
    """
    Multi-modal encoder that projects inputs into polynomial coefficient distributions.
    
    For each polynomial functional φ_k, outputs a distribution over basis coefficients.
    Supports evolutionary saturation for symbolic-first reasoning.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        graph_dim: int = 256,
        num_dim: int = 64,
        hidden_dim: int = 512,
        poly_config: Optional[PolynomialCoprimeConfig] = None,
        use_text: bool = True,
        use_graph: bool = True,
        use_num: bool = True,
        use_saturation: bool = False
    ):
        """
        Args:
            text_dim: Dimension of text embeddings
            graph_dim: Dimension of graph embeddings
            num_dim: Dimension of numerical features
            hidden_dim: Hidden dimension for fusion
            poly_config: Polynomial co-prime configuration
            use_text: Whether to use text modality
            use_graph: Whether to use graph modality
            use_num: Whether to use numerical modality
        """
        super().__init__()
        
        if poly_config is None:
            poly_config = PolynomialCoprimeConfig(k=5, degree=4)
        
        self.config = poly_config
        self.K = poly_config.k
        self.D = poly_config.degree + 1
        self.use_text = use_text
        self.use_graph = use_graph
        self.use_num = use_num
        self.use_saturation = use_saturation
        
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
        
        # Per-functional coefficient predictors
        # Projects to distribution over polynomial basis coefficients
        self.coeff_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.D) for _ in range(self.K)
        ])
        
        # Learned Primitive Perturbation (Phase 6 optimization)
        # Allows adaptive quantization grid deformation
        self.primitive_perturbation = LearnedPrimitivePerturbation(dim=self.D)
    
    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        graph_emb: Optional[torch.Tensor] = None,
        num_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-modal inputs into polynomial coefficient distributions.
        
        Args:
            text_emb: [batch, text_dim] text embeddings
            graph_emb: [batch, graph_dim] graph embeddings
            num_features: [batch, num_dim] numerical features
            
        Returns:
            Dictionary containing:
                - 'residue_distributions': [batch, K, D] coefficient distributions
                - 'primitive_field': [batch, K, D] fixed-point field
                - 'coeff_logits': List of [batch, D] logits per functional
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
        
        # Compute per-functional coefficient distributions
        coeff_logits = []
        coeff_probs = []
        
        for k, head in enumerate(self.coeff_heads):
            logits_k = head(h)  # [batch, D]
            probs_k = torch.softmax(logits_k, dim=-1)  # [batch, D]
            
            coeff_logits.append(logits_k)
            coeff_probs.append(probs_k)
        
        # Stack into tensor: [batch, K, D]
        residue_distributions = torch.stack(coeff_probs, dim=1)
        
        # Apply saturation if enabled (Directly on initial predicted residues)
        if self.use_saturation and hasattr(self.config, 'saturation_gate'):
            residue_distributions = self.config.saturation_gate(residue_distributions)
        
        # Quantize to Fixed Point Operational Primitive
        # "Floating-point arithmetic introduces nondeterminism"
        fixed_field = FixedPointField(residue_distributions)
        
        # Apply learned perturbation (on the primitive integers)
        perturbed_field = self.primitive_perturbation(fixed_field)
        
        return {
            'residue_distributions': residue_distributions, # Continuous (Legacy compatibility) or Saturated
            'primitive_field': perturbed_field,             # Operational Primitive (Invariant Optimization)
            'coeff_logits': coeff_logits,
            'fused_hidden': h
        }
    
    def compute_expected_residues(
        self,
        residue_distributions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute expected polynomial coefficients.
        
        Args:
            residue_distributions: [batch, K, D]
            
        Returns:
            expected_residues: [batch, K, D] expected coefficients
        """
        # For polynomial functionals, distributions already represent  
        # expected coefficients (softmax-weighted)
        return residue_distributions


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
    """Simple graph encoder using node features."""
    
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
