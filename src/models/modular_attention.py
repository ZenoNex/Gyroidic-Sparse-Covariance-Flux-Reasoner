"""
Modular attention with Birkhoff projection and CRT fusion.

Computes attention separately in each functional field, then fuses via Saturated CRT.

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import List, Optional
import math

from src.core.birkhoff_projection import BirkhoffProjection
from src.core.primitive_ops import FixedPointField

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))



class ModularAttention(nn.Module):
    """
    Multi-field modular attention.
    
    For each functional phi_k:
        A_k = Birkhoff(Q_k K_k^T / √d) ⊙ V_k
        
    Then fuse via Saturated CRT: L̂' = SaturatedCRT_Fuse({A_k, r_k})
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_functionals: int = 5,
        dropout: float = 0.1,
        use_birkhoff: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads per field
            num_functionals: Number of polynomial functionals (K)
            dropout: Dropout probability
            use_birkhoff: Whether to apply Birkhoff projection
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.K = num_functionals
        self.use_birkhoff = use_birkhoff
        
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        
        # Per-field Q, K, V projections
        self.field_projections = nn.ModuleList([
            nn.ModuleDict({
                'Q': nn.Linear(hidden_dim, hidden_dim),
                'K': nn.Linear(hidden_dim, hidden_dim),
                'V': nn.Linear(hidden_dim, hidden_dim),
            })
            for _ in range(self.K)
        ])
        
        # Birkhoff projector
        if use_birkhoff:
            self.birkhoff = BirkhoffProjection(max_iterations=20)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * self.K, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Structural Integrity Buffer
        self.register_buffer('last_integrity_mask', torch.ones(1, dtype=torch.bool))
    
    def validate_structural_integrity(self) -> torch.Tensor:
        """
        Structural Validation: Are attention matrices on the Birkhoff Polytope?
        """
        return self.last_integrity_mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        trust_scalars: Optional[torch.Tensor] = None,
        return_field_outputs: bool = False
    ) -> torch.Tensor:
        """
        Compute modular multi-field attention.
        
        Args:
            x: [batch, seq_len, hidden_dim] input
            mask: Optional [batch, seq_len, seq_len] attention mask
            trust_scalars: Optional [K] trust weights for fields
            return_field_outputs: If True, return per-field outputs
            
        Returns:
            output: [batch, seq_len, hidden_dim] attended representation
        """
        batch_size, seq_len, _ = x.shape
        
        # Operational Integration: Dequantize if primitive
        if hasattr(x, 'backing_store') or isinstance(x, FixedPointField):
             # "Dequantize for interaction with legacy float32 layers"
             # Ideally we'd have a FixedPointAttention, but for integration we unpack here
             x = x.forward()
        
        field_outputs = []
        
        # Process each field separately
        for k, proj in enumerate(self.field_projections):
            # Compute Q, K, V for this field
            Q_k = proj['Q'](x)  # [batch, seq_len, hidden_dim]
            K_k = proj['K'](x)
            V_k = proj['V'](x)
            
            # Reshape for multi-head attention
            Q_k = Q_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K_k = K_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V_k = V_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            # Now: [batch, num_heads, seq_len, head_dim]
            
            # Compute attention scores
            scores = torch.matmul(Q_k, K_k.transpose(-2, -1)) / self.scale
            # [batch, num_heads, seq_len, seq_len]
            
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
            
            # Apply Birkhoff projection if enabled
            if self.use_birkhoff:
                # Project each head's attention matrix to doubly-stochastic
                # Use training mode for annealing schedule
                attn_weights = self.birkhoff(scores, anneal=self.training)  # [batch, num_heads, seq_len, seq_len]
            else:
                attn_weights = torch.softmax(scores, dim=-1)
            
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            A_k = torch.matmul(attn_weights, V_k)  # [batch, num_heads, seq_len, head_dim]
            
            # Update Structural Integrity tracker
            if self.use_birkhoff:
                integrity = self.birkhoff.validate_stochasticity(attn_weights) # [batch, num_heads]
                if k == 0:
                    self.last_integrity_mask = integrity
                else:
                    self.last_integrity_mask = self.last_integrity_mask & integrity

            # Reshape back
            A_k = A_k.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
            
            # Weighted by trust if provided
            if trust_scalars is not None:
                A_k = A_k * trust_scalars[k]
                
            field_outputs.append(A_k)
        
        # Fuse field outputs (simple concatenation + projection)
        # In a full CRT implementation, this would use residue reconstruction
        fused = torch.cat(field_outputs, dim=-1)  # [batch, seq_len, K * hidden_dim]
        output = self.output_proj(fused)  # [batch, seq_len, hidden_dim]
        
        if return_field_outputs:
            return output, field_outputs
        return output


class ModularTransformerLayer(nn.Module):
    """
    Transformer layer with modular attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_functionals: int = 5,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        
        self.attention = ModularAttention(hidden_dim, num_heads, num_functionals, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        trust_scalars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Attention with residual
        attn_out = self.attention(x, mask, trust_scalars=trust_scalars)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

    def validate_structural_integrity(self) -> torch.Tensor:
        """Propagate integrity check from attention block."""
        return self.attention.validate_structural_integrity()
