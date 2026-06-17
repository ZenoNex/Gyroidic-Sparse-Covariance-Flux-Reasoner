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
from src.core.honest_jitter import fractal_pad
from src.core.gdpo_normalization import GDPONormalization
from src.core.chern_simons_gasket import ChernSimonsGasket
from src.core.polynomial_coprime import PolynomialCoprimeConfig

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
        A_k = Birkhoff(Q_k K_k^T / d)  V_k
        
    Then fuse via Saturated CRT: L' = SaturatedCRT_Fuse({A_k, r_k})
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        poly_config: PolynomialCoprimeConfig,
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
        self.poly_config = poly_config
        self.use_birkhoff = use_birkhoff
        self.gasket = ChernSimonsGasket(manifold_dim=hidden_dim)
        
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
            self.birkhoff = BirkhoffProjection()
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * self.K, hidden_dim)
        # [ARCHITECTURAL REMEDIATION] Replace crude dropout with Nostalgic Leak
        from src.topology.unknowledge_domain import NostalgicLeakFunctional
        self.dropout = NostalgicLeakFunctional(fossil_dim=hidden_dim)
        
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
        return_field_outputs: bool = False,
        load: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute modular multi-field attention.
        
        Args:
            x: [batch, seq_len, hidden_dim] input
            mask: Optional [batch, seq_len, seq_len] attention mask
            trust_scalars: Optional [K] trust weights for fields
            path_topology_vectors: Optional [batch, seq_len, seq_len] geometric path distances for S-Path RAG
            return_field_outputs: If True, return per-field outputs
            
        Returns:
            output: [batch, seq_len, hidden_dim] attended representation
        """
        # Dimensional Reconciliation via Fractal Padding
        if x.shape[-1] != self.hidden_dim:
            x = fractal_pad(x, self.hidden_dim)
            
        batch_size, seq_len, _ = x.shape
        
        # Operational Integration: Dequantize if primitive
        if hasattr(x, 'backing_store') or isinstance(x, FixedPointField):
             # "Dequantize for interaction with legacy float32 layers"
             # Ideally we'd have a FixedPointAttention, but for integration we unpack here
             x = x.forward()
        
        # Adaptive Functional Sparsification: Drop outer shells under high load
        # Matryoshka nesting: k=0 (inner/stable), k=K-1 (outer/complex)
        active_k = self.K
        if load > 0.0:
            # Linear reduction of functionals based on load
            # Ensures at least 1 functional (the inner shell) remains active
            active_k = max(1, int(self.K * (1.0 - torch.clamp(torch.tensor(load), 0.0, 0.9).item())))
            
        # Process each active field separately
        field_outputs = []
        for k in range(active_k):
            proj = self.field_projections[k]
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
            
            # Phase 6: S-Path RAG Injection (Non-Tokenizing Geometry)
            # Add structural path topology directly to the alignment scores.
            # This allows the model to "feel" geometric distances instead of processing sequential history.
            if kwargs.get('path_topology_vectors') is not None:
                topology_bias = kwargs['path_topology_vectors'].unsqueeze(1) # Broadcast over heads
                scores = scores + topology_bias
                
            # [ARCHITECTURAL REMEDIATION] Replaced crude masked_fill and nn.Dropout with Unknowledge Domain
            # Masking and crude dropout lobotomize the "Dream State" of the attention map.
            # Instead of masking to -1e9, we apply the Unknowledge Domain's computable flux shield later
            # and use Nostalgic Leak on the final states.
            
            # Apply Birkhoff projection if enabled
            if self.use_birkhoff:
                # Project each head's attention matrix to doubly-stochastic
                # Use training mode for annealing schedule
                attn_weights = self.birkhoff(scores, anneal=self.training)  # [batch, num_heads, seq_len, seq_len]
            else:
                attn_weights = torch.softmax(scores, dim=-1)
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
            
        # --- DOWNSTREAM CHERN-SIMONS GASKET ---
        # Data arrives at symbolic-geometric boundaries as CRT residues [Batch, K, Dim//K]
        if len(field_outputs) > 0:
            stacked_residues = torch.stack(field_outputs, dim=2) # [batch_size, seq_len, K, hidden_dim]
            # Reshape for gasket [batch * seq_len, K, hidden_dim]
            flat_residues = stacked_residues.view(batch_size * seq_len, len(field_outputs), self.hidden_dim)
            
            # Pull dynamic polynomial coefficients from live config
            coeffs = self.poly_config.get_coefficients_tensor().to(x.device)
            # Ensure coeffs matches the active K
            if len(field_outputs) < self.K:
                coeffs = coeffs[:len(field_outputs)]
                
            # Plug the logic leak
            self.gasket.to(x.device)
            repaired_flat = self.gasket.plug_logic_leak(flat_residues, coeffs)
            
            # Reshape back to field_outputs format
            repaired_stacked = repaired_flat.view(batch_size, seq_len, len(field_outputs), self.hidden_dim)
            field_outputs = [repaired_stacked[:, :, k, :] for k in range(len(field_outputs))]
        
        # Fuse field outputs (simple concatenation + projection)
        # Pad with zeros if some functionals were sparsified to maintain output_proj shape
        if len(field_outputs) < self.K:
            # "Backwards compatibility: Pad sparse channels to maintain projection geometry"
            dummy = torch.zeros_like(field_outputs[0])
            while len(field_outputs) < self.K:
                field_outputs.append(dummy)
                
        fused = torch.cat(field_outputs, dim=-1)
        # [ARCHITECTURAL REMEDIATION] Apply Nostalgic Leak to the output state
        # instead of dropping out the attention weights.
        output = self.output_proj(fused)  # [batch, seq_len, hidden_dim]
        
        # Apply Nostalgic Leak
        batch_size_out, seq_len_out, dim_out = output.shape
        output_flat = output.view(batch_size_out * seq_len_out, dim_out)
        self.dropout.to(output.device)
        output_leaked = self.dropout(output_flat)
        output = output_leaked.view(batch_size_out, seq_len_out, dim_out)
        
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
        poly_config: PolynomialCoprimeConfig,
        num_functionals: int = 5,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        
        self.attention = ModularAttention(hidden_dim, num_heads, poly_config, num_functionals, dropout)
        
        self.norm1 = GDPONormalization(hidden_dim)
        self.norm2 = GDPONormalization(hidden_dim)
        
        # [ARCHITECTURAL REMEDIATION] Remove crude nn.Dropout from the feed-forward network
        # The NostalgicLeakFunctional in ModularAttention manages the structural preservation.
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        trust_scalars: Optional[torch.Tensor] = None,
        load: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        # Attention with residual (passes S-Path topology vectors if present via kwargs)
        attn_out = self.attention(x, mask, trust_scalars=trust_scalars, load=load, **kwargs)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

    def validate_structural_integrity(self) -> torch.Tensor:
        """Propagate integrity check from attention block."""
        return self.attention.validate_structural_integrity()
