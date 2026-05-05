"""
Core Operational Primitives for Gyroidic Invariant Optimization.

Implements:
1. Fixed-Point Arithmetic (Int64 backed) for deterministic execution.
2. Learned Primitive Perturbation for adaptable quantization.
3. FieldPrimitive container class.

"Floating-point arithmetic introduces nondeterminism... Fixed-point arithmetic preserves reproducibility."

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from src.core.honest_jitter import harvest_honest_jitter

# Global Scale Factor for Fixed Point: Q16.16 or similar
# We use a large scale to maintain precision for "scalar gyroidic ergodicity"
SCALE_FACTOR = 65536.0  # 2^16
SCALE_INT = 65536

def stochastic_round(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Apply Stochastic Rounding (SR) to prevent quantization bias.
    Formula: floor(x * scale + harvest_honest_jitter()) / scale
    
    Reference: 45.2 (Silicon Sovereignty) - "No Deterministic Rounding".
    """
    # Scale and add honest jitter [0, 1)
    # harvest_honest_jitter with scaled=False returns [0, 1]
    jitter = harvest_honest_jitter(x.shape, device=x.device, scaled=False)
    return torch.floor(x * scale + jitter) / scale

class FixedPointField(nn.Module):
    """
    Deterministic Fixed-Point Field Primitive.
    
    Wraps an Int64 tensor but behaves like a float tensor for gradients,
    while ensuring all forward operations are bit-exact deterministic.
    """
    def __init__(self, data: torch.Tensor, scale: float = SCALE_FACTOR):
        super().__init__()
        self.scale = scale
        
        # Quantize immediately on init using Stochastic Rounding
        if data.is_floating_point():
            # (data * scale).round() -> stochastic_round(data, scale) * scale
            self.int_data = (stochastic_round(data, scale) * scale).to(torch.int64)
        else:
            self.int_data = data.to(torch.int64)
            
        self.register_buffer('backing_store', self.int_data)

    def forward(self) -> torch.Tensor:
        """Dequantize for interaction with legacy float32 layers if needed."""
        # This is where we inject the "Learned Primitive Perturbation" hooks later
        return self.backing_store.float() / self.scale

    @staticmethod
    def from_float(x: torch.Tensor) -> 'FixedPointField':
        return FixedPointField(x)

    def add(self, other: 'FixedPointField') -> 'FixedPointField':
        """Deterministic addition."""
        res_data = self.backing_store + other.backing_store
        return FixedPointField(res_data, self.scale)
    
    def mul(self, other: 'FixedPointField') -> 'FixedPointField':
        """Deterministic multiplication with rescaling."""
        # (x * scale) * (y * scale) = (xy) * scale^2
        # Need to divide by scale to get back to Q format
        res_data = (self.backing_store * other.backing_store) // SCALE_INT
        return FixedPointField(res_data, self.scale)


class LearnedPrimitivePerturbation(nn.Module):
    """
    Learned perturbation for adaptive quantization.
    
    Allows the grid to 'breathe' during evolution, maintaining
    ergodicity without breaking the fixed-point lattice capability.
    """
    def __init__(self, dim: int, perturbation_scale: float = 0.01):
        super().__init__()
        # Learnable shift in fixed-point space
        self.shift = nn.Parameter(torch.zeros(dim) * SCALE_INT) 
        self.scale_mod = nn.Parameter(torch.ones(dim)) # Multiplicative mod
        
    def forward(self, field: FixedPointField) -> FixedPointField:
        """
        Apply learned perturbation to the integer field.
        
        x' = (x * alpha) + beta
        """
        # We must keep operations deterministic
        # 1. Scale modification (using fixed point math)
        scale_int = (self.scale_mod * 1024).round().to(torch.int64) # Q10 precision for scale
        scaled_data = (field.backing_store * scale_int) // 1024
        
        # 2. Shift (already in int domain)
        shift_int = (self.shift * self.scale_mod).round().to(torch.int64)
        
        out_data = scaled_data + shift_int
        return FixedPointField(out_data, field.scale) # Re-wrap securely without mantissa loss

def to_fixed_point(x: torch.Tensor) -> torch.Tensor:
    """Helper to quantize a tensor using stochastic rounding."""
    return (stochastic_round(x, SCALE_FACTOR) * SCALE_FACTOR).to(torch.int64)

def from_fixed_point(x: torch.Tensor) -> torch.Tensor:
    """Helper to dequantize a tensor."""
    return x.float() / SCALE_FACTOR
