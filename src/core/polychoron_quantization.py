"""
Meta-Polytope Quantization: 600-Cell (Tetraplex) Mapping & Polychron Quantizer.

Provides high-dimensional symmetry for quantization, preserving chirality,
fixed-point accuracy, and carry-free XOR residue Moiré dynamics.
"""

import torch
import torch.nn as nn
import numpy as np
from itertools import permutations, product
from typing import Dict


class Polychoron600Quantizer(nn.Module):
    """
    Quantizes 4D signals by projecting them onto the vertices of a 600-cell.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('vertices', self._generate_600_cell_vertices())

    def _generate_600_cell_vertices(self) -> torch.Tensor:
        """Generates the 120 vertices of the 600-cell."""
        phi = (1 + 5**0.5) / 2
        vertices = []

        # 1. 8 permutations of (+-1, 0, 0, 0)
        for i in range(4):
            for s in [-1, 1]:
                v = [0, 0, 0, 0]
                v[i] = s
                vertices.append(v)

        # 2. 16 combinations of (+-1/2, +-1/2, +-1/2, +-1/2)
        for s in product([-0.5, 0.5], repeat=4):
            vertices.append(list(s))

        # 3. 96 even permutations of (+-phi/2, +-1/2, +-1/(2phi), 0)
        base_96 = [phi/2, 0.5, 1/(2*phi), 0]
        
        vertices_96 = []
        all_p = list(permutations([0, 1, 2, 3]))
        even_p = [p for p in all_p if self._permutation_parity(p) == 0]
        
        for p_idx in even_p:
            for s in product([-1, 1], repeat=4):
                v = [0, 0, 0, 0]
                for i in range(4):
                    v[i] = base_96[p_idx[i]] * s[i]
                vertices_96.append(v)
        
        # Remove duplicates
        unique_v = set()
        for v in vertices:
            unique_v.add(tuple(np.round(v, 8)))
        for v in vertices_96:
            unique_v.add(tuple(np.round(v, 8)))
            
        return torch.tensor(list(unique_v), dtype=torch.float32)

    def _permutation_parity(self, p):
        """Returns 0 for even, 1 for odd permutations."""
        parity = 0
        p = list(p)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    parity = 1 - parity
        return parity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects input onto the nearest 600-cell vertex.
        Assume x shape is (..., 4)
        """
        orig_shape = x.shape
        x_flat = x.view(-1, 4)
        dist = torch.cdist(x_flat, self.vertices)
        indices = torch.argmin(dist, dim=1)
        quantized = self.vertices[indices]
        return quantized.view(orig_shape)


class PolychronQuantizer(nn.Module):
    """
    Polychron Quantizer: Moiré-via-Modular-Algebra on the 600-Cell (H4 Tetraplex).
    
    Implements:
    1. Multiplicative Layer (Primes): Incommensurate frequencies f_{p_k} = 2*pi*ln(p_k).
    2. Additive Layer (Carry-Free XOR): Bit-plane XOR residue channels preventing intermodulation.
    3. Geometric Layer (phi-Seesaw): 600-cell golden coordinates (phi/2 - 1/(2*phi) = 1/2) bounding quantization drift.
    4. Fringe-Contrast Head & Holonomy: Computes PAS_h phase alignment and updates BerryPhaseTracker.
    """
    def __init__(self, num_channels: int = 4, prime_base_size: int = 5):
        super().__init__()
        from src.core.invariants import get_prime_ladder, PHI, PI
        from src.core.fgrt_primitives import BerryPhaseTracker
        
        self.num_channels = num_channels
        self.quantizer = Polychoron600Quantizer()
        
        # Incommensurate Prime Frequencies f_{p_k} = 2*pi*ln(p_k)
        primes = get_prime_ladder(prime_base_size)
        frequencies = 2.0 * float(PI) * torch.log(primes.float())
        self.register_buffer('primes', primes)
        self.register_buffer('frequencies', frequencies)
        
        self.phi = float(PHI)
        self.pi = float(PI)
        self.berry_tracker = BerryPhaseTracker()
        self.register_buffer('prev_indices', torch.zeros(1, dtype=torch.long))

    def compute_moire_beat_spectrum(self) -> torch.Tensor:
        """Computes pairwise beat frequencies |ln(p_i) - ln(p_j)| for i < j."""
        log_p = torch.log(self.primes.float())
        beats = []
        n = len(log_p)
        for i in range(n):
            for j in range(i + 1, n):
                beats.append(torch.abs(log_p[i] - log_p[j]))
        return torch.stack(beats)

    def forward(self, x: torch.Tensor, t: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Quantizes 4D signal, applies per-prime phase clocking, computes carry-free XOR 
        residue channel, evaluates the phi-seesaw error bound, and computes PAS_h contrast.
        """
        orig_shape = x.shape
        x_flat = x.view(-1, 4)
        
        # 1. Quantize onto nearest 600-cell vertex
        dist = torch.cdist(x_flat, self.quantizer.vertices)
        indices = torch.argmin(dist, dim=1)
        quantized_flat = self.quantizer.vertices[indices]
        quantized = quantized_flat.view(orig_shape)
        
        # 2. Compute phi-seesaw quantization error: e(t) = x - Q(x)
        quant_error = x_flat - quantized_flat
        phi_seesaw_bound = torch.norm(quant_error, dim=-1).mean()
        
        # 3. Carry-free XOR addition between consecutive nearest vertex indices
        if self.prev_indices.shape != indices.shape:
            self.prev_indices = torch.zeros_like(indices)
        xor_codewords = torch.bitwise_xor(indices, self.prev_indices)
        self.prev_indices = indices.detach()
        
        # 4. Per-prime Phase Clocking: theta_k(t) = (t * f_{p_k}) mod 2*pi
        phase_clocks = torch.fmod(t * self.frequencies, 2.0 * self.pi)
        
        # 5. Fringe Contrast (PAS_h Phase Alignment across channels)
        pas_h = torch.cos(phase_clocks).mean()
        
        # 6. Update Geometric Berry Phase
        berry_phase = self.berry_tracker.update(x_flat, quantized_flat)
        
        return {
            "quantized": quantized,
            "vertex_indices": indices,
            "xor_codewords": xor_codewords,
            "quant_error": quant_error.view(orig_shape),
            "phi_seesaw_bound": phi_seesaw_bound,
            "phase_clocks": phase_clocks,
            "pas_h": pas_h,
            "berry_phase": berry_phase,
            "beat_spectrum": self.compute_moire_beat_spectrum()
        }
