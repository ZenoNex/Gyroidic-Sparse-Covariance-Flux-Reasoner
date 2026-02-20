"""
Meta-Polytope Quantization: 600-Cell (Tetraplex) Mapping.

Provides high-dimensional symmetry for quantization, preserving chirality
and fixed-point accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from itertools import permutations, product

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
        base = [phi/2, 0.5, 1/(2*phi), 0.0]
        perms = set()
        # Get all permutations first
        for p in permutations(base):
            for signs in product([-1, 1], repeat=4):
                # Apply signs
                v = [p[i] * signs[i] for i in range(4)]
                
                # Check even permutation condition
                # Parity check
                if self._is_even_permutation(p, base):
                    # We need to be careful with 'even' definition here.
                    # Standard 600-cell uses even permutations of signs and coordinates.
                    # Simplified: just add the known 600-cell vertices structure.
                    perms.add(tuple(v))
        
        # Actually, the standard definition is:
        # All 96 even permutations of (+-phi/2, +-1/2, +-1/2phi, 0)
        # Let's do it properly.
        vertices_96 = []
        base_96 = [phi/2, 0.5, 1/(2*phi), 0]
        
        # All permutations
        all_p = list(permutations([0, 1, 2, 3]))
        even_p = [p for p in all_p if self._permutation_parity(p) == 0]
        
        for p_idx in even_p:
            for s in product([-1, 1], repeat=4):
                v = [0, 0, 0, 0]
                for i in range(4):
                    v[i] = base_96[p_idx[i]] * s[i]
                vertices_96.append(v)
        
        # Use a set to remove duplicates (especially since 0 is in base)
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
        # Reshape to (Batch, 4) for distance calculation
        orig_shape = x.shape
        x_flat = x.view(-1, 4)
        
        # Calculate distances to all 120 vertices
        # (N, 4) and (120, 4) -> (N, 120)
        dist = torch.cdist(x_flat, self.vertices)
        
        # Find nearest vertex index
        indices = torch.argmin(dist, dim=1)
        
        # Map to vertex
        quantized = self.vertices[indices]
        
        return quantized.view(orig_shape)

    @staticmethod
    def _is_even_permutation(p, base):
        # Placeholder for complex parity logic if needed
        return True
