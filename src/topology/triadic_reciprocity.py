"""
Triadic Reciprocity Check.

For processing Generative Art "Word Salad" (DeepDream, early DALL-E).
Instead of checking visual features against Euclidean object datasets (e.g. "Is this a real dog?"),
this module validates the topological flow logic between conflicting features.

If the triadic flow forms a closed reciprocity loop, it is structurally honest, even
if the semantic content is a hallucination.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from src.core.cayley_cubic_probe import CayleyCubicProbe


class TriadicReciprocityChecker(nn.Module):
    """
    Checks if a set of three feature flows A, B, and C exhibit topological reciprocity.
    Reciprocity is defined as the flows mutually reinforcing their geometric loops
    rather than scattering entropically.
    Delegates Cayley cubic singularity checks to canonical CayleyCubicProbe.
    """
    
    def __init__(self, tolerance: float = 1e-3, high_mischief_threshold: float = 0.5):
        super().__init__()
        self.tolerance = tolerance
        # Canonical Cayley-Birkhoff Singularity Probe
        self.cayley_probe = CayleyCubicProbe(high_mischief_threshold=high_mischief_threshold)
        
    def check_flow_reciprocity(
        self, 
        flow_a: torch.Tensor, 
        flow_b: torch.Tensor, 
        flow_c: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluates the structural honesty of the triadic flow.
        
        A flow is reciprocal if (A -> B), (B -> C), and (C -> A) form a closed continuum.
        We estimate this via the cyclic triple scalar product (or determinant of the flow matrix)
        indicating a well-defined orientation/volume in the feature space, avoiding
        topological collapse.
        
        Args:
            flow_a, flow_b, flow_c: Tensors representing visual feature flows [batch, dim]
            
        Returns:
            reciprocity_score: [batch] indicating the strength of the closed loop.
        """
        # Normalize the flows
        fa = torch.nn.functional.normalize(flow_a, dim=-1)
        fb = torch.nn.functional.normalize(flow_b, dim=-1)
        fc = torch.nn.functional.normalize(flow_c, dim=-1)
        
        # If dim >= 3, compute the generalized triple scalar product (volume of parallelepiped)
        # For arbitrary dimensions, we can look at the Frobenius norm of their mutual skew-symmetric relations
        # Simple proxy: 1 - |(fa . fb) + (fb . fc) + (fc . fa)|/3. 
        # If they are mutually orthogonal, this approaches 1 (perfect "corner" forming a volume)
        # If they are collinear, this approaches 0 or negative.
        
        ab = torch.sum(fa * fb, dim=-1)
        bc = torch.sum(fb * fc, dim=-1)
        ca = torch.sum(fc * fa, dim=-1)
        
        # A low sum of dot products indicates they span a distinct volume (high reciprocity)
        # A high sum means they are collapsing into a single vector (low reciprocity)
        reciprocity_score = 1.0 - (torch.abs(ab) + torch.abs(bc) + torch.abs(ca)) / 3.0
        
        return torch.clamp(reciprocity_score, min=0.0, max=1.0)

    def check_cayley_variety(
        self,
        flow_a: torch.Tensor,
        flow_b: torch.Tensor,
        flow_c: torch.Tensor,
        cycle_debt: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluates triadic coordinates on the Cayley cubic surface V(C): x^2 + y^2 + z^2 - xyz = 4.
        Projects flows towards sovereign loci if cycle debt / mischief is high.
        """
        fa = torch.nn.functional.normalize(flow_a, dim=-1)
        fb = torch.nn.functional.normalize(flow_b, dim=-1)
        fc = torch.nn.functional.normalize(flow_c, dim=-1)
        
        ab = torch.sum(fa * fb, dim=-1, keepdim=True)
        bc = torch.sum(fb * fc, dim=-1, keepdim=True)
        ca = torch.sum(fc * fa, dim=-1, keepdim=True)
        
        # Triad coordinates in R^3
        triad_coords = torch.cat([ab, bc, ca], dim=-1)
        
        # Cayley cubic residual: |x^2 + y^2 + z^2 - xyz - 4|
        x = triad_coords[..., 0]
        y = triad_coords[..., 1]
        z = triad_coords[..., 2]
        cayley_residual = torch.abs(x**2 + y**2 + z**2 - x * y * z - 4.0)
        
        # Apply Cayley singularity projection if cycle debt is active
        projected_triad = self.cayley_probe(triad_coords, cycle_debt=cycle_debt)
        
        return {
            "cayley_residual": cayley_residual,
            "projected_triad": projected_triad,
            "reciprocity_score": self.check_flow_reciprocity(flow_a, flow_b, flow_c)
        }

