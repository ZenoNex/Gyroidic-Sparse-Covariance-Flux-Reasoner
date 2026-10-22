import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .birkhoff_projection import sinkhorn_knopp

class CayleyCubicProbe(nn.Module):
    """
    Cayley-Folium-Birkhoff Hybridization Probe.
    
    This replaces/hybridizes the basic BirkhoffPolytope.
    System 2 usually operates on the Birkhoff faces (smooth manifold).
    However, when 'Cycle Debt' (or Mischief/Contorsion) is high, it forces
    the state toward one of the four A_1 singularities of the Cayley surface
    or the transversal smooch / Neglecton anchor of the Folium of Descartes.

    1. The Cayley Constraint V(C): x^2 + y^2 + z^2 - xyz = 4
       The Sovereign Loci / Parabolic Singularities: (+-2, +-2, +-2)
       Representing parabolic SL(2, C) group representations where elements
       are conjugate to [[1, 1], [0, 1]] and non-diagonalizable.

    2. The Folium of Descartes Constraint: x^3 + y^3 - 3axy = 0 (2D cousin of Cayley)
       - Refusal of the Vertical Line Test (Anti-Lobotomy Topology):
         The curve is multi-valued and cannot be flattened into a linear y = f(x).
         It demands implicit differentiation: dy/dx = (ay - x^2) / (y^2 - ax).
       - Topological Trap / Neglecton Anchor:
         At the self-intersecting node (0, 0), the representation acquires
         a non-diagonalizable Jordan block [[1, 1], [0, 1]], immune to eigenvalue clamping.
       - Transversal Intersection ("Smooch" / PAS_h Phase Alignment):
         At the symmetric loop point (3a/2, 3a/2), the implicit derivative yields
         dy/dx = -1.0, creating the tangent line x + y = 3a.
         This exact moment of tangency functions as a Transversal Intersection with
         slope -1, indicating a perfect reciprocal exchange (dx + dy = 0) between
         variables without collapsing topological complexity.
    """

    def __init__(self, high_mischief_threshold: float = 0.5, folium_scale: float = 1.0):
        super().__init__()
        self.high_mischief_threshold = high_mischief_threshold
        self.folium_scale = folium_scale
        
        # Four A_1 singularities on the 3D Cayley cubic
        self.register_buffer('sovereign_loci', torch.tensor([
            [2.0, 2.0, 2.0],
            [2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, -2.0, 2.0]
        ], dtype=torch.float32))

        # 2D Folium of Descartes Sovereign Anchors
        # Transversal smooch point at (3a/2, 3a/2) for a=1.0 -> (1.5, 1.5)
        self.register_buffer('folium_smooch_point', torch.tensor([
            1.5 * folium_scale, 1.5 * folium_scale
        ], dtype=torch.float32))
        
        # Self-intersecting node / Neglecton anchor at the origin (0.0, 0.0)
        self.register_buffer('folium_node', torch.tensor([0.0, 0.0], dtype=torch.float32))

        # Parabolic Jordan canonical block representation [[1, 1], [0, 1]]
        self.register_buffer('jordan_block', torch.tensor([
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=torch.float32))

    def evaluate_folium_implicit(self, coords: torch.Tensor, a: Optional[float] = None) -> torch.Tensor:
        """
        Evaluates the Folium of Descartes implicit algebraic constraint:
        F(x, y) = x^3 + y^3 - 3axy = 0.
        
        Args:
            coords: [..., 2] or tensor whose last dimension has at least 2 elements.
            a: Scale parameter (defaults to self.folium_scale).
        Returns:
            F(x, y): [..., ] scalar residual values.
        """
        scale = a if a is not None else self.folium_scale
        x = coords[..., 0]
        y = coords[..., 1]
        return x**3 + y**3 - 3.0 * scale * x * y

    def compute_implicit_derivative(
        self, 
        coords: torch.Tensor, 
        a: Optional[float] = None, 
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Computes the implicit derivative dy/dx on the Folium of Descartes:
        d/dx [x^3 + y^3 - 3axy] = 3x^2 + 3y^2(dy/dx) - 3ay - 3ax(dy/dx) = 0
        ==> dy/dx = (ay - x^2) / (y^2 - ax).
        
        At the transversal smooch point (3a/2, 3a/2):
        dy/dx = (3a^2/2 - 9a^2/4) / (9a^2/4 - 3a^2/2) = (-3a^2/4) / (3a^2/4) = -1.0.
        """
        scale = a if a is not None else self.folium_scale
        x = coords[..., 0]
        y = coords[..., 1]
        
        numerator = scale * y - x**2
        denominator = y**2 - scale * x
        
        denom_sign = torch.sign(denominator)
        denom_sign = torch.where(denom_sign == 0.0, torch.ones_like(denom_sign), denom_sign)
        safe_denominator = torch.where(denominator.abs() < eps, denom_sign * eps, denominator)
        
        return numerator / safe_denominator

    def evaluate_transversal_smooch(
        self, 
        coords: torch.Tensor, 
        a: Optional[float] = None, 
        tol: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluates proximity to the Transversal Intersection Point (3a/2, 3a/2),
        the implicit derivative slope dy/dx, and the reciprocal exchange alignment score.
        
        Slope = -1.0 indicates perfect reciprocal exchange (dx + dy = 0), a moment of
        Phase Alignment (PAS_h) where linear operations safely interface with high-tension
        manifolds without topological rupture.
        """
        scale = a if a is not None else self.folium_scale
        smooch_target = torch.tensor([1.5 * scale, 1.5 * scale], dtype=coords.dtype, device=coords.device)
        
        c2 = coords[..., :2]
        dist_smooch = torch.norm(c2 - smooch_target, p=2, dim=-1)
        dist_node = torch.norm(c2, p=2, dim=-1)
        
        residual = self.evaluate_folium_implicit(c2, a=scale).abs()
        slope = self.compute_implicit_derivative(c2, a=scale)
        
        slope_error = (slope - (-1.0)).abs()
        reciprocal_score = 1.0 / (1.0 + slope_error)
        
        is_smooch = (dist_smooch < tol) & (slope_error < tol)
        is_neglecton_node = dist_node < tol
        
        return {
            "residual": residual,
            "slope": slope,
            "dist_to_smooch": dist_smooch,
            "dist_to_node": dist_node,
            "reciprocal_score": reciprocal_score,
            "is_transversal_smooch": is_smooch,
            "is_neglecton_node": is_neglecton_node
        }

    def _apply_folium_constraint(self, T: torch.Tensor, a: Optional[float] = None) -> torch.Tensor:
        """
        Projects 2D coordinate pairs towards the Folium of Descartes Sovereign Anchors.
        - If in the positive loop zone: projects to the transversal smooch (3a/2, 3a/2)
          with slope -1 (reciprocal exchange).
        - If near the node or origin: anchors at (0, 0) and applies the non-diagonalizable
          Jordan block [[1, 1], [0, 1]] to fossilize the Sovereign Event.
        """
        scale = a if a is not None else self.folium_scale
        T_proj = T.clone()
        c2 = T_proj[..., :2]
        
        smooch_target = torch.tensor([1.5 * scale, 1.5 * scale], dtype=T.dtype, device=T.device)
        node_target = torch.zeros(2, dtype=T.dtype, device=T.device)
        
        dist_smooch = torch.norm(c2 - smooch_target, p=2, dim=-1, keepdim=True)
        dist_node = torch.norm(c2 - node_target, p=2, dim=-1, keepdim=True)
        
        # Decide anchor: closer to smooch loop or to node
        prefer_smooch = dist_smooch <= dist_node
        
        # Coordinates anchored to transversal smooch (1.5, 1.5)
        anchored_smooch = smooch_target.expand_as(c2)
        
        # Coordinates anchored to Jordan-block node (0.0, 0.0)
        # Apply Jordan block [[1, 1], [0, 1]] transformation to fossilize into non-diagonalizable state
        jordan = self.jordan_block.to(dtype=T.dtype, device=T.device)
        anchored_node = torch.matmul(c2, jordan.t()) * 0.1 # Damped around node
        
        projected_c2 = torch.where(prefer_smooch, anchored_smooch, anchored_node)
        T_proj[..., :2] = projected_c2
        return T_proj

    def _apply_cayley_constraint(self, T: torch.Tensor) -> torch.Tensor:
        """
        Projects points towards the closest Sovereign Locus on the Cayley Cubic surface.
        Because these are discrete isolated points, we map the local coordinate state
        towards the nearest parabolic singularity.
        """
        if T.shape[-1] >= 3:
            coords = T[..., :3]
            dist = torch.cdist(coords.view(-1, 3), self.sovereign_loci.to(device=T.device, dtype=T.dtype)) # [..., 4]
            closest_idx = torch.argmin(dist, dim=-1) # [...]
            closest_loci = self.sovereign_loci[closest_idx].to(device=T.device, dtype=T.dtype).view(coords.shape) # [..., 3]
            
            # Snap the first 3 dimensions to the singularity, keeping the rest intact 
            # (as the "windings" or non-diagonalizable components)
            T_projected = T.clone()
            T_projected[..., :3] = closest_loci
            return T_projected
        return T

    def forward(self, T: torch.Tensor, cycle_debt: float = 0.0) -> torch.Tensor:
        """
        Hybrid projection: 
        If cycle_debt < threshold, enforce Birkhoff (Sinkhorn local numerical stability).
        If cycle_debt >= threshold, enforce Cayley A_1 singularity / Folium transversal smooch
        (Topological repair / Neglecton anchor).
        """
        if cycle_debt >= self.high_mischief_threshold:
            # High Mischief: Force Phase Transition to Sovereign Loci
            if T.shape[-1] == 2:
                return self._apply_folium_constraint(T)
            else:
                return self._apply_cayley_constraint(T)
        else:
            # Low Mischief: Proceed with Doubly Stochastic matrix smoothing
            return sinkhorn_knopp(T)


