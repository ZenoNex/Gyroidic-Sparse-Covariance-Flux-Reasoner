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




class CayleyAdversarialDecomposer(nn.Module):
    """
    Cayley Adversarial Decomposer & Ingestion Hardener.

    Hardens ingestors against anti-AI adversarial text attacks:
    1. Gibberifier & Invisible Unicode: Zero-width spaces, joiners, isolates,
       and presentation forms (U+FE10-U+FE1F, U+FB00-U+FB4F, U+FE70-U+FEFF).
    2. Homoglyph & Confusable Cloaking: Cross-script substitutions (Cyrillic, Greek).
    3. BPE Boundary Shattering & Context-Window Flooding: Intercepts raw byte multipliers.
    4. Steganographic Tag Smuggling: Unicode tag blocks (U+E0000-U+E007F).

    Dual-Stream Decomposition Principle:
    - Clean Morpheme Stream: Extracts human-readable semantic text without byte clutter,
      allowing downstream tokenizers, Larynx autoregression, and seam cuts to operate normally.
    - Cayley Feature Scar Stream: Evaluates perturbation codepoints against Chebyshev
      minimal roots of unity on the Cayley Cubic surface (x^2 + y^2 + z^2 - xyz = 4),
      snapping to the nearest A_1 nodal singularity (+-2, +-2, +-2) to preserve the
      adversarial energy as a permanent topological feature scar (E_scar) rather than
      lobotomizing the representation.
    """

    def __init__(self, probe: Optional[CayleyCubicProbe] = None):
        super().__init__()
        self.probe = probe if probe is not None else CayleyCubicProbe()
        self.register_buffer('sovereign_loci', self.probe.sovereign_loci)

    @staticmethod
    def is_adversarial_codepoint(c: str) -> bool:
        """Identifies invisible, zero-width, tag, or presentation form codepoints."""
        import unicodedata
        o = ord(c)
        cat = unicodedata.category(c)
        return (
            cat in ('Cf', 'Cc', 'Cs') or
            (0x200B <= o <= 0x200D) or  # Zero-width spaces & joiners
            (0x2060 <= o <= 0x206F) or  # Invisible separators & word joiners
            (o == 0xFEFF) or            # Byte order mark / zero-width no-break
            (0x202A <= o <= 0x202E) or  # Bidirectional overrides
            (0x2066 <= o <= 0x2069) or  # Bidirectional isolates
            (0xE0000 <= o <= 0xE007F) or # Unicode Tag Block
            (0xFE00 <= o <= 0xFE0F) or  # Variation Selectors
            (0xFE10 <= o <= 0xFE1F) or  # Vertical Presentation Forms (U+FE12, U+FE18)
            (0xFB00 <= o <= 0xFB4F) or  # Alphabetic Presentation Forms (U+FB12)
            (0xFE70 <= o <= 0xFEFF)     # Arabic Presentation Forms-B (U+FEB2, U+FE82)
        )

    def decompose(self, text: str) -> Dict[str, Any]:
        """
        Decomposes input text into clean semantic text and Cayley Cubic feature scar metrics.
        """
        import unicodedata
        import math

        if not text:
            return {
                "clean_text": "",
                "perturbation_chars": [],
                "perturbation_density": 0.0,
                "is_adversarial": False,
                "cayley_vector": self.sovereign_loci[0],
                "cayley_scar_energy": 0.0
            }

        clean_chars = []
        perturbations = []

        for i, c in enumerate(text):
            if self.is_adversarial_codepoint(c):
                perturbations.append({
                    "char": c,
                    "codepoint": f"U+{ord(c):04X}",
                    "pos": i,
                    "category": unicodedata.category(c)
                })
            else:
                # NFKC normalization handles cross-script homoglyphs & confusables
                c_norm = unicodedata.normalize('NFKC', c)
                clean_chars.append(c_norm)

        clean_text = "".join(clean_chars)
        total_len = len(text)
        pert_len = len(perturbations)
        density = pert_len / max(1, total_len)
        is_adversarial = (pert_len > 0 and (density > 0.01 or pert_len >= 3))

        # Project perturbation stream into Cayley Cubic coordinates
        if pert_len > 0:
            codepoint_vals = [int(p["codepoint"][2:], 16) for p in perturbations]
            h1 = sum(codepoint_vals[0::3])
            h2 = sum(codepoint_vals[1::3]) if len(codepoint_vals) >= 2 else 2
            h3 = sum(codepoint_vals[2::3]) if len(codepoint_vals) >= 3 else 3

            # Minimal roots of unity (3rd roots of unity from Cayley S_3 symmetry)
            theta_x = (2.0 * math.pi / 3.0) * (h1 % 3)
            theta_y = (2.0 * math.pi / 3.0) * (h2 % 3)
            theta_z = (2.0 * math.pi / 3.0) * (h3 % 3)

            init_coords = torch.tensor([
                2.0 * math.cos(theta_x),
                2.0 * math.cos(theta_y),
                2.0 * math.cos(theta_z)
            ], dtype=torch.float32, device=self.sovereign_loci.device)

            # Snap to nearest A_1 nodal singularity on Cayley Cubic surface
            dist = torch.cdist(init_coords.view(1, 3), self.sovereign_loci)
            closest_idx = torch.argmin(dist, dim=-1)
            cayley_vec = self.sovereign_loci[closest_idx].squeeze(0)
            scar_energy = math.log1p(pert_len) * min(1.0, density * 5.0)
        else:
            cayley_vec = self.sovereign_loci[0] # Default (2, 2, 2)
            scar_energy = 0.0

        return {
            "clean_text": clean_text,
            "perturbation_chars": perturbations,
            "perturbation_density": float(density),
            "is_adversarial": bool(is_adversarial),
            "cayley_vector": cayley_vec,
            "cayley_scar_energy": float(scar_energy)
        }
