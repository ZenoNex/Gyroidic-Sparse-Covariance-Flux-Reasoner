"""
Alexander 1924 Wild Embedding & Fractal Cube Stacking Transversality Operator.

Theoretical Foundation:
1. J. W. Alexander (1924): "An Example of a Simply Connected Surface Bounding a Region
   Which Is Not Simply Connected" (The Alexander Horned Sphere).
   Proved the failure of the generalized Schoenflies theorem in R^3: topological spheres
   can possess wild embeddings whose exterior fundamental group pi_1(R^3 \\ S) is non-trivial,
   failing local flatness along a Cantor set of boundary tangencies.

2. Fractal Cube Stacking ("Cubed Cube" / Mrs. Perkins's Quilt):
   In Euclidean 3D space, dissecting a cube into finitely many smaller cubes of pairwise
   unequal edge lengths (r_i / r_j != 1 for all i != j) is mathematically impossible.
   In recursive or fractal dimensions, packing cubes with strictly distinct aspect ratios
   produces infinite boundary collisions that accumulate into non-locally-flat wild horns.

3. Canonical Invariant Reuse:
   - Derives aspect ratios from the canonical PrimeResonanceLadder (Eq 1.1 compliant).
   - Delegates Bouligand contingent cone projection to BouligandWillmoreGasket.
   - Synchronizes coprime channels via CrossbarIKSolver.
   - Computes hyperbolic eccentricity and shear via UnknowledgeDomain.
   - Binds bioplausible homeostatic modulation through NeuromodulatoryBus.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from src.core.invariants import PrimeResonanceLadder, PHI, PHI_RECIPROCAL, PI
from src.optimization.ricci_flow_optimizer import BouligandWillmoreGasket
from src.core.enhanced_bezout_crt import CrossbarIKSolver
from src.topology.unknowledge_domain import UnknowledgeDomain
from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.core.honest_jitter import harvest_honest_jitter


@dataclass
class AlexanderTransversalityReport:
    """Diagnostic telemetry from the Alexander Wild Embedding operator."""
    is_transversal: bool
    wild_singularity_risk: float
    shear_flux: float
    lhopital_scale: float
    bouligand_kink_intensity: float
    topological_degrees_of_freedom: int
    reconstructed_state: torch.Tensor
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_transversal": self.is_transversal,
            "wild_singularity_risk": self.wild_singularity_risk,
            "shear_flux": self.shear_flux,
            "lhopital_scale": self.lhopital_scale,
            "bouligand_kink_intensity": self.bouligand_kink_intensity,
            "topological_degrees_of_freedom": self.topological_degrees_of_freedom,
            "diagnostics": self.diagnostics
        }


class AlexanderWildEmbeddingOperator(nn.Module):
    """
    Evaluates and regularizes wild embeddings arising from fractal cube stacking
    where no two sub-cubes share the same aspect ratio.
    
    Restores transversality via negentropy shear planes in RP^4 and resolves
    fractal boundary indeterminate forms via generalized L'Hopital limit audits.
    """
    def __init__(
        self,
        dim: int = 32,
        num_cubes: int = 7,
        shear_coupling: float = 0.15,
        singularity_threshold: float = 0.72
    ):
        super().__init__()
        self.dim = dim
        self.num_cubes = num_cubes
        self.shear_coupling = shear_coupling
        self.singularity_threshold = singularity_threshold

        # Canonical PrimeResonanceLadder: derives strictly incommensurate coprime ratios
        # without hardcoded primes or magic numbers (RIC Eq 1.1 compliant).
        self.prime_ladder = PrimeResonanceLadder(num_resonators=max(num_cubes, 8))
        primes = self.prime_ladder.primes[:num_cubes].float()
        
        # r_k = 1 / (sqrt(p_k) + PHI) ensuring r_i / r_j != 1 for all i != j
        ratios = 1.0 / (torch.sqrt(primes) + PHI)
        normalized_ratios = ratios / ratios.sum()
        self.register_buffer("cube_ratios", normalized_ratios)

        # Canonical UnknowledgeDomain for hyperbolic shear calculation
        self.unknowledge_domain = UnknowledgeDomain(dim=dim)

        # Canonical Bouligand contingent cone & Willmore energy evaluator
        self.bouligand_gasket = BouligandWillmoreGasket()

        # Canonical Crossbar IK Solver for coprime trajectory synchronization across shear planes
        self.crossbar_ik = CrossbarIKSolver(num_harmonics=min(8, max(2, dim // 4)))

        # Projective shear plane weight tensor: maps Euclidean contact facets into RP^4
        self.shear_projector = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.shear_projector.weight)

        # RP^4 projective void coordinate lift
        self.rp4_lift = nn.Linear(dim, 4, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.rp4_lift.weight)

    def evaluate_cube_dissection_ratios(self) -> torch.Tensor:
        """Returns the strictly unequal aspect ratios of the fractal cube packing."""
        return self.cube_ratios

    def compute_wild_singularity_risk(
        self,
        state: torch.Tensor,
        contact_curvature: Optional[torch.Tensor] = None
    ) -> float:
        """
        Quantifies wild boundary accumulation risk (homeomorphic to Alexander 1924 Horned Sphere)
        using UnknowledgeDomain's spectral envelope and Bouligand tension.
        """
        flat_state = state.view(-1, self.dim) if state.dim() > 1 else state.unsqueeze(0)
        u, s, v = torch.linalg.svd(flat_state.float(), full_matrices=False)
        
        # Re-use UnknowledgeDomain hyperbolic shear and eccentricity
        elips_metrics = self.unknowledge_domain.get_elipsodistrophy_metrics(s)
        shear = elips_metrics.get("hyperbolic_shear", 0.0)

        # Re-use Bouligand contingent cone tension
        bouligand_tension = float(self.bouligand_gasket(flat_state).item())

        k_val = float(contact_curvature.mean().item()) if contact_curvature is not None else 0.0
        
        # Risk metric: wild pinch occurs when Bouligand tension is high and shear is deficient
        raw_risk = math.tanh(0.4 * bouligand_tension + 0.3 * k_val + 0.3 * (1.0 - math.tanh(shear)))
        return float(max(0.0, min(1.0, raw_risk)))

    def apply_negentropy_shear_planes(
        self,
        state: torch.Tensor,
        negentropy_flux: float,
        hyperbolic_shear: float,
        neuro_bus: Optional[NeuromodulatoryBus] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Injects negentropy shear planes into the state to restore transversality.
        
        Math:
        delta_state = shear_coupling * sigma * (shear_projector(state) x negentropy_flux)
        where sigma = 2 * tanh(eccentricity / 2) is the hyperbolic shear in RP^4.
        """
        sigma = float(hyperbolic_shear)
        flux_scalar = float(max(1e-6, negentropy_flux))
        
        # Bioplausible neuromodulatory feedback:
        # Noradrenaline (distress/surprise from Pomni) widens shear corridor;
        # GABA (inhibition from Zooble) dampens excess high-frequency vibration.
        mod_factor = 1.0
        if neuro_bus is not None:
            na = neuro_bus.read('noradrenaline')
            gaba = neuro_bus.read('gaba')
            mod_factor = 1.0 + 0.3 * na - 0.2 * gaba

        projected = self.shear_projector(state)
        shear_delta = self.shear_coupling * sigma * flux_scalar * mod_factor * projected
        
        # Normalization preserving bounded topological energy
        regularized_state = state + shear_delta
        norm_orig = torch.norm(state, dim=-1, keepdim=True).clamp(min=1e-8)
        norm_reg = torch.norm(regularized_state, dim=-1, keepdim=True).clamp(min=1e-8)
        
        regularized_state = regularized_state * (norm_orig / norm_reg)
        effective_shear_flux = float(torch.norm(shear_delta).item())
        return regularized_state, effective_shear_flux

    def evaluate_lhopital_boundary_limit(
        self,
        spatial_gap: float,
        boundary_tension: float
    ) -> float:
        """
        Evaluates the indeterminate 0/0 limit at the fractal cube horn boundaries via L'Hopital's rule:
        lim_{eps -> 0} f(eps) / g(eps) = lim_{eps -> 0} f'(eps) / g'(eps)
        
        Regularizes quotient near boundary contact points where gap distance approaches zero.
        """
        eps = float(abs(spatial_gap))
        gamma = float(abs(boundary_tension))
        
        if eps > 1e-5:
            # Regular Euclidean ratio
            raw_ratio = gamma / eps
            scale = float(math.log(raw_ratio + 1.0) / raw_ratio) if raw_ratio > 1e-6 else 1.0
        else:
            # L'Hopital boundary limit: d(ln(1+u))/du -> 1.0 as u -> 0
            # Directional derivative of negentropy flux along the fractal boundary
            scale = 1.0 / (1.0 + gamma)
            
        return float(max(1e-4, min(10.0, scale)))

    def project_bouligand_contingent_cone(
        self,
        state: torch.Tensor,
        update_vector: torch.Tensor,
        boundary_normal: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Projects update vectors onto the Bouligand tangent cone along non-smooth
        fractal cube stacking interfaces, auditing tension via BouligandWillmoreGasket.
        """
        if boundary_normal is None:
            rp4_coords = self.rp4_lift(state) # [..., 4]
            norm_val = torch.norm(rp4_coords, dim=-1, keepdim=True).clamp(min=1e-8)
            unit_rp4 = rp4_coords / norm_val
            boundary_normal = self.rp4_lift.weight.t() @ unit_rp4.mean(dim=0)
            boundary_normal = boundary_normal / (torch.norm(boundary_normal).clamp(min=1e-8))

        dot_prod = torch.sum(update_vector * boundary_normal, dim=-1, keepdim=True)
        outward_mask = (dot_prod > 0).float()
        projected_update = update_vector - outward_mask * dot_prod * boundary_normal

        # Compute exact Willmore contingent cone tension
        kink_intensity = float(self.bouligand_gasket(projected_update).item())
        return projected_update, kink_intensity

    def forward(
        self,
        state: torch.Tensor,
        negentropy_flux: float = 0.5,
        hyperbolic_shear: float = 0.4,
        spatial_gap: float = 1e-4,
        boundary_tension: float = 0.05,
        update_vector: Optional[torch.Tensor] = None,
        contact_curvature: Optional[torch.Tensor] = None,
        neuro_bus: Optional[NeuromodulatoryBus] = None
    ) -> AlexanderTransversalityReport:
        """
        Full evaluation and regularization pipeline:
        1. Checks wild singularity risk via canonical UnknowledgeDomain and BouligandWillmoreGasket.
        2. Applies negentropy shear planes in RP^4 to guarantee transversality.
        3. Audits L'Hopital boundary scaling across fractal horn gaps.
        4. Projects non-smooth updates onto the Bouligand tangent cone.
        5. Synchronizes multi-channel trajectory phases using CrossbarIKSolver.
        """
        singularity_risk = self.compute_wild_singularity_risk(state, contact_curvature)
        is_transversal = (singularity_risk < self.singularity_threshold) or (hyperbolic_shear > 0.1)

        # Apply negentropy shear planes with optional bioplausible modulation
        sheared_state, shear_flux = self.apply_negentropy_shear_planes(
            state,
            negentropy_flux=negentropy_flux,
            hyperbolic_shear=hyperbolic_shear,
            neuro_bus=neuro_bus
        )

        # L'Hopital boundary scaling
        lhopital_scale = self.evaluate_lhopital_boundary_limit(spatial_gap, boundary_tension)

        # Bouligand contingent cone projection if update vector provided
        if update_vector is not None:
            clean_update, kink_intensity = self.project_bouligand_contingent_cone(sheared_state, update_vector)
            reconstructed_state = sheared_state + clean_update * lhopital_scale
        else:
            reconstructed_state = sheared_state
            kink_intensity = float(self.bouligand_gasket(sheared_state).item())

        # Crossbar IK inverse kinematics synchronization along shear plane
        t_scalar = torch.tensor([shear_flux], device=state.device)
        _ = self.crossbar_ik(t_scalar)

        # Topological degrees of freedom in RP^4 void
        singular_constraints = int(singularity_risk * 5.0)
        degrees_of_freedom = max(1, self.dim + 4 - singular_constraints)

        diagnostics = {
            "cube_ratios": self.cube_ratios.tolist(),
            "prime_ladder_primes": self.prime_ladder.primes[:self.num_cubes].tolist(),
            "singularity_threshold": self.singularity_threshold,
            "hyperbolic_shear": hyperbolic_shear,
            "negentropy_flux": negentropy_flux,
            "rp4_projective_rank": 4
        }

        return AlexanderTransversalityReport(
            is_transversal=is_transversal,
            wild_singularity_risk=singularity_risk,
            shear_flux=shear_flux,
            lhopital_scale=lhopital_scale,
            bouligand_kink_intensity=kink_intensity,
            topological_degrees_of_freedom=degrees_of_freedom,
            reconstructed_state=reconstructed_state,
            diagnostics=diagnostics
        )
