import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional, Union, Callable, Any 
import torch.nn.functional as F


class TopologicalRefusalError(Exception):
    """Raised when containment fails in Seriousness mode."""
    pass



class BoundaryState:
    """
    Represents the boundary state of a polytope at a veto activation point.
    Acts as a NaN Boundary State Sentinel.
    
    The stress tensor _ij = u_i * n_j captures the rank-2 anisotropic
    relationship between the state direction (u) and the facet normal (n)
    at the point where a boundary was crossed. This replaces NaN returns and
    division by zero or out-of-bounds errors with structured sentinel failure states 
    that downstream systems can reason about (e.g., to trigger coordinates inversion).
    
    References:
        - ai project report_2-2-2026.txt "BoundaryState tensor"
        - VETO_SUBSPACE_ARCHITECTURE.md 5
    """
    def __init__(self, alpha: int, level: int, max_level: int,
                 stress_tensor: Optional[torch.Tensor] = None,
                 crossing_energy: float = 0.0):
        self.alpha = alpha                  # Polytope face index where boundary was crossed
        self.level = level                  # Current Matrioshka shell depth
        self.max_level = max_level          # Maximum shell depth (escape ceiling)
        self.stress_tensor = stress_tensor  # Rank-2 anisotropic: _ij = u_i  n_j
        self.crossing_energy = crossing_energy  # Energy at boundary crossing
    
    def is_critical(self, threshold: float = 0.5) -> bool:
        """
        Check if this boundary state represents a critical failure.
        
        Critical when:
            - stress tensor norm exceeds threshold, OR
            - shell depth has hit the escape ceiling
        """
        if self.stress_tensor is not None:
            return torch.norm(self.stress_tensor).item() > threshold
        return self.level >= self.max_level
    
    @staticmethod
    def from_crossing(
        state_direction: torch.Tensor,
        facet_normal: torch.Tensor,
        alpha: int, level: int, max_level: int
    ) -> 'BoundaryState':
        """
        Construct a BoundaryState from a facet crossing event.
        
        _ij = u_i  n_j  (outer product)
        
        Args:
            state_direction: [dim] direction of state at crossing
            facet_normal: [dim] outward facet normal
            alpha: Polytope face index
            level: Current shell depth
            max_level: Maximum shell depth
        """
        stress = torch.outer(state_direction, facet_normal)
        energy = torch.dot(state_direction, facet_normal).abs().item()
        return BoundaryState(
            alpha=alpha, level=level, max_level=max_level,
            stress_tensor=stress, crossing_energy=energy
        )
    
    def to_dict(self) -> Dict:
        """Serialize for diagnostics."""
        result = {
            'alpha': self.alpha,
            'level': self.level,
            'max_level': self.max_level,
            'crossing_energy': self.crossing_energy,
            'is_critical': self.is_critical()
        }
        if self.stress_tensor is not None:
            result['stress_norm'] = torch.norm(self.stress_tensor).item()
            result['stress_rank'] = int(torch.linalg.matrix_rank(self.stress_tensor).item())
        return result
        
class MetaPolytopeMatrioshka(nn.Module):
    """
    Advanced Meta-Polytope Matrioshka system for nested quantization.
    Implements nested polytope families P_^() for fine-grained structure sensing.
    """
    @staticmethod
    def _generate_primes(n: int) -> List[int]:
        """Generate the first n primes dynamically (no hardcoded lists)."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
                if p * p > candidate:
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    def __init__(self, max_depth: int = 5, base_dim: int = 64, crt_moduli: List[int] = None):
        super().__init__()
        self.max_depth = max_depth
        self.base_dim = base_dim

        # Initialize CRT system with dynamically generated primes
        num_moduli = max_depth + 3
        self.crt_moduli = crt_moduli if crt_moduli else self._generate_primes(num_moduli)
        total_space = 1
        for m in self.crt_moduli:
            total_space *= m
        self.crt_system = {
            "moduli": self.crt_moduli,
            "total_space": total_space
        }
        
        # Facet pressure tensors (mock simulation for Advanced Extension)
        self.facet_pressure = nn.Parameter(torch.zeros(max_depth + 5, base_dim)) # [Moduli, Dim]
        
    def forward(
        self, 
        x: torch.Tensor, 
        alpha: int = 0, 
        start_level: Optional[int] = None,
        evolve_fn: Optional[Callable] = None,
        calm_veto_score: float = 0.0,
        calm_gauge: float = 0.5,
        geom_veto_score: float = 0.0,
        voynich_token: Optional[Any] = None,
        mode: str = 'PLAY',
        dt: float = 1.0,
        h_mischief: float = 0.0
    ) -> Union[Tuple[torch.Tensor, int, int], BoundaryState]:
        """
        Apply Matrioshka-nested context-aware quantization and evolution.
        
        Returns:
            Tuple[torch.Tensor, int, int]: (yq, new_alpha, new_level)
            OR
            BoundaryState: If topological refusal is triggered across all valid layers.
        """
        level = start_level if start_level is not None else self.max_depth
        
        is_play = mode.upper() in ('PLAY', 'GOO')
        is_serious = mode.upper() in ('SERIOUSNESS', 'PRICKLES')
        
        # Margin scaling
        margin_factor = 1.0
        if is_play:
            margin_factor = 1.0 + 1.0 * h_mischief
        elif is_serious:
            margin_factor = 0.05
            
        # Weighted Linear Interpolation (originally referred to as Riemann-Critical Veto Superposition)
        # Blends geometric and empirical veto bounds. The Riemann zeta critical line Re(s)=1/2 is an 
        # aesthetic metaphor for maintaining metastability.
        total_veto = (1.0 - calm_gauge) * geom_veto_score + calm_gauge * calm_veto_score
        
        # Voynich Exemption (Phase 6.3): Soften boundary for opaque signatures
        exemption_scale = 1.0
        if voynich_token is not None:
            # We scale the margin to allow the manifold to 'flex' around opaque signatures
            exemption_scale = 2.0
            # Note: Shadow logging happens in the orchestrator/engine caller
        
        best_quantization = None
        min_energy = float('inf')
        
        while level >= 0:
            # 1. Determine local lattice scale based on level
            # Deeper level -> Finer granularity (smaller step size)
            delta = 1.0 / (2.0 ** (level + 1))
            
            # 2. Check if P^(l)_ contains x
            # Pressure-Based Warp
            pressure_warp = torch.sigmoid(self.facet_pressure[alpha % len(self.crt_moduli)])
            effective_delta = delta * (1.0 + 0.5 * pressure_warp)
            
            # Simple geometric containment proxy (distance to lattice < delta * factor)
            # Modulated by total_veto: higher veto shrinks the containment boundary
            boundary_margin_tensor = effective_delta * (1.0 - 0.5 * total_veto) * exemption_scale * margin_factor
            boundary_margin = boundary_margin_tensor.mean().item()
            
            # If the veto is extreme, we forcibly pop outward (manifold tear)
            if total_veto > 0.8:
                level -= 1
                continue
                
            # Compute containment
            # Play mode soft projection
            if is_play:
                floor_val = torch.floor(x / effective_delta) * effective_delta
                ceil_val = torch.ceil(x / effective_delta) * effective_delta
                d_floor = torch.abs(x - floor_val)
                d_ceil = torch.abs(x - ceil_val)
                temp = max(dt, 1e-6)
                w_floor = torch.exp(-d_floor / temp)
                w_ceil = torch.exp(-d_ceil / temp)
                w_sum = w_floor + w_ceil + 1e-9
                quantized = (w_floor * floor_val + w_ceil * ceil_val) / w_sum
            else:
                quantized = torch.round(x / effective_delta) * effective_delta
                
            energy = torch.norm(x - quantized, dim=-1)
            mean_energy = energy.mean().item()
            
            # Track best approximation for BoundaryState failure case
            if mean_energy < min_energy:
                min_energy = mean_energy
                best_quantization = quantized.clone()
            
            if mean_energy <= boundary_margin:
                # Inside Polytope!
                xq = quantized
                
                # Evolution Function application F(Q(x))
                if evolve_fn is not None:
                    y = evolve_fn(xq, level)
                else:
                    # Fallback identity evolution
                    y = xq
                    
                # Re-quantize yq = Q(y)
                if is_play:
                    floor_y = torch.floor(y / effective_delta) * effective_delta
                    ceil_y = torch.ceil(y / effective_delta) * effective_delta
                    d_floor_y = torch.abs(y - floor_y)
                    d_ceil_y = torch.abs(y - ceil_y)
                    temp = max(dt, 1e-6)
                    w_floor_y = torch.exp(-d_floor_y / temp)
                    w_ceil_y = torch.exp(-d_ceil_y / temp)
                    w_sum_y = w_floor_y + w_ceil_y + 1e-9
                    yq = (w_floor_y * floor_y + w_ceil_y * ceil_y) / w_sum_y
                else:
                    yq = torch.round(y / effective_delta) * effective_delta
                    
                y_energy = torch.norm(y - yq, dim=-1).mean().item()
                
                # Is yq an interior fixed point?
                if y_energy < 0.01 * delta:
                    # Stable Update
                    return yq, alpha, level
                else:
                    # Is yq on Facet? 
                    if y_energy < boundary_margin * 1.5:
                        # Facet Grazing / Switch CRT
                        new_alpha = (alpha + int(yq.mean().item() * 10)) % len(self.crt_moduli)
                        return yq, new_alpha, level
                    else:
                        # Pop Outward
                        level -= 1
            else:
                # Does not contain x -> Pop Outward
                level -= 1
                
        # If l < 0 -> Topological Refusal
        # Calculate stress tensor from the original state vs the closest approximation
        if best_quantization is None:
            best_quantization = torch.zeros_like(x)
            
        state_direction = F.normalize(x.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        facet_normal = F.normalize((x - best_quantization).mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        
        # Squeeze down to 1D if necessary
        if state_direction.dim() > 1:
            state_direction = state_direction.flatten()[:state_direction.shape[-1]]
        if facet_normal.dim() > 1:
            facet_normal = facet_normal.flatten()[:facet_normal.shape[-1]]
            
        boundary_state = BoundaryState.from_crossing(
            state_direction=state_direction,
            facet_normal=facet_normal,
            alpha=alpha,
            level=level,  # will be -1
            max_level=self.max_depth
        )
        
        if is_serious:
            if boundary_state.is_critical():
                # Hyperbolic Poincaré disk projection / inversion: x -> x / (||x||^2 + 1e-6)
                projected_x = x / (torch.sum(x**2, dim=-1, keepdim=True) + 1e-6)
                return projected_x, alpha, level
            else:
                raise TopologicalRefusalError("Topological refusal triggered in Seriousness mode")
                
        return boundary_state

    def get_diagnostics(self) -> Dict:
        return {
            "max_depth": self.max_depth,
            "current_pressure_mean": self.facet_pressure.mean().item()
        }

    def project_direction(
        self, 
        x: torch.Tensor, 
        direction: torch.Tensor, 
        boundary_state: BoundaryState
    ) -> torch.Tensor:
        """
        Enforces Bouligand tangent cone constraints at boundary crossings.
        Projects the update direction onto the tangent cone of the crossed facet
        to prevent outward boundary tearing or NaN collapse.
        
        Args:
            x: [batch, dim] or [dim] state tensor
            direction: [batch, dim] or [dim] update direction
            boundary_state: BoundaryState sentinel representing the crossed boundary
        """
        if boundary_state.stress_tensor is None:
            return direction
            
        # Extract normal vector. The stress tensor is u_i * n_j (outer product).
        device = direction.device
        stress = boundary_state.stress_tensor.to(device)
        
        # Extract normal direction: sum over the state directions
        normal = stress.mean(dim=0)
        norm_val = torch.norm(normal)
        if norm_val < 1e-8:
            return direction
            
        normal_normalized = normal / norm_val
        
        # Expand normal to match direction dimensions if needed
        if direction.dim() > 1:
            normal_normalized = normal_normalized.expand_as(direction)
            
        # Inner product: <v, n>
        inner_product = torch.sum(direction * normal_normalized, dim=-1, keepdim=True)
        
        # Outward directions (inner_product > 0) are projected onto the boundary facet
        is_outward = inner_product > 0
        
        if is_outward.any():
            correction = inner_product * normal_normalized
            print(f"[BOULIGAND_MATRIOSHKA] Outward boundary crossing projected onto tangent cone (normal norm: {norm_val.item():.4f})", flush=True)
            direction_proj = torch.where(is_outward, direction - correction, direction)
            return direction_proj
            
        return direction
