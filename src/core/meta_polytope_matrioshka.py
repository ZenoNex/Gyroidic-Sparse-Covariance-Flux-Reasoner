import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional


class BoundaryState:
    """
    Represents the boundary state of a polytope at a veto activation point.
    
    The stress tensor Σ_ij = u_i · n_j captures the rank-2 anisotropic
    relationship between the state direction (u) and the facet normal (n)
    at the point where a boundary was crossed. This replaces NaN returns
    with structured failure states that downstream systems can reason about.
    
    References:
        - ai project report_2-2-2026.txt §"BoundaryState tensor"
        - VETO_SUBSPACE_ARCHITECTURE.md §5
    """
    def __init__(self, alpha: int, level: int, max_level: int,
                 stress_tensor: Optional[torch.Tensor] = None,
                 crossing_energy: float = 0.0):
        self.alpha = alpha                  # Polytope face index where boundary was crossed
        self.level = level                  # Current Matrioshka shell depth
        self.max_level = max_level          # Maximum shell depth (escape ceiling)
        self.stress_tensor = stress_tensor  # Rank-2 anisotropic: Σ_ij = u_i · n_j
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
        
        Σ_ij = u_i · n_j  (outer product)
        
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
    Implements nested polytope families P_α^(ℓ) for fine-grained structure sensing.
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
        
    def forward(self, x: torch.Tensor, alpha: int = 0, level: int = 0) -> Tuple[torch.Tensor, int, int]:
        """
        Apply context-aware quantization based on Matrioshka depth.
        Returns: (quantized_x, new_alpha, new_level)
        """
        # 1. Determine local lattice scale based on level
        # Deeper level -> Finer granularity (smaller step size)
        # Step size Δ ~ 1 / (2^level)
        delta = 1.0 / (2.0 ** (level + 1))
        
        # 2. Apply Pressure-Based Warp
        # If facet pressure is high at this CRT index, we warp the lattice
        pressure_warp = torch.sigmoid(self.facet_pressure[alpha % len(self.crt_moduli)])
        effective_delta = delta * (1.0 + 0.5 * pressure_warp) # Warp expands quantization bins
        
        # 3. Quantize: Q(x) = round(x / Δ) * Δ
        quantized = torch.round(x / effective_delta) * effective_delta
        
        # 4. State Transition Logic (Simulated)
        # Calculate transition energy E = ||x - Q(x)||^2
        energy = torch.norm(x - quantized, dim=-1)
        
        mean_energy = energy.mean().item()
        
        new_level = level
        new_alpha = alpha
        
        # If approximation is bad (high energy), dive deeper (increase level)
        if mean_energy > 0.1 * delta and level < self.max_depth:
            new_level += 1
        # If approximation is perfect, maybe surface (decrease level) or switch CRT index
        elif mean_energy < 0.01 * delta and level > 0:
            new_level -= 1
            
        # Cyclic CRT index transition based on x magnitude
        new_alpha = (alpha + int(x.mean().item() * 10)) % len(self.crt_moduli)
        
        return quantized, new_alpha, new_level

    def get_diagnostics(self) -> Dict:
        return {
            "max_depth": self.max_depth,
            "current_pressure_mean": self.facet_pressure.mean().item()
        }