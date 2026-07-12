"""
Discrete Hyper-Ring Circulation.

The ring is an ordered cycle of constraint states, not a smooth manifold.
Continuous formulations are fiction here.

_H    (C_i), C_i

Author: Implementation from Structural Design Decisions
Created: January 2026
"""
import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional


class DiscreteHyperRingCirculation(nn.Module):
    """
    Compute discrete line integral over constraint cycle.
    
    _H    (C_i), C_i
    
    Adaptive Resolution: Increase ONLY when phase slippage (non-zero circulation) 
    or soliton nucleation is suspected. Fixed high-res is wasteful.
    """
    
    def __init__(
        self,
        base_resolution: int = 8,
        max_resolution: int = 64,
        slippage_threshold: float = 0.1
    ):
        super().__init__()
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.slippage_threshold = slippage_threshold
        
        # Track circulation history for slippage detection
        self.register_buffer('prev_circulation', torch.tensor(0.0))
        self.register_buffer('expected_circulation', torch.tensor(0.0))
    
    def compute_circulation(
        self,
        constraint_cycle: List[torch.Tensor],
        functional: Callable[[torch.Tensor], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Discrete line integral over constraint cycle.
        """
        n = len(constraint_cycle)
        if n < 2:
            return {
                'circulation': torch.tensor(0.0),
                'slippage': torch.tensor(0.0),
                'needs_refinement': torch.tensor(False)
            }
        
        device = constraint_cycle[0].device
        total = torch.tensor(0.0, device=device)
        
        for i in range(n):
            C_i = constraint_cycle[i]
            C_next = constraint_cycle[(i + 1) % n]
            
            # (C_i) - The force field at point i
            phi_i = functional(C_i)
            
            # C_i = C_{i+1} - C_i - The displacement
            delta_C = C_next - C_i
            
            # Inner product: Force . Displacement
            if phi_i.shape == delta_C.shape:
                contribution = (phi_i * delta_C).sum()
            else:
                contribution = phi_i.flatten().sum() * delta_C.norm()
            
            total = total + contribution
        
        # Phase slippage detection: |Observed - Expected|
        slippage = torch.abs(total - self.expected_circulation)
        needs_refinement = slippage > self.slippage_threshold
        
        self.prev_circulation = total.detach()
        
        return {
            'circulation': total,
            'slippage': slippage,
            'needs_refinement': needs_refinement,
            'resolution': torch.tensor(n)
        }
    
    def refine_cycle(
        self,
        constraint_cycle: List[torch.Tensor],
        target_resolution: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Increase resolution via midpoint interpolation."""
        if target_resolution is None:
            target_resolution = min(len(constraint_cycle) * 2, self.max_resolution)
        
        if len(constraint_cycle) >= target_resolution:
            return constraint_cycle
        
        refined = []
        n = len(constraint_cycle)
        for i in range(n):
            C_i = constraint_cycle[i]
            C_next = constraint_cycle[(i + 1) % n]
            refined.append(C_i)
            refined.append((C_i + C_next) / 2.0)
        
        return refined[:target_resolution]

    def forward(
        self,
        constraint_cycle: List[torch.Tensor],
        functional: Callable[[torch.Tensor], torch.Tensor],
        auto_refine: bool = True
    ) -> Dict[str, torch.Tensor]:
        result = self.compute_circulation(constraint_cycle, functional)
        
        if auto_refine and result['needs_refinement']:
            refined_cycle = self.refine_cycle(constraint_cycle)
            result = self.compute_circulation(refined_cycle, functional)
            result['was_refined'] = torch.tensor(True)
        else:
            result['was_refined'] = torch.tensor(False)
        
        return result


class RecurrentHyperRingConnectivity(nn.Module):
    """
    Speculative Neural Connectivity Hyper-Ring.
    
    Acts like a non-Euclidean, recurrent network supporting dynamic 
    amortization over local polytopes (Text Gardens).
    
    [ANTI-LOBOTOMY REWRITE]:
    Removed ML proxies (nn.Linear) and generic sigmoidal gating.
    Integrates PolynomialCoprimeConfig for state-to-polytope projection and
    OmipedialDeflagrator for ley-line potential topological jumps.
    """
    def __init__(self, num_polytopes: int, state_dim: int = 64, coupling_init: float = 0.1):
        super().__init__()
        self.num_polytopes = num_polytopes
        self.state_dim = state_dim
        
        # Adaptive coupling matrix (base)
        self.omega = nn.Parameter(torch.ones(num_polytopes, num_polytopes) * coupling_init)
        self.gamma = nn.Parameter(torch.tensor(0.5)) # Dark matter influence scale
        
        # System 1: Polynomial Co-Prime Config replaces nn.Linear ML proxies
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        self.poly_config = PolynomialCoprimeConfig(k=num_polytopes, degree=4, basis_type='chebyshev', learnable=True)
        
        # System 2: Deflagration Scout handles topological jumps instead of sigmoid gates
        from src.core.deflagration_scout import OmipedialDeflagrator
        self.deflagrator = OmipedialDeflagrator(dim=state_dim, threshold_jump=0.7)

    def _project_to_polytope(self, state: torch.Tensor) -> torch.Tensor:
        """Projects high-dim state to polytope functional space via Coprime basis."""
        # Mean pooling to single scalar per batch element for polynomial evaluation
        x_norm = state.mean(dim=-1, keepdim=True)
        # Evaluate polynomial basis to get K polytope functionals
        polytope_vals = self.poly_config.evaluate(x_norm)
        return polytope_vals.squeeze(1) if polytope_vals.dim() == 3 else polytope_vals

    def forward(
        self, 
        polytope_functionals: torch.Tensor, 
        dark_matter: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            polytope_functionals: [batch, num_polytopes] f_j scores
            dark_matter: [batch, num_polytopes] D_dark speculative traces
        """
        batch_size = polytope_functionals.size(0)
        
        # 1. Functional difference matrix [batch, num_p, num_p]
        f_diff = polytope_functionals.unsqueeze(2) - polytope_functionals.unsqueeze(1)
        
        # 2. Add Dark Matter influence if available
        if dark_matter is not None:
             f_diff = f_diff + self.gamma * dark_matter.unsqueeze(1)
             
        # 3. Base Connectivity Matrix H_ij = omega_ij * tanh(f_diff)
        # Replacing non-topological sigmoid with symmetric tanh
        base_connectivity = self.omega.unsqueeze(0) * torch.tanh(f_diff)
        
        # 4. Omipedial Deflagration (Ley-line topological jumps)
        # Ley potential proxy from functional variance
        ley_potential = torch.abs(f_diff).mean(dim=-1)
        jumps = self.deflagrator.omipedial_jump(ley_potential)
        
        # Apply jumps: open topological shortcuts where ley potential > threshold
        connectivity = base_connectivity + jumps.unsqueeze(2) * 1.5
        
        return connectivity

    def flow_step(self, polytope_states: torch.Tensor, connectivity: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Computes the neural-like flow across polytopes modulated by ManifoldClock's dt:
        S_{t+dt} = S_t + (dt) * sum_j H_ij S_j
        """
        # states: [batch, num_p, hidden_dim]
        # connectivity: [batch, num_p, num_p]
        flow_delta = torch.bmm(connectivity, polytope_states)
        return polytope_states + (flow_delta * dt)
