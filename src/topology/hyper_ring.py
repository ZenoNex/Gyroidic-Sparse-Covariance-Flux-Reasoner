"""
Discrete Hyper-Ring Circulation.

The ring is an ordered cycle of constraint states, not a smooth manifold.
Continuous formulations are fiction here.

∮_H Φ ≈ Σ ⟨Φ(C_i), ΔC_i⟩

Author: Implementation from Structural Design Decisions
Created: January 2026
"""
import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional


class DiscreteHyperRingCirculation(nn.Module):
    """
    Compute discrete line integral over constraint cycle.
    
    ∮_H Φ ≈ Σ ⟨Φ(C_i), ΔC_i⟩
    
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
            
            # Φ(C_i) - The force field at point i
            phi_i = functional(C_i)
            
            # ΔC_i = C_{i+1} - C_i - The displacement
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
    
    H_ij = omega_ij * sigma( f_i - f_j + gamma D_dark )
    """
    def __init__(self, num_polytopes: int, state_dim: int = 64, coupling_init: float = 0.1):
        super().__init__()
        self.num_polytopes = num_polytopes
        self.state_dim = state_dim
        
        # Adaptive coupling matrix ω
        self.omega = nn.Parameter(torch.ones(num_polytopes, num_polytopes) * coupling_init)
        self.gamma = nn.Parameter(torch.tensor(0.5)) # Dark matter influence scale
        
        # Learnable Projection: State Dim -> Polytope Functionals
        # This bridges high-dimensional state space to polytope-level features
        self.project_to_polytope = nn.Linear(state_dim, num_polytopes)
        
        # Learnable Projection: Polytope Functionals -> State Dim
        # This bridges back from polytope-level output to state space
        self.project_from_polytope = nn.Linear(num_polytopes, state_dim)

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
        # 1. Functional difference matrix [batch, num_p, num_p]
        # (f_i - f_j)
        f_diff = polytope_functionals.unsqueeze(2) - polytope_functionals.unsqueeze(1)
        
        # 2. Add Dark Matter influence if available
        if dark_matter is not None:
             # Dark matter influence from the destination polytope j
             # gamma * D_dark_j
             f_diff = f_diff + self.gamma * dark_matter.unsqueeze(1)
             
        # 3. Connectivity Matrix H_ij = omega_ij * sigmoid(f_diff)
        # Sgn(f_diff) allows flow toward "higher resonance" or lower "hole energy"
        connectivity = self.omega.unsqueeze(0) * torch.sigmoid(f_diff)
        
        return connectivity

    def flow_step(self, polytope_states: torch.Tensor, connectivity: torch.Tensor) -> torch.Tensor:
        """
        Computes the neural-like flow across polytopes:
        dS_i/dt = sum_j H_ij S_j
        """
        # states: [batch, num_p, hidden_dim]
        # connectivity: [batch, num_p, num_p]
        flow = torch.bmm(connectivity, polytope_states)
        return flow
