"""
Adversarial Stress Tester (formerly Collapse Path Poisoner).

Role: Generates synthetic "rupture" events to verify the Speculative Homology Engine.
Instead of poisoning training, we purposefully inject "Anti-Aligned" constraints
to check if the system correctly detects the rupture and engages System 2.

Techniques:
1. Constraint Anti-Alignment (Synthetic Rupture)
2. Cycle Debt (Topological Stress)
3. Synthetic Inconsistency

Author: Implementation from Structural Design Decisions
Created: January 2026
Refactored: January 2026 (Anti-Lobotomy Integration)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import math


class CollapsePathPoisoner(nn.Module):
    """
    Stress-tests the topology by injecting ruptures.
    
    Used to verify:
    1. SpeculativeHomologyEngine's PAS_h check (should fail on rupture).
    2. DyadicTransferMap's robustness to perturbation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_constraints: int = 4,
        cycle_history_size: int = 100,
        debt_threshold: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_constraints = num_constraints
        self.debt_threshold = debt_threshold
        
        # Cycle debt tracking
        self.register_buffer('homotopy_history', torch.zeros(cycle_history_size, hidden_dim))
        self.register_buffer('homotopy_ptr', torch.tensor(0, dtype=torch.long))
        self.cycle_history_size = cycle_history_size
        
        # Orthogonal constraint generator
        self.constraint_generator = nn.Linear(hidden_dim, hidden_dim * num_constraints)
    
    def generate_synthetic_rupture(
        self,
        current_manifold: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates a "Rupture" perturbation: a set of constraints strictly orthogonal
        to the current manifold. Applying this should create a hole (betti number change).
        """
        # Flatten if needed
        if current_manifold.dim() > 2:
            current_manifold = current_manifold.mean(dim=1)
        
        batch_size = current_manifold.shape[0]
        
        # Generate raw constraints
        raw = self.constraint_generator(current_manifold)
        constraints = raw.view(batch_size, self.num_constraints, self.hidden_dim)
        
        # Orthogonalize via Gram-Schmidt
        orthogonal = []
        for i in range(self.num_constraints):
            c = constraints[:, i, :]
            for prev in orthogonal:
                proj = (c * prev).sum(dim=-1, keepdim=True)
                norm_sq = (prev.norm(dim=-1, keepdim=True) ** 2) + 1e-8
                c = c - (proj / norm_sq) * prev
            c = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
            orthogonal.append(c)
        
        perturbation = torch.stack(orthogonal, dim=1).sum(dim=1) # [batch, hidden_dim]
        return perturbation
    
    def compute_cycle_debt(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Calculates topological "boredom" or repetition (Cycle Debt).
        High debt = system is looping.
        """
        # Flatten state
        if current_state.dim() > 1:
            state_flat = current_state.flatten()[:self.hidden_dim]
        else:
            state_flat = current_state[:self.hidden_dim]
        
        filled = min(self.homotopy_ptr.item(), self.cycle_history_size)
        
        if filled < 2:
            self._update_homotopy(state_flat)
            return torch.tensor(0.0, device=current_state.device)
        
        history = self.homotopy_history[:filled]
        
        # Normalize for cosine similarity
        current_norm = state_flat / (state_flat.norm() + 1e-8)
        history_norm = history / (history.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Similarities
        similarities = torch.mv(history_norm, current_norm)
        
        # Count near-matches (homotopy class repeats)
        repeats = (similarities > 0.9).float().sum()
        
        # Debt grows with repeats
        debt = repeats / max(filled, 1)
        
        self._update_homotopy(state_flat)
        return debt
    
    def _update_homotopy(self, state: torch.Tensor):
        """Update homotopy history."""
        ptr = self.homotopy_ptr.item()
        # Ensure correct size
        if state.shape[0] >= self.hidden_dim:
            self.homotopy_history[ptr % self.cycle_history_size] = state[:self.hidden_dim].detach()
        else:
            padded = torch.zeros(self.hidden_dim, device=state.device)
            padded[:state.shape[0]] = state.detach()
            self.homotopy_history[ptr % self.cycle_history_size] = padded
        self.homotopy_ptr += 1
    
    def forward(
        self,
        current_manifold: torch.Tensor,
        trigger_rupture: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Stress Test Step.
        """
        cycle_debt = self.compute_cycle_debt(current_manifold.mean(dim=(0,1)) if current_manifold.dim()>2 else current_manifold.mean(dim=0))
        
        results = {
            'cycle_debt': cycle_debt,
            'debt_warning': cycle_debt > self.debt_threshold
        }
        
    def compute_kelly_calm_allocation(self, cycle_debt: torch.Tensor, kappa: torch.Tensor, max_mass: float = 1.0) -> float:
        """
        Fractional Kelly Bets & CALM Veto Superposition.
        Prevents global traffic and t_RFC stalls.
        CALM Veto blends Mohr-Coulomb yield stress (geometric) with cycle debt (empirical).
        """
        k_val = kappa.item() if isinstance(kappa, torch.Tensor) else float(kappa)
        c_val = cycle_debt.item() if isinstance(cycle_debt, torch.Tensor) else float(cycle_debt)
        
        # Mohr-Coulomb Yield Stress approximation based on Non-Commutativity (kappa)
        yield_stress = math.tanh(k_val * 2.0)
        
        # Superposition of vetos
        superposition = (0.6 * c_val) + (0.4 * yield_stress)
        p = max(0.01, min(0.99, 1.0 - superposition))
        
        # Kelly Criterion: f = 2p - 1 (for 1:1 odds)
        kelly_fraction = max(0.0, 2.0 * p - 1.0)
        
        # Half-Kelly for stability
        fractional_kelly = kelly_fraction * 0.5
        
        return float(max_mass * fractional_kelly)
        
        if trigger_rupture:
            results['run_rupture'] = self.generate_synthetic_rupture(current_manifold)
        
        return results

    def reset(self):
        """Reset tracking state."""
        self.homotopy_history.zero_()
        self.homotopy_ptr.zero_()

# Backward compatibility alias
AdversarialStressTester = CollapsePathPoisoner
