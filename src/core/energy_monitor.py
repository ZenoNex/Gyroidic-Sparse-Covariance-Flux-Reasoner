"""
Structural Energy Monitor: Formalizing Pressure as Energy.

This module implements the mapping from the Energy-Based Models (EBM) 
framework to the Project's topological manifold.

Energy E(Y, X) <=> Structural Pressure Phi
Free Energy F <=> Topological Free Energy F_topo
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class StructuralEnergyMonitor(nn.Module):
    """
    Monitors the 'Energy' (Pressure) of the structural manifold.
    
    In the EBM ontology:
    - E_correct (E_C): The pressure of the current symbolic lock.
    - E_offending (E_I): The pressure of alternative, unstable configurations.
    - F_topo: The log-partition function (Free Energy) of the manifold.
    """
    
    def __init__(
        self,
        margin: float = 0.1,
        beta_init: float = 1.0,
        device: str = None
    ):
        super().__init__()
        self.margin = margin
        self.device = device
        
        # Internal state
        self.register_buffer('E_correct', torch.tensor(0.0, device=device))
        self.register_buffer('E_offending', torch.tensor(0.0, device=device))
        self.register_buffer('beta', torch.tensor(beta_init, device=device))
        self.register_buffer('free_energy', torch.tensor(0.0, device=device))

    def update(self, current_pressure: torch.Tensor, alternative_pressures: Optional[torch.Tensor] = None):
        """
        Update energy metrics based on current manifold state.
        
        Args:
            current_pressure: Scalar or [batch] pressure of the accepted state.
            alternative_pressures: [batch, K] pressures of K alternative configurations.
        """
        # E_correct is the mean pressure of the current state
        self.E_correct.copy_(current_pressure.mean().detach())
        
        if alternative_pressures is not None:
            # E_offending is the 'most offending' incorrect answer (lowest alternative pressure)
            # or the expected energy of alternatives.
            # Following EBM tutorial (Definition 1), Y_bar is the lowest energy incorrect answer.
            min_alt = alternative_pressures.min(dim=1)[0]
            self.E_offending.copy_(min_alt.mean().detach())
            
            # Compute Topological Free Energy (F_topo)
            # F = -1/beta * log(sum(exp(-beta * E_i)))
            # We treat the current state and all alternatives as the ensemble.
            combined_energies = torch.cat([current_pressure.unsqueeze(-1), alternative_pressures], dim=1)
            self.free_energy.copy_(self.compute_free_energy(combined_energies))

    def compute_free_energy(self, energies: torch.Tensor) -> torch.Tensor:
        """
        F = -1/beta * log_sum_exp(-beta * E)
        """
        # Using logsumexp for numerical stability
        # -1/beta * log(sum(exp(-beta * E)))
        scaled_energies = -self.beta * energies
        lse = torch.logsumexp(scaled_energies, dim=1)
        f_topo = - (1.0 / (self.beta + 1e-8)) * lse
        return f_topo.mean()

    def set_temperature(self, dt_ratio: float):
        """
        Link temperature/beta to the coordinate time step ratio.
        As dt shrinks (Seriousness), beta increases (Cooling).
        """
        # beta is inverse temperature. 
        # When dt_ratio = 1.0 (base), beta = beta_init.
        # When dt_ratio -> 0, beta -> infinity.
        self.beta.copy_(torch.tensor(1.0 / (dt_ratio + 1e-8), device=self.device))

    def get_gap(self) -> float:
        """Measure the Energy Gap: E_offending - E_correct."""
        return (self.E_offending - self.E_correct).item()

    def get_metrics(self) -> Dict[str, float]:
        return {
            'E_correct': self.E_correct.item(),
            'E_offending': self.E_offending.item(),
            'energy_gap': self.get_gap(),
            'free_energy': self.free_energy.item(),
            'temperature_inv': self.beta.item()
        }

