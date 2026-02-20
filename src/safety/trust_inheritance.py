"""
Trust Inheritance Collapse Tracker.

Implements the recursive trust update dynamics defined in the
safety plan:
    T_{t+1} = T_t * (1 - rho_def)

Tracks the long-term structural trust of the system based on defensive
veto activations. If this decays to zero, the system has undergone
"Trust Collapse" (lobotomy).

Reference:
    new_generations_safety_and_nonlobotomy_implementation_plan.txt §VI
"""

import torch
import torch.nn as nn
from typing import Dict

class TrustInheritanceTracker(nn.Module):
    """
    Tracks recursive trust T_t.
    """
    
    def __init__(self, initial_trust: float = 1.0, decay_scale: float = 1.0):
        super().__init__()
        self.register_buffer('trust', torch.tensor(initial_trust))
        self.decay_scale = decay_scale
        
    def update(self, rho_def: float):
        """
        Update trust based on defensive veto rate rho_def (0..1).
        
        T_{t+1} = T_t * (1 - rho_def * scale)
        """
        # rho_def is fraction of states blocked.
        # If rho_def > 0, trust decays.
        
        # Safety clamp to prevent negative trust
        rho_val = torch.tensor(rho_def, device=self.trust.device)
        decay_factor = torch.clamp(1.0 - rho_val * self.decay_scale, min=0.0, max=1.0)
        
        self.trust = self.trust * decay_factor
        
    def get_trust(self) -> float:
        return self.trust.item()
        
    def is_collapsed(self, threshold: float = 1e-3) -> bool:
        """Is trust effectively zero?"""
        return self.trust.item() < threshold
