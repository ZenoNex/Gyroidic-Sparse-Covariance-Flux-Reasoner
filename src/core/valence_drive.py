"""
Valence & Hunger Drive.

Implements the valency functional that measures the 'need' or 'hunger' 
of the manifold based on the negempirical gap.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ValenceFunctional(nn.Module):
    """
    Valence: The drive to resolve structural dissonance.
    
    Measures the gap between current structural pressure and a 
    historical 'Satisfaction' baseline (Saturated Trust).
    """
    
    def __init__(
        self,
        decay: float = 0.99,
        hunger_scale: float = 1.0,
        device: str = None
    ):
        super().__init__()
        self.decay = decay
        self.hunger_scale = hunger_scale
        self.device = device
        
        # Historical Satisfaction baseline 
        self.register_buffer('satisfaction', torch.tensor(0.0, device=device))

    def forward(self, current_pressure: torch.Tensor) -> torch.Tensor:
        """
        Computes the Training Valence (Hunger).
        V = hunger_scale * max(0, current_pressure - satisfaction)
        """
        # Update baseline (asymptotic satisfaction)
        self.satisfaction.mul_(self.decay).add_((1.0 - self.decay) * current_pressure.mean().detach())
        
        # Hunger is the positive dissonance gap
        hunger = torch.clamp(current_pressure - self.satisfaction, min=0.0)
        
        return hunger * self.hunger_scale

    def get_metrics(self) -> Dict[str, float]:
        return {
            'asymptotic_satisfaction': self.satisfaction.item(),
            'current_hunger_drive': self.hunger_scale * self.satisfaction.item() # Approximation
        }

