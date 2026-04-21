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

    def forward(
        self, 
        current_pressure: torch.Tensor,
        mischief: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the Training Valence (Hunger).
        V = hunger_scale * (max(0, current_pressure - satisfaction) + Dissonance)
        
        Following MATHEMATICAL_DETAILS.md §15.2: 
        Hunger should be high if system is in 'Babbling' or 'Degenerate' states.
        """
        # 1. Update baseline (asymptotic satisfaction)
        # We detach pressure to keep satisfaction as a non-differentiable reference
        self.satisfaction.mul_(self.decay).add_((1.0 - self.decay) * current_pressure.mean().detach())
        
        # 2. Compute primary pressure gap (Surprise)
        surprise = torch.clamp(current_pressure - self.satisfaction, min=0.0)
        
        # 3. Inject Structural Dissonance (Entropy/Mischief)
        # If mischief is high (babbling) or entropy is high (noise), increase hunger
        dissonance = 0.0
        if mischief is not None:
            # Mischief rewards logic-breaks, but high mischief without pas_h 
            # should drive the system to find structured associations (Hunger)
            dissonance += 0.5 * mischief.mean()
            
        if entropy is not None:
            # High spectral entropy means the proposal is noise/babble
            dissonance += 0.3 * entropy.mean()

        # 4. Total Hunger
        hunger = (surprise + dissonance) * self.hunger_scale
        
        return hunger


    def get_metrics(self) -> Dict[str, float]:
        return {
            'asymptotic_satisfaction': self.satisfaction.item(),
            'current_hunger_drive': self.hunger_scale * self.satisfaction.item() # Approximation
        }

