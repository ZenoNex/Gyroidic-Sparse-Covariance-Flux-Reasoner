import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus

class RagathaBonding(nn.Module):
    """
    Ragatha: Caregiving/Affiliation (Fast Timescale 1-10s).
    
    Oxytocinergic caregiving driven by Pomni's Noradrenaline distress signal.
    Applies a dissociative mask buffer to the gradient if distress is too high.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.care_layer = nn.Linear(state_dim, state_dim)
        
    def forward(self, state: torch.Tensor, bus: NeuromodulatoryBus) -> torch.Tensor:
        """
        Calculates oxytocinergic bonding and dissociation mask.
        """
        noradrenaline = bus.read('noradrenaline')
        
        # If distress is high, Ragatha produces Oxytocin to compensate
        oxytocin_release = min(1.0, noradrenaline * 1.2)
        bus.broadcast('oxytocin', oxytocin_release)
        
        # Apply dissociation mask if distress is overwhelming (>0.7)
        dissociation_mask = 1.0
        if noradrenaline > 0.7:
            dissociation_mask = 0.5  # Dampen the signal
            
        # Care layer attempts to smooth the state
        smoothed_state = self.care_layer(state)
        
        # Blend based on oxytocin and dissociation
        final_state = state * (1.0 - oxytocin_release * 0.5) + smoothed_state * (oxytocin_release * 0.5)
        return final_state * dissociation_mask
