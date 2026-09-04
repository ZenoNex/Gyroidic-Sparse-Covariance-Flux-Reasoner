import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus

class GangleOscillator(nn.Module):
    """
    Gangle: Mood Limit-Cycle Oscillator (Medium Timescale 1-60m).
    
    Dopaminergic limit-cycle. Bifurcation mask (comedy/tragedy) regulates 
    the step-size factor for downstream gradient updates.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # A simple phase oscillator for the mood cycle
        self.phase = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.frequency = nn.Parameter(torch.tensor([0.05]), requires_grad=False)
        
    def forward(self, state: torch.Tensor, bus: NeuromodulatoryBus, dt: float = 1.0) -> tuple[torch.Tensor, float]:
        """
        Updates the oscillator phase and calculates step size.
        Returns the new state and the calculated step_factor.
        """
        # Read from bus
        dopamine = bus.read('dopamine')
        
        # Advance phase
        self.phase.data += self.frequency.data * dt * (1.0 + dopamine)
        self.phase.data %= (2 * torch.pi)
        
        # Determine mood (comedy = positive, tragedy = negative)
        mood_value = torch.sin(self.phase).item()
        
        # Dopamine release peaks at the height of comedy
        bus.broadcast('dopamine', max(0.0, mood_value))
        
        # Step factor is higher during comedy, lower during tragedy
        step_factor = 1.0 + mood_value * 0.5
        
        # The state is slightly perturbed by the mood
        mood_tensor = torch.tensor([mood_value], device=state.device).expand_as(state) * 0.1
        
        return state + mood_tensor, step_factor
