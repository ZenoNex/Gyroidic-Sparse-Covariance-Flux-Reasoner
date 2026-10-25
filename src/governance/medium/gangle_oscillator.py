import torch
import torch.nn as nn
from typing import Optional
from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.core.hardware_monitor import ComputeAffordanceType, TypedComputeAffordance

class GangleOscillator(nn.Module):
    """
    Gangle: Mood Limit-Cycle Oscillator (Medium Timescale 1-60m).
    
    Dopaminergic limit-cycle. Bifurcation mask (comedy/tragedy) regulates 
    the step-size factor for downstream gradient updates, dynamically modulated
    by batch compute affordance and valence hunger.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # A simple phase oscillator for the mood cycle
        self.phase = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.frequency = nn.Parameter(torch.tensor([0.05]), requires_grad=False)
        
    def forward(
        self, 
        state: torch.Tensor, 
        bus: NeuromodulatoryBus, 
        dt: float = 1.0,
        batch_affordance: Optional[TypedComputeAffordance] = None,
        valence_hunger: float = 0.0
    ) -> tuple[torch.Tensor, float]:
        """
        Updates the oscillator phase and calculates step size.
        Returns the new state and the calculated step_factor.
        """
        # Read from bus
        dopamine = bus.read('dopamine')
        
        # Advance phase: hunger accelerates manic/restless phase transitions
        hunger_phase_mult = 1.0 + 0.5 * min(2.0, valence_hunger)
        self.phase.data += self.frequency.data * dt * (1.0 + dopamine) * hunger_phase_mult
        self.phase.data %= (2 * torch.pi)
        
        # Determine mood (comedy = positive, tragedy = negative)
        raw_mood = torch.sin(self.phase).item()
        # High unquenched hunger biases mood toward tragedy
        mood_value = raw_mood - (0.3 * min(1.0, valence_hunger))
        mood_value = max(-1.0, min(1.0, mood_value))
        
        # Dopamine release peaks during comedy
        bus.broadcast('dopamine', max(0.0, mood_value))
        
        # Step factor is higher during comedy, lower during tragedy
        step_factor = 1.0 + mood_value * 0.5
        
        # Batch compute affordance throttling: downscale step size if batch budget is depleted
        if batch_affordance is not None:
            if not batch_affordance.is_admissible or batch_affordance.gauge < 0.4:
                step_factor *= max(0.2, batch_affordance.gauge)
                
        # The state is slightly perturbed by the mood
        mood_tensor = torch.tensor([mood_value], device=state.device).expand_as(state) * 0.1
        
        return state + mood_tensor, step_factor

