import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus

class JaxShell(nn.Module):
    """
    Jax: Social-Approach/Avoidance Conflict (Fast Timescale 1-10s).
    
    Avoidant shell that uses Serotonin/GABA toggle based on Community Zeta.
    The Panic flag ties into the CALM predictor's meditation state.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.shell_layer = nn.Linear(state_dim, state_dim)
        self.panic_detector = nn.Linear(state_dim, 1)
        
    def forward(self, state: torch.Tensor, bus: NeuromodulatoryBus) -> tuple[torch.Tensor, bool]:
        """
        Calculates avoidant shell cracking and panic state.
        """
        # Read neurobus for upstream context
        gaba = bus.read('gaba')
        serotonin = bus.read('serotonin')
        
        # Shell strength depends on the balance of serotonin (approach) vs gaba (inhibition)
        shell_strength = torch.clamp(torch.tensor(gaba - serotonin), 0.0, 1.0)
        
        # Determine panic state
        panic_score = torch.sigmoid(self.panic_detector(state)).item()
        panic = panic_score > 0.8
        
        if panic:
            # Broadcast serotonin depletion
            bus.broadcast('serotonin', max(0.0, serotonin - 0.2))
        else:
            bus.broadcast('serotonin', min(1.0, serotonin + 0.1))
            
        # Apply shell filtering to state
        filtered_state = state + torch.tanh(self.shell_layer(state)) * shell_strength
        return filtered_state, panic
