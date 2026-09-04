import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus

class PomniUncertaintyPredictor(nn.Module):
    """
    Pomni: Uncertainty-Minimization (Interoceptive Timescale).
    
    Reads gyroid_entropy and computes a surrogate Free-Energy surprise.
    Broadcasts Noradrenaline to signal systemic distress to downstream modules.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # A simple linear projection to estimate KL divergence / Free Energy bounds
        self.surprise_estimator = nn.Linear(1, 1)
        
    def forward(self, state: torch.Tensor, gyroid_entropy: float, bus: NeuromodulatoryBus) -> torch.Tensor:
        """
        Calculates surprise and updates the neuromodulatory bus.
        """
        entropy_tensor = torch.tensor([gyroid_entropy], dtype=torch.float32, device=state.device)
        surprise = torch.sigmoid(self.surprise_estimator(entropy_tensor)).item()
        
        # Broadcast Noradrenaline based on surprise (distress)
        bus.broadcast('noradrenaline', surprise)
        
        # Pomni doesn't explicitly mutate the state, she just floods the bus with panic
        return state
