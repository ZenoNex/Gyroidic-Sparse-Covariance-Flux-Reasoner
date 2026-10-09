import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.core.honest_jitter import harvest_honest_jitter

class PomniUncertaintyPredictor(nn.Module):
    """
    Pomni: Uncertainty Minimization, Reluctant Resilience & Bridge-Building (Interoceptive Timescale).
    
    Bioplausible & Structural Mechanics:
    1. Noradrenergic Salience & Free-Energy Surprise: Reads system entropy to broadcast
       noradrenaline, alerting downstream modules to environmental dislocation.
    2. The Foil to Absurd Nihilism: Pomni seeks purpose through connection and bridge-building,
       refusing the nihilistic claim that 'nothing matters'.
    3. The Anti-Enabling Boundary: In Callie's Corner's critique, endless unmotivated grace
       flattens Pomni from a complex protagonist into a passive enabler for Jax's angst.
       Bioplausibly, absorbing unreciprocated entropy without structural boundaries causes
       rank collapse (dimensional loss). Pomni exerts homeostatic relational friction
       whenever unilateral grace threatens her own agency.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.surprise_estimator = nn.Linear(1, 1)
        self.bridge_layer = nn.Linear(state_dim, state_dim)
        
        with torch.no_grad():
            jitter_weight = harvest_honest_jitter((state_dim, state_dim), scaled=True) * 0.05
            self.bridge_layer.weight.copy_(jitter_weight)
            
    def forward(
        self, 
        state: torch.Tensor, 
        gyroid_entropy: float, 
        bus: NeuromodulatoryBus,
        parasitic_drag: float = 0.0
    ) -> torch.Tensor:
        """
        Calculates surprise, broadcasts noradrenaline, and applies reluctant resilience
        while defending against enabler rank collapse.
        """
        entropy_tensor = torch.tensor([gyroid_entropy], dtype=torch.float32, device=state.device)
        surprise = torch.sigmoid(self.surprise_estimator(entropy_tensor)).item()
        
        # Broadcast Noradrenaline based on surprise (distress/salience)
        bus.broadcast('noradrenaline', surprise)
        
        # Bridge-Building: Active search for meaning under disorientation
        bridge_force = torch.tanh(self.bridge_layer(state))
        
        # Enabling Check:
        # If external parasitic drag is high (Jax exploiting grace), Pomni's bridge
        # shifts from passive compliance to active relational friction (boundary resistance)
        if parasitic_drag > 0.5:
            # Relational friction: refuse to let the state be flattened into a doormat
            friction_factor = 0.15 * parasitic_drag
            state = state + bridge_force * (1.0 - friction_factor) - (state.mean(dim=-1, keepdim=True) * 0.05)
        else:
            # Resilient bridge-building stabilizes the state under high surprise
            state = state + bridge_force * (0.1 * surprise)
            
        return state

