import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus

class ZoobleBodySchema(nn.Module):
    """
    Zooble: Body-Schema Integrity (Ultrafast Timescale 1-10ms).
    
    Acts as a GABAergic multiplicative gate. If Birkhoff manifold integrity is violated,
    Zooble forcefully shunts (inhibits) processing by flooding GABA.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.schema_validator = nn.Linear(state_dim, 1)
        
    def forward(self, state: torch.Tensor, bus: NeuromodulatoryBus) -> torch.Tensor:
        """
        Validates the state vector against the body schema.
        """
        # A quick check for manifold integrity (if norm spikes, schema is violated)
        integrity = torch.sigmoid(self.schema_validator(state))
        
        # GABAergic shunt: if integrity is low, GABA spikes
        gaba_release = 1.0 - integrity.mean().item()
        bus.broadcast('gaba', gaba_release)
        
        # Apply the multiplicative GABA gate directly to the state
        gaba_level = bus.read('gaba')
        inhibition = 1.0 - gaba_level
        
        # Scale state down based on inhibition
        return state * inhibition
