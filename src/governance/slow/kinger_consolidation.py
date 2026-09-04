import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus

class KingerConsolidation(nn.Module):
    """
    Kinger: Sleep-Dependent Consolidation (Slow Timescale 1-24h).
    
    Hippocampal Acetylcholine replay. Activated by low environmental_luminosity 
    (Dark Lucidity). Tied to structural plasticity.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # Memory buffer for replay
        self.memory_buffer = []
        self.max_buffer_size = 100
        
    def forward(self, state: torch.Tensor, bus: NeuromodulatoryBus, luminosity: float) -> tuple[torch.Tensor, bool]:
        """
        Returns the (possibly consolidated) state and a boolean indicating 
        if consolidation is actively occurring.
        """
        # Store current state in memory buffer
        if len(self.memory_buffer) >= self.max_buffer_size:
            self.memory_buffer.pop(0)
        self.memory_buffer.append(state.detach().clone())
        
        # If luminosity is low (Dark Lucidity), trigger consolidation
        consolidating = luminosity < 0.3
        
        if consolidating:
            # Replay and average memories (simplified placeholder for actual replay)
            if self.memory_buffer:
                replay_tensor = torch.stack(self.memory_buffer).mean(dim=0)
                # Blend current state with replay
                consolidated_state = state * 0.8 + replay_tensor * 0.2
                
                # Broadcast Acetylcholine during consolidation
                bus.broadcast('acetylcholine', 0.8)
                return consolidated_state, True
        else:
            # Baseline Acetylcholine
            bus.broadcast('acetylcholine', 0.2)
            
        return state, False
