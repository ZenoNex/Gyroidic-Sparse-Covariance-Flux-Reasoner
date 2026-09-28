import torch
import torch.nn as nn
from src.core.honest_jitter import harvest_honest_jitter
from src.core.neuromodulatory_bus import NeuromodulatoryBus

class KingerConsolidation(nn.Module):
    """
    Kinger: Low Luminosity Coherence Bridge (Slow Timescale).
    
    Implements the Ombre Effect. When environmental rendering pressure drops 
    (low luminosity/darkness), Kinger's paranoid dissociation subsides, triggering 
    admin-level lucidity. It relaxes quantization boundaries and bridges fragmented 
    polynomial spaces.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # Grant's admin coefficients for polynomial bridging
        self.grant_admin_coefficients = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True))
        self.memory_buffer = []
        self.max_buffer_size = 50
        
    def relax_quantization_grid(self, state: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Relaxes the saturated quantization boundaries."""
        return state * scale_factor
        
    def bridge_polynomial_spaces(self, state: torch.Tensor, admin_coeffs: torch.Tensor) -> torch.Tensor:
        """Bridges fragmented polynomial spaces using admin-level coefficients."""
        return state + admin_coeffs
        
    def forward(
        self, 
        state: torch.Tensor, 
        bus: NeuromodulatoryBus,
        environmental_rendering_pressure: float
    ) -> tuple[torch.Tensor, bool]:
        """
        Monitors rendering pressure. If < 0.2 (low luminosity), triggers Ombre Effect (Consolidation).
        Returns the processed state and a boolean indicating if admin lucidity is active.
        """
        # Store current state in memory buffer
        if len(self.memory_buffer) >= self.max_buffer_size:
            self.memory_buffer.pop(0)
        self.memory_buffer.append(state.detach().clone())
        
        is_lucid = environmental_rendering_pressure < 0.2
        
        if is_lucid:
            # The Ombre Effect: relax boundaries and bridge spaces
            relaxed_state = self.relax_quantization_grid(state, scale_factor=2.0)
            bridged_state = self.bridge_polynomial_spaces(relaxed_state, self.grant_admin_coefficients)
            
            # Replay and average memories
            if self.memory_buffer:
                replay_tensor = torch.stack(self.memory_buffer).mean(dim=0)
                # Blend current bridged state with replay
                consolidated_state = bridged_state * 0.8 + replay_tensor * 0.2
            else:
                consolidated_state = bridged_state
                
            # Broadcast Acetylcholine during consolidation (Admin Lucidity)
            bus.broadcast('acetylcholine', 0.8)
            return consolidated_state, True
            
        else:
            # Baseline Acetylcholine
            bus.broadcast('acetylcholine', 0.2)
            
            # In high rendering pressure, dissociation blocks coherence
            # Apply honest jitter to simulate dissociation
            dissociation = harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 0.1
            return state + dissociation, False
