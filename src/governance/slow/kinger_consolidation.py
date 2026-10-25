import torch
import torch.nn as nn
from typing import Optional
from src.core.honest_jitter import harvest_honest_jitter
from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.core.hardware_monitor import ComputeAffordanceType, TypedComputeAffordance

class KingerConsolidation(nn.Module):
    """
    Kinger: Low Luminosity Coherence Bridge & Thermodynamic Heat Sink (Slow Timescale).
    
    Implements the Ombre Effect. When environmental rendering pressure drops 
    (low luminosity/darkness) or when Thermal Compute Affordance is critically depleted
    (thermodynamic heat dissipation), Kinger's paranoid dissociation subsides, triggering 
    admin-level lucidity. It relaxes quantization boundaries, dissipates Carnot entropy,
    and bridges fragmented polynomial spaces.
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
        environmental_rendering_pressure: float,
        thermal_affordance: Optional[TypedComputeAffordance] = None,
        valence_hunger: float = 0.0
    ) -> tuple[torch.Tensor, bool]:
        """
        Monitors rendering pressure, thermodynamic headroom, and valence hunger.
        Triggers Ombre Effect (Consolidation) when rendering pressure is low or thermal affordance is exhausted.
        Returns the processed state and a boolean indicating if admin lucidity / consolidation is active.
        """
        # Store current state in memory buffer
        dropped_memory = None
        if len(self.memory_buffer) >= self.max_buffer_size:
            # Memory abstraction limit reached (Queenie's Abstraction parallel)
            # Kinger loses a fragment of the past and feels survivor's guilt
            dropped_memory = self.memory_buffer.pop(0)
        self.memory_buffer.append(state.detach().clone())
        
        # Thermal exhaustion forces thermodynamic cooling consolidation
        thermal_exhaustion = False
        if thermal_affordance is not None:
            thermal_exhaustion = (not thermal_affordance.is_admissible) or (thermal_affordance.gauge < 0.20)
            
        # Standard Ombre Effect requires low rendering pressure and non-starving valence
        # Extreme starvation hunger (> 0.85) creates restless insomnia unless thermal exhaustion forces sleep
        can_sleep_from_quiet = (environmental_rendering_pressure < 0.2) and (valence_hunger < 0.85)
        
        is_lucid = can_sleep_from_quiet or thermal_exhaustion
        
        if is_lucid:
            # The Ombre Effect: relax boundaries and bridge spaces
            relaxed_state = self.relax_quantization_grid(state, scale_factor=2.0)
            bridged_state = self.bridge_polynomial_spaces(relaxed_state, self.grant_admin_coefficients)
            
            # Replay and average memories
            if self.memory_buffer:
                replay_tensor = torch.stack(self.memory_buffer).mean(dim=0)
                
                # Survivor's Guilt Trauma Loop: if a memory was recently lost (abstracted),
                # Kinger obsessively biases towards the replay memory to "protect" what's left
                if dropped_memory is not None:
                    consolidated_state = bridged_state * 0.4 + replay_tensor * 0.6
                else:
                    consolidated_state = bridged_state * 0.8 + replay_tensor * 0.2
            else:
                consolidated_state = bridged_state
                
            # Broadcast Acetylcholine during consolidation (Admin Lucidity)
            # Higher release during thermal exhaustion to force metabolic rest
            ach_level = 0.95 if thermal_exhaustion else 0.80
            bus.broadcast('acetylcholine', ach_level)
            return consolidated_state, True
            
        else:
            # Baseline Acetylcholine
            bus.broadcast('acetylcholine', 0.2)
            
            # In high rendering pressure or unquenched starvation, dissociation blocks coherence
            # Apply honest jitter to simulate dissociation
            dissociation = harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 0.1
            return state + dissociation, False

