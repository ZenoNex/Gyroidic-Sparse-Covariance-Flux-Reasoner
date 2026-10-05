import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.core.honest_jitter import harvest_honest_jitter

class JaxShell(nn.Module):
    r"""
    Jax: Cynical Shell (Fast Timescale).
    
    Acts as an active defensive shell protecting internal high-entropy states.
    External pressure or direct probing INCREASES shell rigidity (cynical posturing)
    rather than breaking it. The shell only yields when surrounded by a warm 
    gravity well of community support (\zeta_{community}).
    """
    def __init__(self, state_dim: int, threshold_warmth: float = 0.6, critical_limit: float = 0.8):
        super().__init__()
        self.state_dim = state_dim
        self.shell_layer = nn.Linear(state_dim, state_dim)
        self.threshold_warmth = threshold_warmth
        self.critical_limit = critical_limit
        
        # Initialize cynical mask with honest jitter instead of pseudo-randomness
        with torch.no_grad():
            jitter_weight = harvest_honest_jitter((state_dim, state_dim), scaled=True) * 0.1
            self.shell_layer.weight.copy_(jitter_weight)
            
    def forward(
        self, 
        state: torch.Tensor, 
        surrounding_pas_h: torch.Tensor, 
        batch_tensors: torch.Tensor,
        internal_entropy: float,
        external_pressure: float,
        bus: NeuromodulatoryBus
    ) -> tuple[torch.Tensor, float]:
        """
        Calculates avoidant shell cracking based on community zeta.
        Returns the filtered state and the calculated shell rigidity.
        """
        # Calculate community_support_factor (\zeta_{community})
        pas_mean = torch.mean(surrounding_pas_h)
        batch_std = torch.std(batch_tensors) if batch_tensors.shape[0] > 1 else torch.tensor(0.0, device=state.device)
        zeta_community = pas_mean * (1.0 - batch_std)
        
        # Internal shame metric: how far the internal entropy is from the 'expected' stable self
        internal_shame = max(0.0, internal_entropy - 0.5)
        
        # External pressure and internal shame both INCREASE shell rigidity
        # (The abrasive mask is an overcompensation to hide the vulnerable identity)
        shell_rigidity = 1.0 + external_pressure + (internal_shame * 2.0)
        
        if zeta_community > self.threshold_warmth and internal_shame < self.critical_limit:
            # allow_safe_abstraction_crack()
            # The community is warm and internal shame is manageable: the shell yields
            filtered_state = state + harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 0.1
            shell_rigidity = 0.1
            bus.broadcast('serotonin', 0.8) # Feeling safe / accepted
        else:
            # enforce_cynical_mask_deflection()
            # Apply cynical, abrasive mask to hide identity vulnerability
            deflection = torch.tanh(self.shell_layer(state))
            filtered_state = state + deflection * shell_rigidity
            bus.broadcast('serotonin', 0.1) # Anxious/Cynical/Overcompensating
            
        return filtered_state, shell_rigidity
