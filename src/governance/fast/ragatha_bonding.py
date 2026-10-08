import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.core.honest_jitter import harvest_honest_jitter

class RagathaBonding(nn.Module):
    """
    Ragatha: Caregiving, Affiliation, and Suppressed Grief (Fast Timescale 1-10s).
    
    Bioplausible & Structural Mechanics:
    1. The Caregiver Trap: Reflexive oxytocinergic caretaking and people-pleasing
       used as a defense mechanism against fears of abandonment and rejection.
    2. Suppressed Grief Dynamics: In Callie's Corner's critique, Ragatha's grief over
       multiple abstracted friends (Kaufmo, Queenie, Ribbit) is repeatedly suppressed
       to coddle others. This unilateral smoothing causes dissociative boundary thinning.
    3. Homeostatic Boundary: Continuous over-smoothing without reciprocal boundaries
       strains the network's metabolic capacity (caregiver burnout / excitotoxic exhaustion),
       requiring authentic communal grief acknowledgment to restore balance.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.care_layer = nn.Linear(state_dim, state_dim)
        
        # Suppressed Grief Vector: Memory trace of unexpressed loss
        self.suppressed_grief = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True) * 0.15)
        self.grief_counter = 0.0
        
    def forward(self, state: torch.Tensor, bus: NeuromodulatoryBus, boundary_friction: float = 0.0) -> torch.Tensor:
        """
        Calculates oxytocinergic bonding, tracking caregiver burnout and suppressed grief.
        """
        noradrenaline = bus.read('noradrenaline')
        serotonin = bus.read('serotonin')
        
        # If external distress is high, Ragatha reflexively produces Oxytocin
        oxytocin_release = min(1.0, noradrenaline * 1.2)
        bus.broadcast('oxytocin', oxytocin_release)
        
        # Suppressed Grief Accumulation:
        # When distress is high but serotonin (authenticity/safety) is low,
        # Ragatha swallows her grief to maintain the positive facade.
        if noradrenaline > 0.5 and serotonin < 0.4:
            self.grief_counter = min(2.0, self.grief_counter + 0.05)
        elif serotonin > 0.6:
            # Genuine safety allows grief to be safely processed and discharged
            self.grief_counter = max(0.0, self.grief_counter - 0.05)
            
        # Dissociative Mask:
        # High distress or high accumulated grief forces dissociative damping
        dissociation_mask = 1.0
        if noradrenaline > 0.7 or self.grief_counter > 1.0:
            dissociation_mask = max(0.4, 0.9 - 0.2 * self.grief_counter)
            
        # Care layer attempts to smooth the state
        smoothed_state = self.care_layer(state)
        
        # Blend based on oxytocin and dissociation
        blend_factor = oxytocin_release * 0.5
        final_state = state * (1.0 - blend_factor) + smoothed_state * blend_factor
        
        # If suppressed grief is dangerously high, it leaks as topological strain
        if self.grief_counter > 0.8:
            grief_leak = self.suppressed_grief.unsqueeze(0) if final_state.dim() > 1 else self.suppressed_grief
            final_state = final_state + 0.05 * grief_leak * self.grief_counter
            
        return final_state * dissociation_mask

