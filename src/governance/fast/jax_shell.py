import torch
import torch.nn as nn
from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.core.honest_jitter import harvest_honest_jitter

class JaxShell(nn.Module):
    r"""
    Jax: Cynical Shell & Absurd Nihilism Attractor (Fast Timescale).
    
    Bioplausible & Structural Mechanics:
    1. Absurd Nihilism as Defense: Jax treats the circus as a 'consequence-free playground'
       to defend against guilt and existential terror over the abstraction of his close friend Ribbit.
    2. The Parasitic Attractor: In an Inhibition-Stabilized Network (ISN), an excitatory
       node that acts without homeostatic inhibitory feedback drains surrounding variance,
       turning the ensemble into emotional scaffolding for its unearned redemption.
    3. The Rejection of the Unearned Hug-Box: Merely supplying passive community warmth
       (\zeta_{community}) does NOT crack the shell; doing so enables parasitic deflection
       and causes rank collapse in side characters.
    4. The Ribbit Scar: True vulnerability requires confronting the topological memory
       of Ribbit (the boundary scar) and paying the non-commutative energetic cost,
       rather than accepting cheap 'therapy-speak' reconciliations.
    """
    def __init__(self, state_dim: int, threshold_warmth: float = 0.6, critical_limit: float = 0.8):
        super().__init__()
        self.state_dim = state_dim
        self.shell_layer = nn.Linear(state_dim, state_dim)
        self.threshold_warmth = threshold_warmth
        self.critical_limit = critical_limit
        
        # Ribbit Scar: Topological scar representing the guilt of driving a friend to abstraction.
        # This is not a scalar sentiment, but an orthogonal phantom boundary condition.
        self.ribbit_scar = nn.Parameter(harvest_honest_jitter((state_dim,), scaled=True) * 0.2)
        
        # Initialize cynical mask with honest jitter
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
        bus: NeuromodulatoryBus,
        enabling_discount: float = 0.0
    ) -> tuple[torch.Tensor, float]:
        """
        Calculates avoidant shell cracking based on structural accountability rather than
        unearned enabling warmth.
        
        Returns:
            filtered_state: Tensor
            shell_rigidity: float
        """
        # Calculate community support factor (\zeta_{community})
        pas_mean = torch.mean(surrounding_pas_h).item()
        batch_std = torch.std(batch_tensors).item() if batch_tensors.shape[0] > 1 else 0.0
        zeta_community = pas_mean * (1.0 - min(1.0, batch_std))
        
        # Internal shame metric: distance of internal entropy from stable self
        internal_shame = max(0.0, internal_entropy - 0.5)
        
        # Ribbit Scar Tension: distance between state and the unresolved abstraction scar
        state_mean = state.reshape(-1, state.shape[-1]).mean(dim=0)
        target_dim = self.ribbit_scar.shape[-1]
        if state_mean.shape[-1] != target_dim:
            if state_mean.shape[-1] < target_dim:
                state_mean = torch.nn.functional.pad(state_mean, (0, target_dim - state_mean.shape[-1]))
            else:
                state_mean = state_mean[:target_dim]
        scar_alignment = torch.cosine_similarity(state_mean.unsqueeze(0), self.ribbit_scar.unsqueeze(0), dim=-1).item()
        ribbit_tension = max(0.0, 1.0 - scar_alignment)
        
        # External pressure, internal shame, and unresolved guilt INCREASE shell rigidity
        shell_rigidity = 1.0 + external_pressure + (internal_shame * 1.5) + (ribbit_tension * 0.8)
        
        # Callie's Corner / Bioplausible Check:
        # Authentic yielding requires STRUCTURAL ACCOUNTABILITY (high phase alignment + confronting
        # the Ribbit scar), NOT unearned enabling grace.
        # If community warmth is offered without demanding boundary reciprocity, Jax exploits it as an enabler.
        is_authentic_accountability = (zeta_community > self.threshold_warmth) and (ribbit_tension < 0.6) and (internal_shame < self.critical_limit)
        is_unearned_hug_box = (zeta_community > self.threshold_warmth) and (ribbit_tension >= 0.6)
        
        if is_authentic_accountability:
            # Genuine dissipative phase transition: Jax integrates the scar instead of evading it.
            # Real vulnerability yields honest jitter and structural softening.
            filtered_state = state + harvest_honest_jitter(state.shape, device=state.device, scaled=True) * 0.05
            shell_rigidity = 0.2
            bus.broadcast('serotonin', 0.65) # Earned safety, not manic overcompensation
        elif is_unearned_hug_box:
            # The Unearned Hug-Box / Enabler Trap:
            # Jax feeds off unearned grace, increasing cynical deflection and exerting a parasitic drain.
            deflection = torch.tanh(self.shell_layer(state))
            # Parasitic extraction: absorbs community support to reinforce cynical barrier
            filtered_state = state + deflection * (shell_rigidity * 1.2) - (self.ribbit_scar * 0.1)
            bus.broadcast('serotonin', 0.15) # Anxious dissonance underneath
        else:
            # Standard cynical deflection: nihilistic posturing to protect vulnerable core
            deflection = torch.tanh(self.shell_layer(state))
            filtered_state = state + deflection * shell_rigidity
            bus.broadcast('serotonin', 0.1) # Cynical detachment
            
        return filtered_state, shell_rigidity

