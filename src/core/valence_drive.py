import torch
import torch.nn as nn
from typing import Dict, Optional
import time


class ValenceFunctional(nn.Module):
    """
    Valence: The drive to resolve structural dissonance.
    
    Measures the gap between current structural pressure and a 
    historical 'Satisfaction' baseline (Saturated Trust).
    """
    
    def __init__(
        self,
        decay: float = 0.99,
        hunger_scale: float = 1.0,
        device: str = None
    ):
        super().__init__()
        self.decay = decay
        self.hunger_scale = hunger_scale
        self.device = device
        
        # Historical Satisfaction baseline 
        self.register_buffer('satisfaction', torch.tensor(0.0, device=device))
        self.last_update_time = time.time()

    def forward(
        self, 
        current_pressure: torch.Tensor,
        mischief: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the Training Valence (Hunger).
        V = hunger_scale * (max(0, current_pressure - satisfaction) + Dissonance)
        
        Following MATHEMATICAL_DETAILS.md 15.2 (Metaphysical Entropy Bands):
        The total information entropy of the system is decomposed into multi-scale "disorder" channels:
        H_meta = H_dementia + H_schizo + H_mischief
        
        - Dementia Band (H_dementia): Decays historical anchors that lack current resonance.
        - Schizo Band (H_schizo): Fragments fixed categories into playful multi-modal clusters.
        - Mischief Band (H_mischief): Rewards topological violations (Good Bugs) to prevent scale-induced lobotomy.
        """
        self.last_update_time = time.time()
        
        # 1. Dynamic Decay Schedule (Wattsian Unlearning)
        # Under normal conditions, decay = self.decay.
        # Under stagnation (low entropy or flatline variance), decay accelerates to 0.85
        # to speed up unlearning of satisfaction and spike hunger.
        effective_decay = self.decay
        is_rigid = False
        stagnation_reason = ""
        
        if entropy is not None:
            entropy_val = entropy.mean().item()
            if entropy_val < 0.05:
                is_rigid = True
                stagnation_reason = f"low entropy ({entropy_val:.3f})"
                
        if not is_rigid and current_pressure.numel() > 1:
            pressure_var = current_pressure.var().item()
            if pressure_var < 1e-6:
                is_rigid = True
                stagnation_reason = f"low variance ({pressure_var:.2e})"
                
        if is_rigid:
            effective_decay = 0.85
            if getattr(self, '_last_log_time', 0) < time.time() - 5.0:
                print(f"[VALENCE] Stagnation detected via {stagnation_reason}. "
                      f"Accelerating unlearning: decay={effective_decay:.4f}")
                self._last_log_time = time.time()

        # We detach pressure to keep satisfaction as a non-differentiable reference
        self.satisfaction.mul_(effective_decay).add_((1.0 - effective_decay) * current_pressure.mean().detach())
        
        # 2. Compute primary pressure gap (Surprise)
        surprise = torch.clamp(current_pressure - self.satisfaction, min=0.0)
        
        # 3. Inject Structural Dissonance via Metaphysical Entropy Bands (H_meta)
        # H_meta = H_dementia + H_schizo + H_mischief
        dissonance = 0.0
        if mischief is not None:
            # Mischief Band (H_mischief): Rewards topological violations (Good Bugs), 
            # but high mischief drives the system to find structured associations (Hunger).
            dissonance += 0.5 * mischief.mean()
            
        if entropy is not None:
            # Combined Dementia and Schizo Bands (H_dementia + H_schizo):
            # High spectral entropy representing historical decay and category fragmentation.
            dissonance += 0.3 * entropy.mean()

        # Coordinate category dispersion when the local correlation metric reaches 1.0
        if categories is not None:
            from src.core.martinova_correlation import compute_bounded_correlation
            cat_corr = compute_bounded_correlation(categories)
            if (cat_corr >= 0.99).any():
                # Add a strong dissonance boost to coordinate dispersion (urge the system to disperse)
                dissonance += 1.5
                print("[VALENCE] Category clustering reached 1.0. Coordinating category dispersion.")

        # 4. Total Hunger
        hunger = (surprise + dissonance) * self.hunger_scale
        
        # 5. Persistent tracking for diagnostics
        self._last_hunger = hunger.mean().detach()
        
        return hunger


    def get_metrics(self) -> Dict[str, float]:
        """Return metrics for the diegetic terminal."""
        elapsed = time.time() - getattr(self, 'last_update_time', time.time())
        
        # Biologically inspired: satisfaction decays when the system is starved of fresh inputs
        # 5-minute half-life (300 seconds) for satisfaction decay
        decay_factor = 0.5 ** (elapsed / 300.0)
        current_satisfaction = self.satisfaction.item() * decay_factor
        
        # Starvation hunger naturally rises towards hunger_scale as satisfaction decays
        last_hunger_val = getattr(self, '_last_hunger', torch.tensor(0.0)).item()
        starvation_hunger = self.hunger_scale * (1.0 - current_satisfaction)
        
        # Merge active hunger and starvation hunger
        current_hunger = last_hunger_val * decay_factor + starvation_hunger * (1.0 - decay_factor)
        
        # Ensure hunger has a baseline minimum (e.g. 0.15) during background learning so it never freezes
        current_hunger = max(current_hunger, 0.15)
        
        return {
            'asymptotic_satisfaction': current_satisfaction,
            'current_hunger_drive': current_hunger
        }


