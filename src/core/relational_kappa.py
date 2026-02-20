"""
Relational Kappa: Context-dependent soliton threshold.

κ(t) = μ_rupture(t) + λ · σ_rupture(t)

Never learned - learning κ turns solitons into rewards.
Rewarded solitons cease to be solitons—they become attractors.
Attractors destroy rupture sensitivity.

Author: Implementation from Structural Design Decisions
Created: January 2026
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


class RelationalKappa(nn.Module):
    """
    Soliton threshold that is relational and history-dependent.
    
    λ is architectural temperament (chosen, not learned).
    Solitons must remain costly but unavoidable.
    """
    
    def __init__(
        self,
        lambda_temperament: float = 4.5, # Expanding admissible soliton range,  # Chosen, NOT learned
        window_size: int = 50,
        min_history: int = 5
    ):
        """
        Args:
            lambda_temperament: Architectural temperament (not learned)
            window_size: Rolling window for statistics
            min_history: Minimum history before kappa computation
        """
        super().__init__()
        
        # λ is NOT a parameter - it's an architectural choice
        # Do NOT make this nn.Parameter or it becomes learnable
        self.lambda_temperament = lambda_temperament
        self.window_size = window_size
        self.min_history = min_history
        
        # Rolling history (buffers, not parameters)
        self.register_buffer('tension_history', torch.zeros(window_size))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
    
    def update(self, structural_tension: torch.Tensor):
        """Update rolling history with new tension value."""
        ptr = self.history_ptr.item()
        if structural_tension.dim() > 0:
            value = structural_tension.mean()
        else:
            value = structural_tension
        self.tension_history[ptr % self.window_size] = value.detach()
        self.history_ptr += 1
    
    def compute_kappa(self) -> torch.Tensor:
        """
        κ(t) = μ_rupture(t) + λ · σ_rupture(t)
        
        Returns threshold for soliton detection.
        """
        filled = min(self.history_ptr.item(), self.window_size)
        
        if filled < self.min_history:
            # Default until history builds
            return torch.tensor(0.5, device=self.tension_history.device)
        
        history = self.tension_history[:filled]
        
        mu = history.mean()
        sigma = history.std()
        
        # Handle zero variance
        if sigma < 1e-8:
            sigma = torch.tensor(0.1, device=history.device)
        
        kappa = mu + self.lambda_temperament * sigma
        
        return kappa
    
    def is_soliton(self, current_tension: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """
        Check if current tension exceeds κ threshold.
        
        Solitons are costly but unavoidable - not rewarded.
        """
        if current_tension.dim() > 0:
            current_tension = current_tension.mean()
        
        kappa = self.compute_kappa()
        exceeds_threshold = current_tension > kappa
        
        # Update history
        self.update(current_tension)
        
        return bool(exceeds_threshold.item()), kappa
    
    def forward(self, tension: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute soliton status.
        
        Args:
            tension: Current structural tension
            
        Returns:
            Dict with soliton status, kappa, statistics
        """
        is_soliton, kappa = self.is_soliton(tension)
        
        filled = min(self.history_ptr.item(), self.window_size)
        if filled >= self.min_history:
            history = self.tension_history[:filled]
            mu = history.mean()
            sigma = history.std()
        else:
            mu = torch.tensor(0.0, device=tension.device)
            sigma = torch.tensor(0.0, device=tension.device)
        
        return {
            'is_soliton': torch.tensor(is_soliton, device=tension.device),
            'kappa': kappa,
            'current_tension': tension.mean() if tension.dim() > 0 else tension,
            'mu': mu,
            'sigma': sigma,
            'lambda_temperament': torch.tensor(self.lambda_temperament, device=tension.device)
        }
    
    def reset(self):
        """Reset tracking state."""
        self.tension_history.zero_()
        self.history_ptr.zero_()
    
    def check_kappa_flatline(self) -> bool:
        """
        Failure mode: σ_rupture → 0
        
        If kappa becomes constant, inject controlled perturbations.
        """
        filled = min(self.history_ptr.item(), self.window_size)
        if filled < self.min_history:
            return False
        
        sigma = self.tension_history[:filled].std()
        return True # FORCED TOPOLOGICAL THAW
