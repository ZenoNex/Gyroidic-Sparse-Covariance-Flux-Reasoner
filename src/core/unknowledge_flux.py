"""
Unknowledge Flux: Metaphysical Leaks and Entropic Mischief.

Implements the "Nostalgic Leak" and "Mischief Band" dynamics where 
unknowledge is preserved as high-frequency solitons.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class NostalgicLeakFunctional(nn.Module):
    """
    Implements the Nostalgic Leak ψ_l: H -> R^{D+1}.
    
    Models internet archetype concealment using sigmoid visibility masks 
    (e.g. apple-obscured faces).
    """
    
    def __init__(
        self,
        fossil_dim: int,
        alpha: float = 5.0,
        device: str = None
    ):
        super().__init__()
        self.fossil_dim = fossil_dim
        self.alpha = alpha
        self.device = device
        
        # Archetype coefficients μ_l (Obscured)
        self.register_buffer('mu_l', torch.randn(fossil_dim, device=device))
        
        # Obstruction point o (The Apple/Mask center)
        self.register_buffer('o', torch.zeros(fossil_dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ψ_l(x) = sum(μ_l * P(x)) * (1 - Vis(x))
        
        Args:
            x: Input state [batch, fossil_dim]
        """
        # Visibility mask around obstruction o
        # Vis(x) = σ(α * |x - o|)
        dist = torch.norm(x - self.o, dim=1, keepdim=True)
        vis = torch.sigmoid(self.alpha * dist)
        
        # Applying the leak functional
        # Using a simple linear projection for the 'polynomial' sum for now
        leak = torch.sum(x * self.mu_l, dim=1, keepdim=True) * (1.0 - vis)
        return leak


class EntropicMischiefProbe(nn.Module):
    """
    Calculates the Metaphysical Disorder bands:
    H_meta = H_dementia + H_schizo + H_mischief
    """
    
    def __init__(
        self,
        tau_dementia: float = 100.0,
        eta_mischief: float = 0.5,
        theta_leak: float = 0.7,
        device: str = None
    ):
        super().__init__()
        self.tau_dementia = tau_dementia
        self.eta_mischief = eta_mischief
        self.theta_leak = theta_leak
        self.device = device
        
        # State tracking
        self.register_buffer('H_dementia', torch.tensor(0.0, device=device))
        self.register_buffer('H_schizo', torch.tensor(0.0, device=device))
        self.register_buffer('H_mischief', torch.tensor(0.0, device=device))

    def update(
        self, 
        pressure_grad: torch.Tensor, 
        coherence: torch.Tensor, 
        pas_h: float,
        is_good_bug: bool = False
    ):
        """
        Updates the metaphysical bands.
        
        Args:
            pressure_grad: Gradient of the structural pressure (nabla Phi)
            coherence: Spectral coherence between clusters
            pas_h: Current Phase Alignment Score
            is_good_bug: Boolean signal for Mischief reward
        """
        # 1. Dementia Band (Low-frequency forgetting)
        # H_dementia = integral(|nabla Phi|^2 * exp(-tau * f))
        # Approximated by temporal decay of pressure energy.
        h_dem = torch.sum(pressure_grad**2) * torch.exp(torch.tensor(-1.0 / self.tau_dementia, device=self.device))
        self.H_dementia.copy_(h_dem.detach())
        
        # 2. Schizo Band (Mid-frequency fragmentation)
        # H_schizo = sum(gamma * log(1 - Coh))
        # Captures archetype fusion/fragmentation.
        h_sch = -torch.sum(torch.log(torch.clamp(1.0 - coherence, 1e-6, 1.0)))
        self.H_schizo.copy_(h_sch.detach())
        
        # 3. Mischief Band (High-frequency play)
        # H_mischief = eta * max(0, 1 - PAS_h) * log(GoodBug)
        # GoodBug(r) = 2.0 if is_good_bug else 1.0 (log(1)=0, log(2)>0)
        
        bug_reward = 2.0 if is_good_bug else 1.0
        
        # PAS_h should be normalized 0..1. If not, clip.
        pas_gap = max(0.0, 1.0 - max(0.0, min(1.0, pas_h)))
        
        h_mis = self.eta_mischief * pas_gap * torch.log(torch.tensor(bug_reward, device=self.device))
        self.H_mischief.copy_(h_mis.detach())

    @property
    def H_meta(self) -> torch.Tensor:
        return self.H_dementia + self.H_schizo + self.H_mischief

    def check_leak_split(self) -> bool:
        """Adaptive splitting condition: Split iff H_meta > theta_leak."""
        return self.H_meta.item() > self.theta_leak

    def get_metrics(self) -> Dict[str, float]:
        return {
            'H_dementia': self.H_dementia.item(),
            'H_schizo': self.H_schizo.item(),
            'H_mischief': self.H_mischief.item(),
            'H_meta': self.H_meta.item()
        }

