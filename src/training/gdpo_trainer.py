"""
GDPO Structural Adaptation for Configuration Stabilizing.

Implements PPO-style adaptation with Signal Sovereignty decoupled pressures
for multi-pressure constraint satisfaction problems.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from src.core.gdpo_normalization import SignalSovereignty

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))



class GDPOSovereigntyPressureComputer:
    """
    Computes GDPO-style decoupled sovereignty pressures from multi-pressure rollouts.
    
    Key difference from standard GRPO:
        - GRPO: Normalize sum of pressures → collapse
        - GDPO: Normalize each pressure separately, then aggregate → sovereignty preserved
    """
    
    def __init__(
        self,
        num_pressures: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon: float = 1e-8
    ):
        """
        Args:
            num_pressures: Number of pressure dimensions
            gamma: Discount factor
            gae_lambda: GAE lambda
            epsilon: Numerical stability
        """
        self.num_pressures = num_pressures
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        
        self.sovereignty = SignalSovereignty(num_pressures, epsilon=epsilon)
    
    def compute_sovereignty_pressures(
        self,
        pressures: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute GDPO sovereignty pressures.
        
        Args:
            pressures: [batch, num_steps, num_pressures] multi-pressure trajectories
            values: [batch, num_steps] value estimates
            dones: [batch, num_steps] episode done flags
            weights: [num_pressures] pressure aggregation weights
            group_ids: Optional [batch] group assignments
            
        Returns:
            advantages: [batch, num_steps] GDPO sovereignty pressures
            diagnostics: Additional information
        """
        batch_size, num_steps, num_pressures = pressures.shape
        
        # Flatten batch and steps for processing
        pressures_flat = pressures.view(-1, num_pressures)  # [batch*steps, num_pressures]
        
        # Extend group_ids to match steps
        if group_ids is None:
            group_ids_extended = torch.zeros(batch_size * num_steps, dtype=torch.long,
                                            device=pressures.device)
        else:
            group_ids_extended = group_ids.unsqueeze(1).expand(batch_size, num_steps).reshape(-1)
        
        # Apply GDPO decoupled sovereignty normalization
        aggregated_pressures, gdpo_diag = self.sovereignty(
            pressures_flat,
            weights,
            group_ids_extended
        )
        
        # Reshape back
        aggregated_pressures = aggregated_pressures.view(batch_size, num_steps)
        
        # Compute GAE advantages per dimension
        advantages = torch.zeros((batch_size, num_steps, num_pressures), device=pressures.device)
        
        # diagnostics['decoupled_pressures'] is [batch, steps, num_pressures]
        decoupled_p = gdpo_diag['decoupled'].view(batch_size, num_steps, num_pressures)
        
        for k in range(num_pressures):
            last_gae_k = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_value_k = 0
                else:
                    next_value_k = values[:, t + 1] # Simplified: single value function for all
                
                # TD error for dimension k
                delta_k = decoupled_p[:, t, k] + self.gamma * next_value_k * (1 - dones[:, t]) - values[:, t]
                
                # GAE for dimension k
                advantages[:, t, k] = delta_k + self.gamma * self.gae_lambda * (1 - dones[:, t]) * last_gae_k
                last_gae_k = advantages[:, t, k]
        
        diagnostics = {
            'decoupled_pressures': decoupled_p,
            'returns': advantages + values.unsqueeze(-1)
        }
        
        return advantages, diagnostics


class GDPOSovereigntyAdaptor:
    """
    PPO-style structural adaptor with GDPO sovereignty pressures.
    
    Total objective:
        J(θ) = E[min(ρ·Â^GDPO, clip(ρ)·Â^GDPO)] - β·KL(π_θ || π_ref)
    
    Where Â^GDPO uses decoupled multi-pressure normalization.
    """
    
    def __init__(
        self,
        configuration_model: nn.Module,
        value_model: Optional[nn.Module] = None,
        num_pressures: int = 4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = None,
        # Functional hyperparameters
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.01,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        # GDPO hyperparameters
        pressure_weights: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Args:
            configuration_model: Configuration network (e.g., GyroidicFluxReasoner)
            value_model: Optional separate value network
            num_pressures: Number of pressure dimensions
            optimizer: Optimizer for configuration
            device: Device to adapt on
            clip_epsilon: Clipping parameter
            kl_coef: KL penalty coefficient β
            value_coef: Value pressure coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            pressure_weights: Fixed weights or None for learnable
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        self.configuration = configuration_model.to(device, non_blocking=True)
        self.value_model = value_model.to(device, non_blocking=True) if value_model else None
        self.num_pressures = num_pressures
        self.device = device
        
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.configuration.parameters(), lr=3e-4)
        else:
            self.optimizer = optimizer
        
        # GDPO sovereignty computer
        if pressure_weights is None:
            pressure_weights = torch.ones(num_pressures, device=device) / num_pressures
        
        self.sovereignty_computer = GDPOSovereigntyPressureComputer(
            num_pressures=num_pressures,
            gamma=gamma,
            gae_lambda=gae_lambda
        )
        self.pressure_weights = pressure_weights.to(device, non_blocking=True)
    
    def compute_adaptation_pressure(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute clipped structural adaptation pressure.
        
        L_adaptation = -E[min(ρ·Â, clip(ρ,1-ε,1+ε)·Â)]
        
        Args:
            log_probs: [batch, steps] current configuration log probs
            old_log_probs: [batch, steps] old configuration log probs
            advantages: [batch, steps] GDPO sovereignty pressures
            
        Returns:
            pressures: [num_pressures] Scalar adaptation pressures
            metrics: Dictionary with statistics
        """
        # Importance ratio: ρ = π_θ / π_old
        ratio = torch.exp(log_probs - old_log_probs)
        
        adaptation_pressures = []
        for k in range(advantages.shape[-1]):
            adv_k = advantages[:, :, k]
            surr1 = ratio * adv_k
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_k
            loss_k = -torch.mean(torch.min(surr1, surr2))
            adaptation_pressures.append(loss_k)
        
        # Metrics
        metrics = {
            'adaptation_pressure_mean': torch.stack(adaptation_pressures).mean().item(),
            'ratio_mean': ratio.mean().item(),
            'sovereignty_mean': advantages.mean().item()
        }
        
        return torch.stack(adaptation_pressures), metrics
    
    def adaptation_step(
        self,
        rollout_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single GDPO adaptation step.
        
        Args:
            rollout_batch: Dictionary containing:
                - 'pressures': [batch, steps, num_pressures]
                - 'log_probs': [batch, steps]
                - 'values': [batch, steps]
                - 'dones': [batch, steps]
                - 'group_ids': Optional [batch]
                
        Returns:
            metrics: Adaptation metrics
        """
        self.optimizer.zero_grad()
        
        pressures = rollout_batch['pressures'].to(self.device, non_blocking=True)
        old_log_probs = rollout_batch['log_probs'].to(self.device, non_blocking=True)
        values = rollout_batch['values'].to(self.device, non_blocking=True)
        dones = rollout_batch['dones'].to(self.device, non_blocking=True)
        group_ids = rollout_batch.get('group_ids')
        
        # Compute GDPO sovereignty pressures
        advantages, adv_diagnostics = self.sovereignty_computer.compute_sovereignty_pressures(
            pressures,
            values,
            dones,
            self.pressure_weights,
            group_ids
        )
        
        # Normalize sovereignty (standard practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Re-evaluate under current configuration
        # (This would require forward pass through configuration - simplified here)
        log_probs = old_log_probs  # Placeholder
        
        # Adaptation pressures (multi-domain)
        adaptation_pressures, adaptation_metrics = self.compute_adaptation_pressure(
            log_probs,
            old_log_probs,
            advantages
        )
        
        # Value pressure (if using value function)
        if self.value_model is not None:
            returns = adv_diagnostics['returns'].mean(dim=-1) # Mean returns across domains for value head
            value_pressure = 0.5 * ((values - returns) ** 2).mean()
        else:
            value_pressure = torch.tensor(0.0, device=self.device)
        
        # Backward each adaptation pressure independently to prevent scalarization trap
        for p in adaptation_pressures:
            p.backward(retain_graph=True)
            
        if self.value_model is not None:
            (self.value_coef * value_pressure).backward()
        
        torch.nn.utils.clip_grad_norm_(self.configuration.parameters(), self.max_grad_norm)
        self.optimizer.step()
        # --- GDPO BIRKHOFF ALIGNMENT ---
        with torch.no_grad():
            from src.core.birkhoff_projection import project_to_birkhoff
            for p in self.configuration.parameters():
                if p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.copy_(project_to_birkhoff(p.data))
        
        # Compile metrics
        metrics = {
            'value_pressure': value_pressure.item(),
            **adaptation_metrics
        }
        
        return metrics


