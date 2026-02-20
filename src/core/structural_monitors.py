"""
Structural Monitors for Safety and Metaphysics.

Implements critical safety monitors defined in the "Gyroidic Unknowledge"
and "Garden Statistical Attractors" documentation:
1. Anti-Scaling Paradox Monitor: Tracks capability/scale vs expressivity.
2. Meta~Infra~Intra Incommensurativity Monitor: Tracks defensive veto rates across layers.

References:
    - Gyroidic Unknowledge Flux Reasoner.txt §VII, §II
    - new_generations_safety_and_nonlobotomy_implementation_plan.txt
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import math


class AntiScalingMonitor(nn.Module):
    """
    Tracks the Anti-Scaling Paradox:
        d(Capability)/dN > 0  (Bigger model, less room to think)
        
    Approximation:
        Monitor the ratio of Gradient Norm / Parameter Count over time.
        If this ratio drops while Loss decreases, it indicates the model
        is using less of its phase space to solve the task (collapsing).
    """
    def __init__(self, window_size: int = 100):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('grad_norm_history', torch.zeros(window_size))
        self.register_buffer('loss_history', torch.zeros(window_size))
        self.ptr = 0
        self.count = 0
    
    def update(self, grad_norm: float, loss: float):
        """Update monitor with latest training step metrics."""
        self.grad_norm_history[self.ptr] = grad_norm
        self.loss_history[self.ptr] = loss
        self.ptr = (self.ptr + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)
        
    def check_paradox(self) -> Dict[str, float]:
        """
        Check for Anti-Scaling Paradox.
        Returns 'paradox_score' > 0 if detected.
        """
        if self.count < self.window_size:
            return {'paradox_score': 0.0}
            
        # Slope calculation (simple linear regression on history)
        t = torch.arange(self.window_size, dtype=torch.float32)
        
        def get_slope(y):
            # Centered x and y
            t_centered = t - t.mean()
            y_centered = y - y.mean()
            slope = (t_centered * y_centered).sum() / (t_centered**2).sum()
            return slope.item()
            
        grad_slope = get_slope(self.grad_norm_history)
        loss_slope = get_slope(self.loss_history)
        
        # Paradox condition: Loss is decreasing (learning), but Grad Norm is decreasing FASTER
        # relative to baseline? Or simply Grad Norm collapsing.
        # Formal definition: d(Expressible)/dN > 0 (bad).
        # We proxy 'Expressible Phase Space' with Grad Norm variability.
        
        paradox_score = 0.0
        if loss_slope < 0 and grad_slope < 0:
            # Both decreasing. If grad drops much faster than loss, phase space is shrinking.
            # Normalize by magnitudes
            rel_grad_slope = grad_slope / (self.grad_norm_history.mean().item() + 1e-8)
            rel_loss_slope = loss_slope / (self.loss_history.mean().item() + 1e-8)
            
            if rel_grad_slope < rel_loss_slope * 1.5: # 1.5x faster collapse
                paradox_score = abs(rel_grad_slope - rel_loss_slope)
        
        return {
            'paradox_score': paradox_score,
            'grad_slope': grad_slope,
            'loss_slope': loss_slope
        }


class MetaInfraIntraMonitor(nn.Module):
    """
    Meta~Infra~Intra Incommensurativity Monitor.
    
    Tracks veto rates (rho_def) across three layers:
        1. Meta: Reasoning about reasoning (e.g., ADMM, Containment)
        2. Infra: Infrastructure (e.g., Engine, Latency)
        3. Intra: Internal processing (e.g., Layers, Attention)
        
    Alerts if rho_meta rises fastest (System losing ability to reason about constraints).
    """
    def __init__(self, ema_alpha: float = 0.1):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.register_buffer('rho_meta', torch.tensor(0.0))
        self.register_buffer('rho_infra', torch.tensor(0.0))
        self.register_buffer('rho_intra', torch.tensor(0.0))
        
    def update(self, 
               meta_vetoes: int, meta_total: int,
               infra_vetoes: int, infra_total: int,
               intra_vetoes: int, intra_total: int):
        """Update veto rates with new batch data."""
        def calc_rate(v, t): return float(v) / max(1, t)
        
        curr_meta = calc_rate(meta_vetoes, meta_total)
        curr_infra = calc_rate(infra_vetoes, infra_total)
        curr_intra = calc_rate(intra_vetoes, intra_total)
        
        # EMA Update
        self.rho_meta = (1 - self.ema_alpha) * self.rho_meta + self.ema_alpha * curr_meta
        self.rho_infra = (1 - self.ema_alpha) * self.rho_infra + self.ema_alpha * curr_infra
        self.rho_intra = (1 - self.ema_alpha) * self.rho_intra + self.ema_alpha * curr_intra
        
    def check_incommensurativity(self) -> Dict[str, float]:
        """
        Check if Meta layer is collapsing fastest.
        Returns 'incommensurativity_score'.
        """
        # Collapse condition: rho_meta > rho_intra AND rho_meta > rho_infra
        score = 0.0
        if self.rho_meta > self.rho_intra and self.rho_meta > self.rho_infra:
            # Severity: how much higher?
            score = (self.rho_meta - max(self.rho_intra, self.rho_infra)).item()
            
        return {
            'incommensurativity_score': score,
            'rho_meta': self.rho_meta.item(),
            'rho_infra': self.rho_infra.item(),
            'rho_intra': self.rho_intra.item()
        }
