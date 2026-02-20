"""
Manifold Time Dynamics: The Asymptotic Manifold Interplay.

Implements the "Breathing Time" logic where the coordinate time step (dt)
is modulated by the "Seriousness" (Pressure) of the manifold state.

Equation:
-int Seriousness * dt + int Play * dt

Thermodynamic Mapping:
dt acts as Inverse Temperature (beta). 
Free Energy F_topo = - (1/beta) * log(Z)
High Pressure -> Small dt -> High Beta (Freezing structure)
Low Pressure -> Large dt -> Low Beta (Playful flux)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math


class ManifoldClock(nn.Module):
    """
    Dynamically scales coordinate time (t) relative to proper time (tau)
    based on structural pressure feedback.
    
    This is the "Watts Move": take the universe seriously enough to dance (fine steps),
    not seriously enough to freeze (infinite steps).
    """
    
    def __init__(
        self,
        dt_base: float = 1.0,
        dt_min: float = 0.001,
        dt_max: float = 2.0,
        lambda_seriousness: float = 2.0,
        lambda_play: float = 0.5,
        device: str = None
    ):
        """
        Args:
            dt_base: Default time step
            dt_min: Minimum allowable dt (prevent freezing)
            dt_max: Maximum allowable dt (prevent instability)
            lambda_seriousness: Sensitivity to pressure (high = faster shrinkage)
            lambda_play: Sensitivity to smoothness (high = faster expansion)
        """
        super().__init__()
        self.dt_base = dt_base
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.lambda_seriousness = lambda_seriousness
        self.lambda_play = lambda_play
        self.device = device
        
        # Internal state
        self.register_buffer('coordinate_time', torch.tensor(0.0, device=device))
        self.register_buffer('proper_time', torch.tensor(0.0, device=device))
        self.register_buffer('current_dt', torch.tensor(dt_base, device=device))
        self.register_buffer('accumulated_seriousness', torch.tensor(0.0, device=device))
        self.register_buffer('accumulated_play', torch.tensor(0.0, device=device))

    def tick(self, pressure: torch.Tensor) -> float:
        """
        Update clock based on observed pressure.
        
        Args:
            pressure: Scalar pressure from the manifold (e.g., Gyroid Pressure)
            
        Returns:
            dt: The calculated time step for the next iteration.
        """
        p = pressure.detach()
        
        # 1. Calculate Seriousness vs Play
        # Seriousness is high when pressure is high.
        # Play is high when pressure is low (smoothness).
        seriousness = torch.tanh(self.lambda_seriousness * p)
        play = torch.exp(-self.lambda_play * p)
        
        # 2. Update dt
        # dt = dt_base * (Play / (1 + Seriousness))
        # We clamp to ensure we stay within physical limits.
        new_dt = self.dt_base * (play / (1.0 + seriousness))
        new_dt = torch.clamp(new_dt, self.dt_min, self.dt_max)
        
        # 3. Step time
        self.current_dt.copy_(new_dt)
        self.coordinate_time += new_dt
        self.proper_time += 1.0  # Constant increment in proper time
        
        # 4. History tracking
        self.accumulated_seriousness += seriousness * new_dt
        self.accumulated_play += play * new_dt
        
        return new_dt.item()

    @property
    def dt_ratio(self) -> float:
        """The ratio of current dt to base dt."""
        return self.current_dt.item() / self.dt_base

    def get_state(self) -> Dict[str, float]:
        """Return human-readable time state."""
        return {
            't': self.coordinate_time.item(),
            'tau': self.proper_time.item(),
            'dt': self.current_dt.item(),
            'dt_ratio': self.dt_ratio,
            'dilation': (self.proper_time / self.coordinate_time).item() if self.coordinate_time > 0 else 1.0,
            'total_seriousness': self.accumulated_seriousness.item(),
            'total_play': self.accumulated_play.item()
        }

    def reset(self):
        """Reset the clock."""
        self.coordinate_time.zero_()
        self.proper_time.zero_()
        self.current_dt.fill_(self.dt_base)
        self.accumulated_seriousness.zero_()
        self.accumulated_play.zero_()

