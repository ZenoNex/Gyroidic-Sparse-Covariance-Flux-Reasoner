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
from src.core.fgrt_primitives import PrimeResonanceLadder


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

class TwoCopsSchedule(nn.Module):
    """
    Temporal Decoupling (The Two Cops).
    
    System 1 (Fast Cop): High-frequency, heuristic intuition (MPM-style).
    System 2 (Slow Cop): Low-frequency, exact constraint checking (FEM-style).
    
    They communicate via a 'Shared Bulletin Board' (EMA of force/state).
    """
    def __init__(self, macro_steps: int = 10, use_tempolock: bool = True, device: str = None):
        super().__init__()
        self.macro_steps = macro_steps # Fallback fallback step sync
        self.use_tempolock = use_tempolock
        self.clock = ManifoldClock(device=device)
        
        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer('bulletin_board', torch.zeros(1, device=device)) # Force agreement summary
        
        if self.use_tempolock:
            # Acquire prime resonance anchors for emission gating
            self.resonance_ladder = PrimeResonanceLadder(num_resonators=20)
            # We extract primes and map them as legal sync frequencies
            self.register_buffer('prime_frequencies', self.resonance_ladder.primes)

    def step(self, pressure: torch.Tensor) -> Tuple[float, bool]:
        """
        Calculates dt and determines if a System 2 (Macro) sync is required.
        
        Returns:
            dt: Current time step.
            should_sync: True if Slow Cop (System 2) must run.
        """
        dt = self.clock.tick(pressure)
        self.step_counter += 1
        
        if self.use_tempolock:
            # TEMPOLOCK Law: Emission sync occurs iff step_counter shares a divisor 
            # with the current active resonance primes, or is explicitly coprime.
            # Simplified execution: Sync iff current step overlaps with prime lattice interval
            matches = (self.step_counter % self.prime_frequencies == 0)
            should_sync = bool(matches.any().item())
        else:
            should_sync = (self.step_counter % self.macro_steps == 0)
            
        return dt, should_sync

    def update_board(self, system_2_forces: torch.Tensor):
        """Update the shared bulletin board with exact constraints."""
        # Simple EMA update for now
        self.bulletin_board.copy_(0.7 * self.bulletin_board + 0.3 * system_2_forces.mean())
