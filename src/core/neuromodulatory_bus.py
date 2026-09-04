from typing import Dict
import torch

class NeuromodulatoryBus:
    """
    Shared Neuromodulatory Bus.
    
    Maintains a stateful snapshot of neurochemical channels that modules read from and write to.
    Mirrors diffusion over time.
    """
    def __init__(self, modulators: list = None):
        if modulators is None:
            self.modulators = ['gaba', 'serotonin', 'dopamine', 'oxytocin', 'acetylcholine', 'noradrenaline']
        else:
            self.modulators = modulators
            
        self._state: Dict[str, float] = {mod: 0.0 for mod in self.modulators}
        
    def broadcast(self, channel: str, value: float):
        """Update a neurochemical channel with a new concentration (simulating diffusion)."""
        if channel in self._state:
            # Simple EMA diffusion to prevent instantaneous jumps
            self._state[channel] = 0.8 * self._state[channel] + 0.2 * value
            
    def read(self, channel: str) -> float:
        """Read current concentration of a neurochemical channel."""
        return self._state.get(channel, 0.0)
        
    def snapshot(self) -> Dict[str, float]:
        """Return a copy of the current bus state."""
        return self._state.copy()
