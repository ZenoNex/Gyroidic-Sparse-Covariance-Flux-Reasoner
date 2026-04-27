"""
Topological Erosion FBM (Fractional Brownian Motion).

Phase 6 hybrid implementation:
Instead of simulating every "drop" of user interaction, we apply an Erosion Noise Function
(using multi-octave FBM) on top of the latent manifold.

This "carves" gullies into the feature space along the gradient of user pressure.
The resulting "ridges" become stable Morphological Set Points (Fossils).
This replaces standard teleological gradient descent with a purely topographical
weathering process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TopologicalErosionFBM(nn.Module):
    def __init__(self, octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0):
        super().__init__()
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        
    def _pseudo_random_hash(self, x: torch.Tensor) -> torch.Tensor:
        """Simple deterministic hash for noise generation based on coordinates."""
        # A simple high-frequency sinusoidal hash
        dot = torch.einsum('...d,...d->...', x, torch.sin(x * 12.9898) * 43758.5453)
        return torch.frac(dot).unsqueeze(-1)

    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        """Basic continuous noise proxy."""
        # We use a periodic combination of sines as a fast differentiable noise proxy
        return torch.sin(x * math.pi) * torch.cos(x * math.e)

    def fbm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fractional Brownian Motion over the vector field.
        """
        total = torch.zeros_like(x)
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0  # Used for normalizing
        
        for _ in range(self.octaves):
            total += self._noise(x * frequency) * amplitude
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
            
        return total / max_value

    def forward(self, state: torch.Tensor, pressure_grad: torch.Tensor, intensity: float = 0.1) -> torch.Tensor:
        """
        Carve gullies into the feature space using FBM directed by the pressure gradient.
        
        Args:
            state: [batch, dim] the latent manifold state
            pressure_grad: [batch, dim] topological tension gradient
            intensity: depth of the erosion cut
            
        Returns:
            Eroded state tensor
        """
        if pressure_grad is None or not float(intensity):
            return state
            
        normalized_pressure = F.normalize(pressure_grad, dim=-1)
        
        # PyOpenCL Hardware Offload Attempt
        try:
            from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
            if not hasattr(self, 'silicon_engine'):
                self.silicon_engine = SiliconSovereigntyEngine()
                
            device = state.device
            state_np = state.detach().cpu().numpy()
            pgrad_np = normalized_pressure.detach().cpu().numpy()
            
            result_np = self.silicon_engine.apply_erosion_fbm(
                state=state_np,
                pressure_grad_normalized=pgrad_np,
                octaves=self.octaves,
                persistence=self.persistence,
                lacunarity=self.lacunarity,
                intensity=intensity
            )
            
            return torch.from_numpy(result_np).to(device)
            
        except Exception as e:
            # Fallback to PyTorch
            pass
            
        # The noise amplitude is maximized where the gradient is steepest
        # We carve "down" the gradient.
        noise_field = self.fbm(state)
        
        # Erosion pushes the state along the negative gradient, scaled by the chaotic FBM terrain
        erosion_vector = -normalized_pressure * torch.abs(noise_field)
        
        return state + intensity * erosion_vector
