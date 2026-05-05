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


class FossilizedSurvivalLattice:
    """
    Archaeological Survival Lattice (The 'Cheat' Basis).
    
    Generates prime resonance frequencies dynamically using PrimeResonanceLadder.
    Used ONLY as an emergency fallback when spectral atrophy (PAS_h collapse) 
    is detected in the dynamic polynomial functionals.
    """
    def __init__(self, device: str = 'cpu'):
        from src.core.fgrt_primitives import PrimeResonanceLadder
        self.ladder = PrimeResonanceLadder(num_resonators=8).to(device)
        self.survival_event_logged = False

    def get_primes(self):
        if not self.survival_event_logged:
            import logging
            logging.warning("CONSTITUTIONAL VIOLATION: Spectral Atrophy detected. Triggering Archaeological Survival Lattice.")
            self.survival_event_logged = True
        return self.ladder.primes.float()

class TopologicalErosionFBM(nn.Module):
    def __init__(self, octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0, poly_config=None):
        super().__init__()
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        
        # dynamic co-prime functional system
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        self.poly_config = poly_config or PolynomialCoprimeConfig(k=octaves, degree=4)
        self.survival_lattice = FossilizedSurvivalLattice()

    def _pseudo_random_hash(self, x: torch.Tensor) -> torch.Tensor:
        """Simple deterministic hash for noise generation based on coordinates."""
        # A simple high-frequency sinusoidal hash
        dot = torch.einsum('...d,...d->...', x, torch.sin(x * 12.9898) * 43758.5453)
        return torch.frac(dot).unsqueeze(-1)

    def _noise(self, x: torch.Tensor, freq: float = 1.0) -> torch.Tensor:
        """Basic continuous noise proxy."""
        # We use a periodic combination of sines as a fast differentiable noise proxy
        return torch.sin(x * math.pi * freq) * torch.cos(x * math.e * freq)

    def fbm(self, x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        """
        Fractional Brownian Motion over the vector field using dynamic frequencies.
        """
        total = torch.zeros_like(x)
        amplitude = 1.0
        max_value = 0.0  # Used for normalizing
        
        num_freqs = len(frequencies)
        for i in range(self.octaves):
            f = frequencies[i % num_freqs]
            total += self._noise(x, f.item()) * amplitude
            max_value += amplitude
            amplitude *= self.persistence
            # Lacunarity is handled by the prime-like distribution of frequencies
            
        return total / max_value

    def forward(self, state: torch.Tensor, pressure_grad: torch.Tensor, intensity: float = 0.1, pas_h: float = 1.0) -> torch.Tensor:
        """
        Carve gullies into the feature space using the "Fractional Anisotropic 
        Fractal Polynomial Functionals encoded Brownian Motion" paradigm.
        """
        if pressure_grad is None or intensity == 0:
            return state
            
        normalized_pressure = F.normalize(pressure_grad, dim=-1)
        
        # 1. Determine Frequencies (Dynamic vs Fossilized)
        # Spectral Atrophy Detection (PAS_h < 0.2 indicates collapse)
        if pas_h < 0.2:
            frequencies = self.survival_lattice.get_primes().to(state.device)
        else:
            # Use dynamic polynomial functionals
            # We sample frequencies from the theta-weighted basis
            with torch.no_grad():
                # Extract 'prime-like' anchors from the polynomial coefficients
                # We use the mean magnitude per functional as a frequency proxy
                frequencies = torch.norm(self.poly_config.theta, dim=1) * 10.0 # Scale to audible range
                # Ensure they are incommensurate (co-primality is enforced in poly_config)
        
        # PyOpenCL Hardware Offload
        try:
            from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
            if not hasattr(self, 'silicon_engine'):
                self.silicon_engine = SiliconSovereigntyEngine()
                
            device = state.device
            state_np = state.detach().cpu().numpy()
            pgrad_np = normalized_pressure.detach().cpu().numpy()
            freq_np = frequencies.detach().cpu().numpy()
            
            result_np = self.silicon_engine.apply_erosion_fbm(
                state=state_np,
                pressure_grad_normalized=pgrad_np,
                octaves=self.octaves,
                persistence=self.persistence,
                lacunarity=self.lacunarity,
                intensity=intensity,
                primes=freq_np
            )
            
            return torch.from_numpy(result_np).to(device)
            
        except Exception:
            # Fallback to PyTorch Polynomial Functional Logic
            pass
            
        total = torch.zeros_like(state)
        amp_scale = 1.0
        max_val = 0.0
        
        num_freqs = len(frequencies)
        for i in range(self.octaves):
            p = frequencies[i % num_freqs]
            phase = state * p
            total += torch.sin(phase) * amp_scale
            max_val += amp_scale
            amp_scale *= self.persistence
            
        noise_field = total / max_val
        # Anisotropic erosion: push along the gradient modulated by resonant gullies
        return state + intensity * (-normalized_pressure * torch.abs(noise_field))
