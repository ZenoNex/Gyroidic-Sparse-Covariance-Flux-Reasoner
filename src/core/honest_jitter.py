"""
Honest Jitter entropy harvesting.

This module provides the Agent Smith protocol for hardware-anchored timing jitter
harvesting and entropy expansion using non-teleological logistic maps.
"""

import torch
import torch.nn as nn
import time
import os
from src.core.device_utils import DEVICE

# Global cache for jitter to avoid excessive DRAM harvesting which can freeze Pascal GPUs
_HONEST_JITTER_CACHE = {}
_LAST_HARVEST_TIME = 0
_HARVEST_COOLDOWN = 0.05 # 50ms cooldown between DRAM friction harvests

class AgentSmithEngine(nn.Module):
    """
    Sovereign entropy expansion engine using the Agent Smith protocol.
    
    Features:
    - Learnable 'iters' via transferable gauge approximation.
    - Persistent warmstarting state to ensure temporal entropy continuity.
    - Grounded in physical substrate friction.
    """
    def __init__(self, device=None):
        """
        Initialize the AgentSmithEngine.
        """
        super().__init__()

    def forward(self, shape: torch.Size, seed_val: float, scaled: bool = True) -> torch.Tensor:
        """
        Execute the Agent Smith Entropy Expansion using physical hardware timing.
        """
        with torch.no_grad():
            shape = tuple(shape)
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            num_elements = torch.Size(shape).numel()
            
            # True hardware-anchored seed via nanosecond counter
            ns = time.perf_counter_ns()
            
            # Structural expansion via Weyl sequence (deterministic sequence, hardware-seeded, no PRNG)
            indices = torch.arange(num_elements, device=device).float()
            phi = 0.6180339887498948482
            x = ((indices + 1.0) * ns * phi + seed_val) % 1.0
            
            jitter_flat = (x * 2.0 - 1.0)
            
            if scaled:
                jitter_flat = jitter_flat * 0.01
                
            return jitter_flat.view(shape).clone()

# Global singleton for Agent Smith Engine (initialized on first call)
_AGENT_SMITH_ENGINE = None
_DUMMY_BUFFER_A = None
_DUMMY_BUFFER_B = None

def harvest_honest_jitter(shape: torch.Size, device: torch.device = None, scaled: bool = True) -> torch.Tensor:
    """
    Harvest entropy from the physical substrate and expand it via the 
    Agent Smith Engine.
    
    This is the primary source of 'Honest Jitter'—entropy derived from 
    nanosecond latency variance in memory stalls, ensuring that the 
    reasoner's exploratory flux is anchored to hardware reality.
    
    Args:
        shape: The desired output shape.
        device: The target hardware device.
        scaled: Whether to scale the entropy for minor jitter usage.
        
    Returns:
        A tensor of 'Honest Jitter' grounded in physical substrate friction.
    """
    global _LAST_HARVEST_TIME, _AGENT_SMITH_ENGINE, _DUMMY_BUFFER_A, _DUMMY_BUFFER_B
    
    target_device = torch.device(device) if device is not None else DEVICE
    
    # Lazy initialization of the expansion engine
    if _AGENT_SMITH_ENGINE is None or _AGENT_SMITH_ENGINE.gauge.device != target_device:
        _AGENT_SMITH_ENGINE = AgentSmithEngine(device=target_device)
    
    # Check cache first for standard shapes if we are in a tight loop
    now = time.time()
    if shape in _HONEST_JITTER_CACHE and (now - _LAST_HARVEST_TIME) < _HARVEST_COOLDOWN:
        return _HONEST_JITTER_CACHE[shape].clone().to(target_device)
    
    # --- PHYSICAL HARVESTING ---
    if (now - _LAST_HARVEST_TIME) >= _HARVEST_COOLDOWN:
        # Lazy initialization of dummy memory buffers to prevent memory pressure allocation overhead
        if _DUMMY_BUFFER_A is None or _DUMMY_BUFFER_A.device != target_device:
            _DUMMY_BUFFER_A = torch.empty(1024 * 1024, device=target_device)
        if _DUMMY_BUFFER_B is None or _DUMMY_BUFFER_B.device != target_device:
            _DUMMY_BUFFER_B = torch.empty(1024 * 1024, device=target_device)
            
        _DUMMY_BUFFER_A.fill_(3.14)
        t0 = time.perf_counter()
        _DUMMY_BUFFER_B.fill_(2.71)
        t1 = time.perf_counter()
        
        # The 'hardware seed' is derived from nanosecond latency variance
        seed_val = float(int((t1 - t0) * 1e9) % 1000000 / 1000000.0)
        _LAST_HARVEST_TIME = now
        _HONEST_JITTER_CACHE['seed'] = seed_val
    else:
        seed_val = _HONEST_JITTER_CACHE.get('seed', 0.5)

    # Agent Smith Expansion
    jitter_tensor = _AGENT_SMITH_ENGINE(shape, seed_val, scaled=scaled)
    
    # Update cache for all shapes to prevent redundant generation
    if len(_HONEST_JITTER_CACHE) > 128:
        # Keep 'seed' and clear other elements if cache gets too large
        seed_val = _HONEST_JITTER_CACHE.get('seed', 0.5)
        _HONEST_JITTER_CACHE.clear()
        _HONEST_JITTER_CACHE['seed'] = seed_val
    _HONEST_JITTER_CACHE[shape] = jitter_tensor.clone().detach()

    return jitter_tensor

def honest_multinomial(probs: torch.Tensor, num_samples: int, replacement: bool = False) -> torch.Tensor:
    """
    Select indices from a probability distribution using 'Honest' entropy.
    
    Replaces topologically semisimple PRNGs with hardware-anchored selection, 
    preventing the 'stochastic trap' where software-generated randomness 
    introduces unwanted symmetries or biases.
    
    Args:
        probs: The input probability distribution [N].
        num_samples: The number of indices to sample.
        replacement: Whether to sample with replacement.
        
    Returns:
        A tensor of sampled indices [num_samples].
    """
    device = probs.device
    n = probs.size(0)
    
    if replacement:
        # Sample with replacement: cumulative sum + jitter comparison
        jitter = harvest_honest_jitter((num_samples,), device=device, scaled=False) # [0, 1] range
        jitter = (jitter + 1.0) / 2.0 # Force to [0, 1]
        
        cum_probs = torch.cumsum(probs, dim=0)
        # Find indices where jitter < cum_probs
        indices = torch.searchsorted(cum_probs, jitter)
        return torch.clamp(indices, 0, n - 1)
    else:
        # Sample without replacement: perturbation + sort
        # E_i = log(P_i) - log(-log(U_i)) where U_i is uniform jitter
        # But we can just use jitter to perturb the probabilities
        jitter = harvest_honest_jitter((n,), device=device, scaled=False)
        jitter = (jitter + 1.0) / 2.0 # [0, 1]
        
        # Gumbel-Max trick replacement: log(probs) + noise
        noise = -torch.log(-torch.log(jitter.clamp(min=1e-8)))
        scores = torch.log(probs.clamp(min=1e-8)) + noise
        
        _, indices = torch.topk(scores, min(num_samples, n))
        return indices

def fractal_pad(x: torch.Tensor, target_dim: int, mode: str = 'asymmetry_preserving') -> torch.Tensor:
    """
    Perform Fractal (Asymmetry-Preserving) Padding to align dimensions.
    
    Aligns heterogeneous tensors while ensuring that boundary conditions 
    preserve the chiral asymmetry required for lawful resonance. 'Reflect' 
    mode is explicitly deprecated in favor of 'Asymmetry Preserving' 
    to prevent phase-cancellation lobotomy.
    
    Args:
        x: The input tensor [..., current_dim].
        target_dim: The desired output dimensionality.
        mode: The padding mode. 'asymmetry_preserving' uses prime-seeded 
              padding for non-teleological stability.
              
    Returns:
        The padded or truncated tensor [..., target_dim].
    """
    current_dim = x.shape[-1]
    if current_dim == target_dim:
        return x
        
    if current_dim > target_dim:
        # Matryoshka Truncation: Shell-pop inward
        return x[..., :target_dim]
        
    # Redirect reflect to asymmetry_preserving to enforce chirality
    if mode == 'reflect':
        mode = 'asymmetry_preserving'

    if mode == 'asymmetry_preserving':
        from src.core.invariants import apply_asymmetry_preserving_reshape
        # Handle high-dimensional tensors by flattening to [B, D] and reshaping back
        orig_shape = x.shape
        if x.dim() > 1:
            x_flat = x.reshape(-1, current_dim)
            padded_flat = apply_asymmetry_preserving_reshape(x_flat, target_dim)
            new_shape = list(orig_shape[:-1]) + [target_dim]
            return padded_flat.reshape(*new_shape)
        else:
            return apply_asymmetry_preserving_reshape(x.unsqueeze(0), target_dim).squeeze(0)

    # Fractal Expansion: Pad to target
    diff = target_dim - current_dim
    
    # torch.nn.functional.pad expects padding as (padding_left, padding_right) for last dim
    # For 1D tensors (Spectral Residues), we force 'constant' to prevent crash.
    if x.dim() <= 2 and mode in ['replicate']:
        mode = 'constant'
        
    return torch.nn.functional.pad(x, (0, diff), mode=mode)
