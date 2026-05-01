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
        super().__init__()
        # Internal state for learnable expansion
        self.register_buffer('iters_base_small', torch.tensor(30.0, device=device))
        self.register_buffer('iters_base_large', torch.tensor(50.0, device=device))
        
        # Learnable parameters (Agent Smith Protocol)
        self.gauge = nn.Parameter(torch.tensor(3.99, device=device))
        self.iter_multiplier = nn.Parameter(torch.tensor(1.0, device=device))
        
        # Warmstart state tracking (persistent across calls)
        self.warmstart_states = {}

    def forward(self, shape: torch.Size, seed_val: float, scaled: bool = True) -> torch.Tensor:
        """
        Execute the Agent Smith Entropy Expansion.
        
        Using the Logistic Map (x_{n+1} = gauge * x_n * (1 - x_n)), this method 
        takes a physical seed value and expands it into a high-dimensional 
        entropy field that is deterministic for a given seed but has high 
        structural complexity.
        
        Args:
            shape: The desired output shape for the jitter tensor.
            seed_val: The physical seed value [0, 1] harvested from 
                      substrate friction.
            scaled: If True, scales the jitter to [-0.01, 0.01] to minimize 
                    invariant disruption.
                    
        Returns:
            A tensor of the specified shape containing 'Honest Jitter'.
            
        CODES v40 Invariant: 
            Substrate Sovereignty: 1.1. Entropy must be grounded in physical 
            timing variance, not pseudorandom algorithms.
        """
        shape = tuple(shape) # Normalize to tuple for consistent dict keys
        device = self.gauge.device
        num_elements = torch.Size(shape).numel()
        
        # Check for warmstart state compatibility
        if shape in self.warmstart_states and self.warmstart_states[shape].device == device:
            x = self.warmstart_states[shape]
            # Inject tiny hardware seed variance to prevent attractor lock-in
            x = (x + seed_val * 1e-4) % 1.0
        else:
            # Initialize from linspace + hardware seed
            x = (torch.linspace(0.1, 0.9, num_elements, device=device) + seed_val) % 1.0
            
        # Determine iteration count (learnable)
        base = self.iters_base_small if num_elements < 1024 else self.iters_base_large
        iters = int(torch.clamp(base * self.iter_multiplier, 10, 150).item())
        
        # Logistic Map Expansion: x_{n+1} = gauge * x_n * (1 - x_n)
        for _ in range(iters):
            x = self.gauge * x * (1.0 - x)
            
        # Update warmstart state (detach to prevent graph growth)
        self.warmstart_states[shape] = x.detach()
        
        # Scale to [-1, 1] for jitter
        jitter_flat = (x * 2.0 - 1.0)
        
        if scaled:
            # Scale by a small amount to avoid disrupting global invariants
            jitter_flat = jitter_flat * 0.01
            
        return jitter_flat.view(shape)

# Global singleton for Agent Smith Engine (initialized on first call)
_AGENT_SMITH_ENGINE = None

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
    global _LAST_HARVEST_TIME, _AGENT_SMITH_ENGINE
    
    if device is None:
        device = DEVICE
    
    # Lazy initialization of the expansion engine
    if _AGENT_SMITH_ENGINE is None:
        _AGENT_SMITH_ENGINE = AgentSmithEngine(device=device)
    
    # Check cache first for standard shapes if we are in a tight loop
    now = time.time()
    if shape in _HONEST_JITTER_CACHE and (now - _LAST_HARVEST_TIME) < _HARVEST_COOLDOWN:
        return _HONEST_JITTER_CACHE[shape].to(device)
    
    # --- PHYSICAL HARVESTING ---
    if (now - _LAST_HARVEST_TIME) >= _HARVEST_COOLDOWN:
        # Dummy memory pressure to trigger stalls
        _ = torch.empty(1024 * 1024, device=device).fill_(3.14)
        t0 = time.perf_counter()
        _ = torch.empty(1024 * 1024, device=device).fill_(2.71)
        t1 = time.perf_counter()
        
        # The 'hardware seed' is derived from nanosecond latency variance
        seed_val = float(int((t1 - t0) * 1e9) % 1000000 / 1000000.0)
        _LAST_HARVEST_TIME = now
        _HONEST_JITTER_CACHE['seed'] = seed_val
    else:
        seed_val = _HONEST_JITTER_CACHE.get('seed', 0.5)

    # Agent Smith Expansion
    jitter_tensor = _AGENT_SMITH_ENGINE(shape, seed_val, scaled=scaled)
    
    # Update cache for frequently used shapes
    if shape in [(64, 64), (1,)]:
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
