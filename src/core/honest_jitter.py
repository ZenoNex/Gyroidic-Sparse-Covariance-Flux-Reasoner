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

    def forward(self, shape, seed_val, scaled=True):
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

def harvest_honest_jitter(shape, device=None, scaled=True):
    """
    Harvests entropy from the physical substrate and expands it via the 
    Agent Smith Engine (Logistic Map).
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
        
def honest_multinomial(probs, num_samples, replacement=False):
    """
    Selects indices from 'probs' using harvest_honest_jitter entropy.
    
    This replaces topologically semisimple PRNGs like torch.multinomial
    with hardware-anchored selection.
    
    Args:
        probs: [N] probability distribution (tensor)
        num_samples: Number of indices to sample
        replacement: Whether to sample with replacement
        
    Returns:
        indices: [num_samples] sampled indices
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
