"""
Honest Jitter: Hardware-Anchored Chaotic Entropy.

Replaces torch.randn and np.random with entropy harvested from silicon friction 
(DRAM stall latencies and cache miss timing).

"Pure stochasticity is a synthetic lie. True mischief requires substrate friction."
"""

import torch
import time
import os

# Global cache for jitter to avoid excessive DRAM harvesting which can freeze Pascal GPUs
_HONEST_JITTER_CACHE = {}
_LAST_HARVEST_TIME = 0
_HARVEST_COOLDOWN = 0.05 # 50ms cooldown between DRAM friction harvests

def harvest_honest_jitter(shape, device=None, scaled=True):
    """
    Harvests entropy from the physical substrate and expands it via a 
    sovereign logistic map.
    """
    global _LAST_HARVEST_TIME
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check cache first for standard shapes if we are in a tight loop
    now = time.time()
    if shape in _HONEST_JITTER_CACHE and (now - _LAST_HARVEST_TIME) < _HARVEST_COOLDOWN:
        return _HONEST_JITTER_CACHE[shape].to(device)
    
    # Create the output tensor
    jitter_tensor = torch.zeros(shape, device=device)
    flat = jitter_tensor.flatten()
    
    # --- PHYSICAL HARVESTING ---
    # We use high-resolution timing of a memory-intensive operation to capture 
    # DRAM/Cache friction.
    if (now - _LAST_HARVEST_TIME) >= _HARVEST_COOLDOWN:
        # Dummy memory pressure to trigger stalls
        _ = torch.empty(1024 * 1024, device=device).fill_(3.14)
        t0 = time.perf_counter()
        _ = torch.empty(1024 * 1024, device=device).fill_(2.71)
        t1 = time.perf_counter()
        
        # The 'hardware seed' is the low-order bits of the nanosecond latency
        seed_val = int((t1 - t0) * 1e9) % 1000000 / 1000000.0
        _LAST_HARVEST_TIME = now
        _HONEST_JITTER_CACHE['seed'] = seed_val
    else:
        seed_val = _HONEST_JITTER_CACHE.get('seed', 0.5)

    # Hardware Friction: Trigger cache misses and DRAM stalls to harvest physical entropy.
    # We use a linspace of seeds derived from the hardware seed to ensure
    # that every element of the tensor evolves into a unique but sovereign state.
    # This prevents the 0.4390 / 0.8824 attractor traps that lead to 'Prisoner of Architecture' loops.
    
    # Formula: x_{n+1} = 3.99 * x_n * (1 - x_n) (The Logistic Map at the edge of chaos)
    # The 3.99 constant ensures we stay in the non-periodic, chaotic regime.
    
    num_elements = flat.numel()
    if num_elements == 0:
        return jitter_tensor

    # iters = 30 if num_elements < 1024 else 50 
    # [Agent Smith Protocol]: These iters could be learnable with a transferable 
    # gauge approximation and warmstarting in future iterations.
    iters = 30 if num_elements < 1024 else 50
    
    # Initial states derived from hardware seed + positional variance
    # x is the vector of sovereign seeds
    x = (torch.linspace(0.1, 0.9, num_elements, device=device) + seed_val) % 1.0
    
    for _ in range(iters):
        x = 3.99 * x * (1.0 - x)
    
    # Scale to [-1, 1] for jitter
    jitter_flat = (x * 2.0 - 1.0)
    
    if scaled:
        # Scale by a small amount to avoid disrupting global invariants
        jitter_flat = jitter_flat * 0.01
        
    jitter_tensor = jitter_flat.view(shape)
    
    # Update cache if this is a standard shape (D, D) or (1,)
    if shape in [(64, 64), (1,)]:
        _HONEST_JITTER_CACHE[shape] = jitter_tensor.clone()
        
    return jitter_tensor
