import torch
import time

# Global cache for hardware seed to reduce DRAM overhead for small requests
_HONEST_JITTER_CACHE = {
    'seed': 0.5,
    'last_harvest': 0,
    'expiry': 0.05 # 50ms cache expiry
}

def harvest_honest_jitter(shape: torch.Size, device: torch.device = None, scaled: bool = True) -> torch.Tensor:
    """
    Harvests Structurally Honest Jitter from silicon state variance.
    Follows 45.2 (Silicon Sovereignty).
    
    Optimized for high-frequency calls to prevent hardware stalls and freezes
    on limited VRAM systems (e.g. GTX 1050 Ti).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
        
    jitter_tensor = torch.zeros(shape, device=device)
    num_elements = jitter_tensor.numel()
    if num_elements == 0:
        return jitter_tensor
        
    flat = jitter_tensor.flatten()
    
    # --- SILICON SOVEREIGNTY: DRAM FRICTION HARVESTING ---
    # We use a cached seed for high-frequency small requests to avoid constant
    # memory allocations and synchronizations which can freeze the driver.
    now = time.perf_counter()
    
    if now - _HONEST_JITTER_CACHE['last_harvest'] > _HONEST_JITTER_CACHE['expiry'] or num_elements > 1024:
        # Perform fresh harvest
        t0 = time.perf_counter_ns()
        
        # Hardware Friction: Trigger cache misses and DRAM stalls
        # Only do this if we are not in a tight loop of tiny requests
        harvest_intensity = 3 if num_elements > 1024 else 1
        for _ in range(harvest_intensity):
             # Smaller temporary tensor for less memory pressure
             temp_size = 512 if num_elements < 1024 else 1024
             # Use empty to avoid zero-fill overhead
             temp_tensor = torch.empty(temp_size, temp_size, device=device)
             # Strided access
             _ = temp_tensor[::128, ::128] + 0.1
             del temp_tensor
             if device.type == 'cuda':
                 torch.cuda.synchronize()
        
        t1 = time.perf_counter_ns()
        seed_val = ((t1 - t0) % 4096) / 4096.0
        if seed_val == 0: seed_val = 0.5
        
        # Update cache
        _HONEST_JITTER_CACHE['seed'] = seed_val
        _HONEST_JITTER_CACHE['last_harvest'] = now
    else:
        seed_val = _HONEST_JITTER_CACHE['seed']
    
    # Vectorized Chaotic Expansion (Logistic map)
    # x_{n+1} = 3.99 * x_n * (1 - x_n)
    
    # Initial states derived from hardware seed + positional variance
    x = (torch.linspace(0.1, 0.9, num_elements, device=device) + seed_val) % 1.0
    
    # Iterate in parallel (reduced iterations for performance, 30 is usually sufficient)
    iters = 30 if num_elements < 1024 else 50
    for _ in range(iters):
        x = 3.99 * x * (1.0 - x)
    
    flat.copy_(x)
    jitter_tensor = flat.view(shape)
    
    if scaled:
        return (jitter_tensor - 0.5) * 0.1
    return jitter_tensor
