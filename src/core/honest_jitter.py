import torch
import time

def harvest_honest_jitter(shape: torch.Size, device: torch.device = None, scaled: bool = True) -> torch.Tensor:
    """
    Harvests Structurally Honest Jitter from silicon state variance.
    Follows §45.2 (Silicon Sovereignty).
    
    This is a standalone version of the DiegeticPhysicsEngine method.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
        
    jitter_tensor = torch.zeros(shape, device=device)
    flat = jitter_tensor.flatten()
    
    # --- SILICON SOVEREIGNTY: DRAM FRICTION HARVESTING ---
    # We perform memory-intensive operations to maximize the impact of DRAM stalls 
    # (t_RFC, t_CAS, etc.) on the timing measurement.
    t0 = time.perf_counter_ns()
    
    # 1. Hardware Friction: Large-stride memory access to trigger cache misses and DRAM stalls
    # We use a deterministic but memory-heavy pattern.
    for _ in range(3):
         # Non-contiguous access to bypass CPU prefetchers
         temp_tensor = torch.zeros(1024, 1024, device=device)
         _ = temp_tensor[::128, ::128] + 0.1
         del temp_tensor
         torch.cuda.synchronize() if device.type == 'cuda' else None
         
    # 2. Silicon-Native Jitter: Harvest the nanosecond-level variance
    t1 = time.perf_counter_ns()
    
    # Formula: Delta_t incorporates t_RFC (Refresh Cycle) and DRAM queuing stalls.
    # This value is unique to the physical substrate at this exact moment.
    seed_val = ((t1 - t0) % 4096) / 4096.0
    if seed_val == 0: seed_val = 0.5
    
    # Vectorized Chaotic Expansion (Logistic map)
    # We use a linspace of seeds derived from the hardware seed to ensure
    # that every element of the tensor evolves into a unique but sovereign state.
    # Formula: x_{n+1} = 3.99 * x_n * (1 - x_n)
    
    # Create a vector of seeds
    num_elements = flat.numel()
    if num_elements == 0:
        return jitter_tensor
        
    # Initial states derived from hardware seed + positional variance
    # This ensures that even for large tensors, we don't just have one seed.
    x = (torch.linspace(0.1, 0.9, num_elements, device=device) + seed_val) % 1.0
    
    # Iterate in parallel to reach chaotic regime (50 iterations is typically enough)
    for _ in range(50):
        x = 3.99 * x * (1.0 - x)
    
    flat.copy_(x)
    jitter_tensor = flat.view(shape)
    
    if scaled:
        return (jitter_tensor - 0.5) * 0.1
    return jitter_tensor
