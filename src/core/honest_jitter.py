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
        
    jitter_tensor = torch.zeros(shape, device=device)
    flat = jitter_tensor.flatten()
    
    # Warm up cache and measure nano-variance friction
    t0 = time.perf_counter_ns()
    # Small matrix ops to generate hardware friction
    # We use a deterministic but heavy op here
    for _ in range(5):
         # Deterministic matrix multiplication to generate heat/friction
         a = torch.ones((8, 8), device=device) * 0.5
         _ = torch.mm(a, a)
    t1 = time.perf_counter_ns()
    
    # Harvest the 'least significant nanoseconds' as a seed val
    # This is the physical "anchor"
    seed_val = ((t1 - t0) % 1000) / 1000.0
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
