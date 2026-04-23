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
    seed_val = ((t1 - t0) % 1000) / 1000.0
    if seed_val == 0: seed_val = 0.5
    
    # Deterministic chaotic expansion (Logistic map)
    # x_{n+1} = 3.99 * x_n * (1 - x_n) -- chaotic regime
    x = seed_val
    for i in range(len(flat)):
        x = 3.99 * x * (1.0 - x)
        flat[i] = x
    
    if scaled:
        return (jitter_tensor - 0.5) * 0.1
    return jitter_tensor
