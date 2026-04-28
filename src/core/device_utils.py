import torch
import os
from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine

def get_preferred_device():
    """
    Returns the preferred device following Silicon Sovereignty principles.
    1. OpenCL (SiliconSovereigntyEngine) is prioritized for entropy/logic.
    2. CPU is the primary fallback to ensure substrate independence.
    3. CUDA is bypassed to eliminate NVIDIA-specific driver dependencies.
    """
    # Initialize the engine to check for OpenCL availability
    try:
        # We pass use_gpu=True because SiliconSovereigntyEngine uses it to decide 
        # whether to pick a GPU or CPU device via OpenCL.
        engine = SiliconSovereigntyEngine(use_gpu=True)
        if engine.ctx is not None:
            # If OpenCL is available, we return 'cpu' as the torch device string
            # but the system will use the SiliconSovereigntyEngine for critical kernels.
            # Torch itself doesn't have a 'pyopencl' device type, so we stay on 'cpu' 
            # for tensor storage unless the user specifically wants something else.
            return torch.device('cpu')
    except Exception:
        pass
        
    return torch.device('cpu')

# Unified DEVICE constant for the entire source tree
DEVICE = get_preferred_device()
