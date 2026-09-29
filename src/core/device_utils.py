import torch
import os

def get_preferred_device():
    """
    Returns the preferred device following Silicon Sovereignty principles.
    1. Attempts to use PyOpenCL (TailSlayer architecture) via SiliconSovereigntyEngine.
    2. CPU is the primary fallback to ensure substrate independence.
    3. CUDA is bypassed to eliminate NVIDIA-specific driver dependencies.
    """
    try:
        from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
        # Test if the engine can be instantiated (will fail if PyOpenCL is missing or no device)
        _engine = SiliconSovereigntyEngine()
        return 'opencl'
    except Exception as e:
        return torch.device('cpu')

# Unified DEVICE constant for the entire source tree
DEVICE = get_preferred_device()
