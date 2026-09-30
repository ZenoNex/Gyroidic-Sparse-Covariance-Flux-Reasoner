import torch
import os

def get_torch_device(target_hardware: str) -> str:
    """
    Prevents PyTorch from triggering the Caffe2 legacy device panic.
    Returns 'cpu' as the host staging ground for OpenCL targets.
    """
    if target_hardware.lower() == "opencl":
        return "cpu"
    return target_hardware

def get_preferred_device():
    """
    Returns the preferred device following Silicon Sovereignty principles.
    1. Attempts to use PyOpenCL (TailSlayer architecture) via SiliconSovereigntyEngine.
    2. CPU is the primary fallback to ensure substrate independence.
    3. CUDA is bypassed to eliminate NVIDIA-specific driver dependencies.
    """
    try:
        from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
        # Test if the engine can be instantiated
        _engine = SiliconSovereigntyEngine()
        return "opencl"
    except Exception as e:
        return "cpu"

# Unified DEVICE constant for the entire source tree, routed safely for PyTorch
DEVICE = torch.device(get_torch_device(get_preferred_device()))
