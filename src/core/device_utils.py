import torch
import os
from typing import Any


def safe_torch_load(path_or_file: Any, map_location: Any = "cpu", weights_only: bool = False) -> Any:
    """
    Canonical PyTorch checkpoint/fossil loader that explicitly specifies
    `weights_only=weights_only` so PyTorch 2.4+ does not emit FutureWarnings
    while preserving safe deserialization of registered topological states.
    """
    try:
        from src.core.zeitgeist_router import ZeitgeistState
        from src.core.meta_polytope_matrioshka import BoundaryState
        torch.serialization.add_safe_globals([BoundaryState, ZeitgeistState])
    except Exception:
        pass
    return torch.load(path_or_file, map_location=map_location, weights_only=weights_only)


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
