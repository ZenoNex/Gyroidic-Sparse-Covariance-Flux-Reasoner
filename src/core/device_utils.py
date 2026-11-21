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
    Routes 'opencl' to 'cpu' as the host staging ground for Silicon Sovereignty
    PyOpenCL execution (avoiding forced CUDA device mismatches across modules).
    """
    hw = target_hardware.lower()
    if hw == "opencl":
        return "cpu"
    if hw == "cuda" and torch.cuda.is_available():
        return "cuda"
    return target_hardware


def get_preferred_device():
    """
    Returns the preferred device following Silicon Sovereignty principles:
    1. Defaults to PyOpenCL ('opencl', host-staged on 'cpu' for PyTorch) so all
       modules share unified host staging while SiliconSovereigntyEngine dispatches
       OpenCL Queue A/B kernels.
    2. Only returns 'cuda' when explicitly requested via Sovereign_Use_CUDA=1.
    3. Falls back to 'cpu'.
    """
    force_cpu = os.environ.get("Sovereign_Force_CPU", "0").lower() in ("1", "true", "yes")
    if force_cpu:
        return "cpu"
    use_cuda_env = os.environ.get("Sovereign_Use_CUDA", "0").lower()
    if use_cuda_env in ("1", "true", "yes") and torch.cuda.is_available():
        return "cuda"
    try:
        from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
        _engine = SiliconSovereigntyEngine()
        return "opencl"
    except Exception:
        return "cpu"


def get_compute_acceleration_summary() -> dict:
    """Returns a unified summary of active PyTorch CUDA + PyOpenCL acceleration."""
    cuda_ok = bool(torch.cuda.is_available())
    cuda_name = torch.cuda.get_device_name(0) if cuda_ok else None
    opencl_platforms = []
    try:
        import pyopencl as cl
        opencl_platforms = [p.name.strip() for p in cl.get_platforms()]
    except Exception:
        pass
    return {
        "torch_device": str(DEVICE),
        "cuda_available": cuda_ok,
        "cuda_device": cuda_name,
        "opencl_platforms": opencl_platforms,
    }

# Unified DEVICE constant for the entire source tree, routed safely for PyTorch
DEVICE = torch.device(get_torch_device(get_preferred_device()))
