import torch
import os
from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine

def get_preferred_device():
    """
    Returns the preferred device following Silicon Sovereignty principles.
    1. CPU is the primary fallback to ensure substrate independence.
    2. CUDA is bypassed to eliminate NVIDIA-specific driver dependencies.
    """
    return torch.device('cpu')

# Unified DEVICE constant for the entire source tree
DEVICE = get_preferred_device()
