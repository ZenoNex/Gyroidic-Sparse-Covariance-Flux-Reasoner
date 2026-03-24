"""
Conformal Log-Polar Projector.

Implements the $f(z) = \\log(z)$ foveal unrolling inspired by M.C. Escher's Print Gallery
and mammalian retinal mapping.

Applying this complex conformal map to a continuous Euclidean image transforms:
1. Scaling (Zooming in/out) -> Horizontal Translation in log space
2. Rotation (Spinning) -> Vertical Translation in log space

By feeding this Log-Polar representation into the Gyroidic Codec, the Reasoner 
achieves strict Scale and Rotation Invariance organically, as the operations 
merely slide the topological invariants along the manifold.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class ConformalLogPolarProjector(nn.Module):
    """
    Applies the Conformal Log-Polar wrap to Euclidean image tensors.
    """
    def __init__(self, resolution_r: int = 64, resolution_theta: int = 64):
        """
        Args:
            resolution_r: Resolution along the logarithmic radial axis (horizontal log-space)
            resolution_theta: Resolution along the angular axis (vertical log-space)
        """
        super().__init__()
        self.res_r = resolution_r
        self.res_theta = resolution_theta

    def forward(self, image: torch.Tensor, center: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Unrolls a Cartesian image [C, H, W] or [H, W] into its Log-Polar representation.

        Args:
            image: Input spatial tensor
            center: Optional (cy, cx) focal point. Defaults to image center.

        Returns:
            log_polar_image: [C, res_r, res_theta] (or [res_r, res_theta])
        """
        is_2d = image.dim() == 2
        if is_2d:
            image = image.unsqueeze(0)  # [1, H, W]

        C, H, W = image.shape
        device = image.device

        # Determine geometric center
        cy = center[0] if center is not None else (H - 1) / 2.0
        cx = center[1] if center is not None else (W - 1) / 2.0

        # Max radius to cover the image
        max_r = math.sqrt(cy**2 + cx**2)

        # Create Log-polar grid
        # log_r ranges from a small value (to avoid log(0)) up to log(max_r)
        min_log_r = math.log(1.0) # Assume center 1 pixel is the singularity hole
        max_log_r = math.log(max_r)
        
        log_r = torch.linspace(min_log_r, max_log_r, self.res_r, device=device)
        theta = torch.linspace(0, 2 * math.pi, self.res_theta, device=device)

        # Meshgrid in log-polar space
        log_r_grid, theta_grid = torch.meshgrid(log_r, theta, indexing='ij')

        # Convert back to Cartesian for grid_sample mapping
        # r = e^(log_r)
        r_grid = torch.exp(log_r_grid)
        x_grid = cx + r_grid * torch.cos(theta_grid)
        y_grid = cy + r_grid * torch.sin(theta_grid)

        # Normalize grid for F.grid_sample [-1, 1]
        x_norm = (x_grid / (W - 1)) * 2 - 1
        y_norm = (y_grid / (H - 1)) * 2 - 1
        
        # [batch=1, H_out, W_out, 2]
        sample_grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)

        # Sample the image
        img_batch = image.unsqueeze(0) # [1, C, H, W]
        unrolled = torch.nn.functional.grid_sample(
            img_batch, 
            sample_grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        ).squeeze(0) # [C, res_r, res_theta]

        if is_2d:
            unrolled = unrolled.squeeze(0)

        return unrolled
