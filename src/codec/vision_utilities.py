import torch
import torch.nn.functional as F

def get_russian_doll_projection(coeffs: torch.Tensor, k_image_max: int = 32) -> torch.Tensor:
    """
    Extracts nested 'Russian Doll' Chebyshev projections from raw coefficients.
    
    Args:
        coeffs: Raw coefficients [3 * k_image_max] (L, Cr, Cb)
        k_image_max: Max degree per channel (default 32)
        
    Returns:
        composite_residue: Normalized average of Degrees 8, 16, and 32 projections.
    """
    target = k_image_max * 3
    if coeffs.numel() < target:
        coeffs = F.pad(coeffs, (0, target - coeffs.numel()))
    else:
        coeffs = coeffs[:target]
        
    # Degree 8 projection (Outer Shell)
    deg8_mask = torch.zeros_like(coeffs)
    deg8_indices = [i for i in range(target) if (i % k_image_max) < 8]
    deg8_mask[deg8_indices] = 1.0
    p8 = (coeffs * deg8_mask)
    
    # Degree 16 projection (Middle Shell)
    deg16_mask = torch.zeros_like(coeffs)
    deg16_indices = [i for i in range(target) if (i % k_image_max) < 16]
    deg16_mask[deg16_indices] = 1.0
    p16 = (coeffs * deg16_mask)
    
    # Degree 32 projection (Inner Shell) - uses full coefficients
    p32 = coeffs
    
    # Composite Russian Doll residue (raw coefficient space)
    # This represents the interlaced structural signal
    return (p8 + p16 + p32) / 3.0
