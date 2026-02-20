#!/usr/bin/env python3
"""
Debug the fingerprint generation and reconstruction.
"""

import torch
import numpy as np

def debug_fingerprint():
    """Debug fingerprint values."""
    print("🔍 Debugging Fingerprint Generation")
    print("=" * 40)
    
    # Create a red square fingerprint like in the test
    fingerprint = torch.zeros(137)
    
    # Red histogram with strong red signal
    red_values = [3.0] + [0.1] * 31
    fingerprint[:32] = torch.softmax(torch.tensor(red_values), dim=0)
    
    # Green and blue histograms (low values)
    fingerprint[32:64] = torch.softmax(torch.tensor([0.1] * 32), dim=0)
    fingerprint[64:96] = torch.softmax(torch.tensor([0.1] * 32), dim=0)
    
    # Luminance
    fingerprint[96:128] = torch.softmax(torch.tensor([1.0] * 32), dim=0)
    
    # Texture and edges
    fingerprint[128] = 0.8
    fingerprint[129:137] = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.1, 0.7, 0.1, 0.0])
    
    print(f"Fingerprint shape: {fingerprint.shape}")
    print(f"Red histogram sum: {fingerprint[:32].sum():.6f}")
    print(f"Red histogram max: {fingerprint[:32].max():.6f}")
    print(f"Red histogram values (first 5): {fingerprint[:5].tolist()}")
    
    # Test color calculation
    fingerprint_np = fingerprint.cpu().numpy()
    r_hist = fingerprint_np[:32]
    g_hist = fingerprint_np[32:64]
    b_hist = fingerprint_np[64:96]
    
    bin_centers = np.linspace(0, 1, 32)
    r_color = np.sum(r_hist * bin_centers)
    g_color = np.sum(g_hist * bin_centers)
    b_color = np.sum(b_hist * bin_centers)
    
    print(f"\nColor calculation:")
    print(f"  R color: {r_color:.6f}")
    print(f"  G color: {g_color:.6f}")
    print(f"  B color: {b_color:.6f}")
    
    # Check if red should be dominant
    print(f"\nExpected: Red should be dominant")
    print(f"Actual: R > G? {r_color > g_color}, R > B? {r_color > b_color}")
    
    # Test with different approach - use argmax
    r_peak = np.argmax(r_hist) / 32.0
    g_peak = np.argmax(g_hist) / 32.0
    b_peak = np.argmax(b_hist) / 32.0
    
    print(f"\nArgmax approach:")
    print(f"  R peak: {r_peak:.6f}")
    print(f"  G peak: {g_peak:.6f}")
    print(f"  B peak: {b_peak:.6f}")
    
    # Test with raw values (before softmax)
    print(f"\nRaw values before softmax:")
    raw_red = [3.0] + [0.1] * 31
    raw_green = [0.1] * 32
    raw_blue = [0.1] * 32
    
    print(f"  Raw red max: {max(raw_red)}")
    print(f"  Raw green max: {max(raw_green)}")
    print(f"  Raw blue max: {max(raw_blue)}")
    
    # Test better color calculation
    print(f"\nBetter color calculation:")
    # Use the position of the maximum value, scaled properly
    r_color_better = (np.argmax(r_hist) / 31.0) * 0.8 + 0.2  # Scale to [0.2, 1.0]
    g_color_better = (np.argmax(g_hist) / 31.0) * 0.8 + 0.2
    b_color_better = (np.argmax(b_hist) / 31.0) * 0.8 + 0.2
    
    print(f"  Better R: {r_color_better:.6f}")
    print(f"  Better G: {g_color_better:.6f}")
    print(f"  Better B: {b_color_better:.6f}")

if __name__ == "__main__":
    debug_fingerprint()