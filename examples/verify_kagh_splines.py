
import torch
import torch.nn as nn
from src.surrogates.kagh_networks import KANLayer, SaturatedQuantizer

def verify_b_spline_properties():
    print("=== Verifying Hybrid-Quantized B-Splines ===")
    
    # 1. Setup Layer
    in_dim = 4
    out_dim = 8
    grid_size = 5
    spline_order = 3
    layer = KANLayer(in_dim, out_dim, grid_size, spline_order, quantization_levels=16)
    
    # 2. Test Partition of Unity (B-splines should sum to 1 in the defined range)
    print("\n[Test 1] Partition of Unity")
    x = torch.linspace(-0.5, 0.5, 100).unsqueeze(1).repeat(1, in_dim) # [100, 4]
    
    # KANLayer uses Tanh internal normalization? No, we removed x_norm = torch.tanh(x) 
    # to rely on external normalization or assume inputs are in range. 
    # Actually, looking at the code I replaced, I *removed* the tanh line:
    # "With fixed grid, we rely on upstream normalization (e.g. Tanh or LayerNorm)"
    # So input x should be in range.
    
    basis = layer.b_splines(x) # [100, in, coeffs]
    basis_sum = basis.sum(dim=-1) # [100, in]
    
    # Note: B-splines sum to 1 only strictly within [-1 + delta, 1 - delta] 
    # depending on order and grid. Our grid is -1 to 1.
    # Check center behavior.
    center_variance = basis_sum.var().item()
    center_mean = basis_sum.mean().item()
    print(f"Basis Sum Mean (should be ~1.0): {center_mean:.4f}")
    print(f"Basis Sum Variance: {center_variance:.6f}")
    
    if abs(center_mean - 1.0) < 0.1:
        print("✅ Partition of Unity holds approximately.")
    else:
        print("⚠️ Partition of Unity deviation (expected at boundaries).")

    # 3. Test Quantization
    print("\n[Test 2] Saturated Quantization")
    # Forward pass uses quantized weights
    original_weights = layer.spline_weight.detach().clone()
    
    # Force some gradients
    x = torch.randn(10, in_dim)
    y = layer(x)
    loss = y.mean()
    loss.backward()
    
    # Check that gradients exist
    if layer.spline_weight.grad is not None:
        print("✅ Gradients flow through SaturatedQuantizer (STE working).")
    else:
        print("❌ No gradients on spline_weight!")
        
    # Check if effective weights used were quantized?
    # We can invoke the quantizer manually to check logic
    q_weights = SaturatedQuantizer.apply(original_weights, 16)
    unique_vals = torch.unique(q_weights)
    print(f"Number of unique weight values (Levels=16): {len(unique_vals)}")
    print(f"Sample values: {unique_vals[:5]}")
    
    # 4. Forward Pass Consistency
    print("\n[Test 3] Forward Pass")
    y = layer(x)
    print(f"Output shape: {y.shape}")
    print("✅ Forward pass successful.")

if __name__ == "__main__":
    verify_b_spline_properties()
