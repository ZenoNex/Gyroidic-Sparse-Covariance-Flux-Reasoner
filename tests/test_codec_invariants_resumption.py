
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.codec.gyroidic_codec import GyroidicCodec, CodecConfig
from src.core.veto_subspace import VetoSubspace, RecoveryStatus

def test_non_abelian_gap():
    print("\n--- Testing Non-Abelian Gap ---")
    codec = GyroidicCodec(CodecConfig(K=3, n=4))
    text = "Sovereign"
    image = torch.randn(32, 32)
    
    # Text then Image (Order sensitive)
    res_ti = codec.encode(text, image, commutativity='text_first')
    # Image then Text
    res_it = codec.encode(text, image, commutativity='media_first')
    
    gap = torch.norm(res_ti.encoded - res_it.encoded).item()
    print(f"Commutativity Gap (bmm-based): {gap:.6f}")
    assert gap > 1e-6, "Gap must be non-zero for non-abelian encoding"
    print("SUCCESS: Non-Abelian Gap verified.")

def test_fixed_point_crt():
    print("\n--- Testing Fixed-Point CRT ---")
    codec = GyroidicCodec(CodecConfig(K=3, n=4))
    text = "Algebraic"
    
    # Text Residues (K, n, n)
    res_exp = codec.encode(text)
    
    # Check that residues are in GL(n) (determinant != 0)
    for k in range(3):
        det = torch.det(res_exp.text_residues[k]).item()
        print(f"Channel {k} det: {det:.6f}")
        assert abs(det) > 1e-6, "Matrix exp should guarantee GL(n)"

    # Test Fixed-point mode in CRT
    target_channels = res_exp.text_residues
    # Reshape for CRT (B, K, D) - here B=1, K=3, D=n*n=16
    reshaped = target_channels.view(1, 3, 16)
    
    recon_fp = codec.crt_bridge.crt.forward(reshaped, mode='fixed_point')
    
    print(f"Fixed-Point Reconstruction Shape: {recon_fp.shape}")
    assert recon_fp.shape == (1, 16), "Shape mismatch"
    print("SUCCESS: Fixed-Point CRT executed.")

def test_caq_shells():
    print("\n--- Testing Canonical CAQ (src.core) ---")
    codec = GyroidicCodec(CodecConfig(K=2, n=4))
    
    # Test CAQ directly
    x = torch.randn(1, 16)
    q_out, boundary = codec.text_projector.caq(x)
    
    print(f"CAQ output shape: {q_out.shape}")
    assert q_out.shape == (1, 16)
    
    # Verify quantization effect
    err = torch.norm(x - q_out).item()
    print(f"Quantization Error: {err:.4f}")
    assert err > 0, "Should have some quantization error"
    print("SUCCESS: Canonical CAQ verified.")

def test_meliponini_isolation():
    print("\n--- Testing Meliponini Isolation (Algebraic Mask) ---")
    codec = GyroidicCodec(CodecConfig(K=3, n=4))
    
    # Create residues with distinct channel profiles to prevent argmax collision
    residues = torch.randn(1, 3, 16)
    for k in range(3):
        residues[0, k, k] += 5.0
    # Veto mask for channel 1 [B, K]
    veto_mask = torch.tensor([[False, True, False]]) 
    
    # Reconstruction without veto (fixed-point algebraic path)
    res_full_fp = codec.crt_bridge.crt.forward(residues, mode='fixed_point')
    # Reconstruction with veto
    res_veto_fp = codec.crt_bridge.crt.forward(residues, veto_mask=veto_mask, mode='fixed_point')
    
    diff_fp = torch.norm(res_full_fp - res_veto_fp).item()
    print(f"Veto Impact (Fixed-Point): {diff_fp:.6f}")
    assert diff_fp > 1e-6, "Fixed-point veto mask should change reconstruction"

    # Reconstruction with majority symbolic path
    res_full_maj = codec.crt_bridge.crt.forward(residues, mode='majority')
    res_veto_maj = codec.crt_bridge.crt.forward(residues, veto_mask=veto_mask, mode='majority')
    diff_maj = torch.norm(res_full_maj - res_veto_maj).item()
    print(f"Veto Impact (Majority): {diff_maj:.6f}")
    assert diff_maj > 1e-6, "Majority veto mask should change reconstruction"
    print("SUCCESS: Meliponini Isolation verified.")

if __name__ == "__main__":
    try:
        test_non_abelian_gap()
        test_fixed_point_crt()
        test_caq_shells()
        test_meliponini_isolation()
        print("\nALL INVARIANT TESTS PASSED.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
