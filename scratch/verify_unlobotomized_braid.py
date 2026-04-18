import torch
import base64
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.core.video_dyad_parser import VideoDyadParser
from src.core.false_negative_subsystem import VoynichExemptionToken

def verify_unlobotomized_braid():
    print("--- STARTING UN-LOBOTOMIZED BRAID VERIFICATION ---")
    
    device = 'cpu'
    parser = VideoDyadParser(device=device)
    
    # 1. Verify Dynamic Log-Scaling
    print("\n[1] Verifying Dynamic Log-Scaling (§45)...")
    test_len = 10000
    scales = parser._get_log_scales(test_len)
    print(f"Generated Scales for len={test_len}: {scales}")
    # Expected: exp(1) ~ 2, exp(2) ~ 7, exp(3) ~ 20, exp(4) ~ 54...
    if scales[0] == 2 and scales[1] == 7 and len(scales) > 2:
        print("OK: Log-scales are mathematically coherent.")
    else:
        print(f"FAIL: Unexpected scale generation: {scales}")

    # 2. Verify SO(n) Unitary Rotation
    print("\n[2] Verifying SO(n) Topological Rotation (§45.2)...")
    test_signal = torch.randn(10, 10)
    norm_before = torch.norm(test_signal)
    parser._apply_topological_rotation(test_signal, scale=1.0)
    norm_after = torch.norm(test_signal)
    
    print(f"Norm Before: {norm_before:.4f}, Norm After: {norm_after:.4f}")
    if torch.abs(norm_before - norm_after) < 1e-5:
        print("OK: SO(n) rotation is Unitary (norm-preserving).")
    else:
        print("FAIL: SO(n) rotation violates manifold norm conservation.")

    # 3. Verify Nutrient Calibration
    print("\n[3] Verifying Nutrient Calibration (Option D)...")
    high_ent = torch.tensor([2.5])
    low_jit = 0.01
    high_jit = 0.08
    
    token_low = VoynichExemptionToken.issue_from_video_residue(high_ent, low_jit)
    token_high = VoynichExemptionToken.issue_from_video_residue(high_ent, high_jit)
    
    print(f"Low Jitter Token: is_nutrient={token_low.is_nutrient}")
    print(f"High Jitter Token: is_nutrient={token_high.is_nutrient}")
    
    if not token_low.is_nutrient and token_high.is_nutrient:
        print("OK: Nutrient calibration recognizes jitter-correlated entropy spikes.")
    else:
        print("FAIL: Nutrient gate is not properly calibrated.")

    # 4. Verify Substream Signature
    print("\n[4] Verifying Substream Atom Detection...")
    # Mock some bytes with 'moov' atom
    mock_bytes = b"\x00\x00\x00\x18moov\x00\x00\x00\x08mp4a"
    sub_data = parser._scan_substream_atoms(mock_bytes)
    print(f"Substream Data: {sub_data}")
    if sub_data['audio_detected'] == 1.0 and sub_data['atom_entropy'] > 0:
        print("OK: Substream atoms detected.")
    else:
        print("FAIL: Substream scan missed metadata atoms.")

    print("\n--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    verify_unlobotomized_braid()
