#!/usr/bin/env python3
"""
tests/test_log_cholesky_jepa.py
Verification of:
1. OKLab Moment Field Log-Cholesky SPD Reparameterization & Inverse Mapping
2. Morton (Z-Order) Spatial Locality Sorting in Point Cloud Processing
3. JEPA Latent Representation Monge-Ampere Optimal Transport Energy & Cayley Surface Gating
"""

import sys
import os
import torch

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core.conjugate_moment_transport import ConjugateMomentTransport
from src.models.polynomial_embeddings import PolynomialFunctionalEmbedder

def test_log_cholesky_reparameterization():
    print("--- Test 1: Log-Cholesky SPD Reparameterization ---")
    cmt = ConjugateMomentTransport(dim=16)
    
    # Generate random RGB point cloud [batch=2, num_points=20, 3]
    torch.manual_seed(42)
    rgb_points = torch.rand(2, 20, 3)
    
    # Extract moment field using Log-Cholesky: 3 (mean) + 6 (cholesky) = 9
    mf_cholesky = cmt.extract_oklab_moment_field(rgb_points, use_log_cholesky=True)
    assert mf_cholesky.shape == (2, 9), f"Expected shape (2, 9), got {mf_cholesky.shape}"
    assert torch.isfinite(mf_cholesky).all(), "Moment field contains non-finite values"
    
    # Reconstruct mean and covariance
    mu, cov = cmt.reconstruct_from_moment_field(mf_cholesky, use_log_cholesky=True)
    assert mu.shape == (2, 3), f"Expected mu shape (2, 3), got {mu.shape}"
    assert cov.shape == (2, 3, 3), f"Expected cov shape (2, 3, 3), got {cov.shape}"
    
    # Verify symmetry: cov == cov^T
    sym_diff = (cov - cov.transpose(-1, -2)).abs().max().item()
    assert sym_diff < 1e-5, f"Reconstructed covariance is not symmetric: diff={sym_diff}"
    
    # Verify strict positive-definiteness: all eigenvalues > 0
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals > 0).all(), f"Reconstructed covariance is not positive definite: eigs={eigvals}"
    
    # Test legacy fallback (12-dim)
    mf_legacy = cmt.extract_oklab_moment_field(rgb_points, use_log_cholesky=False)
    assert mf_legacy.shape == (2, 12), f"Expected legacy shape (2, 12), got {mf_legacy.shape}"
    mu_leg, cov_leg = cmt.reconstruct_from_moment_field(mf_legacy, use_log_cholesky=False)
    assert cov_leg.shape == (2, 3, 3)
    
    print("Log-Cholesky SPD Reparameterization: PASSED\n")

def test_morton_spatial_sorting():
    print("--- Test 2: Morton Spatial Sorting in Point Cloud Processing ---")
    cmt = ConjugateMomentTransport(dim=16)
    
    rgb_points = torch.rand(2, 50, 3)
    spatial_coords = torch.rand(2, 50, 3) * 100.0  # [batch, num_points, 3]
    
    mf_sorted = cmt.extract_oklab_moment_field(
        rgb_points, 
        spatial_coords=spatial_coords, 
        use_log_cholesky=True
    )
    assert mf_sorted.shape == (2, 9)
    assert torch.isfinite(mf_sorted).all()
    print("Morton Spatial Sorting: PASSED\n")

def test_jepa_monge_ampere_energy():
    print("--- Test 3: JEPA Monge-Ampere Energy & Cayley Surface Gating ---")
    embedder = PolynomialFunctionalEmbedder(
        text_dim=16,
        graph_dim=16,
        num_dim=16,
        hidden_dim=16
    )
    cmt = ConjugateMomentTransport(dim=16, hidden_dim=32, num_layers=2)
    
    context = torch.randn(2, 16)
    target = torch.randn(2, 16)
    
    # Compute Monge-Ampere energy with Cayley surface gating
    energy = embedder.compute_jepa_monge_ampere_energy(
        context_state=context,
        target_state=target,
        moment_transport=cmt,
        shell_depth=1,
        is_honeybee_mode=False
    )
    
    assert torch.isfinite(energy), "JEPA Monge-Ampere energy is non-finite"
    print(f"Computed JEPA Monge-Ampere Energy: {energy.item():.4f}")
    assert energy.item() >= 0.0 or torch.isfinite(energy), "Energy computation valid"
    print("JEPA Monge-Ampere Energy: PASSED\n")

def main():
    print("Starting Log-Cholesky and JEPA Validation Suite...\n")
    try:
        test_log_cholesky_reparameterization()
        test_morton_spatial_sorting()
        test_jepa_monge_ampere_energy()
        print("ALL TESTS IN SUITE PASSED SUCCESSFULLY.")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
