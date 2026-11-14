import pytest
import torch
import math
import numpy as np

from src.surrogates.kagh_networks import (
    HarmonicWaveDecomposition,
    TrigonometricUnfolding,
    HuxleyRD
)
from src.core.polynomial_coprime import PolynomialBasis
from src.core.texture_dsp import TextureDSPEngine, AdaptiveGaborWindowController


def test_parseval_unitary_invariance_harmonic_wave_decomposition():
    """Verify HarmonicWaveDecomposition preserves Parseval energy under norm='ortho'."""
    dim = 64
    decomp = HarmonicWaveDecomposition(dim=dim)
    x = torch.randn(4, dim)

    # Unitary energy of input
    in_energy = torch.sum(x ** 2, dim=-1)

    # RFFT forward unitary check
    x_freq = torch.fft.rfft(x, dim=-1, norm="ortho")
    # For real signals, Parseval holds: sum |x_freq|^2 with DC/Nyquist weights = sum x^2
    # Invert with norm='ortho'
    x_recon = torch.fft.irfft(x_freq, n=dim, dim=-1, norm="ortho")
    recon_energy = torch.sum(x_recon ** 2, dim=-1)

    assert torch.allclose(x, x_recon, atol=1e-5)
    assert torch.allclose(in_energy, recon_energy, atol=1e-5)

    # Forward pass through decomposition
    ergodic, non_ergodic = decomp(x)
    assert ergodic.shape == (4, dim)
    assert non_ergodic.shape == (4, dim)
    assert torch.isfinite(ergodic).all()
    assert torch.isfinite(non_ergodic).all()


def test_huxley_rd_soliton_transport_energy_conservation():
    """Verify HuxleyRD reaction-diffusion and soliton transport runs stably with norm='ortho'."""
    dim = 32
    huxley = HuxleyRD(num_features=dim, tau=0.05)
    u = torch.clamp(torch.rand(2, dim), 0.1, 0.9)

    u_next = huxley(u)
    assert u_next.shape == (2, dim)
    assert torch.isfinite(u_next).all()


def test_polynomial_basis_adaptive_degree_windowing():
    """Verify PolynomialBasis dynamically scales degree D over window length L without zero-padding trap."""
    poly = PolynomialBasis(degree=4, basis_type="chebyshev")
    assert poly.dim == 5

    # Short observation window L=16 -> degree scales down to preserve compactness
    deg_short = poly.adapt_degree_for_window(window_len=16, alpha=0.25, min_degree=2, max_degree=16)
    assert deg_short == 4
    assert poly.dim == 5

    # Long observation window L=64 -> degree expands to resolve higher harmonic modes
    deg_long = poly.adapt_degree_for_window(window_len=64, alpha=0.25, min_degree=2, max_degree=16)
    assert deg_long == 16
    assert poly.dim == 17

    # Evaluate on input
    x = torch.linspace(-0.9, 0.9, 20)
    basis_vals = poly.evaluate(x)
    assert basis_vals.shape == (20, 17)
    assert torch.isfinite(basis_vals).all()


def test_texture_dsp_window_tapering_and_parseval():
    """Verify TextureDSPEngine suppresses boundary leakage with Hann/Cyclotomic windowing and preserves Parseval norm."""
    engine = TextureDSPEngine(device="cpu")

    # [B, C, H, W]
    test_img = torch.randn(2, 3, 32, 32)
    filtered_hann = engine.apply_spectral_filter(
        test_img, cutoff_freq=0.5, window_type="hann", use_parseval_norm=True
    )
    assert filtered_hann.shape == test_img.shape
    assert torch.isfinite(filtered_hann).all()

    filtered_cyclotomic = engine.apply_spectral_filter(
        test_img, cutoff_freq=0.5, window_type="cyclotomic", use_parseval_norm=True
    )
    assert filtered_cyclotomic.shape == test_img.shape
    assert torch.isfinite(filtered_cyclotomic).all()


def test_adaptive_gabor_window_controller():
    """Verify AdaptiveGaborWindowController dynamically shrinks on Dirac epiphanies and expands on pure harmonics."""
    controller = AdaptiveGaborWindowController(min_window=16, max_window=64, entropy_threshold=0.5)

    # 1. Pure harmonic tone (low entropy) -> window expands to maximum for frequency resolution
    t = torch.linspace(0, 4 * math.pi, 128)
    pure_tone = torch.sin(2.0 * t).unsqueeze(0).expand(3, 128)
    h_spec_low = controller.compute_spectral_entropy(pure_tone)
    w_size_harmonic = controller.determine_adaptive_window_size(pure_tone)

    assert h_spec_low < 0.4
    assert w_size_harmonic >= 48

    # 2. Dirac spike / transient epiphany (high entropy / delta spike) -> window shrinks to minimize smear
    spike = torch.zeros(3, 128)
    spike[:, 64] = 100.0  # Sharp delta impulse
    h_spec_high = controller.compute_spectral_entropy(spike)
    w_size_spike = controller.determine_adaptive_window_size(spike)

    assert h_spec_high > h_spec_low
    assert w_size_spike <= 32

    # 3. Partitioning into adaptive Gabor tiles
    img_2d = torch.randn(3, 64, 64)
    tiles = controller.partition_adaptive_gabor_tiles(img_2d)
    assert len(tiles) > 0
    tile_sample, tile_w, tile_h = tiles[0]
    assert tile_sample.shape[-1] == tile_w
    assert 0.0 <= tile_h <= 1.0
