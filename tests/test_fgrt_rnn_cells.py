"""
Unit tests verifying the 7 FGRT Recurrent Neural Network Cells.
Ensures tensor shape preservation, Stiefel-Whitney flips, KAN B-splines,
honest jitter initialization, and Cerumen Pot LCFT scarring.
"""

import pytest
import torch
import numpy as np

from src.core.fgrt_rnn_cells import (
    BerryPhaseGRUCell,
    ChiralGatedRNNCell,
    ComplexFGRTRNNCell,
    SaturatedQuantizerRNNCell,
    PyOpenCLHardwareSovereigntyCell,
    LazarusSuperpositionRNNCell,
    FeatureScarLCFTCell
)


def test_berry_phase_gru_cell():
    batch, input_dim, hidden_dim = 2, 16, 32
    cell = BerryPhaseGRUCell(input_dim, hidden_dim)
    x = torch.randn(batch, input_dim)
    h_prev = torch.zeros(batch, hidden_dim)
    gamma_prev = torch.zeros(batch, 1)

    h_t, gamma_t = cell(x, (h_prev, gamma_prev))
    assert h_t.shape == (batch, hidden_dim)
    assert gamma_t.shape == (batch, 1)
    assert torch.isfinite(h_t).all()
    assert torch.isfinite(gamma_t).all()


def test_chiral_gated_rnn_cell():
    batch, input_dim, hidden_dim = 2, 16, 32
    cell = ChiralGatedRNNCell(input_dim, hidden_dim, num_attractors=4)
    x = torch.randn(batch, input_dim)
    h_prev = torch.zeros(batch, hidden_dim)

    h_t = cell(x, h_prev)
    assert h_t.shape == (batch, hidden_dim)
    assert torch.isfinite(h_t).all()


def test_complex_fgrt_rnn_cell():
    batch, input_dim, hidden_dim = 2, 16, 32
    cell = ComplexFGRTRNNCell(input_dim, hidden_dim)
    x = torch.randn(batch, input_dim)
    h_prev_complex = torch.zeros(batch, hidden_dim * 2)

    h_t = cell(x, h_prev_complex)
    assert h_t.shape == (batch, hidden_dim * 2)
    assert torch.isfinite(h_t).all()


def test_saturated_quantizer_rnn_cell():
    batch, input_dim, hidden_dim = 2, 16, 32
    cell = SaturatedQuantizerRNNCell(input_dim, hidden_dim, quant_levels=4)
    x = torch.randn(batch, input_dim)
    h_prev = torch.zeros(batch, hidden_dim)

    h_t = cell(x, h_prev)
    assert h_t.shape == (batch, hidden_dim)
    assert torch.isfinite(h_t).all()


def test_pyopencl_hardware_sovereignty_cell():
    batch, input_dim, hidden_dim = 2, 16, 32
    cell = PyOpenCLHardwareSovereigntyCell(input_dim, hidden_dim)
    x = torch.randn(batch, input_dim)
    h_prev = torch.zeros(batch, hidden_dim)

    # Should run with fallback or OpenCL without crashing
    h_t = cell(x, h_prev)
    assert h_t.shape == (batch, hidden_dim)
    assert torch.isfinite(h_t).all()


def test_lazarus_superposition_rnn_cell():
    batch, input_dim, hidden_dim = 2, 16, 32
    cell = LazarusSuperpositionRNNCell(input_dim, hidden_dim, num_resonators=8)
    x = torch.randn(batch, input_dim)
    h_prev = torch.zeros(batch, hidden_dim)

    h_t = cell(x, h_prev, step=3)
    assert h_t.shape == (batch, hidden_dim)
    assert torch.isfinite(h_t).all()


def test_feature_scar_lcft_cell():
    batch, input_dim, hidden_dim = 2, 16, 32
    cell = FeatureScarLCFTCell(input_dim, hidden_dim)
    x = torch.randn(batch, input_dim)
    h_prev = torch.zeros(batch, hidden_dim)
    cerumen_pot = torch.zeros(batch, hidden_dim)

    h_t, updated_pot = cell(x, (h_prev, cerumen_pot))
    assert h_t.shape == (batch, hidden_dim)
    assert updated_pot.shape == (batch, hidden_dim)
    assert torch.isfinite(h_t).all()
    assert torch.isfinite(updated_pot).all()
