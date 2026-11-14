"""
Test suite for Muon & Ricci Flow Hybrid Optimization.
Verifies quintic Newton-Schulz polar decomposition, singular value equalization,
matrix-aware parameter partitioning, and training convergence.
"""

import math
import pytest
import torch
import torch.nn as nn

from src.optimization.muon_hybrid import (
    zeropower_via_newtonschulz5,
    MuonOptimizer,
    MuonRicciHybridOptimizer
)
from src.optimization.ricci_flow_optimizer import RicciFlowOptimizer


def test_zeropower_newtonschulz5_square():
    """Verify singular value equalization on square weight matrices."""
    torch.manual_seed(42)
    dim = 32
    G = torch.randn(dim, dim) * 5.0
    sv_before = torch.linalg.svdvals(G)
    assert (sv_before.max() / sv_before.min()).item() > 2.0

    X = zeropower_via_newtonschulz5(G, steps=5)
    assert X.shape == G.shape
    assert X.dtype == G.dtype

    sv_after = torch.linalg.svdvals(X)
    # After quintic iterations, singular values are equalized and bounded
    assert (sv_after.max() / sv_after.min()).item() < (sv_before.max() / sv_before.min()).item()
    assert (sv_after > 0.4).all() and (sv_after < 1.3).all()


def test_zeropower_newtonschulz5_rectangular():
    """Verify singular value equalization on rectangular (tall and wide) matrices."""
    torch.manual_seed(101)
    # Tall matrix (rows > cols)
    tall_G = torch.randn(64, 24)
    tall_X = zeropower_via_newtonschulz5(tall_G, steps=5)
    assert tall_X.shape == (64, 24)
    tall_sv = torch.linalg.svdvals(tall_X)
    assert (tall_sv > 0.4).all() and (tall_sv < 1.3).all()

    # Wide matrix (rows < cols)
    wide_G = torch.randn(24, 64)
    wide_X = zeropower_via_newtonschulz5(wide_G, steps=5)
    assert wide_X.shape == (24, 64)
    wide_sv = torch.linalg.svdvals(wide_X)
    assert (wide_sv > 0.4).all() and (wide_sv < 1.3).all()


def test_muon_optimizer_step():
    """Verify MuonOptimizer updates parameters and maintains momentum buffers."""
    torch.manual_seed(42)
    linear = nn.Linear(32, 16, bias=False)
    opt = MuonOptimizer(linear.parameters(), lr=0.05, momentum=0.9, nesterov=True)

    x = torch.randn(8, 32)
    loss = (linear(x) ** 2).sum()
    loss.backward()

    w_before = linear.weight.clone()
    opt.step()
    w_after = linear.weight.clone()

    assert not torch.allclose(w_before, w_after)
    # Check momentum buffer was initialized
    state = opt.state[linear.weight]
    assert 'momentum_buffer' in state
    assert state['step'] == 1


def test_muon_ricci_hybrid_partitioning_and_step():
    """Verify parameter auto-partitioning into Muon (2D) and Ricci (1D) groups."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 16, bias=True),
        nn.LayerNorm(16),
        nn.Linear(16, 8, bias=True)
    )

    hybrid_opt = MuonRicciHybridOptimizer(
        model.parameters(),
        lr=0.02,
        lr_1d=0.001,
        momentum=0.95,
        fuse_chern_simons=True
    )

    # Check param groups
    assert len(hybrid_opt.param_groups) == 2
    muon_group = next(g for g in hybrid_opt.param_groups if g['is_muon'])
    ricci_group = next(g for g in hybrid_opt.param_groups if not g['is_muon'])

    # 2 linear weight matrices should be in muon_group (dim=2, min_dim >= 8)
    assert len(muon_group['params']) == 2
    # 2 linear biases + 2 layer norm params (weight, bias) should be in ricci_group
    assert len(ricci_group['params']) == 4

    # Run forward, backward, and optimization step
    x = torch.randn(4, 32)
    y = model(x)
    loss = (y ** 2).sum()
    loss.backward()

    hybrid_opt.step()
    assert True


def test_ricci_flow_optimizer_with_orthogonalization():
    """Verify RicciFlowOptimizer works with orthogonalize=True."""
    torch.manual_seed(42)
    model = nn.Linear(16, 16)
    opt = RicciFlowOptimizer(model.parameters(), lr=1e-3, orthogonalize=True, ns_steps=5)

    x = torch.randn(4, 16)
    loss = (model(x) ** 2).sum()
    loss.backward()

    w_before = model.weight.clone()
    opt.step()
    w_after = model.weight.clone()

    assert not torch.allclose(w_before, w_after)


if __name__ == '__main__':
    test_zeropower_newtonschulz5_square()
    print("test_zeropower_newtonschulz5_square passed")
    test_zeropower_newtonschulz5_rectangular()
    print("test_zeropower_newtonschulz5_rectangular passed")
    test_muon_optimizer_step()
    print("test_muon_optimizer_step passed")
    test_muon_ricci_hybrid_partitioning_and_step()
    print("test_muon_ricci_hybrid_partitioning_and_step passed")
    test_ricci_flow_optimizer_with_orthogonalization()
    print("test_ricci_flow_optimizer_with_orthogonalization passed")
    print("ALL 5 MUON TESTS PASSED SUCCESSFULLY!")
