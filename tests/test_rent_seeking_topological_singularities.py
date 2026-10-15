"""
Unit and Integration Tests for Rent-Seeking Topological Singularities.

Validates the 5 operational layers:
1. Substrate & Feed Initialization (Spatial metric g_mu_nu + PointerlessOctree)
2. Void Friction & Geodesic Deflection (F_void scaling + Chern-Simons coprime bypass)
3. Mechanical Shear & Voxel Fracturing (Mohr-Coulomb/Drucker-Prager shear + CarnotMobius N_crit runaway)
4. Necrotic Quarantine & Memory Fossilization (Unknowledge domain stagnation + Meliponini cerumen barrier CRT compression)
5. Visual Rendering (Brenier push-forward gradient displacement + OKLab perceptual luminance absorption)
"""

import math
import torch
import numpy as np
import pytest

from src.ui.diegetic_backend import HybridPhysicsBackend
from src.ui.voxelboxter_simulation import PointerlessOctree
from src.core.zeitgeist_router import ZeitgeistRouter, ZeitgeistState
from src.core.codes_constraint_framework import CODESConstraintFramework
from src.core.carnot_mobius_ledger import CarnotMobiusLedger
from src.topology.unknowledge_domain import UnknowledgeDomain
from src.core.garden_statistical_attractors import MemoryPalaceRoom
from src.ui.diegetic_visualizer import brenier_pushforward, apply_oklab_luminance_absorption, srgb_to_oklab, oklab_to_srgb


def test_layer1_substrate_and_feed_initialization():
    backend = HybridPhysicsBackend(state_dim=32, device="cpu")
    # Mock external network feed to prevent network timeouts during test
    backend.fetch_financial_feeds = lambda feeds=None: ["Equities flux neutral; topological yield curve steady."]
    spatial_data = backend.fetch_spatial_land_value()
    assert "spatial_metric" in spatial_data
    assert "land_rent_density" in spatial_data
    assert spatial_data["spatial_metric"].shape == (4, 4)
    assert spatial_data["land_rent_density"] > 0.0

    octree = PointerlessOctree(max_depth=6)
    nodes_added = backend.map_to_pointerless_octree(octree, spatial_data)
    assert nodes_added > 0
    assert octree.num_nodes > 0

    feed_chain = backend.multiplex_feed_chain()
    assert "combined_flux" in feed_chain
    assert feed_chain["combined_flux"].shape == (32,)


def test_layer2_void_friction_and_geodesic_deflection():
    moduli = (3, 5, 7, 11, 13)
    router = ZeitgeistRouter(dim=32, moduli=moduli)
    state = ZeitgeistState.initial(moduli)

    spatial_metric = torch.eye(4)
    spatial_metric[0, 0] = 4.0
    spatial_metric[3, 3] = 4.0

    x_in = torch.randn(1, 32)
    mode, next_state, diag, x_steer = router(x_in, state, tadc_kwargs={'spatial_metric': spatial_metric})
    assert "f_void" in diag
    assert diag["f_void"] > 0.0

    steered = router.temporal_inverse_kinematics(state, x_in, spatial_metric=spatial_metric)
    assert steered.shape == x_in.shape

    codes = CODESConstraintFramework(state_dim=32, max_constraints=5)
    high_f_void = 0.85
    routed_state, bypass_info = codes.monitor_void_friction(x_in, high_f_void)
    assert bypass_info["coprime_bypass_active"] is True
    assert "gasket_diagnostics" in bypass_info


def test_layer3_mechanical_shear_and_voxel_fracturing():
    octree = PointerlessOctree(max_depth=6)
    for x in range(3):
        for y in range(3):
            octree.add_fractal_node(x, y, 0, material=1, complexity=0.0)

    stress_tensor = torch.tensor([
        [2.5, 1.2, 0.0],
        [1.2, 0.5, 0.0],
        [0.0, 0.0, 0.2]
    ])
    sheared_count = octree.shear_voxels(stress_tensor, yield_limit=1.0)
    assert sheared_count > 0
    assert octree.betti_1 >= 1

    ledger = CarnotMobiusLedger()
    lambda_vac = torch.tensor(1.0)
    lambda_pump = torch.tensor(10.0)
    admr_residues = torch.randn(4, 8)
    depth_N = 10
    uncirculated_capital = torch.full((1, 8), 10.0)
    diag = ledger(
        lambda_vac=lambda_vac,
        lambda_pump=lambda_pump,
        admr_residues=admr_residues,
        depth_N=depth_N,
        uncirculated_capital=uncirculated_capital,
        stagnation_velocity=0.0
    )
    assert diag["effective_depth_N"] >= 10.0
    assert diag["irreversible_n_crit_runaway"] is True


def test_layer4_necrotic_quarantine_and_memory_fossilization():
    domain = UnknowledgeDomain(tau_m=0.3, dim=16)
    stagnant_state = torch.randn(16)
    classification = domain.classify_domain(stagnant_state, h_mischief=0.1, velocity_flow=1e-5)
    assert classification["stagnant"] is True
    assert classification["classification"] == "Unknowledge"

    room = MemoryPalaceRoom(room_id="necrotic_chamber", center=torch.zeros(16))
    well_coords = torch.randn(16, requires_grad=True)
    residues = room.compress_necrotic_well(well_coords)
    assert room.sealed is True
    assert room.metadata["meliponini_cerumen_barrier"] is True
    assert isinstance(residues, tuple)
    assert len(residues) == 3
    assert room.center.requires_grad is False


def test_layer5_visual_rendering():
    coords = np.zeros((8, 8, 2))
    potential = np.ones((8, 8)) * 0.5
    potential[3:5, 3:5] = 2.0
    displaced = brenier_pushforward(coords, potential, kappa=0.2)
    assert displaced.shape == (8, 8, 2)
    assert np.any(displaced != 0.0)

    test_rgb = np.ones((4, 4, 3), dtype=np.float64) * 0.8
    lab = srgb_to_oklab(test_rgb)
    srgb_back = oklab_to_srgb(lab)
    np.testing.assert_allclose(test_rgb, srgb_back, atol=1e-3)

    absorbed = apply_oklab_luminance_absorption(test_rgb, void_friction=0.5, curvature=1.0)
    assert absorbed.shape == test_rgb.shape
    assert np.mean(absorbed) < np.mean(test_rgb)
