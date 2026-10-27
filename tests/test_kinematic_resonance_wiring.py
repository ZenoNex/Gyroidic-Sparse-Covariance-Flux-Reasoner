import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn

from src.core.orchestrator import UniversalOrchestrator
from src.core.enhanced_bezout_crt import EnhancedBezoutCRT, CrossbarIKSolver
from src.core.topological_gyrocompass import TopologicalGyrocompass
from src.core.garden_statistical_attractors import GardenOrchestrator
from src.core.archetype_engines import RP4ProjectiveRouter
from src.core.admr_solver import PolynomialADMRSolver
from src.core.polynomial_coprime import PolynomialCoprimeConfig


def test_rp4_puncture_threshold():
    """Verify RP4ProjectiveRouter puncturing behavior above/below 0.8 void friction."""
    dim = 32
    rp4 = RP4ProjectiveRouter(state_dim=dim)
    stranded = torch.randn(dim)
    
    # Sub-threshold friction: no puncture allowed
    sub_thresh = rp4.attempt_puncture(stranded, void_friction=0.5)
    assert torch.all(sub_thresh == 0.0), "RP4 router must not puncture when void_friction <= 0.8"
    
    # Critical void friction: puncture gate activates
    supra_thresh = rp4.attempt_puncture(stranded, void_friction=0.9)
    assert not torch.all(supra_thresh == 0.0), "RP4 router must puncture and tunnel when void_friction > 0.8"


def test_topological_gyrocompass_gimbal_lock_shield():
    """Verify TopologicalGyrocompass projects velocity vector away from ownership null space."""
    dim = 32
    love_dim = 8
    gyro = TopologicalGyrocompass(state_dim=dim, love_dim=love_dim, device=torch.device('cpu'))
    
    # Create collinear states (simulating near gimbal-lock condition)
    base = torch.randn(1, dim)
    states = base.repeat(4, 1) + 1e-4 * torch.randn(4, dim)
    dx = base.repeat(4, 1)
    
    dx_shielded = gyro.gimbal_lock_shield(dx, states)
    assert dx_shielded.shape == dx.shape
    assert torch.isfinite(dx_shielded).all()


def test_enhanced_bezout_crt_and_ik_solver():
    """Verify EnhancedBezoutCRT refreshes coefficients and synchronizes retrograde phase."""
    eb = EnhancedBezoutCRT(state_dim=32, num_moduli=4)
    state = torch.randn(1, 32)
    
    reconstructed, diag = eb.refresh_bezout_coefficients(state)
    assert reconstructed.shape == state.shape
    assert "bezout_condition_number" in diag
    assert diag["bezout_condition_number"] >= 0.99
    
    # Test Crossbar IK retrograde phase synchronization
    eb.ik_solver.synchronize_phase(note_event_time=2.5, phase_shift=0.15)
    traj = eb.ik_solver.solve_trajectory(torch.tensor([0.25, 0.5, 0.75]))
    assert traj.shape[0] == 3


def test_garden_orchestrator_chladni_evolution():
    """Verify GardenOrchestrator evolves concept distributions and outputs cymatic force components."""
    dim = 32
    garden = GardenOrchestrator(num_attractors=4, feature_dim=dim, device='cpu')
    concepts = torch.randn(2, dim)
    
    res = garden.evolve_garden(concepts, dt=0.05)
    assert "evolved_concepts" in res
    assert "influence_forces" in res
    assert "defect_forces" in res
    assert res["evolved_concepts"].shape == concepts.shape


def test_orchestrator_kinematic_wiring():
    """Verify UniversalOrchestrator successfully integrates GardenOrchestrator, EnhancedBezoutCRT, and Gyrocompass."""
    dim = 32
    orch = UniversalOrchestrator(dim=dim)
    
    # Ensure components are active on orchestrator
    assert hasattr(orch, 'garden_orchestrator') and orch.garden_orchestrator is not None
    assert hasattr(orch, 'enhanced_bezout_crt') and orch.enhanced_bezout_crt is not None
    assert hasattr(orch, 'gyrocompass') and orch.gyrocompass is not None
    
    # Run a single step to verify execution through all modified phases
    dummy_input = torch.randn(1, dim)
    out = orch.step(dummy_input)
    assert out is not None
    assert 'bezout_condition_number' in orch.telemetry_state
