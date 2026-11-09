import pytest
import torch
import torch.nn as nn

from src.core.orchestrator import UniversalOrchestrator
from src.topology.triadic_reciprocity import TriadicReciprocityChecker
from src.training.enhanced_temporal_training import (
    NonLobotomyTemporalModel,
    NonLobotomyTemporalTrainer,
)
from src.optimization.muon_hybrid import MuonRicciHybridOptimizer
from src.optimization.ricci_flow_optimizer import BouligandWillmoreGasket, SplitBeamInterfactorization
from src.topology.gyroid_differentiation import ForbiddenSmoothingChecker
from src.topology.gyroid_covariance import SaturationFractureDetector, MoebiusFiberBundle, LeyLineGeodesicMetric
from src.topology.soliton_stability import SolitonStability


def test_orchestrator_wired_topological_safety_checks():
    """Verify UniversalOrchestrator wires and audits the newly integrated topological safety detectors."""
    dim = 32
    orch = UniversalOrchestrator(dim=dim)
    
    assert hasattr(orch, "forbidden_smoothing")
    assert isinstance(orch.forbidden_smoothing, ForbiddenSmoothingChecker)
    assert hasattr(orch, "saturation_detector")
    assert isinstance(orch.saturation_detector, SaturationFractureDetector)
    assert hasattr(orch, "soliton_stability")
    assert isinstance(orch.soliton_stability, SolitonStability)
    assert hasattr(orch, "moebius_bundle")
    assert isinstance(orch.moebius_bundle, MoebiusFiberBundle)
    assert hasattr(orch, "ley_line_metric")
    assert isinstance(orch.ley_line_metric, LeyLineGeodesicMetric)

    # Run check_safety with nominal signals
    safety_res = orch.check_safety(rho_def=0.05, grad_norm=0.1, loss=0.2)
    assert "forbidden_smoothing_alert" in safety_res
    assert "saturation_fracture_alert" in safety_res
    assert "homology_drift_alert" in safety_res
    assert "soliton_instability_alert" in safety_res
    assert "safety_alert" in safety_res


def test_triadic_reciprocity_palindromic_fast_bypass():
    """Verify TriadicReciprocityChecker uses PalindromicRoutingCheck for O(K) symmetric cycle bypass."""
    checker = TriadicReciprocityChecker(tolerance=1e-3)
    assert hasattr(checker, "palindromic_check")

    # Construct symmetric (palindromic) unit flows with identical norms
    flow_a = torch.tensor([[1.0, 0.0, 0.0]])
    flow_b = torch.tensor([[0.0, 1.0, 0.0]])
    flow_c = torch.tensor([[0.0, 0.0, 1.0]])

    score = checker.check_flow_reciprocity(flow_a, flow_b, flow_c)
    # Identical norms satisfy palindromic condition and trigger fast bypass -> score 1.0
    assert torch.allclose(score, torch.tensor([1.0]))


def test_non_lobotomy_temporal_trainer_with_muon_ricci():
    """Verify NonLobotomyTemporalTrainer properly initializes and steps with MuonRicciHybridOptimizer."""
    model = NonLobotomyTemporalModel(input_dim=32, hidden_dim=16, num_functionals=3, poly_degree=2)
    trainer = NonLobotomyTemporalTrainer(
        model=model,
        dataset=None,
        evolution_rate=0.02,
        survivorship_threshold=0.7,
        optimizer_type="muon_ricci"
    )
    assert isinstance(trainer.optimizer, MuonRicciHybridOptimizer)
    assert isinstance(trainer.willmore_energy, BouligandWillmoreGasket)

    # Perform a single synthetic training step with sequences and targets
    batch = {
        "sequences": torch.randn(2, 4, 32),
        "targets": torch.randn(2, 4, 32)
    }
    metrics = trainer.train_step(batch)
    assert "survivorship_pressure" in metrics
    assert "temporal_coherence" in metrics
    assert "association_accuracy" in metrics
    assert "pas_h" in metrics


def test_moebius_and_bouligand_willmore_gasket():
    """Verify MoebiusFiberBundle twist and BouligandWillmoreGasket contingent cone evaluation."""
    dim = 16
    bundle = MoebiusFiberBundle(dim=dim, fiber_dim=8)
    state = torch.randn(2, dim)
    gate_on = torch.tensor([1.0, 0.0])
    twisted = bundle(state, gate_on)
    assert twisted.shape == (2, 8)

    gasket = BouligandWillmoreGasket()
    flux = torch.randn(2, dim)
    tension = gasket(flux)
    assert tension.dim() == 0
    assert 0.0 <= float(tension.item()) <= 1.0


def test_pomni_uncertainty_interoceptive_friction():
    """Verify PomniUncertaintyPredictor broadcasts noradrenaline and applies relational friction under parasitic drag."""
    from src.governance.interoceptive.pomni_uncertainty import PomniUncertaintyPredictor
    from src.core.neuromodulatory_bus import NeuromodulatoryBus

    bus = NeuromodulatoryBus()
    pomni = PomniUncertaintyPredictor(state_dim=16)
    state = torch.randn(1, 16)

    # 1. Nominal forward pass: broadcasts noradrenaline on bus
    out_nominal = pomni(state, gyroid_entropy=0.3, bus=bus, parasitic_drag=0.1, valence_hunger=0.2)
    assert out_nominal.shape == (1, 16)
    norad_nominal = bus.read('noradrenaline')
    assert 0.0 <= norad_nominal <= 1.0

    # 2. High parasitic drag (> 0.5): triggers relational friction and salience broadcast
    out_friction = pomni(state, gyroid_entropy=1.5, bus=bus, parasitic_drag=0.8, valence_hunger=1.2)
    assert out_friction.shape == (1, 16)
    norad_high = bus.read('noradrenaline')
    assert norad_high >= norad_nominal

