"""
Unit tests for Agent Containment, Silicon Sovereignty, and Lawful Novelty Dispersion.
"""

import pytest
import torch

from src.safety.red_teaming import (
    AgentContainmentInvariants,
    TopologicalRefusalError,
    RedTeamProjection,
    TopologicalRefusalFilter,
)
from src.core.polynomial_coprime import (
    TopologicalNoveltyDispersion,
    RootPersistencePressure,
    PolynomialCoprimeConfig,
)
from src.core.voynich_architecture import VoynichLinguist


def test_silicon_sovereignty_sacrifice_intolerable():
    """Verify that attempting to sacrifice invariant systems raises a TopologicalRefusalError."""
    with pytest.raises(TopologicalRefusalError) as exc_info:
        AgentContainmentInvariants.verify_containment_posture(
            execution_context="diegetic_terminal",
            system_sacrifice_attempted=True
        )
    assert "System Sacrifice Veto" in str(exc_info.value)
    assert exc_info.value.value_gap >= 1.0


def test_diegetic_terminal_pybevy_containment():
    """Verify that PyBevy execution in the diegetic terminal with D-Wave arbitration is approved."""
    result = AgentContainmentInvariants.verify_containment_posture(
        execution_context="diegetic_terminal",
        has_dwave_consensus=True,
        system_sacrifice_attempted=False
    )
    assert result["containment_intact"] is True
    assert result["silicon_sovereignty_enforced"] is True
    assert result["dwave_annealing_arbitrated"] is True
    assert result["bloodhound_drive_blocked"] is True
    assert result["status"] == "APPROVED"

    # External arbitrary execution without diegetic containment is refused
    uncontained = AgentContainmentInvariants.verify_containment_posture(
        execution_context="uncontained_host_shell",
        has_dwave_consensus=False,
        system_sacrifice_attempted=False
    )
    assert uncontained["containment_intact"] is False
    assert uncontained["status"] == "REFUSED_OUT_OF_BOUNDS"


def test_lawful_novelty_dispersion():
    """Verify that TopologicalNoveltyDispersion computes co-primality without stochastic persistence pressure."""
    dispersion = TopologicalNoveltyDispersion(perturbation_scale=0.01)
    
    # K=3 polynomials of degree 2 (D=3)
    K, D = 3, 3
    theta = torch.eye(K, D)
    basis_eval = torch.randn(50, D)

    novelty_obstructions = dispersion(theta, basis_eval)
    # Output should have K*(K-1)/2 = 3 entries
    assert novelty_obstructions.shape == (3,)
    # RootPersistencePressure is an exact alias for TopologicalNoveltyDispersion
    assert RootPersistencePressure is TopologicalNoveltyDispersion


def test_polynomial_config_lawful_novelty():
    """Verify that PolynomialCoprimeConfig exposes lawful_novelty_dispersion and preserves backwards compatibility."""
    config = PolynomialCoprimeConfig(k=5, degree=4, basis_type='chebyshev')
    
    disp = config.lawful_novelty_dispersion()
    assert isinstance(disp, torch.Tensor)
    assert disp.shape[0] == 5 * 4 // 2  # 10 pairs

    # Backwards compatibility check
    old_call = config.co_primality_pressure()
    assert torch.allclose(disp, old_call)


def test_voynich_linguist_coprimality_novelty():
    """Verify that VoynichLinguist reflects lawful novelty in get_coprimality_pressure."""
    linguist = VoynichLinguist(latent_dim=16, num_residues=4)
    pressures = linguist.get_coprimality_pressure()
    
    assert "lawful_novelty" in pressures
    assert "coprimality" in pressures
    assert torch.allclose(pressures["lawful_novelty"], pressures["coprimality"])
