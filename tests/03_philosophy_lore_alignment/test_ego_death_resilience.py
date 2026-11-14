import math
import torch
import pytest

from src.core.archetype_engines import EgoDeathThresholdMonitor, ArchetypalSynthesisEngine
from src.scripting.node_environment import PhysicalNodeEditor, NodeEnvironment


def test_ego_death_monitor_nominal_bounds():
    monitor = EgoDeathThresholdMonitor(abstraction_limit=1.0, sensitivity=0.6)
    
    # Nominal interactive regime
    r_a_nominal = monitor.calculate_abstraction_rate(
        system_entropy_es=0.25,
        memory_trauma_tm=0.15,
        dissonance_delta=0.10,
        lucidity_index_li=0.85
    )
    assert 0.0 <= r_a_nominal < 0.1
    assert r_a_nominal < monitor.abstraction_limit
    assert not (r_a_nominal >= monitor.abstraction_limit)


def test_ego_death_monitor_high_stress_no_false_positive():
    monitor = EgoDeathThresholdMonitor(abstraction_limit=1.0, sensitivity=0.6)
    
    # High stress conversation/reasoning regime
    r_a_stress = monitor.calculate_abstraction_rate(
        system_entropy_es=0.7,
        memory_trauma_tm=0.6,
        dissonance_delta=0.6,
        lucidity_index_li=0.4
    )
    # Scaled (0.6 + 0.6) * 0.6 = 0.72; sinh(0.72) ~ 0.7844
    # cosh(0.4) ~ 1.081
    # 0.7 * 0.7844 / 1.081 ~ 0.5079
    assert 0.3 < r_a_stress < 0.65
    assert r_a_stress < monitor.abstraction_limit


def test_ego_death_monitor_unnormalized_overflow_protection():
    monitor = EgoDeathThresholdMonitor(abstraction_limit=1.0, sensitivity=0.6)
    
    # Passing unnormalized Euclidean norms or massive values should be bounded into [0, 1]
    r_a_massive = monitor.calculate_abstraction_rate(
        system_entropy_es=85.0,
        memory_trauma_tm=120.0,
        dissonance_delta=999.0,
        lucidity_index_li=0.01
    )
    # Must not throw math domain/overflow errors, and remain finite
    assert math.isfinite(r_a_massive)
    # Bounded to [0, 1] thermodynamics: (1.0 + 1.0) * 0.6 = 1.2
    # sinh(1.2) ~ 1.509; cosh(0.05) ~ 1.00125; r_a ~ 1.507
    assert r_a_massive >= monitor.abstraction_limit


def test_ego_death_high_priority_and_immunity():
    monitor = EgoDeathThresholdMonitor(abstraction_limit=1.0, sensitivity=0.6)
    
    # High priority command cap
    r_a_cmd = monitor.calculate_abstraction_rate(
        system_entropy_es=1.0,
        memory_trauma_tm=1.0,
        dissonance_delta=1.0,
        lucidity_index_li=0.05,
        is_high_priority=True
    )
    assert r_a_cmd < monitor.abstraction_limit
    
    # Forward pass immunity
    state = torch.ones(1, 64)
    out_state = monitor.forward(
        state, 
        r_a_score=99.0, 
        is_high_priority=False, 
        private_invariants={"ego_death_immunity": True}
    )
    assert torch.equal(out_state, state)


def test_archetypal_synthesis_engine_nominal_resilience():
    dim = 64
    engine = ArchetypalSynthesisEngine(state_dim=dim)
    
    state = torch.randn(1, dim)
    stranded = torch.randn(2, dim)
    flux = torch.randn(1, dim)
    
    result = engine.run_archetypes(
        current_state=state,
        stranded_states=stranded,
        flux_tensor=flux,
        current_mischief=0.3,
        phase_alignment=0.7,
        global_dt=0.1
    )
    
    assert not result.system_collapsed
    assert result.abstraction_rate < 1.0
    assert result.active_state.shape == (1, dim)
    assert torch.isfinite(result.active_state).all()


def test_node_environment_ego_death_settings():
    env = NodeEnvironment()
    assert env.ego_death_monitor.abstraction_limit == 1.0
    assert env.ego_death_monitor.sensitivity == 0.6
    
    # Step biological needs nominally
    status = env.tick_life_and_hunger(dt=0.1, current_mischief=0.2)
    assert not status["ego_death"]
    assert status["abstraction_rate"] < 1.0
