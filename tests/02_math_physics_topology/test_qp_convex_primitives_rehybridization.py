"""
Test suite for Quadratic and Convex Primitives Rehybridization.
Validates:
1. LocalActuationQPOperator local feasibility, box constraint clamping, and Gyroid penalty.
2. FailureToken emission on constraint rupture (Silent Failure contract).
3. Parameter evolution via geometric Ricci Flow preserving positive definiteness.
4. Stage 5 DiegeticPhysicsEngine feasibility probing and CODES relaxation.
5. AeronauticContraption QUBO flight subsystem arbitration.
6. RedNet16BundledCable QUBO channel conflict arbitration.
7. RicciFlowOptimizer convex potential factorization regularization.
"""

import pytest
import torch

from src.optimization.local_qp_operator import LocalActuationQPOperator
from src.core.failure_token import FailureToken, FailureTokenType
from src.core.diegetic_physics_engine import DiegeticPhysicsEngine
from src.environment.cleanroom_mechanics import AeronauticContraption, RedNet16BundledCable
from src.optimization.ricci_flow_optimizer import RicciFlowOptimizer
from src.ui.voxelboxter_simulation import VoxelboxterEngine


def test_local_qp_operator_feasibility_and_bounds():
    u_min = torch.tensor([-1.0, -0.5, -2.0, -1.0])
    u_max = torch.tensor([1.0, 0.5, 2.0, 1.0])
    qp = LocalActuationQPOperator(constraint_index=0, actuation_dim=4, u_min=u_min, u_max=u_max)

    # Command beyond bounds
    raw_cmd = torch.tensor([2.5, -1.8, 0.2, 3.0])
    opt_c, failure_token, diag = qp.solve_local_feasibility(raw_cmd, max_iters=25)

    assert failure_token is None
    assert diag["is_feasible"] is True
    # Verify strict box clamping
    assert (opt_c >= u_min - 1e-5).all()
    assert (opt_c <= u_max + 1e-5).all()


def test_local_qp_operator_rupture_and_failure_token():
    # Impossible inequality: c_0 + c_1 >= 100, while box bounds are [-1, 1]
    u_min = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    u_max = torch.tensor([1.0, 1.0, 1.0, 1.0])
    A_ineq = torch.tensor([[-1.0, -1.0, 0.0, 0.0]])  # -c_0 - c_1 <= -100 ==> c_0 + c_1 >= 100
    b_ineq = torch.tensor([-100.0])

    qp = LocalActuationQPOperator(
        constraint_index=7,
        actuation_dim=4,
        u_min=u_min,
        u_max=u_max,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
    )

    raw_cmd = torch.tensor([0.0, 0.0, 0.0, 0.0])
    opt_c, failure_token, diag = qp.solve_local_feasibility(raw_cmd, max_iters=10)

    assert failure_token is not None
    assert failure_token.type == FailureTokenType.BOT
    assert failure_token.constraint_index == 7
    assert failure_token.gradient_enabled is False
    assert diag["is_feasible"] is False


def test_local_qp_operator_ricci_flow_evolution():
    qp = LocalActuationQPOperator(constraint_index=0, actuation_dim=4)
    initial_Q = qp.Q.clone()

    # Apply Ricci flow step with synthetic curvature
    curvature = torch.eye(4) * 0.5
    qp.evolve_parameters_ricci_flow(curvature, dt=0.05)

    # Q must have changed, remained symmetric, and strictly positive definite
    assert not torch.allclose(qp.Q, initial_Q)
    assert torch.allclose(qp.Q, qp.Q.t(), atol=1e-5)
    evals = torch.linalg.eigvalsh(qp.Q)
    assert (evals >= 0.01).all()


def test_diegetic_physics_stage5_feasibility_probe():
    engine = DiegeticPhysicsEngine(device="cpu", state_dim=16)
    assert hasattr(engine, "codes_constraint")
    assert hasattr(engine, "qp_probe")

    v_state = {"velocity": (0.0, 0.0, 0.0), "angular_velocity": (0.0, 0.0, 0.0)}
    track_state = {"betti_0": 1, "betti_1": 0}
    c_input = {"throttle": 0.5, "yaw": 0.1, "pitch": 0.0, "roll": 0.0}

    out = engine.process_input(v_state, track_state, c_input)
    assert "c_out" in out
    assert "betti_shift" in out
    assert torch.isfinite(out["c_out"]).all()
    assert out["c_out"].shape[-1] == 16


def test_voxelboxter_actuation_probe():
    sim = VoxelboxterEngine(device="cpu")
    cmd = torch.tensor([0.5, -0.2, 0.8, -0.1])
    res_c, token, diag = sim.probe_actuation_feasibility(cmd)

    assert token is None
    assert diag["is_feasible"] is True
    assert res_c.shape == cmd.shape


def test_aeronautic_contraption_qubo_arbitration():
    airship = AeronauticContraption(name="SovereignHindenburg")
    Q, c = airship.map_to_qubo_hamiltonian()
    assert Q.shape == (4, 4)
    assert c.shape == (4,)

    best_state, min_energy, summary = airship.solve_qubo_subsystem_arbitration(steps=30)
    assert best_state.shape == (4,)
    assert set(best_state.tolist()).issubset({0.0, 1.0})
    assert "boost_thrust" in summary
    assert "gas_venting" in summary
    assert "ground_energy" in summary


def test_rednet_qubo_channel_arbitration():
    cable = RedNet16BundledCable()
    cable.set_signal("red", 255)
    cable.set_signal("orange", 200)
    cable.set_signal("blue", 180)

    arbitrated, min_energy = cable.arbitrate_channels_qubo(steps=25)
    assert len(arbitrated) == 16
    assert isinstance(arbitrated["red"], bool)
    assert isinstance(min_energy, float)


def test_ricci_flow_with_convex_regularization():
    weight = torch.randn(8, 8, requires_grad=True)
    opt = RicciFlowOptimizer(
        [weight],
        lr=0.01,
        use_convex_regularization=True,
        convex_lambda=0.1,
    )

    # Forward pass mock
    loss = (weight ** 2).sum()
    loss.backward()

    opt.step()
    assert torch.isfinite(weight).all()
