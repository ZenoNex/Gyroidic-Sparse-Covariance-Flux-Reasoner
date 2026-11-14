import unittest
import torch

from src.core.hardware_monitor import (
    HardwareMonitor,
    ComputeAffordanceType,
    TypedComputeAffordance,
    has_compute_affordance,
    get_typed_affordance,
    get_affordance_gauges,
    get_compute_affordance_telemetry
)
from src.core.orchestrator import UniversalOrchestrator
from src.core.veto_subspace import VetoLevel
from src.core.admr_solver import PolynomialADMRSolver
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.topology.bonfire_network import BonfireNetwork
from src.ui.diegetic_backend import DiegeticPhysicsEngine

class TestComputeAffordanceRehybridization(unittest.TestCase):
    def setUp(self):
        self.hw = HardwareMonitor.get_instance()
        self.hw.update_thermodynamics(p_waste=0.0, thermal_runaway=False)

    def test_typed_affordances_across_all_regimes(self):
        for atype in ComputeAffordanceType:
            aff = get_typed_affordance(atype)
            self.assertIsInstance(aff, TypedComputeAffordance)
            self.assertEqual(aff.affordance_type, atype)
            self.assertGreaterEqual(aff.gauge, 0.0)
            self.assertLessEqual(aff.gauge, 1.0)
            self.assertIsInstance(aff.is_admissible, bool)

        telemetry = get_compute_affordance_telemetry()
        self.assertEqual(telemetry["status"], "ok")
        self.assertIn("gauges", telemetry)
        self.assertIn("affordances", telemetry)

    def test_community_p2p_mesh_affordance_synchronization(self):
        # Update P2P mesh affordance via BonfireNetwork
        bonfire = BonfireNetwork(node_id="test_node", local_url="http://localhost:8000")
        try:
            budget = bonfire.get_collective_compute_budget()
            self.assertGreaterEqual(budget, 0.1)
            self.assertLessEqual(budget, 1.0)
            
            # Sync with HardwareMonitor
            self.hw.update_p2p_mesh(budget, freenet_online=True, active_peers=2)
            mesh_aff = get_typed_affordance(ComputeAffordanceType.P2P_MESH)
            self.assertTrue(mesh_aff.is_admissible)
            self.assertGreaterEqual(mesh_aff.gauge, 0.70)
            self.assertTrue(mesh_aff.telemetry.get("freenet_online", False))
            self.assertEqual(mesh_aff.telemetry.get("active_peers", 0), 2)
        finally:
            bonfire.close()

    def test_orchestrator_budget_veto_delegation(self):
        orchestrator = UniversalOrchestrator(dim=32)
        
        # Under normal conditions, budget veto should not trigger
        vetoed = orchestrator.check_veto(VetoLevel.BUDGET)
        self.assertIsInstance(vetoed, bool)
        
        # Test affordance property methods on orchestrator
        self.assertTrue(isinstance(orchestrator.affordance_status, bool))
        gauges = orchestrator.affordance_gauges
        self.assertIn("silicon", gauges)
        self.assertIn("p2p_mesh", gauges)

    def test_admr_solver_batch_affordance_arbitration(self):
        dim = 32
        cfg = PolynomialCoprimeConfig(k=5, degree=4)
        solver = PolynomialADMRSolver(poly_config=cfg, state_dim=dim)
        states = torch.randn(2, dim)
        neighbors = torch.randn(2, 3, dim)
        adj_weight = torch.softmax(torch.randn(2, 3), dim=-1)
        
        # Run tournament drafts - should scale without psutil exception
        out = solver.tournament_differential_step(states, neighbors, adj_weight, num_drafts=4, dt=0.01)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, states.shape)

    def test_diegetic_engine_adaptive_training_step(self):
        engine = DiegeticPhysicsEngine(dim=64, device='cpu', config={'lightweight': True})
        state = torch.randn(1, 64)
        
        # _train_mimicry dynamically scales sequence length via HardwareMonitor without raw psutil
        engine._train_mimicry(input_state=state, text_target="Testing central compute affordance rehybridization.")
        self.assertIsNotNone(engine.larynx)

    def test_500ms_restless_thermal_guard_and_rehybridized_invariants(self):
        from src.core.hardware_monitor import enforce_500ms_restless_thermal_guard
        from src.core.invariants import (
            compute_ihc_bao_shell_modulation,
            compute_fractal_lhopital_boundary_ratio,
            compute_enso_inkblot_emergence,
        )

        # 1. Verify 500ms Restless Thermal Latency Guard
        guard_fast = enforce_500ms_restless_thermal_guard(execution_ms=120.0, cooldown_sleep_sec=0.0)
        self.assertFalse(guard_fast["exceeds_500ms_thermal_guard"])
        guard_slow = enforce_500ms_restless_thermal_guard(execution_ms=650.0, cooldown_sleep_sec=0.01)
        self.assertTrue(guard_slow["exceeds_500ms_thermal_guard"])
        self.assertEqual(guard_slow["routing_mode"], "rp4_void_lazarus_play")
        self.hw.reset_thermodynamics()

        # 2. Verify IHC BAO RP^4 shell & Z_3 standing-wave modulation
        bao = compute_ihc_bao_shell_modulation(z_effective=0.754)
        self.assertAlmostEqual(bao["f_k1_shell_factor"], 1.0183535, places=4)
        self.assertGreater(bao["r_s_ihc_mpc"], 150.0)

        # 3. Verify Modified L'Hopital Fractal Boundary ratio & Enso/Inkblot emergence
        r = torch.sin(torch.linspace(0, 6.28, 64))
        t = torch.cos(torch.linspace(0, 6.28, 64))
        lh = compute_fractal_lhopital_boundary_ratio(r, t)
        self.assertGreater(lh["damping_factor"], 0.2)
        enso = compute_enso_inkblot_emergence(r, r - t)
        self.assertGreater(enso["enso_interior_stillness"], 0.5)
        self.assertGreater(enso["inkblot_boundary_warp"], 0.0)

if __name__ == '__main__':
    unittest.main()
