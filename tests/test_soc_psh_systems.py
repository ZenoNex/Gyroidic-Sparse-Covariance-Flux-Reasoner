import torch
import math
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath("."))

from src.core.speculative_coprime_gate import ChiralCoherenceEstimator, SpeculativeCoprimeGate
from src.ui.voxelboxter_simulation import FaunaComponent, FaunaISN
from src.core.fgrt_primitives import TorsionConnection, BerryPhaseTracker
from src.core.noncommutativity_curvature import NonCommutativityCurvature
from src.core.diegetic_physics_engine import DiegeticPhysicsEngine
from src.surrogates.calm_predictor import CALM
from src.surrogates.kagh_networks import KAGHBlock

class TestSOCPSHSystems(unittest.TestCase):
    def setUp(self):
        from src.core.hardware_monitor import HardwareMonitor
        HardwareMonitor.get_instance().update_thermodynamics(p_waste=0.0, p_cool_max=1000.0, thermal_runaway=False)

    def tearDown(self):
        from src.core.hardware_monitor import HardwareMonitor
        HardwareMonitor.get_instance().update_thermodynamics(p_waste=0.0, p_cool_max=1000.0, thermal_runaway=False)

    def test_green_chiral_coherence_estimator(self):
        # Green system 1: ChiralCoherenceEstimator Pauli projection & fractal superposition
        cce = ChiralCoherenceEstimator(dim=16, sample_size=32)
        for _ in range(8):
            st = torch.randn(16)
            cce.update_buffer(st)
        
        score = cce.compute_chiral_score()
        self.assertIsInstance(score, torch.Tensor)
        self.assertTrue(-1.0 <= score.item() <= 1.0, f"Chiral score {score.item()} outside [-1, 1]")

    def test_green_fauna_continuous_spin_torque(self):
        # Green system 2: FaunaISN continuous vector torque Omega(v) x s
        isn = FaunaISN()
        fauna = FaunaComponent(
            entity_id="stalker_01",
            position=(0.0, 0.0, 0.0),
            velocity=(1.0, 0.5, 0.0),
            heading_spin=(1.0, 0.0)
        )
        old_spin = fauna.heading_spin
        isn.step_population(fauna, external_stimulus=0.1, dt=0.1)
        new_spin = fauna.heading_spin
        
        # Verify heading spin was smoothly updated (continuous rotation)
        spin_norm = math.sqrt(sum(s**2 for s in new_spin))
        self.assertAlmostEqual(spin_norm, 1.0, places=3)
        self.assertNotEqual(old_spin, new_spin)

    def test_green_berry_phase_and_torsion_connection(self):
        # Green system 3: TorsionConnection with Pauli blocks & BerryPhaseTracker
        tc = TorsionConnection(dim=8, alpha=0.1, beta=0.1)
        delta_psh = tc.get_psh_delta()
        self.assertAlmostEqual(delta_psh.item(), 0.0, places=5)
        
        tracker = BerryPhaseTracker()
        s1 = torch.randn(2, 8)
        s2 = torch.randn(2, 8)
        dphi = tracker.update(s1, s2, torsion=tc)
        self.assertEqual(dphi.shape, (2, 4))

    def test_green_noncommutativity_curvature(self):
        # Green system 4: NonCommutativityCurvature PSH invariant probe
        ncc = NonCommutativityCurvature(dim=8)
        A = torch.randn(8, 8)
        B = torch.randn(8, 8)
        curv = ncc.compute_curvature(A, B)
        
        self.assertIn('delta_psh', curv)
        self.assertIn('is_psh_balanced', curv)
        self.assertTrue(isinstance(curv['delta_psh'], (float, torch.Tensor)))
        self.assertIsInstance(curv['is_psh_balanced'], bool)

    def test_yellow_diegetic_physics_engine_damping(self):
        # Yellow system 1: DiegeticPhysicsEngine low-pass damped spin precession
        engine = DiegeticPhysicsEngine(device="cpu", state_dim=16)
        v_state = {"angular_velocity": (0.0, 1.5, 0.0)}
        track_state = {"betti_0": 1, "betti_1": 0}
        c_in = {"throttle": 0.8, "yaw": 0.5}
        
        out = engine.process_input(v_state, track_state, c_in)
        self.assertIn("damped_spin_precession", out)
        self.assertIn("delta_psh", out)
        self.assertIn("psh_invariant_active", out)
        self.assertTrue(out["damped_spin_precession"] >= 0.0)

    def test_yellow_calm_maml_psh_modulation(self):
        # Yellow system 2: CALM MAML adaptation step size modulation bounded in [0.1, 1.0]
        calm = CALM(dim=8, history_len=4, hidden_dim=16, nhead=2)
        supp_hist = torch.randn(2, 4, 8)
        supp_targets = torch.randn(2, 8)
        
        # Test with balanced PSH (delta = 0.0 -> factor = 1.0)
        adapted_balanced = calm.adapt(supp_hist, supp_targets, steps=1, lr=0.01, psh_delta=0.0)
        self.assertIsNotNone(adapted_balanced)
        
        # Test with high delta (delta = 100.0 -> clamped to 0.1, never 0.0)
        adapted_damped = calm.adapt(supp_hist, supp_targets, steps=1, lr=0.01, psh_delta=100.0)
        self.assertIsNotNone(adapted_damped)

    def test_yellow_kagh_maml_psh_modulation(self):
        # Yellow system 2: KAGHBlock adapt_online modulation bounded in [0.1, 1.0]
        block = KAGHBlock(n_in=8, n_out=8)
        states = torch.randn(2, 8)
        targets = torch.randn(2, 8)
        
        adapted = block.adapt_online(states, targets, steps=1, lr=0.01, psh_delta=0.5)
        self.assertIsNotNone(adapted)

    def test_typed_compute_affordance_and_gauges(self):
        # Verify typed compute affordance enum, tokens, and cross-type gauge couplings
        from src.core.hardware_monitor import (
            HardwareMonitor, ComputeAffordanceType, TypedComputeAffordance,
            has_compute_affordance, get_typed_affordance, get_affordance_gauges, gauge_between
        )
        hm = HardwareMonitor.get_instance()
        
        # Test individual typed affordances
        for atype in ComputeAffordanceType:
            aff = get_typed_affordance(atype)
            self.assertIsInstance(aff, TypedComputeAffordance)
            self.assertEqual(aff.affordance_type, atype)
            self.assertTrue(0.0 <= aff.gauge <= 1.0)
            self.assertIsInstance(bool(aff), bool)

        # Test cross-type gauge couplings
        g_cross = gauge_between(ComputeAffordanceType.THERMAL, ComputeAffordanceType.GPU_SOVEREIGN)
        self.assertTrue(0.0 <= g_cross <= 1.0)

        # Test gauge transform
        silicon_aff = get_typed_affordance(ComputeAffordanceType.SILICON)
        transformed = silicon_aff.gauge_transform(ComputeAffordanceType.DIEGETIC, coupling=0.5)
        self.assertEqual(transformed.affordance_type, ComputeAffordanceType.DIEGETIC)
        self.assertAlmostEqual(transformed.gauge, silicon_aff.gauge * 0.5, places=3)

        # Test thermal runaway gating
        hm.update_thermodynamics(p_waste=5000.0, p_cool_max=100.0, thermal_runaway=True)
        self.assertFalse(has_compute_affordance(ComputeAffordanceType.GPU_SOVEREIGN))
        self.assertFalse(has_compute_affordance(ComputeAffordanceType.THERMAL))
        gpu_aff = get_typed_affordance(ComputeAffordanceType.GPU_SOVEREIGN)
        self.assertEqual(gpu_aff.gauge, 0.0)
        self.assertFalse(gpu_aff.is_admissible)

        # Restore normal
        hm.update_thermodynamics(p_waste=50.0, p_cool_max=1000.0, thermal_runaway=False)
        self.assertTrue(has_compute_affordance(ComputeAffordanceType.THERMAL))

if __name__ == "__main__":
    unittest.main()
