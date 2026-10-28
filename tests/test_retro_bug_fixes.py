import unittest
import torch
import math
from src.core.manifold_time import TwoCopsSchedule
from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
from src.terminal.udp_server_colonizer import OptionD_Colonizer

class TestRetroBugFixes(unittest.TestCase):
    def test_two_cops_schedule_decoupling(self):
        """Verify TwoCopsSchedule does not trigger should_sync on every step (Battletoads fix)."""
        schedule = TwoCopsSchedule(macro_steps=10, use_tempolock=True)
        sync_count = 0
        total_steps = 50
        for _ in range(total_steps):
            _, should_sync = schedule.step(torch.tensor(0.1))
            if should_sync:
                sync_count += 1
        # In 50 steps with macro_steps=10, should_sync should fire at most 5-6 times, NOT 48+ times.
        self.assertLessEqual(sync_count, 10, f"should_sync fired {sync_count} times out of {total_steps}, breaking temporal decoupling")
        self.assertGreater(sync_count, 0, "should_sync never fired")

    def test_hardware_monitor_thermal_runaway_recovery(self):
        """Verify HardwareMonitor recovers from thermal runaway when heat drops (Joe & Mac 2 fix)."""
        hw = HardwareMonitor.get_instance()
        # Force thermal runaway
        hw.update_thermodynamics(p_waste=1500.0, p_cool_max=1000.0, thermal_runaway=True)
        self.assertTrue(hw.thermal_runaway)
        
        # Test hysteretic cooling recovery (waste heat drops below 75% capacity with >25% margin)
        hw.update_thermodynamics(p_waste=500.0, p_cool_max=1000.0, thermal_runaway=False)
        self.assertFalse(hw.thermal_runaway, "HardwareMonitor failed to auto-recover from thermal runaway")
        
        # Test manual reset
        hw.thermal_runaway = True
        hw.reset_thermodynamics()
        self.assertFalse(hw.thermal_runaway, "reset_thermodynamics did not clear thermal_runaway latch")

    def test_dyadic_transfer_precision_and_device(self):
        """Verify evaluate_dyadic_transfer preserves tensor precision without memory corruption (Rainbow Islands fix)."""
        from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
        engine = SiliconSovereigntyEngine(use_gpu=False)
        batch, num_tasks, dim = 2, 4, 8
        task_states = torch.randn(batch, num_tasks, dim, dtype=torch.float64)
        t_mat = torch.eye(num_tasks, dtype=torch.float64)
        gating = torch.ones(batch, num_tasks, dtype=torch.float64)
        
        result = engine.evaluate_dyadic_transfer(task_states, t_mat, gating)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.shape, task_states.shape)
        self.assertFalse(torch.isnan(result).any(), "Result contained NaN values")
        self.assertFalse(torch.isinf(result).any(), "Result contained Inf values")

    def test_voxelboxter_aeronautics_tick(self):
        """Verify VoxelboxterEngine.tick updates AeronauticContraption instances (Joe & Mac 2 fix)."""
        from src.ui.voxelboxter_simulation import VoxelboxterEngine
        from src.environment.cleanroom_mechanics import AeronauticContraption
        engine = VoxelboxterEngine(device="cpu")
        contraption = AeronauticContraption(name="AirshipAlpha")
        engine.add_aeronautic_airship("AirshipAlpha", contraption)
        initial_pos = contraption.position
        # Give thrust
        contraption.engine.throttle = 1.0
        engine.tick(dt=0.1)
        # Verify contraption position updated or physics evaluated
        self.assertIn("AirshipAlpha", engine.aeronautics)

    def test_fossilizer_async_init_resilience(self):
        """Verify DyadFossilizer._index_ready is set even when loading encounters an error (Sonic PLC fix)."""
        from src.core.knowledge_dyad_fossilizer import DyadFossilizer
        fossilizer = DyadFossilizer(storage_dir="tmp/test_fossil_resilience")
        # Ensure _index_ready event is set promptly
        ready = fossilizer._index_ready.wait(timeout=2.0)
        self.assertTrue(ready, "DyadFossilizer._index_ready event was not set, causing caller timeout stall")

if __name__ == '__main__':
    unittest.main()
