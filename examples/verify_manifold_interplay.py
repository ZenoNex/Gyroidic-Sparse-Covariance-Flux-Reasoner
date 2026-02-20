"""
Verification of Asymptotic Manifold Interplay.

Tests:
1. ManifoldClock: Pressure -> dt scaling (Seriousness vs Play).
2. SituationalBatchSampler: Entanglement update (Co-arising).
3. StructuralAdaptor Integration: Feedback loop.
"""

import torch
import torch.nn as nn
from src.core.manifold_time import ManifoldClock
from src.core.situational_batching import SituationalBatchSampler
from src.training.trainer import ConstraintDataset, StructuralAdaptor
import matplotlib.pyplot as plt
import os


def verify_breathing():
    print("--- Verifying Manifold Breathing (Time Dilation) ---")
    clock = ManifoldClock(dt_base=1.0, dt_min=0.1, dt_max=2.0)
    
    pressures = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 0.0]
    dt_history = []
    mode_history = []
    
    for p in pressures:
        dt = clock.tick(torch.tensor(p))
        dt_history.append(dt)
        mode = "SERIOUS" if dt < 0.8 else "PLAY" if dt > 1.2 else "NEUTRAL"
        mode_history.append(mode)
        print(f"Pressure: {p:.1f} -> dt: {dt:.3f} [{mode}]")
    
    # Assertions
    assert dt_history[0] > dt_history[3], "dt should shrink as pressure increases"
    assert dt_history[-1] > dt_history[-2], "dt should expand as pressure drops"
    print("✓ Breathing logic verified.")


def verify_entanglement():
    print("\n--- Verifying Situational Entanglement (Co-arising) ---")
    num_samples = 100
    batch_size = 10
    sampler = SituationalBatchSampler(num_samples=num_samples, batch_size=batch_size, play_ratio=0.1)
    
    # Simulate high conflict between indices 0, 1, and 2
    indices_a = [0, 1, 2, 10, 11, 12, 13, 14, 15, 16]
    pressure_high = torch.tensor(2.0)
    
    for _ in range(5):
        sampler.update_entanglement(indices_a, pressure_high)
    
    summary = sampler.get_entanglement_summary()
    print(f"Entanglement Summary: {summary}")
    
    # Check if sampler groups 0, 1, 2 together
    batch_contains_conflict = 0
    num_trials = 20
    for _ in range(num_trials):
        # We need to restart the iterator or just check batches
        for batch in sampler:
            if 0 in batch:
                if 1 in batch or 2 in batch:
                    batch_contains_conflict += 1
                break
    
    ratio = batch_contains_conflict / num_trials
    print(f"Co-occurrence ratio for entangled indices: {ratio:.2f}")
    assert ratio > 0.5, "Entangled indices should co-occur more frequently than random (0.1)"
    print("✓ Situational batching verified.")


def verify_full_loop():
    print("\n--- Verifying Full Adaptor Integration ---")
    # Mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.randn(10))
            self.p_calls = 0
        def forward(self, **kwargs):
            self.p_calls += 1
            # Return increasing pressure to trigger seriousness
            p = self.p_calls * 0.1
            return {
                'gyroid_pressure': torch.tensor(p),
                'selection_pressure': torch.tensor(p),
                'num_cycles': 5
            }
        def validate_structural_integrity(self):
            return torch.tensor([True])

    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adaptor = StructuralAdaptor(model, optimizer)
    
    dataset = ConstraintDataset(num_samples=40)
    
    # Run 1 epoch
    adaptor.adapt(dataset, batch_size=8, num_epochs=1, log_interval=1)
    
    state = adaptor.clock.get_state()
    print(f"Final Clock State: {state}")
    assert state['t'] > 0
    assert state['dt'] < 1.0, "Clock should have dilated due to increasing mock pressure"
    print("✓ Full loop integration verified.")


if __name__ == "__main__":
    verify_breathing()
    verify_entanglement()
    verify_full_loop()
    print("\nALL VERIFICATIONS PASSED.")
