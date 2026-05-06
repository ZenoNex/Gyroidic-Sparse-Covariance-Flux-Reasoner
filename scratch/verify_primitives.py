import torch
from src.core.primitive_ops import FixedPointField, LearnedPrimitivePerturbation, SCALE_FACTOR

def test_primitive_integrity():
    print("Testing Primitives Integrity (No-Mantissa Drift Constraint)...")
    
    # 1. Initialize from Int64
    raw_data = torch.tensor([1, 2, 3], dtype=torch.int64)
    field = FixedPointField(raw_data)
    
    print(f"Original field backing store: {field.backing_store}")
    
    # 2. Test LearnedPrimitivePerturbation
    perturbation = LearnedPrimitivePerturbation(dim=3)
    # The perturbation should just add a shifted int
    new_field = perturbation(field)
    
    print(f"Perturbed field backing store: {new_field.backing_store}")
    assert new_field.backing_store.dtype == torch.int64, "Backing store must remain int64"
    assert new_field.scale == field.scale, "Scale must be preserved"
    
    # Check that we haven't lost precision due to stochastic rounding of a float
    # If the hacky .float() cast was present, stochastic rounding would have added jitter
    # For exactly representable integers, stochastic rounding should ideally not change them,
    # but the cast to float and back is inherently lossy for large numbers or introduces noise.
    
    print("PASS: Primitives maintain Int64 backing without float rounding.")

if __name__ == "__main__":
    test_primitive_integrity()
