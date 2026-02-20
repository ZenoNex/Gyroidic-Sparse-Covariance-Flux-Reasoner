"""
Verification of Non-Scalarization and Pressure Typing.
Ensures that pressures from different domains cannot be summed or ranked.
"""

import torch
import pytest
from src.core.pressure_typing import StructuralPressure

def test_domain_isolation():
    """Assert that summing pressures from different domains raises ValueError."""
    p1 = StructuralPressure(torch.tensor(0.5), "selection")
    p2 = StructuralPressure(torch.tensor(0.3), "containment")
    
    with pytest.raises(ValueError, match="Scalarization Trap"):
        total = p1 + p2
        
def test_same_domain_summation():
    """Assert that pressures from the same domain can be summed (e.g., for multi-residue aggregate)."""
    p1 = StructuralPressure(torch.tensor(0.5), "selection")
    p2 = StructuralPressure(torch.tensor(0.3), "selection")
    
    total = p1 + p2
    assert total.domain == "selection"
    assert total.item() == 0.8

def test_scalar_weighting():
    """Assert that pressures can be weighted by scalars within their domain."""
    p1 = StructuralPressure(torch.tensor(0.5), "selection")
    weighted = 0.5 * p1
    assert weighted.domain == "selection"
    assert weighted.item() == 0.25

def test_pytorch_compatibility():
    """Assert that StructuralPressure proxies common torch methods."""
    x = torch.tensor([1.0], requires_grad=True)
    p = StructuralPressure(x * 2.0, "test")
    
    assert p.requires_grad
    p.backward()
    assert x.grad is not None
    assert x.grad.item() == 2.0

if __name__ == "__main__":
    print("Running Non-Scalarization Verifications...")
    try:
        test_domain_isolation()
        print("  ✅ Domain Isolation Confirmed.")
        test_same_domain_summation()
        print("  ✅ Same-Domain Summation Confirmed.")
        test_scalar_weighting()
        print("  ✅ Scalar Weighting Confirmed.")
        test_pytorch_compatibility()
        print("  ✅ PyTorch Compatibility Confirmed.")
    except Exception as e:
        print(f"  ❌ Verification Failed: {e}")
        exit(1)
    
    print("\nLogic Hardening: Structural Tripwires are ACTIVE.")
