"""
Structural Pressure Typing.

Enforces domain isolation and prevents scalarization traps.
Prevents cross-domain pressure aggregation to maintain non-teleological design.

Author: William Matthew Bryant
Created: January 2026
"""

import torch
from typing import Union

class StructuralPressure:
    """
    Individually typed pressure signal.
    Enforces that pressures from different domains cannot be summed or compared.
    """
    def __init__(self, value: torch.Tensor, domain: str):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.value = value
        self.domain = domain
        
    def backward(self, *args, **kwargs):
        """Proxy for the underlying tensor's backward."""
        return self.value.backward(*args, **kwargs)
        
    def item(self):
        """Proxy for the underlying tensor's item."""
        return self.value.item()
        
    @property
    def requires_grad(self):
        return self.value.requires_grad
        
    def __repr__(self):
        return f"StructuralPressure(domain='{self.domain}', value={self.value.item():.4f})"

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        if isinstance(other, StructuralPressure):
            if self.domain != other.domain:
                raise ValueError(
                    f"Scalarization Trap: Cannot sum pressure from '{self.domain}' "
                    f"and '{other.domain}'. Pressures must remain domain-isolated."
                )
            return StructuralPressure(self.value + other.value, self.domain)
        raise TypeError(f"Cannot add {type(other)} to StructuralPressure.")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other: Union[int, float, torch.Tensor]):
        # Scalar multiplication within domain is allowed for weighting.
        if isinstance(other, (int, float, torch.Tensor)):
            return StructuralPressure(self.value * other, self.domain)
        raise TypeError(f"Cannot multiply StructuralPressure by {type(other)}.")
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other: Union[int, float, torch.Tensor]):
        if isinstance(other, (int, float, torch.Tensor)):
            return StructuralPressure(self.value / other, self.domain)
        raise TypeError(f"Cannot divide StructuralPressure by {type(other)}.")
