"""
Modular Virtualization (Residue Number System - RNS) Layer

Replaces continuous gradient drift with discrete arithmetic geometry.
Virtualizes floats onto a finite field (Z/pZ) utilizing repunit sequences
for cyclic overflow prevention boundaries.
"""

import torch
import torch.nn as nn
from src.core.fgrt_primitives import RepunitHasher, PrimeResonanceLadder
from src.core.primitive_ops import stochastic_round
from typing import Optional

class ModularVirtualizationLayer(nn.Module):
    """
    A hybrid layer mapping floating-point states into a Residue Number System (RNS)
    with Palindromic Symmetry.
    
    Refined Update: Integrates the prime-based torus with palindromic repunit mirrors.
    The hybrid modulus is defined as the product of the prime resonance and its 
    corresponding repunit symmetry: hybrid_modulus = p * R_p.
    """
    def __init__(self, dim: int, base: int = 2, legacy_mode: bool = False, device: str = None):
        super().__init__()
        self.dim = dim
        self.device = device
        self.base = base
        self.legacy_mode = legacy_mode
        self.scale_factor = 1e4
        
        # Prime Resonance Alignment: Fetch (p, R_p) pairs
        self.resonance_ladder = PrimeResonanceLadder(num_resonators=max(dim, 100))
        self.register_buffer('primes', self.resonance_ladder.primes[:dim])
        self.register_buffer('repunits', self.resonance_ladder.repunits[:dim])
        
        # Hasher for auxiliary palindromic markers (backward compatibility)
        self.hasher = RepunitHasher(base=2, sequence_length=max(dim, 10), device=device)
        
    def get_hybrid_modulus(self) -> torch.Tensor:
        """
        Fetch the composite hybrid modulus: p * R_p.
        If legacy_mode is active, returns just the prime basis.
        """
        if self.legacy_mode:
            return self.primes.float()
        return self.primes.float() * self.repunits.float()

    def repunit_crt_sparse_probe(self, candidate: torch.Tensor, target_residue: torch.Tensor) -> torch.Tensor:
        """
        O(1) Parity Filter (Repunit-CRT Sparse Probe)
        Rejects invalid trajectories at zero cost based on LSB parity.
        isValid = (candidate & 1) ^ (target_residue & 1) == 0
        """
        # Ensure integer bitwise compatibility
        c_bits = candidate.long()
        t_bits = target_residue.long()
        
        # XOR bitwise check
        parity_check = (c_bits & 1) ^ (t_bits & 1)
        return parity_check == 0

    def topological_refusal_snap(self, x: torch.Tensor, anchor_sym: torch.Tensor) -> torch.Tensor:
        """
        Topological Refusal: Snaps the state back to the nearest Birkhoff Polytope boundary.
        Biased towards the residue anchor (c_sym) to preserve Symbolic Non-Revisability (Law 1).
        """
        import math
        n = int(self.dim ** 0.5)
        if n * n == self.dim:
            from src.core.birkhoff_projection import DirectBirkhoffProjection
            if not hasattr(self, 'birkhoff_projector') or self.birkhoff_projector.n != n:
                self.birkhoff_projector = DirectBirkhoffProjection(n, device=x.device)
            
            # Project onto the Birkhoff subspace
            projected = self.birkhoff_projector(x)
            
            # Snap factor bias to anchor to preserve Symbolic Non-Revisability (Law 1)
            snap_factor = 0.8
            snapped = projected + snap_factor * (anchor_sym - projected)
            return snapped
        else:
            # Fallback discrete snap when dimension is not a perfect square matrix representation
            snap_factor = 0.8
            snapped = x + snap_factor * (anchor_sym - x)
            return snapped

    def float_to_rns(self, tensor: torch.Tensor, anchor_sym: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Quantize state scaling onto a hybrid modular torus.
        Utilizes the Repunit-CRT Sparse Probe for zero-cost rejection.
        """
        modulus = self.get_hybrid_modulus().to(tensor.device)
        
        # Scale for integer arithmetic (Signal Sovereignty precision)
        # Use stochastic rounding instead of deterministic round
        integerized = (stochastic_round(tensor, self.scale_factor) * self.scale_factor).long()
        
        # 1. Repunit-CRT Sparse Probe (Fast-Reject)
        # If anchor_sym is provided, use it as the target parity anchor
        if anchor_sym is not None:
            target_residue = torch.remainder(
                (stochastic_round(anchor_sym, self.scale_factor) * self.scale_factor).long(), 
                modulus.long()
            )
            is_valid = self.repunit_crt_sparse_probe(integerized, target_residue)
            
            # If invalid, apply topological refusal snap before modulo
            if not torch.all(is_valid):
                integerized = (stochastic_round(
                    self.topological_refusal_snap(tensor, anchor_sym), 
                    self.scale_factor
                ) * self.scale_factor).long()

        # 2. Modulo bound constraints (Hybrid RNS representation)
        modular_residues = torch.remainder(integerized, modulus.long())
        return modular_residues.float()

    def rns_to_float(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Inverse projection from hybrid finite field back to float.
        """
        # De-scale
        scale_factor = 1e4
        return residues / scale_factor

    def fast_congruence_check(self, state_a: torch.Tensor, state_b: torch.Tensor, tolerance: float = 0.10) -> bool:
        """
        Check digit-pattern congruence (warmstart shortcut) using the hybrid basis.
        """
        rns_a = self.float_to_rns(state_a)
        rns_b = self.float_to_rns(state_b)
        
        modulus = self.get_hybrid_modulus().to(state_a.device)
        
        # Wrapped cyclic distance
        diff = torch.abs(rns_a - rns_b)
        cyclic_dist = torch.min(diff, modulus - diff)
        
        mean_dist = torch.mean(cyclic_dist / modulus)
        return mean_dist.item() < tolerance
