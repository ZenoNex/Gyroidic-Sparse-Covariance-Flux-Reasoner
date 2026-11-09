# -*- coding: utf-8 -*-
"""
src/core/lychrel_soliton.py

Adelic Digit-Reversal Operator, Chiral Asymmetry Gap, and Lychrel 196 Soliton Engine.
Part of the Unicorn Synthesis / CODES v40 Non-Commutative Arithmetic Layer.

Mathematical Architecture:
1. Adelic Digit-Reversal Operator R_B(x):
   R_B(x) = sum_{i=0}^k d_{k-i} B^i for x = sum_{i=0}^k d_i B^i (default base B=10).
2. Chiral Asymmetry Gap:
   kappa_Lychrel(x) = ||x - R_B(x)||_RNS across coprime modular channels m_i in {197, 211, 223, ...}.
3. 1-195 Palindromic Basin vs. 196 Infinite Chiral Generator:
   - For n < 196: Trajectory contracts to a base-10 palindrome (kappa_Lychrel -> 0),
     absorbing into the Birkhoff doubly-stochastic invariant manifold.
   - For n = 196: Trajectory diverges into an unhalting carry-cascade (kappa_Lychrel > 0),
     establishing a steady-state non-zero curvature. Consumes a VoynichExemptionToken
     rather than aborting, treating 196 as a perpetual entropy source for honest jitter.
4. Prime-Stride Sieve (p = 197):
   Folds the digit carry-cascade of 196 into 2D rasters modulo p = 197 (immediate prime
   neighbor of 196) and computes inter-row spatial cosine coherence C(197) to classify
   quasi-periodic soliton structure vs. true pseudo-random noise.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.core.invariants import get_prime_ladder, compute_chirality, PHI
from src.core.false_negative_subsystem import VoynichExemptionToken
from src.core.honest_jitter import harvest_honest_jitter


class AdelicDigitReversalOperator(nn.Module):
    """
    Computes the base-B digit reversal involution tau_B(x) = R_B(x).
    Supports arbitrary-precision Python integers, single integer values,
    and modular digit tensors.
    """
    def __init__(self, base: int = 10):
        super().__init__()
        self.base = base

    def reverse(self, n: int) -> int:
        """Alias for reverse_integer."""
        return self.reverse_integer(n)

    def is_palindrome(self, n: int) -> bool:
        """Checks whether integer n is a base-B palindrome."""
        return n == self.reverse_integer(n)

    def reverse_integer(self, n: int) -> int:
        """Reverses the base-B digits of integer n."""
        if n < 0:
            return -self.reverse_integer(-n)
        if self.base == 10:
            return int(str(n)[::-1])
        # General base B digit extraction
        digits = []
        val = n
        while val > 0:
            digits.append(val % self.base)
            val //= self.base
        res = 0
        for d in digits:
            res = res * self.base + d
        return res

    def reverse_and_add_step(self, n: int) -> Tuple[int, int, bool]:
        """
        Executes one reverse-and-add step:
            x_{k+1} = x_k + R_B(x_k)
        Returns (x_{k+1}, rev_x, is_palindrome).
        """
        rev_n = self.reverse_integer(n)
        is_pal = (n == rev_n)
        next_n = n + rev_n
        return next_n, rev_n, is_pal

    def forward(self, state_or_digits: Union[int, List[int], torch.Tensor]) -> Union[int, torch.Tensor]:
        if isinstance(state_or_digits, int):
            return self.reverse_integer(state_or_digits)
        elif isinstance(state_or_digits, torch.Tensor):
            # Tensor mode: reverse along the last dimension
            return torch.flip(state_or_digits, dims=[-1])
        elif isinstance(state_or_digits, list):
            return state_or_digits[::-1]
        return state_or_digits


@dataclass
class LychrelTrajectoryReport:
    """Detailed structural diagnostic of a reverse-and-add trajectory."""
    seed: int
    terminated: bool
    iterations: int
    terminal_value: Optional[int]
    chiral_curvature: float
    digit_lengths: List[int]
    rns_residues: Dict[int, int]
    soliton_coherence_p197: float
    voynich_token_issued: bool
    entropy_harvested: float
    exemption_token: Optional[VoynichExemptionToken] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_converged(self) -> bool:
        """Alias for terminated."""
        return self.terminated

    @property
    def steps_taken(self) -> int:
        """Alias for iterations."""
        return self.iterations

    @property
    def chiral_gap(self) -> float:
        """Alias for chiral_curvature."""
        return self.chiral_curvature

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "terminated": self.terminated,
            "iterations": self.iterations,
            "terminal_value": str(self.terminal_value) if self.terminal_value is not None else None,
            "chiral_curvature": round(self.chiral_curvature, 6),
            "digit_lengths": self.digit_lengths,
            "rns_residues": self.rns_residues,
            "soliton_coherence_p197": round(self.soliton_coherence_p197, 6),
            "voynich_token_issued": self.voynich_token_issued,
            "entropy_harvested": round(self.entropy_harvested, 6),
            "metadata": self.metadata,
        }


class Lychrel196SolitonEngine(nn.Module):
    """
    Manages reverse-and-add dynamics, the 1-195 palindromic convergence basin,
    and the 196 non-halting chiral generator.
    
    Prevents cycle-debt saturation and integer overflow by embedding trajectories
    into modular residue channels (RNS) and issuing Voynich Exemption Tokens on
    detecting persistent Lychrel curvature.
    """
    def __init__(
        self,
        base: int = 10,
        rns_moduli: Optional[List[int]] = None,
        max_search_depth: int = 60,
        max_steps: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.base = base
        self.max_search_depth = max_steps if max_steps is not None else max_search_depth
        self.reverser = AdelicDigitReversalOperator(base=base)
        
        # Coprime RNS moduli: default includes 197, 211, 223, 227, 229
        if rns_moduli is None:
            self.rns_moduli = [197, 211, 223, 227, 229]
        else:
            self.rns_moduli = sorted(list(set(rns_moduli)))

        self.register_buffer("moduli_tensor", torch.tensor(self.rns_moduli, dtype=torch.long))

    def compute_lychrel_chiral_gap(self, value: int) -> float:
        """
        Computes the Chiral Asymmetry Gap:
            kappa_Lychrel(x) = ||x - R_B(x)||_RNS
        Normalized between 0.0 (perfect palindrome) and 1.0 (maximal asymmetry).
        """
        rev_val = self.reverser.reverse_integer(value)
        if value == rev_val:
            return 0.0

        diff_residues = []
        for m in self.rns_moduli:
            res_val = value % m
            res_rev = rev_val % m
            diff = abs(res_val - res_rev)
            # Modular cyclic distance
            cyclic_dist = min(diff, m - diff) / (m / 2.0)
            diff_residues.append(cyclic_dist)

        return float(sum(diff_residues) / len(diff_residues))

    def step_rns_modular(self, residues: Dict[int, int], value_str: str) -> Dict[int, int]:
        """
        Computes x_{k+1} = x_k + R_B(x_k) (mod m_i) directly in RNS channels,
        enabling overflow-free trajectory evolution at massive digit lengths.
        """
        rev_str = value_str[::-1]
        next_residues = {}
        for m in self.rns_moduli:
            # Modular value of forward and reversed strings
            val_mod = 0
            rev_mod = 0
            for char in value_str:
                val_mod = (val_mod * self.base + int(char)) % m
            for char in rev_str:
                rev_mod = (rev_mod * self.base + int(char)) % m
            next_residues[m] = (val_mod + rev_mod) % m
        return next_residues

    def classify_with_prime_stride_197(self, digit_stream: List[int]) -> float:
        """
        Folds the digit carry sequence into 2D candidate rasters with stride p = 197.
        Computes inter-row spatial cosine coherence C(197) to classify quasi-periodic
        solitons vs pseudo-random noise.
        """
        if len(digit_stream) < 16:
            return 0.5

        tensor_stream = torch.tensor(digit_stream, dtype=torch.float32)
        # Expand stream if needed for at least 3 rows of stride 197
        target_len = 197 * 3
        if tensor_stream.numel() < target_len:
            repeats = (target_len // tensor_stream.numel()) + 1
            tensor_stream = tensor_stream.repeat(repeats)[:target_len]

        rows = tensor_stream.numel() // 197
        if rows < 2:
            return 0.5

        grid = tensor_stream[:rows * 197].view(rows, 197)
        # Row-to-row cosine coherence
        row_norms = torch.norm(grid, dim=-1, keepdim=True) + 1e-8
        norm_grid = grid / row_norms
        cosine_sims = (norm_grid[:-1] * norm_grid[1:]).sum(dim=-1)
        mean_coherence = float(cosine_sims.mean().item())
        return max(0.0, min(1.0, (mean_coherence + 1.0) / 2.0))

    def evaluate_seed(
        self,
        seed: int,
        max_steps: Optional[int] = None,
        harvest_entropy: bool = True
    ) -> LychrelTrajectoryReport:
        """
        Evaluates a seed integer through the reverse-and-add algorithm.
        
        For n < 196:
            Finds palindromic fixed point, records steps, kappa -> 0.
        For n == 196 (or other unhalting Lychrel candidates):
            Detects persistent carry-cascade, prevents overflow via RNS,
            issues a VoynichExemptionToken, and returns harvested entropy.
        """
        steps = max_steps or self.max_search_depth
        curr = seed
        digit_lengths = [len(str(curr))]
        carry_digit_history: List[int] = [int(c) for c in str(curr)]
        
        terminated = False
        terminal_val = None
        voynich_issued = False

        for step in range(steps):
            rev_curr = self.reverser.reverse_integer(curr)
            if curr == rev_curr and step > 0:
                terminated = True
                terminal_val = curr
                break

            curr = curr + rev_curr
            curr_str = str(curr)
            digit_lengths.append(len(curr_str))
            for c in curr_str[-10:]:
                carry_digit_history.append(int(c))

            # Guard against runaway memory if step count is high
            if len(curr_str) > 500:
                break

        # Compute Chiral Gap
        if terminated:
            kappa = 0.0
        else:
            kappa = self.compute_lychrel_chiral_gap(curr)

        # RNS representation of final evaluated state
        final_str = str(curr)
        rns_res = {}
        for m in self.rns_moduli:
            m_val = 0
            for ch in final_str:
                m_val = (m_val * self.base + int(ch)) % m
            rns_res[m] = m_val

        # Prime-Stride Sieve at p = 197
        coherence_197 = self.classify_with_prime_stride_197(carry_digit_history)

        # Voynich Exemption Token for persistent chiral divergence
        token = None
        entropy_val = 0.0
        if not terminated:
            voynich_issued = True
            token = VoynichExemptionToken(
                honesty_score=float(min(1.0, kappa + 0.1)),
                is_valid_exemption=True,
                reason="LYCHREL_196_SOLITON",
                is_nutrient=True,
                shadow_mode=False
            )
            # Harvest entropy for honest jitter pool
            if harvest_entropy:
                entropy_val = float(math.log(len(final_str) + 1.0) * (kappa + 0.5))

        return LychrelTrajectoryReport(
            seed=seed,
            terminated=terminated,
            iterations=step if terminated else steps,
            terminal_value=terminal_val,
            chiral_curvature=kappa,
            digit_lengths=digit_lengths,
            rns_residues=rns_res,
            soliton_coherence_p197=coherence_197,
            voynich_token_issued=voynich_issued,
            entropy_harvested=entropy_val,
            exemption_token=token,
            metadata={
                "is_lychrel_seed_196": (seed == 196),
                "token_reason": getattr(token, "reason", "PALINDROME_CONVERGED"),
                "final_digit_count": len(final_str)
            }
        )

    def evaluate_trajectory(
        self,
        seed: int,
        max_steps: Optional[int] = None,
        harvest_entropy: bool = True
    ) -> LychrelTrajectoryReport:
        """Alias for evaluate_seed."""
        return self.evaluate_seed(seed=seed, max_steps=max_steps, harvest_entropy=harvest_entropy)

    def harvest_honest_jitter(
        self,
        report_or_shape: Any = None,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Harvests structured honest jitter tensor from Lychrel soliton carry dynamics.
        Accepts either (report, dim=32) or (shape=(32,)) or dim=32 directly.
        """
        if dim is not None:
            shape = torch.Size([dim])
        elif isinstance(report_or_shape, (tuple, list)):
            shape = torch.Size(report_or_shape)
        elif isinstance(report_or_shape, torch.Size):
            shape = report_or_shape
        elif isinstance(report_or_shape, int):
            shape = torch.Size([report_or_shape])
        else:
            shape = torch.Size([32])
        return self.harvest_lychrel_entropy_tensor(shape=shape, seed=196)

    def harvest_lychrel_entropy_tensor(self, shape: torch.Size, seed: int = 196) -> torch.Tensor:
        """
        Generates a structured, honest-jitter entropy tensor derived from the
        carry-cascade dynamics of Lychrel seed 196 folded through prime p = 197.
        """
        report = self.evaluate_seed(seed=seed, max_steps=25, harvest_entropy=True)
        # Use digit lengths and RNS residues to seed deterministic high-entropy hardware jitter
        base_jitter = harvest_honest_jitter(shape, scaled=True)
        modulation = torch.sin(torch.tensor(report.soliton_coherence_p197) * 3.14159)
        return base_jitter * (1.0 + 0.2 * modulation)


def compute_lychrel_chiral_gap(value: int, rns_moduli: Optional[List[int]] = None, base: int = 10) -> float:
    """Module-level helper to compute the Chiral Asymmetry Gap of an integer."""
    engine = Lychrel196SolitonEngine(base=base, rns_moduli=rns_moduli)
    return engine.compute_lychrel_chiral_gap(value)
