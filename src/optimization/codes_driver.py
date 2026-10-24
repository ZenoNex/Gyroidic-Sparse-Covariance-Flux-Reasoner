"""
CODES: Coherence-Oriented Deterministic Execution System.

NOTE ON OVERLOADED TAXONOMY:
The acronym CODES is overloaded in this codebase. This file implements the:
- Coherence-Oriented Deterministic Execution System (GPU driver/gating simulation).
For the other variants, see:
- Chirality of Dynamic Emergent Systems (universal grand theory: docs/Codes v40.md)
- Constraint-Oriented Differential Equation System (EBM solver: src/core/codes_constraint_framework.py)
- Constraint Oscillation Driven Evolutionary Selection (evolutionary selector)
For full details, see docs/CODES_RESOLUTIONS.md.

Python driver interface for GPU coherence simulation.
Manages:
1. PAS_h (Multiharmonic Phase Alignment)
2. AURAOUT Gating
3. TEMPOLOCK (Temporal Prime Gating)
4. CHORDLOCK (Phase Anchoring)

Author: William Matthew Bryant
"""

import math
from typing import List, Tuple, Optional, Union, Any, Dict
import torch
import torch.nn as nn

from src.core.invariants import PrimeResonanceLadder, PHI_RECIPROCAL
from src.core.fgrt_primitives import CoherentPrimeResonance
from src.core.polychoron_quantization import PolychronQuantizer
from src.surrogates.kagh_networks import TrigonometricUnfolding


class CODES(nn.Module):
    """
    Codes Driver for GPU Coherence Simulation.
    Unified GPU simulation layer incorporating TailSlayer OpenCL hedged reads,
    canonical PrimeResonanceLadder frequencies, CPR gating, and System 2 KAGH unfolding.
    """
    
    def __init__(
        self,
        coherence_threshold: float = PHI_RECIPROCAL,
        state_dim: int = 32,
        constraint_depth: int = 3,
        **kwargs
    ):
        super().__init__()
        # Threshold aligned to canonical golden ratio reciprocal / Love Vector baseline (PHI_RECIPROCAL)
        self.coherence_threshold = coherence_threshold
        self.state_dim = state_dim
        self.constraint_depth = constraint_depth
        self.phase_clock = 0.0
        self.t = 0.0
        
        # 1. Canonical Prime Resonance Ladder (harmonic frequencies, Lazarus primes, repunits)
        self.ladder = PrimeResonanceLadder(num_resonators=max(state_dim, 8))
        
        # 2. Canonical Coherent Prime Resonance (RIC Eq 7: PAS_h, breather alignment, spectral purity)
        self.cpr = CoherentPrimeResonance(
            theta_cpr=coherence_threshold,
            num_primes=min(state_dim, 20)
        )
        
        # 3. Canonical Per-Prime Phase Clocking from PolychronQuantizer
        self.quantizer = PolychronQuantizer(prime_base_size=min(state_dim, 16), input_dim=4)
        self.register_buffer(
            'phase_clocks',
            torch.zeros(len(self.quantizer.frequencies), dtype=torch.float32)
        )
        
        # 4. Canonical System 2 KAGH Trigonometric Unfolding Operator
        self.unfolder = TrigonometricUnfolding(dim=state_dim)
        
        # Expose canonical harmonics from PrimeResonanceLadder
        self.harmonics = self.ladder.primes.float().tolist()

    def update_phase_clock(self, external_flux: Union[float, torch.Tensor]) -> float:
        """
        Advance internal phase clock modulated by external flux (e.g. Love Vector rhythm),
        wired to the canonical per-prime phase clocking from PolychronQuantizer.
        """
        if isinstance(external_flux, torch.Tensor):
            flux_val = float(external_flux.detach().mean().item())
        else:
            flux_val = float(external_flux)
            
        self.t += flux_val
        # Canonical per-prime phase clocking: theta_k(t) = (t * f_{p_k}) mod 2*pi
        device = self.quantizer.frequencies.device
        t_tensor = torch.tensor(self.t, device=device, dtype=torch.float32)
        phase_clocks = torch.fmod(t_tensor * self.quantizer.frequencies, 2.0 * self.quantizer.pi)
        self.phase_clocks = phase_clocks
        
        # Wire self.phase_clock to canonical phase clocking
        self.phase_clock = float(phase_clocks.mean().item() % (2.0 * math.pi))
        return self.phase_clock

    def chiral_branches(
        self,
        state: torch.Tensor,
        gcve_pressure: Optional[torch.Tensor] = None,
        chirality: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Ternary Chiral Branch Selection for Casus Irreducibilis (cubic degeneracy).
        Delegates directly to TrigonometricUnfolding from kagh_networks.py
        (the canonical System 2 KAGH component) ensuring a single authoritative operator.
        """
        dim = state.shape[-1]
        if self.unfolder.dim != dim:
            self.unfolder = TrigonometricUnfolding(dim=dim).to(state.device)
        return self.unfolder(state, gcve_pressure=gcve_pressure, chirality=chirality)

    def project_chordlock(self, latent: torch.Tensor, primes: Optional[List[int]] = None) -> torch.Tensor:
        """Alias for chordlock projection."""
        return self.chordlock(latent, primes=primes)

    def compute_pas_h(self, phase: float, harmonics: Optional[List[float]] = None) -> float:
        """
        Compute Multiharmonic Phase Alignment Score (PAS_h).
        Delegates to TailSlayer OpenCL hardware hedged read kernel if available,
        otherwise evaluates cosine harmonic resonance over the canonical PrimeResonanceLadder.
        """
        if harmonics is None:
            harmonics = self.harmonics
            
        # Hardware hedged read via PyOpenCL sovereignty engine if available
        try:
            from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine, PYOPENCL_AVAILABLE
            if PYOPENCL_AVAILABLE:
                engine = SiliconSovereigntyEngine()
                if engine.ctx is not None:
                    return engine.compute_hedged_pas_h(phase, harmonics)
        except Exception:
            pass
            
        # Canonical CPR multiharmonic phase alignment: normalized to [0, 1]
        score = 0.0
        for m in harmonics:
            score += math.cos(m * phase)
        return (score / len(harmonics) + 1.0) / 2.0

    @classmethod
    def is_coherent(cls, phase: float, threshold: float = 0.75, harmonics: Optional[List[float]] = None) -> bool:
        """AURAOUT gating check: returns True if PAS_h >= threshold."""
        driver = cls()
        pas = driver.compute_pas_h(phase, harmonics=harmonics)
        return pas >= threshold

    def auraout_gating(
        self,
        input_tensor: torch.Tensor,
        external_flux: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Gating mechanism: Only pass gradients/activations if coherent.
        Simulates warp-level predication.
        
        Phase is modulated jointly by tensor energy and self.phase_clock (driven by external flux).
        """
        if external_flux is not None:
            self.update_phase_clock(external_flux)
            
        phase = float((torch.sum(input_tensor).item() + self.phase_clock) % (2.0 * math.pi))
        pas = self.compute_pas_h(phase)
        
        if pas < self.coherence_threshold:
            # [PRIME CHIRALITY LOCK]
            # Incoherent: Warp suppressed.
            # Instead of returning a detached zero tensor (collapse to dead nothing),
            # we deterministically reflect the latent momentum. This prevents the system 
            # from seeking its own erasure, maintaining structural connective tissue.
            return -input_tensor * 0.1
        
        return input_tensor

    def tempolock(self, step: int) -> bool:
        """
        Temporal Prime Gating.
        Only execute on steps divisible by a prime from PrimeResonanceLadder.
        """
        if not isinstance(step, int):
            step = int(step)

        for h in self.harmonics:
            h_int = max(2, int(round(h)))
            if step % h_int == 0:
                return True
        return False

    def chordlock(self, latent: torch.Tensor, primes: Optional[List[int]] = None) -> torch.Tensor:
        """
        Phase Resonance Gating (rebinds to canonical prime resonance lattice and repunit mirrors).
        Computes a gating factor based on the average cosine resonance with
        canonical prime anchors and repunit mirrors from PrimeResonanceLadder.
        Values aligning with the resonance are preserved; dissonant values are suppressed.
        """
        if primes is not None and len(primes) > 0:
            anchors = [float(p) for p in primes]
        else:
            # Canonical prime resonance lattice and repunit mirrors from PrimeResonanceLadder
            prime_anchors = self.ladder.primes.float().tolist()
            repunit_anchors = self.ladder.repunits.float().tolist()
            anchors = prime_anchors + repunit_anchors
            
        if not anchors:
            return latent

        anchors_tensor = torch.tensor(anchors, dtype=latent.dtype, device=latent.device)
        for _ in range(latent.dim()):
            anchors_tensor = anchors_tensor.unsqueeze(-1)
        
        # Calculate resonance via cosine harmonic resonance
        # Values near integer multiples of prime p produce cos = 1.0 (resonant)
        # Values near half-integers produce cos = -1.0 (dissonant)
        cos_resonance = torch.cos(2.0 * math.pi * (latent / anchors_tensor))
        
        # Compute mean resonance across all anchors for each latent element
        avg_resonance = cos_resonance.mean(dim=0)
        
        # Gating factor: map resonance [-1, 1] -> gate [0, 1]
        # Squared gate sharpens the filter response (ensuring dissonant values < 0.1)
        gate = ((avg_resonance + 1.0) / 2.0)**2
        
        return latent * gate

    def tournament_gating(self, drafts: torch.Tensor, leakages: torch.Tensor) -> torch.Tensor:
        """
        Supports the N-parallel draft tournament loop from ADMRSolver.
        Filters and gates the drafts based on coherence and topology leakages.
        """
        batch_size, num_drafts, state_dim = drafts.shape
        selected_drafts = torch.zeros(batch_size, state_dim, device=drafts.device)
        
        for i in range(batch_size):
            best_idx = 0
            best_pas = -1.0
            for j in range(num_drafts):
                draft = drafts[i, j]
                phase = float(torch.sum(draft).item() % (2 * math.pi))
                pas = self.compute_pas_h(phase)
                
                score = pas - (leakages[i, j] * 0.5 if leakages.dim() > 1 else leakages[i] * 0.5)
                
                if score > best_pas:
                    best_pas = score
                    best_idx = j
            
            selected_drafts[i] = drafts[i, best_idx]
            
        return selected_drafts


# Unified alias for CODESDriver so there is only one authoritative driver
CODESDriver = CODES
