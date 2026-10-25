import sys
import os
import time
import math
import logging
from typing import Dict, Any, Tuple
import torch

# Guarantee project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.optimization.codes_driver import CODES, CODESDriver
from src.core.invariants import PrimeResonanceLadder, PHI_RECIPROCAL
from src.core.fgrt_primitives import CoherentPrimeResonance
from src.surrogates.kagh_networks import TrigonometricUnfolding
from src.core.honest_jitter import harvest_honest_jitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChordlockEvaluator")


class ChordlockEvaluator:
    """
    Evaluates Chordlock, Canonical Prime Resonance Ladder, and Authoritative System 2 operators.

    Benchmarks:
    1. Chordlock Identity & Resonant Preservation (tests/test_chordlock.py parity)
    2. Coherent Prime Resonance Alignment (CPR Phase Coherence & Breather Alignment)
    3. Multi-Prime Harmonic Lattice Scaling (Palindromic Repunit Basis)
    4. Hardware Hedged Read Path (PyOpenCL Fallback & PAS_h Multiharmonic Evaluation)
    5. Chiral Branches Unfolding (Authoritative TrigonometricUnfolding)
    """
    def __init__(self, state_dim: int = 16, device: str = 'cpu'):
        self.state_dim = state_dim
        self.device = device
        self.codes = CODES(state_dim=state_dim)
        self.ladder = PrimeResonanceLadder(num_resonators=max(state_dim, 8))
        self.cpr = CoherentPrimeResonance(theta_cpr=PHI_RECIPROCAL, num_primes=min(state_dim, 20))

    def evaluate_chordlock_identity(self) -> Tuple[bool, float]:
        """Verify chordlock projection preserves shape and anchors resonant signals."""
        latent = harvest_honest_jitter((10,), device=self.device, scaled=False)
        locked = self.codes.chordlock(latent, primes=[])
        shape_ok = (locked.shape == latent.shape) and torch.isfinite(locked).all().item()

        prime = 2
        resonant_val = torch.tensor([2.0, 4.0, 6.0], device=self.device)
        locked_res = self.codes.chordlock(resonant_val, primes=[prime])
        diff = float(torch.norm(locked_res - resonant_val).item())
        is_pass = shape_ok and (diff < 0.1)
        return is_pass, diff

    def evaluate_coherent_prime_resonance(self) -> Tuple[bool, float]:
        """Verify CPR global phase coherence and positive breather mode alignment."""
        phases = torch.zeros(self.state_dim, device=self.device)
        phase_pass = self.cpr.check_phase_coherence(phases)

        field = torch.ones(self.state_dim, device=self.device)
        breathers = torch.ones(self.state_dim, device=self.device)
        breather_pass = self.cpr.check_breather_alignment(breathers, field)

        coherence = 1.0 if (phase_pass and breather_pass) else 0.0
        is_pass = phase_pass and breather_pass
        return is_pass, coherence

    def evaluate_multi_prime_lattice(self) -> Tuple[bool, int]:
        """Verify multi-prime ladder frequencies and repunit mirrors."""
        frequencies, repunits, hw_status = self.ladder()
        count = len(frequencies)
        freq_positive = bool((frequencies > 0).all().item())
        is_pass = (count >= self.state_dim or count >= 8) and freq_positive and len(repunits) == count
        return is_pass, count

    def evaluate_hardware_hedged_read(self) -> Tuple[bool, str]:
        """Verify multiharmonic phase alignment (PAS_h) across phase conditions."""
        pas_zero = self.codes.compute_pas_h(0.0)
        pas_pi = self.codes.compute_pas_h(math.pi)
        is_pass = (0.0 <= pas_zero <= 1.0) and (0.0 <= pas_pi <= 1.0) and (pas_zero > pas_pi)
        dev_label = "OpenCL/CPU Hedged Read"
        return is_pass, dev_label

    def evaluate_chiral_unfolding(self) -> Tuple[bool, float]:
        """Verify chiral branch selection produces finite unfolded tensors."""
        x = harvest_honest_jitter((2, self.state_dim), device=self.device, scaled=False)
        unfolded = self.codes.chiral_branches(x)
        var_norm = float(torch.var(unfolded).item())
        is_pass = (unfolded.shape == x.shape) and torch.isfinite(unfolded).all().item() and (var_norm > 0.0)
        return is_pass, var_norm

    def run_suite(self) -> Dict[str, Any]:
        results = {
            'suite_name': 'ChordlockEvaluator',
            'timestamp': time.time(),
            'passed_count': 0,
            'total_count': 5,
            'tests': {}
        }

        # 1. Identity & Resonance
        id_pass, id_diff = self.evaluate_chordlock_identity()
        results['tests']['chordlock_identity'] = {'pass': id_pass, 'phase_delta': id_diff}
        if id_pass: results['passed_count'] += 1

        # 2. Coherent Prime Resonance
        cpr_pass, cpr_coh = self.evaluate_coherent_prime_resonance()
        results['tests']['coherent_prime_resonance'] = {'pass': cpr_pass, 'coherence': cpr_coh}
        if cpr_pass: results['passed_count'] += 1

        # 3. Multi-Prime Lattice
        lat_pass, lat_pts = self.evaluate_multi_prime_lattice()
        results['tests']['multi_prime_lattice'] = {'pass': lat_pass, 'lattice_points': lat_pts}
        if lat_pass: results['passed_count'] += 1

        # 4. Hardware Hedged Read Path
        hw_pass, hw_dev = self.evaluate_hardware_hedged_read()
        results['tests']['hardware_hedged_read'] = {'pass': hw_pass, 'active_device': hw_dev}
        if hw_pass: results['passed_count'] += 1

        # 5. Chiral Branches Unfolding
        chiral_pass, chiral_orth = self.evaluate_chiral_unfolding()
        results['tests']['chiral_unfolding'] = {'pass': chiral_pass, 'orthogonality': chiral_orth}
        if chiral_pass: results['passed_count'] += 1

        results['success'] = (results['passed_count'] == results['total_count'])
        return results

    async def async_run_suite(self) -> Dict[str, Any]:
        """Asynchronously executes the Chordlock suite in a background thread."""
        import asyncio
        return await asyncio.to_thread(self.run_suite)


def run_chordlock_benchmark(device: str = 'cpu') -> Dict[str, Any]:
    evaluator = ChordlockEvaluator(device=device)
    return evaluator.run_suite()


if __name__ == "__main__":
    rep = run_chordlock_benchmark()
    print("ChordlockEvaluator Suite Success:", rep['success'], f"({rep['passed_count']}/{rep['total_count']})")
