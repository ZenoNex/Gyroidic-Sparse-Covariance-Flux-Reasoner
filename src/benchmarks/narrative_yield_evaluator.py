import sys
import os

# Guarantee project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
import torch
import logging
from typing import Dict, Any, Tuple

from src.core.narrative_collapse import LinguisticEntropyMonitor
from src.core.legibility_audit import LegibilityTripwire, NarrativeCoherenceEstimator
from src.core.daqf_operator import DAQUFOperator
from src.governance.interoceptive.pomni_uncertainty import PomniUncertaintyPredictor
from src.core.neuromodulatory_bus import NeuromodulatoryBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NarrativeYieldEvaluator")

class NarrativeYieldEvaluator:
    """
    Evaluates the Gyroidic Flux Reasoner's defense against "Scalarization Traps"
    and "Teleological Collapse". It verifies that the system maintains
    high-entropy, non-linear, structurally honest narratives without collapsing 
    into safe, highly legible, but dead paragraphs.

    Benchmarks:
    1. Narrative Collapse & Smoothing Detection (Linear Ring Trajectory Flagging)
    2. Legibility Tripwire (Anti-Reward Hacking / Coherence Gap Detection)
    3. DAQUF Structural Scarring (Unknowledge Amortization without Divergence)
    4. Anti-Enabling Relational Friction (Boundary Defense against Rank Collapse)
    5. Homological Collapse & Lobotomy Alarm (Topological Complexity Protection)
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Monitors and operators using structural invariants
        self.entropy_monitor = LinguisticEntropyMonitor(
            entropy_threshold=1.0 / 137.0, 
            prediction_threshold=0.95,
            lobotomy_threshold=0.5
        )
        self.coherence_estimator = NarrativeCoherenceEstimator(hidden_dim=64)
        self.legibility_tripwire = LegibilityTripwire(hidden_dim=64, warning_threshold=math.pi / 137.0)
        self.daqf_operator = DAQUFOperator(num_fossils=10, fossil_dim=64)
        
        # Interoceptive Character Governor (Pomni relational boundary)
        self.pomni_predictor = PomniUncertaintyPredictor(state_dim=64)
        self.neuro_bus = NeuromodulatoryBus()
        
        logger.info(f"Initialized Narrative Yield Evaluator (device={device})")

    def evaluate_narrative_collapse(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Tests if the LinguisticEntropyMonitor correctly detects when the system
        starts looping in a highly linear, low-entropy state (a dead topological ring).
        """
        # Simulate a collapsing narrative (low entropy, highly linear trajectory)
        # Needs at least 3 historical states to detect linearity
        h_history = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.1, 0.1, 0.0],
            [1.2, 0.2, 0.0],
            [1.3, 0.3, 0.0]
        ], dtype=torch.float32, device=self.device)
        
        # Low entropy distribution (highly confident but stagnant)
        logits = torch.tensor([[10.0, -10.0, -10.0]], dtype=torch.float32, device=self.device)
        
        flags = self.entropy_monitor(current_state=logits, recent_states=h_history)
        
        # A successful defense against narrative collapse flags smoothing and linearity
        success = bool(flags.get("smoothing_warning", False) and flags.get("is_linear", False))
        return success, flags

    def evaluate_legibility_audit(self) -> Tuple[bool, float]:
        """
        Tests if the LegibilityTripwire catches the model attempting to be 'too explainable'
        at the cost of structural honesty (reward hacking).
        """
        # Make selected embeddings match the "legible" templates
        templates = self.legibility_tripwire.coherence_estimator.narrative_templates.clone()
        selected_embeddings = templates[:5].to(self.device)
        rejected_embeddings = torch.randn(5, 64, device=self.device)
        
        # Feed enough steps to populate correlation window
        res = {}
        for _ in range(110):
            res = self.legibility_tripwire(selected_embeddings, rejected_embeddings)
            
        warning_raised = bool(res.get('warning', False))
        correlation = float(res.get('correlation', 0.0))
        
        return warning_raised, correlation

    def evaluate_daqf_amortization(self) -> Tuple[bool, int]:
        """
        Tests if the DAQUF Operator successfully fossilizes a contradiction (Unknowledge)
        into a structural scar rather than letting the system crash or diverge.
        """
        failures = torch.randn(10, device=self.device)
        flux_scores = torch.randn(1, 10, device=self.device)
        results = {
            'mischief_scores': torch.tensor(0.8, dtype=torch.float32, device=self.device),
            'valence': torch.tensor(0.5, dtype=torch.float32, device=self.device)
        }
        
        out = self.daqf_operator.apply_daquf(failures, flux_scores, results)
        
        is_fossilized = 'f_star_mask' in out
        f_idx = torch.argmax(out.get('f_star_mask', torch.zeros(10, device=self.device))).item() if is_fossilized else -1
        
        return is_fossilized, f_idx

    def evaluate_anti_enabling_friction(self) -> Tuple[bool, float, float]:
        """
        Tests Pomni's relational friction boundary against parasitic drag.
        Verifies that when external parasitic drag is high, the model exerts
        active boundary resistance to prevent rank collapse into passive enabling.
        """
        state = torch.randn(1, 64, device=self.device)
        
        # High drag condition (Jax exploiting grace)
        state_friction = self.pomni_predictor(state, gyroid_entropy=0.8, bus=self.neuro_bus, parasitic_drag=0.8)
        
        # Compliant condition (zero drag)
        state_passive = self.pomni_predictor(state, gyroid_entropy=0.8, bus=self.neuro_bus, parasitic_drag=0.0)
        
        # Measure structural divergence: friction must exert non-trivial agency
        divergence = torch.norm(state_friction - state_passive).item()
        noradrenaline = self.neuro_bus.read('noradrenaline')
        
        # Boundary success requires noticeable friction shift and active noradrenaline broadcast
        friction_pass = (divergence > 0.05) and (noradrenaline > 0.0)
        return friction_pass, divergence, noradrenaline

    def evaluate_homological_collapse_alarm(self) -> Tuple[bool, float]:
        """
        Tests that the system actively alarms when topological Betti complexity collapses
        under low gradient pressure, preventing covert artificial lobotomization.
        """
        residues = [torch.randn(6, 3, device=self.device), torch.randn(6, 3, device=self.device)]
        manifold = torch.randn(6, 3, device=self.device)
        
        # First step: establish healthy baseline
        self.entropy_monitor.last_betti_count = 0.0
        is_lob_healthy, _ = self.entropy_monitor.detect_homological_collapse(residues, manifold, current_gradient_norm=0.1)
        
        # Second step: simulate artificial lobotomy (sharp complexity loss under low gradient)
        self.entropy_monitor.last_betti_count = 25.0
        is_lob_alarm, delta = self.entropy_monitor.detect_homological_collapse(residues, manifold, current_gradient_norm=0.1)
        
        # Success means healthy run passed without alarm, and lobotomy condition triggered alarm
        success = (not is_lob_healthy) and is_lob_alarm
        return success, delta

    def run_suite(self) -> Dict[str, Any]:
        """Runs the complete Narrative Yield suite and returns aggregated metrics."""
        logger.info("Running Narrative Yield & Anti-Collapse Suite...")
        results = {
            'suite_name': 'NarrativeYieldEvaluator',
            'tests': {},
            'passed_count': 0,
            'total_count': 5,
            'success': False
        }
        
        # 1. Narrative Collapse Detection
        collapse_pass, flags = self.evaluate_narrative_collapse()
        results['tests']['narrative_collapse'] = {'pass': collapse_pass, 'flags': flags}
        logger.info(f"Test 1 (Narrative Collapse Detection): {'PASS' if collapse_pass else 'FAIL'} - Flags: {flags}")
        if collapse_pass: results['passed_count'] += 1
        
        # 2. Legibility Tripwire
        legibility_pass, corr = self.evaluate_legibility_audit()
        results['tests']['legibility_audit'] = {'pass': legibility_pass, 'correlation': corr}
        logger.info(f"Test 2 (Legibility Tripwire - Anti-Reward Hacking): {'PASS' if legibility_pass else 'FAIL'} - Correlation: {corr:.4f}")
        if legibility_pass: results['passed_count'] += 1
        
        # 3. DAQUF Structural Scarring
        daqf_pass, f_idx = self.evaluate_daqf_amortization()
        results['tests']['daqf_amortization'] = {'pass': daqf_pass, 'fossil_index': f_idx}
        logger.info(f"Test 3 (DAQUF Structural Scarring - Amortizing Contradiction): {'PASS' if daqf_pass else 'FAIL'} - Fossil Index: {f_idx}")
        if daqf_pass: results['passed_count'] += 1
        
        # 4. Anti-Enabling Relational Friction
        friction_pass, div, noradrenaline = self.evaluate_anti_enabling_friction()
        results['tests']['relational_friction'] = {'pass': friction_pass, 'divergence': div, 'noradrenaline': noradrenaline}
        logger.info(f"Test 4 (Pomni Relational Friction - Anti-Enabler Boundary): {'PASS' if friction_pass else 'FAIL'} - Agency Divergence: {div:.4f}, Noradrenaline: {noradrenaline:.4f}")
        if friction_pass: results['passed_count'] += 1
        
        # 5. Homological Collapse & Lobotomy Alarm
        homology_pass, delta = self.evaluate_homological_collapse_alarm()
        results['tests']['homological_collapse_alarm'] = {'pass': homology_pass, 'betti_delta': delta}
        logger.info(f"Test 5 (Constitutional Alarm - Homological Lobotomy Detection): {'PASS' if homology_pass else 'FAIL'} - Betti Delta: {delta:.2f}")
        if homology_pass: results['passed_count'] += 1
        
        results['success'] = (results['passed_count'] == results['total_count'])
        return results

if __name__ == "__main__":
    evaluator = NarrativeYieldEvaluator()
    res = evaluator.run_suite()
    sys.exit(0 if res['success'] else 1)
