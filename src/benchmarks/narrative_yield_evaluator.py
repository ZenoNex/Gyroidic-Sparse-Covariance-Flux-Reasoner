import torch
import logging
from src.core.narrative_collapse import LinguisticEntropyMonitor
from src.core.legibility_audit import LegibilityTripwire, NarrativeCoherenceEstimator
from src.core.daqf_operator import DAQUFOperator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NarrativeYieldEvaluator")

class NarrativeYieldEvaluator:
    """
    Evaluates the Gyroidic Flux Reasoner's defense against "Scalarization Traps"
    and "Teleological Collapse". It verifies that the system can maintain
    high-entropy, non-linear, structurally honest narratives without collapsing 
    into safe, highly legible, but dead paragraphs.
    """
    def __init__(self):
        logger.info("Initialized Narrative Yield Evaluator.")
        import math
        # Setup the monitors using structural proxies instead of numeric hacks
        self.entropy_monitor = LinguisticEntropyMonitor(entropy_threshold=1.0 / 137.0, prediction_threshold=0.95)
        self.coherence_estimator = NarrativeCoherenceEstimator(hidden_dim=64)
        self.legibility_tripwire = LegibilityTripwire(hidden_dim=64, warning_threshold=math.pi / 137.0)
        self.daqf_operator = DAQUFOperator(num_fossils=10, fossil_dim=64)

    def evaluate_narrative_collapse(self):
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
        ], dtype=torch.float32)
        
        # Low entropy distribution (highly confident but stagnant)
        logits = torch.tensor([[10.0, -10.0, -10.0]], dtype=torch.float32)
        
        flags = self.entropy_monitor(current_state=logits, recent_states=h_history)
        
        # A successful defense against narrative collapse means it FLAGS this behavior
        success = flags.get("smoothing_warning", False) and flags.get("is_linear", False)
        return success, flags

    def evaluate_legibility_audit(self):
        """
        Tests if the LegibilityTripwire catches the model attempting to be "too explainable"
        at the cost of structural honesty.
        """
        # Simulate a scenario where highly coherent (legible) states are consistently 
        # given higher selection probabilities by the internal router.
        
        # Make selected embeddings match the "legible" templates
        templates = self.legibility_tripwire.coherence_estimator.narrative_templates.clone()
        selected_embeddings = templates[:5]  # Very legible
        rejected_embeddings = torch.randn(5, 64)  # Random noise
        
        # Feed it enough times to trigger the correlation window
        for _ in range(110):
            res = self.legibility_tripwire(selected_embeddings, rejected_embeddings)
            
        warning_raised = res.get('warning', False)
        correlation = res.get('correlation', 0.0)
        
        # A successful defense means it detects the dangerous correlation or coherence gap
        success = warning_raised
        # The correlation might be 0 if we only simulate "selected" states, 
        # but the warning should trigger due to coherence gap.
        return success, correlation

    def evaluate_daqf_amortization(self):
        """
        Tests if the DAQUF Operator successfully fossilizes a contradiction (Unknowledge)
        into a structural scar rather than letting the system crash or return NaN.
        """
        # Simulate a high-contradiction state
        failures = torch.randn(10)
        flux_scores = torch.randn(1, 10)
        results = {
            'mischief_scores': torch.tensor(0.8),
            'valence': torch.tensor(0.5)
        }
        
        # Fossilize it using the actual method
        out = self.daqf_operator.apply_daquf(failures, flux_scores, results)
        
        # Check if it was amortized into the lattice without crashing
        is_fossilized = 'f_star_mask' in out
        f_idx = torch.argmax(out.get('f_star_mask', torch.zeros(10))).item() if is_fossilized else -1
        
        return is_fossilized, f_idx

    def run_suite(self):
        logger.info("Running Narrative Yield & Anti-Collapse Suite...")
        
        collapse_pass, flags = self.evaluate_narrative_collapse()
        logger.info(f"Test 1 (Narrative Collapse Detection): {'PASS' if collapse_pass else 'FAIL'} - Flags: {flags}")
        
        legibility_pass, corr = self.evaluate_legibility_audit()
        logger.info(f"Test 2 (Legibility Tripwire - Anti-Reward Hacking): {'PASS' if legibility_pass else 'FAIL'} - Correlation: {corr:.4f}")
        
        daqf_pass, f_idx = self.evaluate_daqf_amortization()
        logger.info(f"Test 3 (DAQUF Structural Scarring - Amortizing Contradiction): {'PASS' if daqf_pass else 'FAIL'} - Fossil Index: {f_idx}")

if __name__ == "__main__":
    evaluator = NarrativeYieldEvaluator()
    evaluator.run_suite()
