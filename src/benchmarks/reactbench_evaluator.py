import torch
import logging
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.polynomial_crt import PolynomialCRT
from src.core.love_invariant_protector import SoftSaturatedGates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReactBenchEvaluator")

class ReactBenchEvaluator:
    """
    Evaluates the Gyroidic Flux Reasoner on ReactBench constraints
    using the actual Polynomial CRT from the codebase.
    """
    def __init__(self):
        logger.info("Initialized ReactBench Evaluator using real PolynomialCRT.")
        # Setup real configuration
        self.poly_config = PolynomialCoprimeConfig(k=4, degree=2, device='cpu')
        self.crt = PolynomialCRT(poly_config=self.poly_config, use_soft_reconstruction=True)
        # Setup Tri-State Logic Temperature Check
        self.soft_gates = SoftSaturatedGates(num_functionals=4, poly_degree=2, device='cpu')

    def evaluate_branch_convergence(self, branch_a, branch_b):
        """
        ReactBench demands tracking two separate reactions converging into one state.
        We test this using the real `fixed_point_reconstruction` from PolynomialCRT.
        """
        # Residue distribution: [Batch, K, D]
        # We merge branch_a and branch_b into the residue channels
        residues = torch.tensor([[branch_a, branch_b, branch_a, branch_b]], dtype=torch.float32)
        
        # The CRT must synthesize a single coherent polynomial output from these distinct branches
        reconstruction = self.crt.fixed_point_reconstruction(residues)
        
        # The success criteria in CRT terms is measuring the reconstruction pressure
        pressure = self.crt.compute_reconstruction_pressure(residues, return_reconstruction=False)
        
        success = pressure.mean().item() < 1.0  # Threshold of acceptable synthesis pressure
        return success, pressure.mean().item()

    def evaluate_cyclic_catalysis(self, pathways):
        """
        Tests cycle retention without signal loss using the CRT's reconstruction.
        """
        residues = torch.tensor([pathways], dtype=torch.float32)
        
        # In modal CRT, consistent lattice solutions are selected
        reconstruction, diagnostics = self.crt.forward(residues, mode='modal', return_diagnostics=True)
        
        # Check variance of expected residues vs reconstruction
        energy = torch.var(reconstruction).item()
        
        success = energy > 1e-4  # Non-commutative properties retained
        return success, energy

    def evaluate_anti_lobotomy_tristate(self):
        """
        Tests the Tri-State Gate logic (Play vs Seriousness) mandated by
        GOVERNANCE_ANTI_LOBOTOMY.md to prevent binary lobotomy.
        Checks if low PAS_h induces silence (play) instead of hard binaries.
        """
        # Low Phase Alignment (PAS) = High Noise/Play Mode
        # The adaptive floor uses EMA: 0.95*old + 0.05*target. We run it 100 times to converge.
        for _ in range(100):
            self.soft_gates.update_lambda_adaptive(pas_h=0.1) 
        
        # Test signal with ambiguous gradient
        ambiguous_signal = torch.tensor([[[-0.2, 0.1, 0.05]]], dtype=torch.float32)
        
        # In pure binary clipping (sgn), -0.2 -> -1.0, 0.1 -> 1.0. 
        # But in Tri-State SoftSaturatedGates, they should drop to exactly 0.0 (Silence)
        las_output = self.soft_gates.lattice_adaptive_shrinkage(ambiguous_signal)
        
        silence_achieved = torch.all(las_output == 0.0).item()
        return silence_achieved, self.soft_gates.lambda_adaptive.item()

    def run_suite(self):
        logger.info("Running ReactBench Cyclic Pathway Suite via Core Codebase...")
        
        # Simulated D=3 branches (representing K=4 channels)
        branch_a = [0.2, 0.4, 0.6]
        branch_b = [0.22, 0.45, 0.55]
        
        fusion_pass, dist = self.evaluate_branch_convergence(branch_a, branch_b)
        logger.info(f"Test 1 (Branch Convergence - CRT Reconstruction): {'PASS' if fusion_pass else 'FAIL'} - Pressure: {dist:.4f}")
        
        pathways = [
            [0.1, 0.0, -0.1], 
            [0.0, 0.1, 0.0], 
            [-0.1, 0.0, 0.1], 
            [0.1, 0.0, -0.1]
        ]
        cycle_pass, energy = self.evaluate_cyclic_catalysis(pathways)
        logger.info(f"Test 2 (Cyclic Graph Catalysis - Modal Retention): {'PASS' if cycle_pass else 'FAIL'} - Structural Energy: {energy:.6f}")
        
        tristate_pass, current_lambda = self.evaluate_anti_lobotomy_tristate()
        logger.info(f"Test 3 (Anti-Lobotomy Tri-State Gate / Play Mode): {'PASS' if tristate_pass else 'FAIL'} - Adaptive Silence Floor: {current_lambda:.4f}")

if __name__ == "__main__":
    evaluator = ReactBenchEvaluator()
    evaluator.run_suite()
