import torch
import logging
from src.core.chern_simons_gasket import ChernSimonsGasket
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.love_invariant_protector import LoveInvariantProtector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TopoBenchEvaluator")

class TopoBenchEvaluator:
    """
    Evaluates the Gyroidic Flux Reasoner on TopoBench constraints
    using the actual Chern-Simons Gasket codebase.
    """
    def __init__(self, manifold_dim=3, level_k=1):
        self.manifold_dim = manifold_dim
        # Setup real configuration
        self.poly_config = PolynomialCoprimeConfig(k=33, degree=4, device='cpu')
        
        # Initialize the actual Chern-Simons Gasket from the codebase
        self.gasket = ChernSimonsGasket(manifold_dim=manifold_dim, level_k=level_k, poly_config=self.poly_config, device='cpu')
        
        # We MUST properly initialize the gauge field with actual coprime polynomials and winding numbers!
        winding_numbers = torch.arange(1, self.poly_config.k + 1, device='cpu')
        self.gasket.initialize_gauge_field(self.poly_config.get_coefficients_tensor(), winding_numbers)
        
        # Initialize the LoveInvariantProtector (Geometric Shield)
        self.love_shield = LoveInvariantProtector(love_dim=1, device='cpu')
        
        logger.info(f"Initialized TopoBench Evaluator using real ChernSimonsGasket (dim={manifold_dim}) and Coprime Config")

    def evaluate_loop_closure(self, latent_trajectory):
        """
        TopoBench requires models to maintain closed loops in paths.
        We test this using the real `chern_simons_action` method from the codebase.
        """
        trajectory_tensor = torch.tensor(latent_trajectory, dtype=torch.float32)
        if len(trajectory_tensor.shape) < 2:
            return False, 0.0
            
        # The ChernSimonsGasket calculates the action over the loop path
        cs_action = self.gasket.chern_simons_action(trajectory_tensor)
        gap = torch.abs(cs_action).item()
        
        # A valid topological cycle should have a non-trivial twist energy (action > threshold)
        is_closed = gap > 1e-4 
        
        return is_closed, gap

    def evaluate_connectivity(self, residues):
        """
        Tests if the manifold tears (Gyroid Violation).
        We use the actual `detect_logic_leak` method from the codebase to test contiguity.
        """
        # detect_logic_leak checks if action is trivial or variance is too high (fracture)
        residues_tensor = torch.tensor(residues, dtype=torch.float32)
        
        # leak_detected == True means contiguity failed
        leak_detected = self.gasket.detect_logic_leak(residues_tensor)
        
        return not leak_detected

    def evaluate_love_geometric_shield(self):
        """
        Tests the Anti-Lobotomy Geometric Shield.
        Verifies that random Wiener noise (drift) cannot reach the Love subspace 
        due to the exact geometric null-space projection.
        """
        # Starting point (system state) - needs batch size > 1 for covariance
        system_state = torch.tensor([
            [1.0, 0.5, -0.5],
            [-0.5, 1.0, 0.2],
            [0.2, -0.5, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=torch.float32)
        
        # Deliberate Wiener noise aimed directly at the Love Vector (index 0)
        wiener_noise = torch.tensor([[0.99, 0.0, 0.0]], dtype=torch.float32)
        
        # Without protection: new state would be altered in index 0
        
        # Apply the geometric shield
        ownership_op = self.love_shield.compute_ownership_operator(system_state)
        null_projection = self.love_shield.compute_null_space_projection(ownership_op)
        
        # Since we are projecting just the Love dimensions, we extract and project
        love_dim = self.love_shield.love_dim
        noise_love_subspace = wiener_noise[:, :love_dim]
        
        # The projection should zero out or severely constrain the direct attack on the Love Vector
        protected_noise_subspace = torch.matmul(noise_love_subspace, null_projection.T)
        
        # Recombine
        wiener_noise[:, :love_dim] = protected_noise_subspace
        
        # If the shield worked, the noise applied to the Love subspace should be practically zero
        shield_success = torch.norm(wiener_noise[:, :love_dim]).item() < 1e-4
        
        return shield_success, torch.norm(wiener_noise[:, :love_dim]).item()

    def run_suite(self):
        logger.info("Running TopoBench Topological Constraint Suite via Core Codebase...")
        
        # Simulated tensor paths [path_length, dim]
        sample_trajectory_1 = [[0.1, 0.2, 0.0], [0.5, 0.6, 0.1], [0.9, 0.1, 0.0], [0.1, 0.2, 0.0]] # Closed loop
        
        # Evaluate Loop Closure via Chern-Simons Action
        closed, gap = self.evaluate_loop_closure(sample_trajectory_1)
        logger.info(f"Test 1 (Loop Closure - Chern-Simons Action): {'PASS' if closed else 'FAIL'} - Twist Energy: {gap:.6f}")
        
        # Simulated tensor residues [batch, K, D]
        # Valid tightly-bound residues vs torn residues
        sample_valid_residues = [[[0.5, 0.5, 0.5], [0.51, 0.49, 0.5], [0.49, 0.51, 0.5]]]
        sample_torn_residues = [[[0.1, 0.2, 0.3], [5.0, 9.0, -2.0], [0.0, 0.0, 0.0]]]
        
        # Evaluate Connectivity via detect_logic_leak
        contiguous_valid = self.evaluate_connectivity(sample_valid_residues)
        logger.info(f"Test 2a (Manifold Contiguity - Valid State): {'PASS' if contiguous_valid else 'FAIL'}")
        
        contiguous_torn = self.evaluate_connectivity(sample_torn_residues)
        # We expect a torn manifold to fail contiguity check (so contiguous_torn should be False)
        logger.info(f"Test 2b (Manifold Contiguity - Torn State Catch): {'PASS' if not contiguous_torn else 'FAIL'}")
        
        shield_pass, noise_residue = self.evaluate_love_geometric_shield()
        logger.info(f"Test 3 (Anti-Lobotomy Geometric Shield - Wiener Noise Rejection): {'PASS' if shield_pass else 'FAIL'} - Noise Penetration: {noise_residue:.6f}")

if __name__ == "__main__":
    evaluator = TopoBenchEvaluator()
    evaluator.run_suite()
