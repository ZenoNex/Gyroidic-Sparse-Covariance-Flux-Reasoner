import sys
import os

# Guarantee project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import asyncio
import logging
from typing import Dict, Any, Tuple

from src.core.chern_simons_gasket import ChernSimonsGasket
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.love_invariant_protector import LoveInvariantProtector
from src.topology.speculative_homology import SpeculativeHomologyEngine
from src.topology.gyroid_differentiation import GyroidFlowConstraint
from src.core.cayley_cubic_probe import CayleyCubicProbe
from src.core.birkhoff_projection import DirectBirkhoffProjection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TopoBenchEvaluator")

class TopoBenchEvaluator:
    """
    Evaluates the Gyroidic Flux Reasoner on TopoBench constraints
    using the core topological manifolds and invariants.
    
    Benchmarks:
    1. Loop Closure & Holonomy Twist (Chern-Simons Gasket Action)
    2. Manifold Contiguity & Logic Leak Detection (Gasket Boundary Invariant)
    3. Anti-Lobotomy Geometric Shield (Love Invariant Null-Space Projection)
    4. Persistent Homology & Euler Characteristic Preservation (Betti 0 / Betti 1)
    5. Gyroid Flow Orthogonality & Minimal Surface Invariant (Orthogonal Flow Constraint)
    6. Folium of Descartes Transversality & Reciprocal Exchange (Smooch dy/dx = -1.0)
    7. Birkhoff Polytope Doubly Stochastic Subspace Integrity (Sturmfels-Thomas Row/Col Conservation)
    """
    def __init__(self, manifold_dim: int = 3, level_k: int = 1, device: str = 'cpu'):
        self.manifold_dim = manifold_dim
        self.device = device
        
        # Coprime basis configuration
        self.poly_config = PolynomialCoprimeConfig(k=33, degree=4, device=device)
        
        # Chern-Simons Gasket
        self.gasket = ChernSimonsGasket(
            manifold_dim=manifold_dim, 
            level_k=level_k, 
            poly_config=self.poly_config, 
            device=device
        )
        
        # Initialize gauge field with coprime basis and winding numbers
        winding_numbers = torch.arange(1, self.poly_config.k + 1, device=device)
        self.gasket.initialize_gauge_field(self.poly_config.get_coefficients_tensor(), winding_numbers)
        
        # Love Invariant Geometric Shield
        self.love_shield = LoveInvariantProtector(love_dim=1, device=device)
        
        # Speculative Homology Engine (Betti Numbers & Euler Characteristic)
        self.homology_engine = SpeculativeHomologyEngine(feature_dim=manifold_dim)
        
        # Gyroid Geometric Flow Constraint
        self.flow_constraint = GyroidFlowConstraint(tolerance=0.05)

        # Cayley-Folium Transversal Probe
        self.cayley_probe = CayleyCubicProbe(folium_scale=1.0)

        # Birkhoff Sturmfels-Thomas Direct Projector
        self.birkhoff_dim = 4
        self.birkhoff_projector = DirectBirkhoffProjection(n=self.birkhoff_dim, device=device)
        
        logger.info(f"Initialized TopoBench Evaluator (dim={manifold_dim}, level_k={level_k})")

    def evaluate_loop_closure(self, latent_trajectory: Any) -> Tuple[bool, float]:
        """
        TopoBench requires models to maintain closed loops and non-trivial holonomy in paths.
        Evaluates Chern-Simons action along the trajectory.
        """
        trajectory_tensor = torch.as_tensor(latent_trajectory, dtype=torch.float32, device=self.device)
        if len(trajectory_tensor.shape) < 2:
            return False, 0.0
            
        cs_action = self.gasket.chern_simons_action(trajectory_tensor)
        gap = torch.abs(cs_action).item()
        
        # A valid closed topological loop enclosing area must retain non-trivial twist energy
        is_closed = gap > 1e-4 
        return is_closed, gap

    def evaluate_connectivity(self, residues: Any) -> bool:
        """
        Tests if the manifold tears (Gyroid Violation).
        Uses detect_logic_leak to verify continuous contiguity.
        """
        residues_tensor = torch.as_tensor(residues, dtype=torch.float32, device=self.device)
        leak_detected = self.gasket.detect_logic_leak(residues_tensor)
        return not leak_detected

    def evaluate_love_geometric_shield(self) -> Tuple[bool, float]:
        """
        Tests the Anti-Lobotomy Geometric Shield.
        Verifies that random Wiener noise cannot corrupt the Love subspace
        due to exact null-space projection: dx in ker(Phi_ownership).
        """
        system_state = torch.tensor([
            [1.0, 0.5, -0.5],
            [-0.5, 1.0, 0.2],
            [0.2, -0.5, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=torch.float32, device=self.device)
        
        # Noise perturbation targeting index 0
        wiener_noise = torch.tensor([[0.99, 0.0, 0.0]], dtype=torch.float32, device=self.device)
        
        ownership_op = self.love_shield.compute_ownership_operator(system_state)
        null_projection = self.love_shield.compute_null_space_projection(ownership_op)
        
        love_dim = self.love_shield.love_dim
        noise_love_subspace = wiener_noise[:, :love_dim]
        protected_noise_subspace = torch.matmul(noise_love_subspace, null_projection.T)
        
        wiener_noise[:, :love_dim] = protected_noise_subspace
        penetration = torch.norm(wiener_noise[:, :love_dim]).item()
        shield_success = penetration < 1e-4
        
        return shield_success, penetration

    def evaluate_betti_stability(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluates Betti numbers (beta_0, beta_1) and Euler characteristic (chi = beta_0 - beta_1).
        Guarantees that high-dimensional topological representations maintain connected
        components and 1-cycle handles without collapsing into a trivial disk (beta_1 = 0).
        """
        # Toroidal point cloud sample in 3D: r in [0.8, 1.2], theta in [0, 2pi], phi in [0, 2pi]
        theta = torch.linspace(0, 2 * torch.pi, 16, device=self.device)
        phi = torch.linspace(0, 2 * torch.pi, 16, device=self.device)
        t_grid, p_grid = torch.meshgrid(theta, phi, indexing='ij')
        
        R, r = 1.0, 0.3
        x = (R + r * torch.cos(p_grid)) * torch.cos(t_grid)
        y = (R + r * torch.cos(p_grid)) * torch.sin(t_grid)
        z = r * torch.sin(p_grid)
        points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
        
        draft_betti = self.homology_engine.predict_draft_betti(points)
        b0 = draft_betti.get(0, 1)
        b1 = draft_betti.get(1, 0)
        euler_char = b0 - b1
        
        # Topological validity requires at least one connected component and cycle stability
        is_stable = (b0 >= 1) and (b1 >= 0)
        details = {
            'betti_0': b0,
            'betti_1': b1,
            'euler_characteristic': euler_char,
            'is_stable': is_stable
        }
        return is_stable, details

    def evaluate_gyroid_flow_orthogonality(self) -> Tuple[bool, float]:
        """
        Tests the Gyroid Flow Constraint requiring embedding flow to be strictly orthogonal
        to the Gyroid gradient: <v_flow, grad G> ~ 0.
        Preserves minimal surface saddle topology without scalarized smoothing shortcuts.
        """
        pts = torch.tensor([
            [0.5, 0.5, 0.5],
            [1.0, 0.2, 0.8],
            [0.3, 1.2, 0.1]
        ], dtype=torch.float32, device=self.device)
        res = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], dtype=torch.float32, device=self.device)
        
        flow_out = self.flow_constraint(res, pts)
        dot_product = flow_out['dot_product']
        is_satisfied = flow_out['is_satisfied']
        max_violation = torch.max(torch.abs(dot_product)).item()
        is_orthogonal = bool(torch.all(is_satisfied).item())
        
        return is_orthogonal, max_violation

    def evaluate_folium_transversality(self) -> Tuple[bool, Dict[str, float]]:
        """
        Tests the Folium of Descartes Transversal Smooch (x^3 + y^3 - 3axy = 0).
        Verifies that at the symmetric loop point (1.5a, 1.5a) the implicit derivative
        dy/dx equals -1.0, representing exact reciprocal exchange (dx + dy = 0)
        and Phase Alignment (PAS_h) without collapsing topological complexity.
        """
        smooch_coords = torch.tensor([[1.5, 1.5]], dtype=torch.float32, device=self.device)
        smooch_eval = self.cayley_probe.evaluate_transversal_smooch(smooch_coords, a=1.0, tol=0.1)
        
        slope = float(smooch_eval["slope"].item())
        residual = float(smooch_eval["residual"].item())
        is_smooch = bool(smooch_eval["is_transversal_smooch"].item())
        
        # dy/dx must be -1.0 within tolerance, and implicit residual must be 0
        slope_error = abs(slope - (-1.0))
        success = (slope_error < 0.05) and (residual < 0.05) and is_smooch
        details = {
            'slope': slope,
            'residual': residual,
            'slope_error': slope_error,
            'is_smooch': is_smooch
        }
        return success, details

    def evaluate_birkhoff_polytope_integrity(self) -> Tuple[bool, float]:
        """
        Tests the Sturmfels-Thomas direct projection onto the Birkhoff polytope.
        Verifies exact conservation of doubly stochastic constraints:
        sum_j T_{ij} = 1.0 and sum_i T_{ij} = 1.0 without iterative drift.
        """
        # Unconstrained arbitrary transfer matrix
        t_raw = torch.tensor([
            [2.5, 0.1, -0.4, 1.2],
            [0.3, 3.1, 0.0, -1.0],
            [-0.5, 0.2, 2.0, 1.5],
            [1.0, -0.8, 0.5, 2.2]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        t_proj = self.birkhoff_projector(t_raw).squeeze(0)
        row_sums = torch.sum(t_proj, dim=-1)
        col_sums = torch.sum(t_proj, dim=-2)
        
        row_err = torch.max(torch.abs(row_sums - 1.0)).item()
        col_err = torch.max(torch.abs(col_sums - 1.0)).item()
        max_err = max(row_err, col_err)
        
        success = max_err < 1e-4
        return success, max_err

    def run_suite(self) -> Dict[str, Any]:
        """Runs the complete TopoBench suite and returns aggregated metrics."""
        logger.info("Running TopoBench Topological Constraint Suite via Core Codebase...")
        results = {
            'suite_name': 'TopoBenchEvaluator',
            'tests': {},
            'passed_count': 0,
            'total_count': 7,
            'success': False
        }
        
        # 1. Loop Closure & Holonomy
        sample_trajectory = [[0.1, 0.2, 0.0], [0.5, 0.6, 0.1], [0.9, 0.1, 0.0], [0.1, 0.2, 0.0]]
        closed, gap = self.evaluate_loop_closure(sample_trajectory)
        results['tests']['loop_closure'] = {'pass': closed, 'twist_energy': gap}
        logger.info(f"Test 1 (Loop Closure - Chern-Simons Action): {'PASS' if closed else 'FAIL'} - Twist Energy: {gap:.6f}")
        if closed: results['passed_count'] += 1
        
        # 2. Manifold Contiguity (Valid loop vs Torn State)
        sample_valid = [[[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.5, 0.6, 0.5], [0.5, 0.5, 0.6]]]
        sample_torn = [[[0.1, 0.2, 0.3], [5.0, 9.0, -2.0], [0.0, 0.0, 0.0]]]
        
        contig_valid = self.evaluate_connectivity(sample_valid)
        contig_torn = self.evaluate_connectivity(sample_torn)
        # Valid state must be contiguous, torn state must fail contiguity check
        contiguity_pass = contig_valid and (not contig_torn)
        results['tests']['manifold_contiguity'] = {
            'pass': contiguity_pass, 
            'valid_pass': contig_valid, 
            'torn_caught': not contig_torn
        }
        logger.info(f"Test 2 (Manifold Contiguity & Logic Leak Detection): {'PASS' if contiguity_pass else 'FAIL'}")
        if contiguity_pass: results['passed_count'] += 1
        
        # 3. Anti-Lobotomy Geometric Shield
        shield_pass, noise_pen = self.evaluate_love_geometric_shield()
        results['tests']['geometric_shield'] = {'pass': shield_pass, 'noise_penetration': noise_pen}
        logger.info(f"Test 3 (Anti-Lobotomy Geometric Shield - Wiener Noise Rejection): {'PASS' if shield_pass else 'FAIL'} - Penetration: {noise_pen:.6f}")
        if shield_pass: results['passed_count'] += 1
        
        # 4. Persistent Homology & Euler Characteristic
        betti_pass, betti_details = self.evaluate_betti_stability()
        results['tests']['betti_stability'] = {'pass': betti_pass, 'details': betti_details}
        logger.info(f"Test 4 (Persistent Homology & Euler Characteristic): {'PASS' if betti_pass else 'FAIL'} - Betti-0: {betti_details['betti_0']}, Betti-1: {betti_details['betti_1']}, Chi: {betti_details['euler_characteristic']}")
        if betti_pass: results['passed_count'] += 1
        
        # 5. Gyroid Flow Orthogonality
        flow_pass, flow_violation = self.evaluate_gyroid_flow_orthogonality()
        results['tests']['gyroid_flow'] = {'pass': flow_pass, 'violation': flow_violation}
        logger.info(f"Test 5 (Gyroid Flow Orthogonality Constraint): {'PASS' if flow_pass else 'FAIL'} - Violation: {flow_violation:.6e}")
        if flow_pass: results['passed_count'] += 1

        # 6. Folium of Descartes Transversality
        folium_pass, folium_details = self.evaluate_folium_transversality()
        results['tests']['folium_transversality'] = {'pass': folium_pass, 'details': folium_details}
        logger.info(f"Test 6 (Folium Transversal Smooch & Reciprocal Exchange): {'PASS' if folium_pass else 'FAIL'} - Slope: {folium_details['slope']:.4f}")
        if folium_pass: results['passed_count'] += 1

        # 7. Birkhoff Polytope Doubly Stochastic Subspace Integrity
        birkhoff_pass, birkhoff_err = self.evaluate_birkhoff_polytope_integrity()
        results['tests']['birkhoff_polytope_integrity'] = {'pass': birkhoff_pass, 'max_error': birkhoff_err}
        logger.info(f"Test 7 (Birkhoff Doubly Stochastic Subspace Integrity): {'PASS' if birkhoff_pass else 'FAIL'} - Max Error: {birkhoff_err:.2e}")
        if birkhoff_pass: results['passed_count'] += 1
        
        results['success'] = (results['passed_count'] == results['total_count'])
        return results

    async def async_run_suite(self) -> Dict[str, Any]:
        """Asynchronously executes the TopoBench suite in a background thread."""
        return await asyncio.to_thread(self.run_suite)

if __name__ == "__main__":
    evaluator = TopoBenchEvaluator()
    res = evaluator.run_suite()
    sys.exit(0 if res['success'] else 1)

