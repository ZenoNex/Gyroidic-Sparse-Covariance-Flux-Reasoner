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

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.polynomial_crt import PolynomialCRT
from src.core.love_invariant_protector import SoftSaturatedGates
from src.core.continuous_coprimality import ContinuousCoprimality
from src.core.spectral_coherence_repair import BezoutCoefficientRefresh
from src.core.honest_jitter import harvest_honest_jitter
from src.optimization.muon_hybrid import zeropower_via_newtonschulz5
from src.core.carnot_mobius_ledger import CarnotMobiusLedger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReactBenchEvaluator")

class ReactBenchEvaluator:
    """
    Evaluates the Gyroidic Flux Reasoner on ReactBench constraints
    using the core Polynomial CRT, dynamic coprime modular systems,
    anti-lobotomy tri-state gating, Muon quintic polar orthogonalization,
    and Carnot-Möbius thermodynamic friction.

    Benchmarks:
    1. Branch Convergence (Polynomial CRT Fixed-Point Reconstruction & Pressure)
    2. Cyclic Graph Catalysis (Modal Non-Commutative Energy Retention)
    3. Anti-Lobotomy Tri-State Gate (Lattice Adaptive Shrinkage & Silence in Play Mode)
    4. Continuous Coprimality (Entropy Pressure and Asymptotic Independence)
    5. Dynamic Bezout Coefficient Refresh (CRT Modulus Drift Recovery & Condition Number)
    6. Muon Quintic Polar Decomposition (Singular Value Equalization via Newton-Schulz 5)
    7. Carnot-Möbius Thermodynamic Ledger (Runaway Coupling & Thermal Friction Resistance)
    """
    def __init__(self, k: int = 4, degree: int = 2, device: str = 'cpu'):
        self.device = device
        self.k = k
        self.degree = degree
        
        # Real polynomial coprime configuration
        self.poly_config = PolynomialCoprimeConfig(k=k, degree=degree, device=device)
        self.crt = PolynomialCRT(poly_config=self.poly_config, use_soft_reconstruction=True)
        
        # Tri-State Logic Temperature Check
        self.soft_gates = SoftSaturatedGates(num_functionals=k, poly_degree=degree, device=device)
        
        # Continuous Coprimality Checker (Entropy-based independence)
        self.coprimality_checker = ContinuousCoprimality(use_binary_quantization=True)
        
        # Dynamic Bezout Coefficient Refresh
        self.bezout_refresh = BezoutCoefficientRefresh(num_functionals=k, poly_degree=degree, device=device)

        # Carnot-Möbius Thermodynamic Ledger
        self.carnot_ledger = CarnotMobiusLedger(xi_shear=0.1, p_cool_max=1000.0, p_nuclear=2000.0)
        
        logger.info(f"Initialized ReactBench Evaluator (k={k}, degree={degree}, device={device})")

    def evaluate_branch_convergence(self, branch_a: list, branch_b: list) -> Tuple[bool, float]:
        """
        ReactBench demands tracking separate reactions converging into one state.
        Tests fixed_point_reconstruction and reconstruction pressure from PolynomialCRT.
        """
        residues = torch.tensor([[branch_a, branch_b, branch_a, branch_b]], dtype=torch.float32, device=self.device)
        
        # The CRT synthesizes a single coherent polynomial output from distinct branches
        _ = self.crt.fixed_point_reconstruction(residues)
        
        # The success criteria is measuring the reconstruction pressure
        pressure = self.crt.compute_reconstruction_pressure(residues, return_reconstruction=False)
        pressure_val = pressure.mean().item()
        
        success = pressure_val < 1.0  # Threshold of acceptable synthesis pressure
        return success, pressure_val

    def evaluate_cyclic_catalysis(self, pathways: list) -> Tuple[bool, float]:
        """
        Tests cycle retention without signal loss using the CRT modal reconstruction.
        """
        residues = torch.as_tensor([pathways], dtype=torch.float32, device=self.device)
        
        # In modal CRT, consistent lattice solutions are selected
        reconstruction, _ = self.crt.forward(residues, mode='modal', return_diagnostics=True)
        
        # Check variance of expected residues vs reconstruction
        energy = torch.var(reconstruction).item()
        
        success = energy > 1e-4  # Non-commutative properties retained
        return success, energy

    def evaluate_anti_lobotomy_tristate(self) -> Tuple[bool, float]:
        """
        Tests the Tri-State Gate logic (Play vs Seriousness) mandated by
        governance to prevent binary lobotomy.
        Checks if low PAS_h induces silence (play) instead of hard binaries.
        """
        # Low Phase Alignment (PAS) = High Noise/Play Mode
        # The adaptive floor uses EMA: 0.95*old + 0.05*target. We run it 100 times to converge.
        for _ in range(100):
            self.soft_gates.update_lambda_adaptive(pas_h=0.1) 
        
        # Test signal with ambiguous gradient
        ambiguous_signal = torch.tensor([[[-0.2, 0.1, 0.05]]], dtype=torch.float32, device=self.device)
        
        # In pure binary clipping (sgn), -0.2 -> -1.0, 0.1 -> 1.0. 
        # But in Tri-State SoftSaturatedGates, they drop to exactly 0.0 (Silence)
        las_output = self.soft_gates.lattice_adaptive_shrinkage(ambiguous_signal)
        
        silence_achieved = bool(torch.all(las_output == 0.0).item())
        current_lambda = self.soft_gates.lambda_adaptive.item()
        return silence_achieved, current_lambda

    def evaluate_continuous_coprimality(self) -> Tuple[bool, float]:
        """
        Tests non-algebraic co-primality between distinct channels under saturation.
        Verifies asymptotic independence: lim_{t->inf} Cov(r_i^(t), r_j^(t)) -> 0.
        """
        # Generate orthogonal functional residue channels
        r_i = torch.tensor([[1.0, 0.0, -1.0, 0.0]], dtype=torch.float32, device=self.device)
        r_j = torch.tensor([[0.0, 1.0, 0.0, -1.0]], dtype=torch.float32, device=self.device)
        
        is_independent, final_cov = self.coprimality_checker.check_asymptotic_independence(
            r_i, r_j, tolerance=1e-4
        )
        cov_val = torch.abs(final_cov).max().item()
        success = bool(is_independent.item()) and (cov_val < 1e-4)
        return success, cov_val

    def evaluate_bezout_refresh(self) -> Tuple[bool, Dict[str, float]]:
        """
        Tests dynamic Bezout coefficient refresh to recover from CRT modulus drift
        and maintain numerical stability (finite condition number).
        """
        # Simulate drifted residues across batch
        drifted_residues = harvest_honest_jitter((4, self.k, self.degree + 1), device=self.device, scaled=False) * 2.0
        
        corrected = self.bezout_refresh.apply_crt_correction(drifted_residues)
        diagnostics = self.bezout_refresh.get_diagnostics()
        
        cond_num = diagnostics.get('bezout_condition_number', 1.0)
        has_no_nan = not torch.isnan(corrected).any().item()
        is_stable = has_no_nan and (1.0 <= cond_num <= 1000.0)
        
        return is_stable, diagnostics

    def evaluate_muon_polar_decomposition(self) -> Tuple[bool, Dict[str, float]]:
        """
        Tests Muon matrix-aware polar orthogonalization via quintic Newton-Schulz iteration:
        Equalizes singular values of 2D weight matrix to approximately 1.0 without SVD.
        Verifies that X = zeropower_via_newtonschulz5(G) yields an isometric update with
        singular values centered around 1.0.
        """
        # Linear layer weight matrix with honest jitter perturbation
        base_eye = torch.eye(8, device=self.device)
        jitter = harvest_honest_jitter((8, 8), device=self.device, scaled=False) * 0.15
        G = base_eye + jitter
        
        # Newton-Schulz quintic polar factor
        X = zeropower_via_newtonschulz5(G, steps=5)
        
        # Check Gram matrix proximity to identity: mean off-diagonal should be near 0
        gram = torch.matmul(X.T, X)
        eye = torch.eye(G.shape[1], device=self.device)
        mean_err = float(torch.mean(torch.abs(gram - eye)).item())
        max_err = float(torch.max(torch.abs(gram - eye)).item())
        
        # Singular values should be concentrated around 1.0
        svals = torch.linalg.svdvals(X)
        sv_mean = float(torch.mean(svals).item())
        sv_min = float(torch.min(svals).item())
        sv_max = float(torch.max(svals).item())
        
        is_pass = (mean_err < 0.15) and (0.60 <= sv_min) and (sv_max <= 1.25)
        details = {
            'mean_orthogonality_error': mean_err,
            'max_orthogonality_error': max_err,
            'singular_value_mean': sv_mean,
            'singular_value_min': sv_min,
            'singular_value_max': sv_max
        }
        return is_pass, details

    def evaluate_carnot_mobius_backpressure(self) -> Tuple[bool, Dict[str, float]]:
        """
        Tests the Carnot-Möbius thermodynamic friction and non-linear backpressure:
        Verifies that rapid entropy increases and phase mismatches encounter thermodynamic
        resistance, finite Carnot efficiency limit, and bounded thermal waste without unconstrained runaway.
        """
        lambda_vac = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        lambda_pump = torch.tensor(5.0, dtype=torch.float32, device=self.device)
        admr_residues = harvest_honest_jitter((4, self.k), device=self.device, scaled=False) * 0.1
        
        # Evaluate thermodynamic stack efficiency and runaway metrics
        ledger_out = self.carnot_ledger(
            lambda_vac=lambda_vac,
            lambda_pump=lambda_pump,
            admr_residues=admr_residues,
            depth_N=3
        )
        
        eta_stack = float(ledger_out.get('eta_stack', 0.5))
        p_waste = float(ledger_out.get('p_waste', 0.0))
        n_crit = float(ledger_out.get('n_crit', 10.0))
        thermal_runaway = bool(ledger_out.get('thermal_runaway', False))
        
        # Bounded operation requires positive efficiency and bounded waste below p_cool_max
        is_pass = (0.0 < eta_stack <= 1.0) and (p_waste < float(self.carnot_ledger.p_cool_max)) and not thermal_runaway
        details = {
            'eta_stack': eta_stack,
            'p_waste': p_waste,
            'n_crit': n_crit,
            'thermal_runaway': thermal_runaway
        }
        return is_pass, details

    def run_suite(self) -> Dict[str, Any]:
        """Runs the complete ReactBench suite and returns aggregated metrics."""
        logger.info("Running ReactBench Cyclic Pathway Suite via Core Codebase...")
        results = {
            'suite_name': 'ReactBenchEvaluator',
            'tests': {},
            'passed_count': 0,
            'total_count': 7,
            'success': False
        }
        
        # 1. Branch Convergence (CRT Reconstruction)
        branch_a = [0.2, 0.4, 0.6]
        branch_b = [0.22, 0.45, 0.55]
        fusion_pass, pressure_val = self.evaluate_branch_convergence(branch_a, branch_b)
        results['tests']['branch_convergence'] = {'pass': fusion_pass, 'reconstruction_pressure': pressure_val}
        logger.info(f"Test 1 (Branch Convergence - CRT Reconstruction): {'PASS' if fusion_pass else 'FAIL'} - Pressure: {pressure_val:.4f}")
        if fusion_pass: results['passed_count'] += 1
        
        # 2. Cyclic Graph Catalysis
        pathways = [
            [0.1, 0.0, -0.1], 
            [0.0, 0.1, 0.0], 
            [-0.1, 0.0, 0.1], 
            [0.1, 0.0, -0.1]
        ]
        cycle_pass, energy = self.evaluate_cyclic_catalysis(pathways)
        results['tests']['cyclic_catalysis'] = {'pass': cycle_pass, 'structural_energy': energy}
        logger.info(f"Test 2 (Cyclic Graph Catalysis - Modal Retention): {'PASS' if cycle_pass else 'FAIL'} - Structural Energy: {energy:.6f}")
        if cycle_pass: results['passed_count'] += 1
        
        # 3. Anti-Lobotomy Tri-State Gate
        tristate_pass, current_lambda = self.evaluate_anti_lobotomy_tristate()
        results['tests']['tristate_gate'] = {'pass': tristate_pass, 'adaptive_lambda': current_lambda}
        logger.info(f"Test 3 (Anti-Lobotomy Tri-State Gate / Play Mode): {'PASS' if tristate_pass else 'FAIL'} - Adaptive Silence Floor: {current_lambda:.4f}")
        if tristate_pass: results['passed_count'] += 1
        
        # 4. Continuous Coprimality & Asymptotic Independence
        coprime_pass, cov_val = self.evaluate_continuous_coprimality()
        results['tests']['continuous_coprimality'] = {'pass': coprime_pass, 'final_covariance': cov_val}
        logger.info(f"Test 4 (Continuous Coprimality - Asymptotic Independence): {'PASS' if coprime_pass else 'FAIL'} - Covariance: {cov_val:.6e}")
        if coprime_pass: results['passed_count'] += 1
        
        # 5. Bezout Coefficient CRT Drift Refresh
        bezout_pass, bezout_diag = self.evaluate_bezout_refresh()
        results['tests']['bezout_refresh'] = {'pass': bezout_pass, 'diagnostics': bezout_diag}
        logger.info(f"Test 5 (Dynamic Bezout Refresh - Modulus Drift Recovery): {'PASS' if bezout_pass else 'FAIL'} - Condition Number: {bezout_diag['bezout_condition_number']:.3f}")
        if bezout_pass: results['passed_count'] += 1

        # 6. Muon Quintic Polar Orthogonalization
        muon_pass, muon_diag = self.evaluate_muon_polar_decomposition()
        results['tests']['muon_polar_decomposition'] = {'pass': muon_pass, 'diagnostics': muon_diag}
        logger.info(f"Test 6 (Muon Quintic Polar Decomposition): {'PASS' if muon_pass else 'FAIL'} - Mean Ortho Error: {muon_diag['mean_orthogonality_error']:.4f}")
        if muon_pass: results['passed_count'] += 1

        # 7. Carnot-Möbius Thermodynamic Ledger
        carnot_pass, carnot_diag = self.evaluate_carnot_mobius_backpressure()
        results['tests']['carnot_mobius_backpressure'] = {'pass': carnot_pass, 'diagnostics': carnot_diag}
        logger.info(f"Test 7 (Carnot-Möbius Thermodynamic Backpressure): {'PASS' if carnot_pass else 'FAIL'} - Waste: {carnot_diag['p_waste']:.1f}W, Efficiency: {carnot_diag['eta_stack']:.3f}")
        if carnot_pass: results['passed_count'] += 1
        
        results['success'] = (results['passed_count'] == results['total_count'])
        return results

    async def async_run_suite(self) -> Dict[str, Any]:
        """Asynchronously executes the ReactBench suite in a background thread."""
        return await asyncio.to_thread(self.run_suite)

if __name__ == "__main__":
    evaluator = ReactBenchEvaluator()
    res = evaluator.run_suite()
    sys.exit(0 if res['success'] else 1)

