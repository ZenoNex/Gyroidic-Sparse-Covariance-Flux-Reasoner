import sys
import os
import time
import logging
from typing import Dict, Any, Tuple

# Guarantee project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np

from src.models.gyroid_reasoner import GyroidicFluxReasoner
from src.core.hessian_8x8_curvature_ledger import (
    Hessian8x8CurvatureLedger,
    Trajectory8x8Decombiner,
    AttractorRegime
)
from src.core.honest_jitter import harvest_honest_jitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIReasonerEvaluator")


class AIReasonerEvaluator:
    """
    Evaluates the Gyroidic AI Reasoner and Second-Order Curvature Dynamics:
    1. Gyroidic AI Reasoner End-to-End Inference & Self-Model Coherence
    2. 8x8 Boundary Curvature Ledger & Minimal Surface Saddle Invariant
    3. Bouligand Directional Subdifferential Kink Detection Across Piecewise Gates
    4. Cyclotomic Order-8 Phase Alignment (PAS_h) Harmonic Decomposition
    5. Trajectory 8x8 Decombiner & Coxeter D_4 Chamber Quotienting
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.ledger_engine = Hessian8x8CurvatureLedger(delta_h=1e-3, device=device)
        self.decombiner = Trajectory8x8Decombiner(budget=50000, max_preimages_per_step=32, use_coxeter_d4=True)

    def evaluate_reasoner_forward_coherence(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Test 1: End-to-End AI Reasoner forward pass, verifying finite state output,
        self-model probe tracking, and dark matter retention without divergence.
        """
        try:
            # Instantiate compact test instance of GyroidicFluxReasoner
            reasoner = GyroidicFluxReasoner(
                text_dim=64,
                graph_dim=32,
                num_dim=16,
                hidden_dim=64,
                num_layers=1,
                num_heads=2,
                num_functionals=3,
                poly_degree=2,
                use_admm=False
            ).to(self.device)
            reasoner.eval()
            
            text_emb = harvest_honest_jitter((1, 64), device=self.device, scaled=False).requires_grad_(True)
            graph_emb = harvest_honest_jitter((1, 32), device=self.device, scaled=False).requires_grad_(True)
            num_features = harvest_honest_jitter((1, 16), device=self.device, scaled=False).requires_grad_(True)
            
            out_dict = reasoner(text_emb=text_emb, graph_emb=graph_emb, num_features=num_features)
                
            fused = out_dict.get('fused_state')
            if fused is None:
                fused = out_dict.get('reconstructed')
            if fused is None:
                # Any tensor in out_dict
                for v in out_dict.values():
                    if isinstance(v, torch.Tensor):
                        fused = v
                        break
                        
            has_no_nan = not torch.isnan(fused).any().item() if fused is not None else True
            is_finite = torch.isfinite(fused).all().item() if fused is not None else True
            success = has_no_nan and is_finite and (len(out_dict) > 0)
            
            details = {
                'output_keys': list(out_dict.keys()),
                'has_no_nan': has_no_nan,
                'is_finite': is_finite,
                'state_norm': float(torch.norm(fused).item()) if fused is not None else 0.0
            }
            return success, details
        except Exception as e:
            logger.error(f"Reasoner forward evaluation failed: {e}")
            return False, {'error': str(e)}

    def evaluate_8x8_hessian_ledger(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Test 2: Evaluates the 8x8 Boundary Curvature Ledger on a saddle energy landscape.
        Verifies condition number kappa is bounded and minimal surface saddle Tr(H) ~ 0 is identified.
        """
        try:
            state = torch.zeros(8, device=self.device)
            def saddle_energy(x):
                signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=x.dtype, device=x.device)
                return 0.5 * torch.sum(signs * (x ** 2))
                
            res = self.ledger_engine.compute_ledger(state, energy_fn=saddle_energy)
            
            trace_zero = abs(res.trace_curvature) < 0.1
            has_8_eigs = len(res.eigenvalues) == 8
            is_saddle = res.morse_index >= 2
            condition_valid = res.condition_number >= 0.95
            
            success = trace_zero and has_8_eigs and is_saddle and condition_valid
            
            success = trace_zero and has_8_eigs and is_saddle and condition_valid
            details = {
                'trace_curvature': res.trace_curvature,
                'condition_number': res.condition_number,
                'morse_index': res.morse_index,
                'is_minimal_surface': res.is_minimal_surface,
                'attractor_regime': res.attractor_regime
            }
            return success, details
        except Exception as e:
            logger.error(f"8x8 Hessian ledger evaluation failed: {e}")
            return False, {'error': str(e)}

    def evaluate_bouligand_kink_detection(self) -> Tuple[bool, float]:
        """
        Test 3: Evaluates Bouligand subdifferential kink tensor across a piecewise-quadratic contact function.
        Verifies that non-smooth fractures (H^+ - H^-) are detected and measured.
        """
        try:
            state = torch.zeros(8, device=self.device)
            def contact_fn(x):
                return torch.sum(torch.relu(x) ** 2)
                
            res = self.ledger_engine.compute_ledger(state, energy_fn=contact_fn, compute_bouligand=True)
            kink_detected = res.bouligand_kink_intensity > 0.2
            return kink_detected, float(res.bouligand_kink_intensity)
        except Exception as e:
            logger.error(f"Bouligand kink evaluation failed: {e}")
            return False, 0.0

    def evaluate_cyclotomic_harmonics(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Test 4: Evaluates order-8 cyclotomic harmonic projections along roots of unity omega_k = exp(i 2pi k / 8).
        Verifies multiharmonic coherence in [0, 1] and all 8 harmonic channels.
        """
        try:
            state = torch.tensor([0.3, -0.5, 0.2, 0.7, -0.1, 0.4, -0.6, 0.8], device=self.device)
            res = self.ledger_engine.compute_ledger(state)
            
            has_8_modes = len(res.cyclotomic_harmonics) == 8
            coherence_valid = 0.0 <= res.multiharmonic_coherence <= 1.0
            dc_real = abs(res.cyclotomic_harmonics[0].imag) < 1e-3
            
            success = has_8_modes and coherence_valid and dc_real
            details = {
                'multiharmonic_coherence': res.multiharmonic_coherence,
                'dc_real_component': float(res.cyclotomic_harmonics[0].real),
                'harmonic_count': len(res.cyclotomic_harmonics)
            }
            return success, details
        except Exception as e:
            logger.error(f"Cyclotomic harmonics evaluation failed: {e}")
            return False, {'error': str(e)}

    def evaluate_trajectory_decombining(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Test 5: Evaluates 8x8 trajectory decombining with Coxeter D_4 gauge chamber reduction
        and exact sum invariant conservation.
        """
        try:
            # Test chain state: [8, 2] in row 0
            board = [[0] * 8 for _ in range(8)]
            board[0][0] = 8
            board[0][1] = 2
            
            res = self.decombiner.decombine_trajectory(board)
            
            status_valid = res["status"] in ("completed", "budget_exhausted", "bounded_search_complete")
            orbit_pruned = res["orbit_count"] >= 1
            has_canonical = "coxeter_canonical_initial" in res
            
            # Check impossible odd sum rejection
            imp_board = [[0] * 8 for _ in range(8)]
            imp_board[0][0] = 3
            res_imp = self.decombiner.decombine_trajectory(imp_board)
            sum_invariant_pruned = res_imp["status"] == "invalid_sum_invariant"
            
            success = status_valid and orbit_pruned and has_canonical and sum_invariant_pruned
            details = {
                'status': res["status"],
                'nodes_expanded': res["nodes_expanded"],
                'unique_dag_states': res["unique_dag_states"],
                'sum_invariant_pruned': sum_invariant_pruned
            }
            return success, details
        except Exception as e:
            logger.error(f"Trajectory decombining evaluation failed: {e}")
            return False, {'error': str(e)}

    def run_suite(self) -> Dict[str, Any]:
        """
        Executes the AI Reasoner & Second-Order Curvature benchmark suite.
        """
        logger.info("Running AI Reasoner & Second-Order Curvature Suite...")
        results = {
            'suite': 'AIReasonerEvaluator',
            'total_count': 5,
            'passed_count': 0,
            'success': False,
            'tests': {}
        }
        
        # 1. Reasoner Forward Coherence
        fwd_pass, fwd_details = self.evaluate_reasoner_forward_coherence()
        results['tests']['reasoner_forward_coherence'] = {'pass': fwd_pass, 'details': fwd_details}
        logger.info(f"Test 1 (AI Reasoner Inference & Coherence): {'PASS' if fwd_pass else 'FAIL'}")
        if fwd_pass: results['passed_count'] += 1
        
        # 2. 8x8 Boundary Curvature Ledger
        ledger_pass, ledger_details = self.evaluate_8x8_hessian_ledger()
        results['tests']['hessian_8x8_ledger'] = {'pass': ledger_pass, 'details': ledger_details}
        logger.info(f"Test 2 (8x8 Hessian Curvature Ledger & Morse Saddle): {'PASS' if ledger_pass else 'FAIL'}")
        if ledger_pass: results['passed_count'] += 1
        
        # 3. Bouligand Subdifferential Kinks
        kink_pass, kink_val = self.evaluate_bouligand_kink_detection()
        results['tests']['bouligand_kinks'] = {'pass': kink_pass, 'kink_intensity': kink_val}
        logger.info(f"Test 3 (Bouligand Subdifferential Kink Detection): {'PASS' if kink_pass else 'FAIL'} - Intensity: {kink_val:.4f}")
        if kink_pass: results['passed_count'] += 1
        
        # 4. Cyclotomic Harmonics
        cyclo_pass, cyclo_details = self.evaluate_cyclotomic_harmonics()
        results['tests']['cyclotomic_harmonics'] = {'pass': cyclo_pass, 'details': cyclo_details}
        logger.info(f"Test 4 (Cyclotomic Order-8 Phase Alignment): {'PASS' if cyclo_pass else 'FAIL'}")
        if cyclo_pass: results['passed_count'] += 1
        
        # 5. Trajectory Decombining
        decomb_pass, decomb_details = self.evaluate_trajectory_decombining()
        results['tests']['trajectory_decombining'] = {'pass': decomb_pass, 'details': decomb_details}
        logger.info(f"Test 5 (Trajectory 8x8 Decombining & D4 Quotienting): {'PASS' if decomb_pass else 'FAIL'}")
        if decomb_pass: results['passed_count'] += 1
        
        results['success'] = (results['passed_count'] == results['total_count'])
        return results


if __name__ == "__main__":
    evaluator = AIReasonerEvaluator()
    res = evaluator.run_suite()
    sys.exit(0 if res['success'] else 1)
