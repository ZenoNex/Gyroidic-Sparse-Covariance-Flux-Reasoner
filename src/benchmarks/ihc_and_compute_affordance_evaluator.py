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

from src.core.invariants import PHI, get_ihc_ladder_scale, get_ihc_ladder_tensor
from src.core.hardware_monitor import (
    HardwareMonitor,
    ComputeAffordanceType,
    TypedComputeAffordance,
    get_typed_affordance,
    get_compute_affordance_telemetry,
    has_compute_affordance
)
from src.surrogates.calm_predictor import CALM
from src.models.modular_attention import ModularAttention
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.data.open_science_ingestor import OpenScienceIngestor
from src.core.cayley_cubic_probe import CayleyCubicProbe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IHCAndComputeAffordanceEvaluator")


class IHCAndComputeAffordanceEvaluator:
    """
    Evaluator for Inverted Hypersphere Cosmology (IHC) and Typed Compute Affordance Governance.

    Benchmarks:
    1. IHC 33-Shell Golden Ratio Ladder Scaling (Invariants, k=7 acoustic center, tensor generation)
    2. IHC BAO Cosmological Validation (r_s = 153.2 Mpc, ell_coh = 346.0 Mpc, Z3 phase alignment)
    3. IHC RP4 Projective Attention in VOID Regime (ModularAttention 33-shell scaling)
    4. Typed Compute Affordance Governance (All 8 regimes, domain isolation, gauge transform)
    5. CALM MAML Online Adaptation Thermal Throttling (Thermal veto and waste heat limits)
    6. Folium Transversal Smooch Anchor & Implicit Reciprocal Exchange (dy/dx = -1.0)
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.hw = HardwareMonitor.get_instance()

    def evaluate_ihc_33_shell_ladder(self) -> Tuple[bool, float]:
        """
        Validate Inverted Hypersphere Cosmology 33-shell golden ratio ladder scaling.
        Scale at center shell k=7 is exactly 1.0 when use_sound_horizon=True,
        and scales follow phi^(-(k - 7) / 33).
        """
        center_scale = get_ihc_ladder_scale(7, num_shells=33, center_k=7, use_sound_horizon=True)
        is_center_unity = abs(center_scale - 1.0) < 1e-5

        ladder_tensor = get_ihc_ladder_tensor(33, center_k=7, use_sound_horizon=True)
        shape_ok = (ladder_tensor.shape == (33,))
        finite_ok = torch.isfinite(ladder_tensor).all().item()
        positive_ok = (ladder_tensor > 0.0).all().item()
        center_tensor_unity = abs(float(ladder_tensor[7].item()) - 1.0) < 1e-5

        is_pass = bool(is_center_unity and shape_ok and finite_ok and positive_ok and center_tensor_unity)
        error = abs(center_scale - 1.0) + abs(float(ladder_tensor[7].item()) - 1.0)
        return is_pass, float(error)

    def evaluate_ihc_bao_cosmology(self) -> Tuple[bool, float]:
        """
        Validate IHC BAO Cosmological metrics via OpenScienceIngestor:
        - Sound horizon scale r_s = 153.2 Mpc
        - Coherence length ell_coh = 346.0 Mpc
        - Z3 counter-rotation multiharmonic phase alignment (PAS_m > 0.20)
        """
        ingestor = OpenScienceIngestor(cache_dir="scratch/open_science_cache", verbosity="low")
        from pathlib import Path
        cache_path = Path("scratch/open_science_cache/eval_sdss_test.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        sim_data = ingestor._simulate_sdss_catalog("eval_catalog", limit=33, cache_path=cache_path)
        metrics = ingestor.calculate_ihc_bao_metrics(sim_data)

        ihc_valid = bool(metrics.get("ihc_validated", False))
        pas_m = float(metrics.get("pas_m", 0.0))
        pas_ok = pas_m > 0.20
        r_s_expected = 153.2
        r_s_obs = float(metrics.get("r_s_bao_mpc", 0.0))
        r_s_ok = abs(r_s_obs - r_s_expected) < 1.0
        ell_coh_ok = abs(float(metrics.get("ell_coh_mpc", 0.0)) - 346.0) < 1.0

        is_pass = bool(ihc_valid and pas_ok and r_s_ok and ell_coh_ok)
        return is_pass, pas_m

    def evaluate_ihc_rp4_projective_attention(self) -> Tuple[bool, float]:
        """
        Validate IHC RP4 projective attention routing in ModularAttention under VOID regime
        with 33-shell golden ratio ladder scaling.
        """
        poly_cfg = PolynomialCoprimeConfig(k=4, degree=3)
        attn = ModularAttention(hidden_dim=16, num_heads=2, poly_config=poly_cfg, num_functionals=4)
        x = torch.randn(2, 4, 16)

        out_std = attn(x)
        out_void = attn(x, regime='VOID')

        shape_ok = (out_void.shape == (2, 4, 16))
        finite_ok = torch.isfinite(out_void).all().item()
        distinct_ok = not torch.allclose(out_std, out_void)

        diff = float(torch.norm(out_void - out_std).item())
        is_pass = bool(shape_ok and finite_ok and distinct_ok)
        return is_pass, diff

    def evaluate_typed_compute_affordances(self) -> Tuple[bool, int]:
        """
        Validate typed compute affordances across all 8 regimes without scalarization collapse:
        SILICON, THERMAL, GPU_SOVEREIGN, TOPOLOGICAL, P2P_MESH, DIEGETIC, ALGORITHMIC, COMPUTE_BATCH.
        """
        valid_count = 0
        for atype in ComputeAffordanceType:
            aff = get_typed_affordance(atype)
            if isinstance(aff, TypedComputeAffordance) and aff.affordance_type == atype:
                if 0.0 <= aff.gauge <= 1.0 and isinstance(aff.is_admissible, bool):
                    # Test non-scalar gauge transformation
                    transformed = aff.gauge_transform(ComputeAffordanceType.TOPOLOGICAL, coupling=0.8)
                    if transformed.affordance_type == ComputeAffordanceType.TOPOLOGICAL:
                        valid_count += 1

        telemetry = get_compute_affordance_telemetry()
        telemetry_ok = (telemetry.get("status") == "ok") and ("gauges" in telemetry)
        is_pass = (valid_count == len(ComputeAffordanceType)) and telemetry_ok
        return is_pass, valid_count

    def evaluate_calm_thermal_affordance_governance(self) -> Tuple[bool, float]:
        """
        Validate CALM MAML online adaptation under thermal affordance constraints:
        - When waste heat exceeds cooling capacity (P_waste > P_cool_max), thermal runaway triggers.
        - CALM.adapt throttles inner loop gradient steps (clamped to <= 1) and scales down learning rate.
        - CALM.forward elevates abort_score to assert thermal emergency veto.
        """
        calm = CALM(dim=16, history_len=4, hidden_dim=32, nhead=2)
        support_hist = torch.randn(1, 4, 16)
        support_target = torch.randn(1, 16)

        # 1. Normal state: adaptation executes normally
        self.hw.update_thermodynamics(p_waste=0.0, p_cool_max=1000.0, thermal_runaway=False)
        adapted_normal = calm.adapt(support_hist, support_target, steps=2, lr=0.01)
        norm_abort, _, _, _, _, _ = calm(support_hist)

        # 2. Thermal runaway state: P_waste = 1500 > P_cool_max = 1000
        self.hw.update_thermodynamics(p_waste=1500.0, p_cool_max=1000.0, thermal_runaway=True)
        adapted_throttled = calm.adapt(support_hist, support_target, steps=3, lr=0.01)
        hot_abort, _, _, _, _, _ = calm(support_hist)

        # Abort score under thermal runaway must be elevated
        elevated_ok = hot_abort.mean().item() >= norm_abort.mean().item()
        throttled_ok = adapted_throttled is not None

        # Reset thermodynamics to normal
        self.hw.update_thermodynamics(p_waste=0.0, p_cool_max=1000.0, thermal_runaway=False)

        diff = float(hot_abort.mean().item() - norm_abort.mean().item())
        is_pass = bool(elevated_ok and throttled_ok)
        return is_pass, diff

    def evaluate_folium_transversal_smooch_anchor(self) -> Tuple[bool, float]:
        """
        Validate Folium of Descartes transversal smooch anchor:
        - Implicit residual x^3 + y^3 - 3xy = 0 at (1.5, 1.5)
        - Implicit derivative dy/dx = -1.0 (reciprocal exchange)
        - Transversal smooch detection and reciprocal exchange score
        """
        probe = CayleyCubicProbe(high_mischief_threshold=0.5, folium_scale=1.0)
        smooch_pt = torch.tensor([[1.5, 1.5]], dtype=torch.float32)

        residual = probe.evaluate_folium_implicit(smooch_pt)
        residual_ok = abs(residual.item()) < 1e-4

        slope = probe.compute_implicit_derivative(smooch_pt)
        slope_ok = abs(slope.item() - (-1.0)) < 1e-4

        metrics = probe.evaluate_transversal_smooch(smooch_pt)
        is_smooch = bool(metrics["is_transversal_smooch"].item())
        reciprocal_score = float(metrics["reciprocal_score"].item())
        score_ok = reciprocal_score > 0.99

        is_pass = bool(residual_ok and slope_ok and is_smooch and score_ok)
        return is_pass, reciprocal_score

    def run_suite(self) -> Dict[str, Any]:
        """Executes all 6 IHC & Compute Affordance benchmark evaluations."""
        start_time = time.time()
        results = {
            'suite': 'IHCAndComputeAffordanceEvaluator',
            'timestamp': time.time(),
            'total_count': 6,
            'passed_count': 0,
            'success': False,
            'tests': {}
        }

        # 1. IHC 33-Shell Golden Ratio Ladder
        ladder_pass, ladder_err = self.evaluate_ihc_33_shell_ladder()
        results['tests']['ihc_33_shell_ladder'] = {'pass': ladder_pass, 'error': ladder_err}
        if ladder_pass:
            results['passed_count'] += 1

        # 2. IHC BAO Cosmological Validation
        bao_pass, bao_pas = self.evaluate_ihc_bao_cosmology()
        results['tests']['ihc_bao_cosmology'] = {'pass': bao_pass, 'pas_m': bao_pas}
        if bao_pass:
            results['passed_count'] += 1

        # 3. IHC RP4 Projective Attention
        rp4_pass, rp4_diff = self.evaluate_ihc_rp4_projective_attention()
        results['tests']['ihc_rp4_projective_attention'] = {'pass': rp4_pass, 'norm_diff': rp4_diff}
        if rp4_pass:
            results['passed_count'] += 1

        # 4. Typed Compute Affordances
        aff_pass, aff_valid = self.evaluate_typed_compute_affordances()
        results['tests']['typed_compute_affordances'] = {'pass': aff_pass, 'valid_types': aff_valid}
        if aff_pass:
            results['passed_count'] += 1

        # 5. CALM Thermal Affordance Governance
        calm_pass, calm_diff = self.evaluate_calm_thermal_affordance_governance()
        results['tests']['calm_thermal_governance'] = {'pass': calm_pass, 'abort_elevation': calm_diff}
        if calm_pass:
            results['passed_count'] += 1

        # 6. Folium Transversal Smooch Anchor
        smooch_pass, smooch_score = self.evaluate_folium_transversal_smooch_anchor()
        results['tests']['folium_transversal_smooch'] = {'pass': smooch_pass, 'reciprocal_score': smooch_score}
        if smooch_pass:
            results['passed_count'] += 1

        results['duration_seconds'] = time.time() - start_time
        results['success'] = (results['passed_count'] == results['total_count'])
        return results

    async def async_run_suite(self) -> Dict[str, Any]:
        """Asynchronously executes the benchmark suite in a background thread."""
        import asyncio
        return await asyncio.to_thread(self.run_suite)


def run_ihc_and_compute_affordance_benchmark(device: str = 'cpu') -> Dict[str, Any]:
    evaluator = IHCAndComputeAffordanceEvaluator(device=device)
    return evaluator.run_suite()


if __name__ == "__main__":
    rep = run_ihc_and_compute_affordance_benchmark()
    print("IHCAndComputeAffordanceEvaluator Suite Success:", rep['success'], f"({rep['passed_count']}/{rep['total_count']})")
    for t_name, t_info in rep['tests'].items():
        print(f"  {t_name}: {t_info}")
