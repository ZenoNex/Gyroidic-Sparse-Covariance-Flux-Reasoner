import sys
import os
import time
import logging
from typing import Dict, Any, Tuple
import torch
import numpy as np

# Guarantee project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.ui.diegetic_backend import HybridPhysicsBackend
from src.ui.voxelboxter_simulation import PointerlessOctree
from src.core.zeitgeist_router import ZeitgeistRouter, ZeitgeistState
from src.core.codes_constraint_framework import CODESConstraintFramework
from src.core.carnot_mobius_ledger import CarnotMobiusLedger
from src.topology.unknowledge_domain import UnknowledgeDomain
from src.core.honest_jitter import harvest_honest_jitter
from src.ui.diegetic_visualizer import brenier_pushforward, apply_oklab_luminance_absorption, srgb_to_oklab, oklab_to_srgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TopologicalSingularityEvaluator")


class TopologicalSingularityEvaluator:
    """
    Evaluates Rent-Seeking Topological Singularities across 5 Operational Layers:
    1. Layer 1: Substrate & Feed Multiplexing (Spatial Land-Value g_mu_nu + PointerlessOctree)
    2. Layer 2: Void Friction & Geodesic Deflection (F_void > 0.80 Coprime Bypass via Chern-Simons Gasket)
    3. Layer 3: Mechanical Shear & Voxel Fracturing (Delta beta_1 = 1 + Carnot-Mobius N_crit Runaway)
    4. Layer 4: Necrotic Quarantine & Memory Fossilization (Zero-velocity Flow Stagnation in UnknowledgeDomain)
    5. Layer 5: Visual Rendering (Brenier Push-Forward Displacement + OKLab Luminance Absorption)
    """
    def __init__(self, state_dim: int = 32, device: str = 'cpu'):
        self.state_dim = state_dim
        self.device = device
        self.backend = HybridPhysicsBackend(state_dim=state_dim, device=device)
        self.backend.fetch_financial_feeds = lambda feeds=None: ["Equities flux neutral; topological yield curve steady."]
        self.router = ZeitgeistRouter(dim=state_dim, moduli=(3, 5, 7, 11, 13))
        self.codes = CODESConstraintFramework(state_dim=state_dim, max_constraints=5)
        self.unknowledge = UnknowledgeDomain(tau_m=0.3, dim=state_dim)

    def evaluate_layer1_substrate_feed_multiplexing(self) -> Tuple[bool, Dict[str, Any]]:
        spatial_data = self.backend.fetch_spatial_land_value()
        octree = PointerlessOctree(max_depth=6)
        nodes_added = self.backend.map_to_pointerless_octree(octree, spatial_data)
        feed_chain = self.backend.multiplex_feed_chain()
        is_pass = (
            nodes_added > 0
            and octree.num_nodes > 0
            and spatial_data["spatial_metric"].shape == (4, 4)
            and feed_chain["combined_flux"].shape == (self.state_dim,)
        )
        return is_pass, {
            "nodes_added": nodes_added,
            "octree_nodes": octree.num_nodes,
            "land_rent_density": float(spatial_data["land_rent_density"])
        }

    def evaluate_layer2_void_friction_coprime_bypass(self) -> Tuple[bool, Dict[str, Any]]:
        state = ZeitgeistState.initial(self.router.moduli)
        spatial_metric = torch.eye(4, device=self.device)
        spatial_metric[0, 0] = 4.0
        spatial_metric[3, 3] = 4.0
        x_in = harvest_honest_jitter((1, self.state_dim), device=self.device, scaled=False)
        
        mode, next_state, diag, x_steer = self.router(x_in, state, tadc_kwargs={'spatial_metric': spatial_metric})
        steered = self.router.temporal_inverse_kinematics(state, x_in, spatial_metric=spatial_metric)
        routed_state, bypass_info = self.codes.monitor_void_friction(x_in, f_void=0.85)

        is_pass = (
            diag.get("f_void", 0.0) > 0.0
            and steered.shape == x_in.shape
            and bypass_info.get("coprime_bypass_active") is True
            and "gasket_diagnostics" in bypass_info
        )
        return is_pass, {
            "f_void": float(diag.get("f_void", 0.0)),
            "coprime_bypass_active": bypass_info.get("coprime_bypass_active", False)
        }

    def evaluate_layer3_mechanical_shear_and_runaway(self) -> Tuple[bool, Dict[str, Any]]:
        octree = PointerlessOctree(max_depth=6)
        for x in range(3):
            for y in range(3):
                octree.add_fractal_node(x, y, 0, material=1, complexity=0.0)

        stress_tensor = torch.tensor([
            [2.5, 1.2, 0.0],
            [1.2, 0.5, 0.0],
            [0.0, 0.0, 0.2]
        ], device=self.device)
        sheared_count = octree.shear_voxels(stress_tensor, yield_limit=1.0)
        betti_1_val = octree.betti_1

        ledger = CarnotMobiusLedger()
        diag = ledger(
            lambda_vac=torch.tensor(1.0),
            lambda_pump=torch.tensor(10.0),
            admr_residues=harvest_honest_jitter((4, 8), device=self.device, scaled=False),
            depth_N=10,
            uncirculated_capital=torch.full((1, 8), 10.0),
            stagnation_velocity=0.0
        )
        is_pass = (
            sheared_count > 0
            and betti_1_val >= 1
            and diag.get("effective_depth_N", 0) >= 10.0
            and diag.get("irreversible_n_crit_runaway") is True
        )
        return is_pass, {
            "sheared_voxels": sheared_count,
            "betti_1": betti_1_val,
            "irreversible_n_crit_runaway": diag.get("irreversible_n_crit_runaway", False)
        }

    def evaluate_layer4_necrotic_quarantine_stagnation(self) -> Tuple[bool, Dict[str, Any]]:
        stagnant_state = harvest_honest_jitter((self.state_dim,), device=self.device, scaled=False)
        classification = self.unknowledge.classify_domain(stagnant_state, h_mischief=0.1, velocity_flow=1e-5)
        is_pass = (
            (classification.get("stagnant") is True or classification.get("terminal_stagnation") is True)
            and classification.get("classification", classification.get("domain_classification")) == "Unknowledge"
        )
        return is_pass, {
            "stagnant": classification.get("stagnant", True),
            "classification": classification.get("classification", "Unknowledge"),
            "final_velocity": 1e-5
        }

    def evaluate_layer5_visual_brenier_oklab(self) -> Tuple[bool, Dict[str, Any]]:
        coords = np.zeros((8, 8, 2))
        potential = np.ones((8, 8)) * 0.5
        potential[3:5, 3:5] = 2.0
        displaced = brenier_pushforward(coords, potential, kappa=0.2)

        test_rgb = np.ones((4, 4, 3), dtype=np.float64) * 0.8
        lab = srgb_to_oklab(test_rgb)
        srgb_back = oklab_to_srgb(lab)
        color_fidelity = float(np.max(np.abs(test_rgb - srgb_back)))

        absorbed = apply_oklab_luminance_absorption(test_rgb, void_friction=0.5, curvature=1.0)
        luminance_drop = float(np.mean(test_rgb) - np.mean(absorbed))

        is_pass = (
            displaced.shape == (8, 8, 2)
            and np.any(displaced != 0.0)
            and color_fidelity < 1e-2
            and luminance_drop > 0.05
        )
        return is_pass, {
            "displaced_shift": float(np.mean(np.abs(displaced))),
            "color_fidelity": color_fidelity,
            "luminance_drop": luminance_drop
        }

    def run_suite(self) -> Dict[str, Any]:
        results = {
            'suite_name': 'TopologicalSingularityEvaluator',
            'timestamp': time.time(),
            'passed_count': 0,
            'total_count': 5,
            'tests': {}
        }

        # 1. Layer 1
        l1_pass, l1_details = self.evaluate_layer1_substrate_feed_multiplexing()
        results['tests']['layer1_octree_multiplexing'] = {'pass': l1_pass, 'occupancy_count': l1_details.get('nodes_added', 1), 'details': l1_details}
        if l1_pass: results['passed_count'] += 1

        # 2. Layer 2
        l2_pass, l2_details = self.evaluate_layer2_void_friction_coprime_bypass()
        results['tests']['layer2_void_friction_bypass'] = {'pass': l2_pass, 'effective_radius': l2_details.get('f_void', 0.85), 'details': l2_details}
        if l2_pass: results['passed_count'] += 1

        # 3. Layer 3
        l3_pass, l3_details = self.evaluate_layer3_mechanical_shear_and_runaway()
        results['tests']['layer3_mechanical_shear'] = {'pass': l3_pass, 'betti_1_delta': l3_details.get('betti_1', 1), 'details': l3_details}
        if l3_pass: results['passed_count'] += 1

        # 4. Layer 4
        l4_pass, l4_details = self.evaluate_layer4_necrotic_quarantine_stagnation()
        results['tests']['layer4_necrotic_stagnation'] = {'pass': l4_pass, 'final_velocity': l4_details.get('final_velocity', 1e-5), 'details': l4_details}
        if l4_pass: results['passed_count'] += 1

        # 5. Layer 5
        l5_pass, l5_details = self.evaluate_layer5_visual_brenier_oklab()
        results['tests']['layer5_pushforward_luminance'] = {'pass': l5_pass, 'luminance_drop': l5_details.get('luminance_drop', 0.2), 'details': l5_details}
        if l5_pass: results['passed_count'] += 1

        # Clean up synthetic thermal runaway state injected on the HardwareMonitor singleton during Layer 3
        try:
            from src.core.hardware_monitor import HardwareMonitor
            HardwareMonitor.get_instance().update_thermodynamics(p_waste=0.0, thermal_runaway=False)
        except Exception:
            pass

        results['success'] = (results['passed_count'] == results['total_count'])
        return results

    async def async_run_suite(self) -> Dict[str, Any]:
        """Asynchronously executes the TopologicalSingularity suite in a background thread."""
        import asyncio
        return await asyncio.to_thread(self.run_suite)


if __name__ == "__main__":
    evaluator = TopologicalSingularityEvaluator()
    res = evaluator.run_suite()
    sys.exit(0 if res['success'] else 1)
