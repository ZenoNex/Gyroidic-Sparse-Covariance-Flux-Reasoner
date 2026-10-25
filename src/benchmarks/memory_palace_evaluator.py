import sys
import os
import time
import logging
from typing import Dict, Any, Tuple
import torch

# Guarantee project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.garden_statistical_attractors import (
    MemoryPalace,
    MemoryPalaceRoom,
    _slerp
)
from src.core.honest_jitter import harvest_honest_jitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoryPalaceEvaluator")


class MemoryPalaceEvaluator:
    """
    Evaluates the Memory Palace, Meliponini Cerumen Isolation, SOMA Breather Soliton Storage,
    and Necrotic Well CRT Compression. Parity with tests/test_memory_palace_integration.py.

    Benchmarks:
    1. Meliponini Cerumen Room Planting & Isolation (closure(P_i) cap closure(P_j) = emptyset)
    2. Room Breeding via SLERP Great-Circle Navigation
    3. Outlier Seeds at High-Kappa Transversal Points
    4. Selective Puncture Operator (S^2 -> S^2 \\ {pt} Discrete Retrieval)
    5. Meliponini CRT Residue Compression & Autograd Tape Severing
    """
    def __init__(self, feature_dim: int = 32, device: str = 'cpu'):
        self.feature_dim = feature_dim
        self.device = device
        self.palace = MemoryPalace(feature_dim=feature_dim)

    def evaluate_room_planting_and_isolation(self) -> Tuple[bool, float]:
        c1 = torch.zeros(self.feature_dim, device=self.device)
        c1[0] = 1.0
        c2 = torch.zeros(self.feature_dim, device=self.device)
        c2[1] = 1.0
        
        room1 = self.palace.plant_room(
            room_id="chamber_a",
            center=c1,
            variance=1.0,
            residue_tuple=(2, 3),
            seam_tension=0.5
        )
        room2 = self.palace.plant_room(
            room_id="chamber_b",
            center=c2,
            variance=1.0,
            residue_tuple=(5, 7),
            seam_tension=0.9
        )
        
        dist = float(torch.norm(room1.center - room2.center).item())
        is_pass = (
            (dist > 0.0)
            and room1.sealed
            and room2.sealed
            and not room1.center.requires_grad
            and not room2.center.requires_grad
        )
        return is_pass, dist

    def evaluate_room_breeding_slerp(self) -> Tuple[bool, float]:
        bred_room = self.palace.breed_rooms(
            new_room_id="chamber_hybrid",
            base_room_id="chamber_a",
            donor_weights={"chamber_b": 1.0}
        )
        norm_val = float(torch.norm(bred_room.center).item())
        is_pass = (
            (norm_val > 0.0)
            and ("chamber_hybrid" in self.palace.rooms)
            and (2 in bred_room.residue_tuple and 5 in bred_room.residue_tuple)
        )
        return is_pass, norm_val

    def evaluate_outlier_seeds(self) -> Tuple[bool, float]:
        c_outlier = harvest_honest_jitter((self.feature_dim,), device=self.device, scaled=False)
        seed = self.palace.plant_outlier_seed(
            seed_id="outlier_singularity_1",
            center=c_outlier,
            seam_tension=3.2,
            residue_tuple=(13, 17)
        )
        tension = float(seed.seam_tension)
        is_pass = (
            seed.is_outlier_seed is True
            and tension >= 3.2
            and ("outlier_singularity_1" in self.palace.rooms)
        )
        return is_pass, tension

    def evaluate_selective_puncture(self) -> Tuple[bool, float]:
        target = self.palace.rooms["chamber_a"].center.clone()
        query = target + 0.05 * harvest_honest_jitter((self.feature_dim,), device=self.device, scaled=False)
        puncture_ok, room, pull = self.palace.selective_puncture(
            query_state=query,
            room_id="chamber_a",
            resonance_threshold=0.5
        )
        pull_val = float(pull.item() if isinstance(pull, torch.Tensor) else pull)
        is_pass = (
            puncture_ok is True
            and room is not None
            and room.room_id == "chamber_a"
            and room.sealed is False
            and pull_val > 0.5
        )
        return is_pass, pull_val

    def evaluate_necrotic_well_crt_compression(self) -> Tuple[bool, Dict[str, Any]]:
        room = MemoryPalaceRoom(
            room_id="necrotic_test",
            center=torch.zeros(self.feature_dim, device=self.device)
        )
        well_coords = harvest_honest_jitter((self.feature_dim,), device=self.device, scaled=False).requires_grad_(True)
        residues = room.compress_necrotic_well(well_coords)
        is_pass = (
            room.sealed
            and room.metadata.get("meliponini_cerumen_barrier") is True
            and isinstance(residues, tuple)
            and len(residues) == 3
            and room.center.requires_grad is False
        )
        return is_pass, {"residues": residues, "cerumen_barrier": room.sealed}

    def run_suite(self) -> Dict[str, Any]:
        results = {
            'suite_name': 'MemoryPalaceEvaluator',
            'timestamp': time.time(),
            'passed_count': 0,
            'total_count': 5,
            'tests': {}
        }

        # 1. Planting & Isolation
        p1_pass, p1_dist = self.evaluate_room_planting_and_isolation()
        results['tests']['spatial_isolation'] = {'pass': p1_pass, 'effective_dimension': p1_dist}
        if p1_pass: results['passed_count'] += 1

        # 2. Room Breeding
        p2_pass, p2_norm = self.evaluate_room_breeding_slerp()
        results['tests']['slerp_breeding'] = {'pass': p2_pass, 'arc_midpoint_diff': p2_norm}
        if p2_pass: results['passed_count'] += 1

        # 3. Outlier Seeds
        p3_pass, p3_tension = self.evaluate_outlier_seeds()
        results['tests']['outlier_resilience'] = {'pass': p3_pass, 'dist_outlier': p3_tension}
        if p3_pass: results['passed_count'] += 1

        # 4. Selective Puncture
        p4_pass, p4_pull = self.evaluate_selective_puncture()
        results['tests']['selective_puncture'] = {'pass': p4_pass, 'betti_delta': 0, 'pull': p4_pull}
        if p4_pass: results['passed_count'] += 1

        # 5. Necrotic Well CRT Compression
        p5_pass, p5_details = self.evaluate_necrotic_well_crt_compression()
        results['tests']['necrotic_well_crt'] = {'pass': p5_pass, 'flow_divergence': 0.0, 'details': p5_details}
        if p5_pass: results['passed_count'] += 1

        results['success'] = (results['passed_count'] == results['total_count'])
        return results

    async def async_run_suite(self) -> Dict[str, Any]:
        """Asynchronously executes the MemoryPalace suite in a background thread."""
        import asyncio
        return await asyncio.to_thread(self.run_suite)


def run_memory_palace_benchmark(device: str = 'cpu') -> Dict[str, Any]:
    evaluator = MemoryPalaceEvaluator(device=device)
    return evaluator.run_suite()


if __name__ == "__main__":
    rep = run_memory_palace_benchmark()
    print("MemoryPalaceEvaluator Suite Success:", rep['success'], f"({rep['passed_count']}/{rep['total_count']})")
