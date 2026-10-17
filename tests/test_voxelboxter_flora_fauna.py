"""
Tests for Voxelboxter Ecological Simulation Layer:
- Flora Dynamics & Cashew Tree of Pirangi Mutation
- B-Splines as True Geometry (BSplineCompiledMod)
- Dual-Scale Plasticity (Mohr-Coulomb vs Drucker-Prager)
- FBM Weathering & Ley Line Carving
- Fauna ISN Dynamics & Defect Propagation PDE
- Emergent Baking Engine & Single Unified Graph Preservation
"""

import unittest
import torch
import math
from typing import Tuple

from src.ui.voxelboxter_simulation import (
    StructuralGraph, Block, RigidBody, InventoryComponent, Role, PermissionsManager,
    BSplineCompiledMod, AddonRoutine, BooleanXORLayer,
    FloraComponent, FloraTreeSapling, PirangiCashewTree, AdventitiousRootNode,
    FaunaComponent, FaunaISN, EmergentBakingEngine,
    VoxelboxterEngine, PointerlessOctree, morton_encode,
    VoxelboxterAdmissibilityGate, SpectralSpeculationEngine
)
from src.environment.cleanroom_mechanics import SubgridRotationalWireHarness, MultipartVoxelCell
from src.scripting.node_environment import PhysicalNodeEditor

class TestVoxelboxterFloraFauna(unittest.TestCase):

    def test_morton_encoding_consistency(self):
        """Verify 3D Morton bit interleaving produces unique deterministic 1D keys."""
        code1 = morton_encode(0, 0, 0)
        code2 = morton_encode(1, 0, 0)
        code3 = morton_encode(0, 1, 0)
        code4 = morton_encode(0, 0, 1)
        self.assertEqual(code1, 0)
        self.assertNotEqual(code2, code3)
        self.assertNotEqual(code3, code4)

    def test_pirangi_cashew_tree_mutation(self):
        """
        Verify the Cashew Tree of Pirangi mutation mechanics:
        - Inverted gravitropism (horizontal branch growth)
        - Cantilever droop under mass accumulation
        - Adventitious rooting on ground contact
        - Secondary trunk metamorphosis spawning radial child branches
        - Single unified organism: single connected StructuralGraph component
        - Mass deduction from inventory
        """
        inventory = InventoryComponent(block_masses={4: 200, 5: 100, 6: 100})
        octree = PointerlessOctree()

        # Plant sapling
        tree = FloraTreeSapling.plant_sapling(
            position=(10, 10, 2),
            inventory=inventory,
            player_id="ADMIN",
            ground_z=0
        )
        self.assertIsNotNone(tree)
        # 4 wood mass was deducted for planting the initial sapling
        self.assertEqual(inventory.block_masses[4], 196)
        
        # Verify initial primary trunk and 4 radial lateral tips (inverted gravitropism)
        self.assertEqual(len(tree.active_branch_tips), 4)
        initial_blocks = len(tree.graph.blocks)
        self.assertGreater(initial_blocks, 0)

        # Run several growth ticks to simulate lateral expansion and droop
        for tick in range(10):
            res = tree.grow_tick(terrain_octree=octree, inventory=inventory, max_new_blocks_per_tick=6)

        # 1. Canopy must have expanded horizontally
        self.assertGreater(tree.flora_comp.canopy_radius, 1.5)
        
        # 2. Adventitious roots must have been formed as branches touched ground_z
        self.assertGreater(tree.flora_comp.adventitious_roots_count, 0)
        
        # 3. Root sites must have metamorphosed into secondary trunks
        self.assertGreater(tree.flora_comp.secondary_trunks_count, 0)
        self.assertGreater(len(tree.root_nodes), 0)

        # 4. SINGLE UNIFIED ORGANISM CHECK:
        # Crucial: The entire tree must form a single connected component in StructuralGraph
        comps = tree.graph.find_disconnected_components()
        self.assertEqual(len(comps), 1, "Pirangi Cashew Tree must remain a single connected topological construct")

        # 5. Inventory mass conservation
        self.assertLess(inventory.block_masses[4], 196)

    def test_bspline_compiled_mod(self):
        """Verify KAN Cox-de Boor B-spline compiled mod non-heuristic generation."""
        mod = BSplineCompiledMod(name="test_hull_spline", latent_dim=3, resolution=10)
        t = torch.linspace(0, 1, 5)
        curve_pts = mod.evaluate_curve(t)
        self.assertEqual(curve_pts.shape, (5, 3))

        # Check layer integration into AddonRoutine
        routine = AddonRoutine()
        routine.add_layer(mod)
        graph = StructuralGraph()
        mod.execute(graph)
        self.assertGreater(len(graph.blocks), 0)

    def test_fauna_isn_dynamics_and_defect_pde(self):
        """
        Verify animal fauna Inhibition-Stabilized Network (ISN):
        - Excitation/Inhibition ratio remains bounded (no runaway explosion or collapse)
        - Defect propagation PDE diffuses environmental disruption
        - Chiral phase steering
        """
        fauna = FaunaComponent(
            entity_id="stalker_01",
            excitation_level=1.0,
            inhibition_level=1.0,
            defect_density=0.1
        )
        isn = FaunaISN()

        # Step population with external disturbance
        for _ in range(20):
            metrics = isn.step_population(
                fauna,
                external_stimulus=0.2,
                local_gyroid_potential=0.1,
                dt=0.05
            )

        self.assertGreater(metrics["excitation"], 0.0)
        self.assertGreater(metrics["inhibition"], 0.0)
        # E/I ratio should be in biologically stable homeostasis
        self.assertGreater(metrics["ei_ratio"], 0.2)
        self.assertLess(metrics["ei_ratio"], 5.0)
        self.assertIn(metrics["chirality_phase"], [-1.0, 1.0])

    def test_emergent_baking_engine(self):
        """Verify emergent baking turns subjective constructs into fossilized topological bodies."""
        perms = PermissionsManager()
        perms.roles["ADMIN_USER"] = Role.ADMIN
        perms.roles["GUEST_USER"] = Role.VISITOR

        graph = StructuralGraph()
        b1 = Block(local_cell=(0, 0, 0), rotation=(0, 0, 0, 1), material_id=1, health=50.0)
        b2 = Block(local_cell=(1, 0, 0), rotation=(0, 0, 0, 1), material_id=1, health=50.0)
        graph.add_block(b1)
        graph.add_block(b2)

        engine = EmergentBakingEngine()
        # Visitor cannot bake
        res_fail = engine.bake_construct("house_01", graph, perms, "GUEST_USER")
        self.assertFalse(res_fail["success"])

        # Admin can bake
        res_ok = engine.bake_construct("house_01", graph, perms, "ADMIN_USER")
        self.assertTrue(res_ok["success"])
        self.assertEqual(res_ok["total_fossilized_blocks"], 2)
        self.assertEqual(b1.health, 999.0)

    def test_voxelboxter_dual_scale_plasticity_and_ley_lines(self):
        """
        Verify VoxelboxterEngine dual-scale plasticity:
        - Mohr-Coulomb brittle shear fracture
        - Drucker-Prager global flow envelope
        - FBM weathering Ley line carving
        """
        engine = VoxelboxterEngine(device="cpu")

        # Setup construct
        graph = StructuralGraph()
        for x in range(3):
            b = Block(local_cell=(x, 0, 0), rotation=(0, 0, 0, 1), material_id=1, health=100.0)
            graph.add_block(b)
        engine.constructs["kart_track"] = graph
        inv = InventoryComponent()

        # High impulse impact: triggers Mohr-Coulomb brittle rupture
        stress = torch.tensor([[5.0]])
        fracture_res = engine.apply_dual_yield_fracture(
            impulse_J=50.0,
            stress_tensor=stress,
            local_cell=(0, 0, 0),
            construct_id="kart_track",
            inventory=inv
        )
        self.assertTrue(fracture_res["is_brittle_rupture"])
        self.assertIn((0, 0, 0), fracture_res["detached_blocks"])
        self.assertIn(1, inv.block_masses)

        # Ley line FBM carving
        track_coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 4.0, 0.0]])
        ley_res = engine.carve_weathering_ley_lines(track_coords, vehicle_mass=2.5, traffic_intensity=1.5)
        self.assertIn("erosion_depth", ley_res)
        self.assertGreater(ley_res["slipstream_boost"], 0.0)

    def test_wager_1_admissibility_gate(self):
        """
        Verify Wager #1 (The 90/10 Rejection Ratio):
        - Geometric probe checks Gyroid violation H_spec
        - Filters proposed voxel batches with O(1) probe before simulation
        """
        gate = VoxelboxterAdmissibilityGate(epsilon=0.08)
        cells = [(0, 0, 0), (1, 1, 1), (10, 20, 30), (5, 5, 5)]
        report = gate.filter_construct_proposal(cells)
        self.assertEqual(report["total_evaluated"], 4)
        self.assertIn("rejection_ratio", report)
        self.assertGreaterEqual(len(report["admitted_cells"]) + len(report["rejected_cells"]), 4)

    def test_wager_2_fossilization_split_80_20(self):
        """
        Verify Wager #2 (The Fossilization Split: 80/20):
        - 80% immutable fossilized bone, 20% plastic flesh
        - Precomputed bone COM and inertia tensor
        - Collision detachment selectively sheds plastic flesh, protecting bone skeleton
        """
        graph = StructuralGraph()
        for x in range(10):
            graph.add_block(Block(local_cell=(x, 0, 0), rotation=(0, 0, 0, 1), material_id=1, health=100.0))
        
        split_res = graph.fossilize_core(bone_ratio=0.8)
        self.assertEqual(split_res["bone_blocks"], 8)
        self.assertEqual(split_res["flesh_blocks"], 2)
        self.assertEqual(len(graph.bone_blocks), 8)
        self.assertEqual(len(graph.flesh_blocks), 2)
        self.assertGreater(split_res["bone_mass"], 0.0)
        self.assertEqual(len(split_res["bone_inertia_tensor"]), 3)

        # RigidBody collision impulse detachment
        rb = RigidBody(mass=10.0)
        inv = InventoryComponent()
        # Mid impulse detaches one flesh block
        col_res = rb.apply_collision_impulse(impulse_J=30.0, graph=graph, inventory=inv)
        self.assertEqual(col_res["tier"], "MID")
        self.assertEqual(len(col_res["detached_blocks"]), 1)
        detached_cell = col_res["detached_blocks"][0]
        # Detached cell must NOT have been a bone block
        self.assertNotIn(detached_cell, graph.bone_blocks)

    def test_wager_3_topological_macro_moves(self):
        """
        Verify Wager #3 (The Structure/Weight Bet: 99/1):
        - Topological macro-move surgery (Delta Topology >= 1) over infinitesimal weight nudges
        - Slices flesh boundary, preserving fossilized bone core
        """
        graph = StructuralGraph()
        for x in range(8):
            graph.add_block(Block(local_cell=(x, 0, 0), rotation=(0, 0, 0, 1), material_id=1, health=100.0))
        # 80/20 split
        graph.fossilize_core(bone_ratio=0.75)
        # Slices across x = 6.0
        macro_res = graph.execute_topological_macro_move(cut_normal=(1.0, 0.0, 0.0), cut_offset=6.0)
        self.assertTrue(macro_res["macro_move_executed"])
        self.assertIn("delta_topology", macro_res)
        self.assertIn("initial_betti", macro_res)

    def test_wager_4_spectral_speculation_engine(self):
        """
        Verify Wager #4 (Spectral Speculation: Frequency > Time):
        - Projects vibrational modes onto PrimeResonanceLadder
        - Low spectral entropy H_spec triggers fastpass bypass of System 2 loops
        """
        spec = SpectralSpeculationEngine(epsilon=0.9, num_resonators=16)
        signal = torch.zeros(16)
        signal[0] = 10.0
        can_fastpass, h_spec = spec.evaluate_spectral_entropy(signal)
        self.assertTrue(can_fastpass)
        self.assertLess(h_spec, 0.9)
        self.assertGreater(spec.fastpass_count, 0)

    def test_non_conservative_circulation_and_local_repair(self):
        """
        Verify Non-Conservative Circulation (oint Phi != 0) & Local O(1) Repair:
        - Relaxes local topological stress in O(1) neighborhood
        - Evaluates rotating wire harness non-conservative circulation
        """
        graph = StructuralGraph()
        b_center = Block(local_cell=(0, 0, 0), rotation=(0, 0, 0, 1), material_id=1, health=100.0, topological_stress=5.0)
        b_nb = Block(local_cell=(1, 0, 0), rotation=(0, 0, 0, 1), material_id=1, health=100.0, topological_stress=3.0)
        graph.add_block(b_center)
        graph.add_block(b_nb)

        rep_res = graph.repair_local_neighborhood(origin_cell=(0, 0, 0), radius=1)
        self.assertEqual(rep_res["complexity"], "O(1)")
        self.assertEqual(b_center.topological_stress, 0.0)
        self.assertEqual(b_nb.topological_stress, 0.0)
        self.assertEqual(rep_res["relaxed_stress"], 8.0)

        # Subgrid wire harness circulation
        wire = SubgridRotationalWireHarness()
        wire.set_channel_signal(0, 15)
        wire.set_channel_signal(4, 5)
        circ_res = wire.evaluate_non_conservative_circulation()
        self.assertIn("closed_loop_circulation", circ_res)
        self.assertEqual(circ_res["local_repair_cost"], "O(1)")

    def test_physical_node_editor_efficiency_singularity_hook(self):
        """
        Verify PhysicalNodeEditor.hook_voxelboxter_efficiency_singularity integration
        across Wagers #1 through #4 and Kelly Defense.
        """
        editor = PhysicalNodeEditor()
        hook_res = editor.hook_voxelboxter_efficiency_singularity()
        self.assertIn("wager_1_admissibility", hook_res)
        self.assertIn("wager_2_fossilization_split", hook_res)
        self.assertIn("wager_3_macro_move", hook_res)
        self.assertIn("wager_4_spectral_speculation", hook_res)
        self.assertIn("non_conservative_local_repair", hook_res)
        self.assertIn("non_ergodic_kelly_growth", hook_res)

    def test_fractal_navigation_scale_expansion_and_contraction(self):
        """
        Verify 4th-dimensional fractal navigation across scale:
        - Expanding x16 transforms a single geometric block into a complete 16x16x4 nested biome.
        - Contracting 1/16 collapses the nested biome into a single macroscopic block.
        """
        octree = PointerlessOctree()
        octree.add_fractal_node(2, 3, 4, material=7)
        self.assertEqual(len(octree.morton_grid), 1)

        # Expand x16
        child_octree = octree.traverse_scale_expand_16x((2, 3, 4))
        self.assertEqual(len(child_octree.morton_grid), 16 * 16 * 4) # 1024 sub-voxels
        # All nested sub-voxels inherited the parent material
        self.assertTrue(all(m == 7 for m in child_octree.morton_grid.values()))

        # Contract 1/16 back to macro scale
        collapsed_octree = child_octree.traverse_scale_contract_16x()
        self.assertEqual(len(collapsed_octree.morton_grid), 1)
        self.assertIn(collapsed_octree._interleave_bits(0, 0, 0), collapsed_octree.morton_grid)
        self.assertEqual(collapsed_octree.morton_grid[collapsed_octree._interleave_bits(0, 0, 0)], 7)

    def test_boundary_tearing_extreme_velocity_puncture(self):
        """
        Verify boundary tearing failure mode analysis:
        - Normal velocity remains Euclidean-constrained.
        - High-velocity impulse (camper van launched across non-Euclidean terrain) punctures simulated rules.
        """
        rb = RigidBody(linear_velocity=(10.0, 5.0, 2.0))
        normal_res = rb.check_boundary_tearing_puncture(extreme_velocity_threshold=45.0)
        self.assertFalse(normal_res["punctured"])
        self.assertEqual(normal_res["puncture_type"], "EUCLIDEAN_CONSTRAINED")
        self.assertFalse(normal_res["substrate_exposed"])

        # Extreme velocity puncture: Skybox launch
        rb_extreme = RigidBody(linear_velocity=(30.0, 30.0, 40.0)) # speed > 58.0
        tear_res = rb_extreme.check_boundary_tearing_puncture(extreme_velocity_threshold=45.0)
        self.assertTrue(tear_res["punctured"])
        self.assertEqual(tear_res["puncture_type"], "SKYBOX_LAUNCH")
        self.assertTrue(tear_res["substrate_exposed"])

    def test_operator_spatial_telemetry_and_drift_alert(self):
        """
        Verify operator spatial navigation and gaze tracking telemetry evaluation:
        - Spatial traversal friction and scan jitter yield bounded interaction workload.
        - High anchor coherence maintains stable tracking state.
        - Degraded anchor coherence triggers tracking drift alert.
        """
        engine = VoxelboxterEngine()

        # Stable tracking state with descriptive telemetry API
        stable_res = engine.evaluate_operator_telemetry(
            spatial_friction=1.5,
            scan_jitter=0.08,
            anchor_coherence=0.96
        )
        self.assertTrue(stable_res["telemetry_recorded"])
        self.assertTrue(stable_res["is_stable"])
        self.assertFalse(stable_res["drift_alert"])
        self.assertEqual(stable_res["status"], "TRACKING_STABLE")
        self.assertAlmostEqual(stable_res["drift_risk"], 0.04, places=4)

        # Degraded tracking state triggering drift alert
        degraded_res = engine.evaluate_operator_telemetry(
            spatial_friction=2.0,
            scan_jitter=0.15,
            anchor_coherence=0.70
        )
        self.assertTrue(degraded_res["drift_alert"])
        self.assertFalse(degraded_res["is_stable"])
        self.assertEqual(degraded_res["status"], "TRACKING_DRIFT_ALERT")
        self.assertAlmostEqual(degraded_res["drift_risk"], 0.30, places=4)

        # Legacy alias compatibility verification
        legacy_stable = engine.harvest_zar_inner_core(
            spatial_maze_friction=1.5,
            ocular_scan_jitter=0.08,
            anchor_coherence=0.96
        )
        self.assertTrue(legacy_stable["inner_core_generated"])
        self.assertTrue(legacy_stable["is_safely_contained"])
        self.assertFalse(legacy_stable["catatonic_hazard"])
        self.assertEqual(legacy_stable["status"], "SUPERPOSITION_STABLE")

        legacy_degraded = engine.harvest_zar_inner_core(
            spatial_maze_friction=2.0,
            ocular_scan_jitter=0.15,
            anchor_coherence=0.70
        )
        self.assertTrue(legacy_degraded["catatonic_hazard"])
        self.assertFalse(legacy_degraded["is_safely_contained"])
        self.assertEqual(legacy_degraded["status"], "SCALE_DISSOCIATION_CONTAINMENT_ALERT")

if __name__ == '__main__':
    unittest.main()

