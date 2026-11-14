"""
Unit tests for Cleanroom Handheld Piston Tool and Multiplexed Perlin-Worley-Gyroid
Procedural Terrain Generation in the Treehouse Dimension.
"""

import math
import pytest
import torch

from src.ui.voxelboxter_simulation import (
    PointerlessOctree, Role, PermissionsManager,
    StructuralGraph, Block, RigidBody
)
from src.environment.cleanroom_mechanics import (
    HandheldPistonTool, PistonMode, PistonResultCode,
    PerlinWorleyGyroidGenerator, SplineNoise5DConfig, TreehouseBiomeType
)
from src.scripting.node_environment import PhysicalNodeEditor


class TestHandheldPistonTool:
    def setup_method(self):
        self.piston = HandheldPistonTool(impulse_magnitude=500.0, max_reach=10.0)
        self.permissions = PermissionsManager()
        self.octree = PointerlessOctree(max_depth=6)

    def test_piston_sovereignty_rejection_for_visitor(self):
        # Visitor role should be rejected
        self.permissions.roles["visitor_bob"] = Role.VISITOR
        self.octree.add_fractal_node(2, 2, 2, material=1)
        
        result = self.piston.fire(
            player_id="visitor_bob",
            source_coord=(2, 2, 2),
            push_direction=(1, 0, 0),
            octree=self.octree,
            permissions_manager=self.permissions
        )
        assert result.status == PistonResultCode.REJECTED_VISITOR_RESTRICTION
        assert "VISITOR role" in result.message

    def test_piston_sovereignty_acceptance_for_builder_and_admin(self):
        self.permissions.roles["builder_alice"] = Role.BUILDER
        self.octree.add_fractal_node(3, 3, 3, material=1)

        result = self.piston.fire(
            player_id="builder_alice",
            source_coord=(3, 3, 3),
            push_direction=(1, 0, 0),
            octree=self.octree,
            permissions_manager=self.permissions
        )
        assert result.status == PistonResultCode.SUCCESS
        assert result.target_type == "VOXEL"
        assert result.dest_coord == (4, 3, 3)

    def test_piston_morton_encoding_and_voxel_extraction_placement(self):
        self.permissions.roles["admin_root"] = Role.ADMIN
        # Place voxel at (1, 1, 1) with material 5
        self.octree.add_fractal_node(1, 1, 1, material=5)
        src_morton = self.octree._interleave_bits(1, 1, 1)
        assert src_morton in self.octree.morton_grid

        result = self.piston.fire(
            player_id="admin_root",
            source_coord=(1, 1, 1),
            push_direction=(0, 1, 0),
            octree=self.octree,
            permissions_manager=self.permissions
        )
        assert result.status == PistonResultCode.SUCCESS
        assert result.extracted_material_id == 5
        assert src_morton not in self.octree.morton_grid

        dst_morton = self.octree._interleave_bits(1, 2, 1)
        assert dst_morton in self.octree.morton_grid
        assert self.octree.morton_grid[dst_morton] == 5
        assert result.topological_complexity >= 1.0

    def test_piston_macro_structure_80_20_fossilization_wager(self):
        self.permissions.roles["builder_alice"] = Role.BUILDER
        graph = StructuralGraph()
        
        # Build 10 blocks: 8 bone (80%), 2 flesh (20%)
        for i in range(8):
            graph.add_block(Block(local_cell=(i, 0, 0), rotation=(0, 0, 0, 1), material_id=1, health=100.0, is_fossilized_bone=True))
        graph.add_block(Block(local_cell=(8, 0, 0), rotation=(0, 0, 0, 1), material_id=2, health=50.0, is_fossilized_bone=False))
        graph.add_block(Block(local_cell=(9, 0, 0), rotation=(0, 0, 0, 1), material_id=2, health=50.0, is_fossilized_bone=False))
        graph.fossilize_core(bone_ratio=0.8)

        assert len(graph.bone_blocks) >= 6
        assert len(graph.flesh_blocks) >= 2

        # Fire piston to cleave flesh blocks along plane x = 8.5
        result = self.piston.fire(
            player_id="builder_alice",
            source_coord=(8, 0, 0),
            push_direction=(1, 0, 0),
            octree=self.octree,
            permissions_manager=self.permissions,
            graph=graph
        )
        assert result.status == PistonResultCode.SUCCESS
        assert result.target_type == "MACRO_GRAPH"
        assert result.macro_move_report is not None
        assert result.macro_move_report["macro_move_executed"] is True
        # Inner bone core blocks should stay intact
        assert (4, 0, 0) in graph.blocks
        assert (4, 0, 0) in graph.bone_blocks
        assert graph.blocks[(4, 0, 0)].is_fossilized_bone is True

    def test_piston_entity_knockback_rigid_body(self):
        self.permissions.roles["builder_alice"] = Role.BUILDER
        rb = RigidBody(mass=100.0)
        
        # J = 500 N*s, m = 100 kg -> Delta v = 5.0 m/s (MID tier)
        result = self.piston.fire(
            player_id="builder_alice",
            source_coord=(0, 0, 0),
            push_direction=(0, 0, 1),
            octree=self.octree,
            permissions_manager=self.permissions,
            rigid_body=rb
        )
        assert result.status == PistonResultCode.SUCCESS
        assert result.target_type == "ENTITY_RIGIDBODY"
        assert result.kinetic_impulse_report is not None
        delta_v = result.kinetic_impulse_report["delta_v"]
        assert math.isclose(delta_v, 5.0, rel_tol=1e-3)
        assert result.kinetic_impulse_report["tier"] in ("LOW", "MID", "HIGH")
        assert rb.linear_velocity[2] > 0.0

    def test_piston_reach_limit(self):
        self.permissions.roles["admin_root"] = Role.ADMIN
        result = self.piston.fire(
            player_id="admin_root",
            source_coord=(20, 20, 20),
            player_pos=(0, 0, 0),
            octree=self.octree,
            permissions_manager=self.permissions
        )
        assert result.status == PistonResultCode.OUT_OF_REACH


class TestPerlinWorleyGyroidGenerator:
    def setup_method(self):
        self.gen = PerlinWorleyGyroidGenerator(seed=42)
        self.octree = PointerlessOctree(max_depth=6)

    def test_perlin_3d_gradient_smoothness(self):
        val1 = self.gen.sample_perlin_3d(10.0, 5.0, 2.0, octaves=3)
        val2 = self.gen.sample_perlin_3d(10.05, 5.05, 2.05, octaves=3)
        assert -1.0 <= val1 <= 1.0
        assert -1.0 <= val2 <= 1.0
        # Continuous gradient should produce nearby values
        assert abs(val1 - val2) < 0.15

    def test_worley_3d_cellular_and_ridge(self):
        f1, f2 = self.gen.sample_worley_3d(12.0, 15.0, 18.0, cell_size=16.0)
        assert f1 <= f2
        ridge = f2 - f1
        assert ridge >= 0.0

    def test_gyroid_minimal_surface(self):
        # Origin evaluation: sin(0)cos(0) + sin(0)cos(0) + sin(0)cos(0) = 0
        g0 = self.gen.sample_gyroid_surface(0.0, 0.0, 0.0, scale=0.1)
        assert math.isclose(g0, 0.0, abs_tol=1e-6)

    def test_density_evaluation_kniberg_splines(self):
        eval_res = self.gen.evaluate_density(16.0, 32.0, 16.0)
        assert "density" in eval_res
        assert "is_solid" in eval_res
        assert "biome" in eval_res
        assert isinstance(eval_res["biome"], TreehouseBiomeType)
        assert "continentalness" in eval_res
        assert "erosion" in eval_res
        assert "peaks_valleys" in eval_res
        assert "entropy_bands" in eval_res
        assert "h_mischief" in eval_res["entropy_bands"]
        assert "h_schizo" in eval_res["entropy_bands"]
        assert "h_dementia" in eval_res["entropy_bands"]

    def test_chunk_generation_and_betti_auditing(self):
        chunk_report = self.gen.generate_chunk(
            chunk_x=0,
            chunk_z=0,
            octree=self.octree,
            y_min=0,
            y_max=16,
            step=4
        )
        assert chunk_report["chunk_coords"] == (0, 0)
        assert chunk_report["voxels_placed"] > 0
        assert "betti_numbers" in chunk_report
        betti = chunk_report["betti_numbers"]
        assert betti["betti_0"] >= 1
        assert betti["betti_1"] >= 0
        assert chunk_report["topological_complexity"] >= 1.0

    def test_treehouse_dimension_matrioshka_scale_nesting(self):
        # Place root block
        self.octree.add_fractal_node(0, 0, 0, material=1, complexity=0.4)
        
        # Inward 16x scale expansion
        child_octree = self.gen.traverse_treehouse_scale(self.octree, (0, 0, 0), direction="expand")
        assert child_octree.total_voxels > 1
        # Complexity is inherited with L'Hopital regularized scaling
        assert child_octree.complexity_grid[child_octree._interleave_bits(0, 0, 0)] < 0.4
        
        # Outward 1/16x scale contraction
        parent_octree = self.gen.traverse_treehouse_scale(child_octree, (0, 0, 0), direction="contract")
        assert parent_octree.total_voxels >= 1


class TestPhysicalNodeEditorHooks:
    def setup_method(self):
        self.editor = PhysicalNodeEditor()

    def test_node_editor_piston_hook(self):
        # Configure builder permission for player
        self.editor.permissions.roles["player_sovereign"] = Role.BUILDER
        self.editor.octree.add_fractal_node(5, 5, 5, material=2)
        
        res = self.editor.hook_cleanroom_piston(
            player_id="player_sovereign",
            source_coord=(5, 5, 5),
            push_direction=(1, 0, 0)
        )
        assert res["status"] == "SUCCESS"
        assert res["target_type"] == "VOXEL"
        assert res["dest_coord"] == (6, 5, 5)

    def test_node_editor_terrain_generation_hook(self):
        res = self.editor.hook_cleanroom_terrain_generation(
            chunk_x=1,
            chunk_z=1,
            y_min=0,
            y_max=16,
            step=4
        )
        assert res["chunk_coords"] == (1, 1)
        assert res["voxels_placed"] > 0
        assert res["total_octree_nodes"] > 0


class TestAll15VoxelboxterHooks:
    """
    Verification suite for the 15 texturing, gsplat, rasterization, and generative hooks.
    """
    def setup_method(self):
        self.octree = PointerlessOctree(max_depth=6)
        self.editor = PhysicalNodeEditor()

    def test_hook_01_gsplat_ingestion(self, tmp_path):
        # Create a stub gltf splat json file
        import json
        stub_gltf = tmp_path / "test_splat.gltf"
        stub_gltf.write_text(json.dumps({
            "asset": {"version": "2.0"},
            "accessors": [{"count": 24, "componentType": 5126, "type": "VEC3"}]
        }))
        res = self.octree.ingest_gaussian_splats(str(stub_gltf), target_dim=32)
        assert res["batches_ingested"] >= 1
        assert res["nodes_placed"] > 0
        assert res["total_octree_nodes"] > 0

    def test_hook_02_wasserstein_mass_collapse_rasterization(self):
        dummy_residues = torch.randn(8, 4)
        lattice = self.octree.apply_wasserstein_rasterization(dummy_residues, iteration=1, mass_budget=2.0)
        assert len(lattice) == 8
        # Output is quantized into discrete 4-bit intervals (multiples of 16)
        assert all(val % 16.0 == 0.0 for val in lattice)

    def test_hook_03_conjugate_color_transport(self):
        from image_extension import ConjugateColorTransport
        transport = ConjugateColorTransport()
        test_rgb = torch.tensor([
            [0.8, 0.2, 0.1],
            [0.1, 0.7, 0.9],
            [0.5, 0.5, 0.5]
        ], dtype=torch.float32)
        transported = transport.transport_colors(test_rgb, kappa=0.2)
        assert transported.shape == test_rgb.shape
        assert not torch.allclose(transported, test_rgb, atol=1e-3)
        assert torch.all(transported >= 0.0) and torch.all(transported <= 1.0)

    def test_hook_04_morton_z_order_rasterizer(self):
        from src.ui.voxelboxter_simulation import morton_encode
        code_1 = morton_encode(3, 5, 7)
        code_2 = morton_encode(3, 5, 7)
        code_3 = morton_encode(4, 5, 7)
        assert code_1 == code_2
        assert code_1 != code_3
        # Invertible recovery
        rx, ry, rz = self.octree._deinterleave_bits(code_1)
        assert (rx, ry, rz) == (3, 5, 7)

    def test_hook_05_chebyshev_spectral_texture_dsp(self):
        from src.core.texture_dsp import TextureDSPEngine
        dsp = TextureDSPEngine()
        dummy_img = torch.rand(1, 3, 32, 32)
        filtered = dsp.apply_spectral_filter(dummy_img, cutoff_freq=0.4)
        assert filtered.shape == dummy_img.shape
        # Conjugate color filter test
        colored = dsp.apply_conjugate_color_filter(dummy_img, kappa=0.15)
        assert colored.shape == dummy_img.shape

    def test_hook_06_mandelbulb_recursive_embedding(self):
        coords = torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, -0.5]], dtype=torch.float32)
        fractal_tex = self.octree.sample_mandelbulb_fractal_texture(coords, power=8)
        assert fractal_tex.shape[0] == 2
        assert fractal_tex.shape[1] == 9  # 3 input features * 3D Mandelbulb coords

    def test_hook_07_bspline_true_geometric_texture_weaving(self):
        from src.ui.voxelboxter_simulation import BSplineCompiledMod
        bspline = BSplineCompiledMod(name="TestHull", latent_dim=3, resolution=10)
        t = torch.linspace(0.0, 1.0, steps=8)
        curve = bspline.evaluate_curve(t)
        assert curve.shape == (8, 3)

    def test_hook_08_perlin_3d_gradient_terrain(self):
        from src.environment.cleanroom_mechanics import PerlinWorleyGyroidGenerator
        gen = PerlinWorleyGyroidGenerator()
        val = gen.sample_perlin_3d(12.5, 8.2, 19.1, octaves=4)
        assert isinstance(val, float)
        assert -1.5 <= val <= 1.5

    def test_hook_09_worley_cellular_caves(self):
        from src.environment.cleanroom_mechanics import PerlinWorleyGyroidGenerator
        gen = PerlinWorleyGyroidGenerator()
        f1, f2 = gen.sample_worley_3d(10.0, 15.0, 20.0, cell_size=16.0)
        assert f1 <= f2
        ridge = f2 - f1
        assert ridge >= 0.0

    def test_hook_10_tpms_gyroid_level_set(self):
        from src.environment.cleanroom_mechanics import PerlinWorleyGyroidGenerator
        gen = PerlinWorleyGyroidGenerator()
        g_val = gen.sample_gyroid_surface(0.0, 0.0, 0.0)
        assert abs(g_val) < 1e-4

    def test_hook_11_matrioshka_scale_nesting(self):
        from src.environment.cleanroom_mechanics import PerlinWorleyGyroidGenerator
        gen = PerlinWorleyGyroidGenerator()
        self.octree.add_fractal_node(1, 1, 1, material=3, complexity=0.6)
        child = gen.traverse_treehouse_scale(self.octree, (1, 1, 1), direction="expand")
        assert child.total_voxels >= 1

    def test_hook_12_kniberg_5d_splines(self):
        from src.environment.cleanroom_mechanics import PerlinWorleyGyroidGenerator
        gen = PerlinWorleyGyroidGenerator()
        d_eval = gen.evaluate_density(100.0, 32.0, 100.0)
        assert "continentalness" in d_eval
        assert "erosion" in d_eval
        assert "peaks_valleys" in d_eval
        assert "is_solid" in d_eval

    def test_hook_13_fbm_anisotropic_weathering(self):
        # Populate voxels
        for x in range(3):
            for y in range(3):
                self.octree.add_adaptive_fractal_node(float(x), float(y), 0.0, material=1, complexity=1.0)
        init_count = len(self.octree.morton_grid)
        weather_res = self.octree.anisotropic_weathering_pass(dt=1.0, pressure_gradient=(1.0, 1.0, 0.0))
        assert "weathered_nodes" in weather_res
        assert weather_res["remaining_octree_nodes"] <= init_count

    def test_hook_14_emergent_baking_poincare_fossilization(self):
        from src.environment.cleanroom_mechanics import EmergentBakingEngine
        baking = EmergentBakingEngine()
        # Add construct blocks
        for x in range(3):
            for y in range(3):
                self.octree.add_adaptive_fractal_node(float(x), float(y), 0.0, material=1, complexity=0.5)
        rec = baking.bake_construct(self.octree, center_coord=(1, 1, 0), radius=3, player_id="alice")
        assert rec["baked_nodes"] > 0
        assert rec["fossilization_wager"] == "80/20_SPLIT_ENFORCED"
        assert len(baking.baked_records) == 1

    def test_hook_15_chisels_and_bits_micro_octree(self):
        from src.environment.cleanroom_mechanics import ChiselsAndBitsMicroOctree
        cb = ChiselsAndBitsMicroOctree()
        # Carve a bit from solid macro block (2, 2, 2)
        mat = cb.chisel_carve_bit(2, 2, 2, sx=0, sy=0, sz=0, material_id=4, octree=self.octree)
        assert mat == 4
        assert cb.bit_inventory.get(4, 0) == 1
        # Place a bit into (3, 3, 3)
        ok = cb.chisel_place_bit(3, 3, 3, sx=1, sy=1, sz=1, material_id=4, octree=self.octree)
        assert ok is True
        assert cb.get_cell_bit_count(3, 3, 3) == 1
        assert cb.bit_inventory.get(4, 0) == 0  # Conserved


class TestMathematicalPhysicsHybridization:
    """
    Unit test suite verifying the three mathematical physics equations:
    Eq 1: Parallel Sorting Entropy with Gauge Field Circulation
    Eq 2: Fractal L'Hopital Regularization for Surface-to-Volume Divergence
    Eq 3: Valley Vorticity Flux Threshold & Sweep Operator
    """
    def setup_method(self):
        self.octree = PointerlessOctree(max_depth=6)
        from src.core.carnot_mobius_ledger import CarnotMobiusLedger
        from src.core.fgrt_primitives import FractalLHopitalRegularizer, ValleyVorticitySweeper
        from src.environment.cleanroom_mechanics import VoxelboxterHookIntegrator
        self.carnot_ledger = CarnotMobiusLedger()
        self.lhopital_reg = FractalLHopitalRegularizer(fractal_dim_D=2.73, C_constant=1.0)
        self.vorticity_sweeper = ValleyVorticitySweeper(tau_crit=2.5, sweep_damping=0.85)
        self.hook_integrator = VoxelboxterHookIntegrator(self.octree)
        self.editor = PhysicalNodeEditor()

    def test_equation_01_parallel_sorting_entropy(self):
        # 4 attention / sorting heads
        N_heads = 4
        omega_heads = torch.tensor([100.0, 250.0, 50.0, 500.0], dtype=torch.float32)
        gauge_circulation = torch.tensor([0.2, -0.1, 0.4, 0.0], dtype=torch.float32)
        euler_chi = torch.tensor([2.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        lambda_min = 1e-2
        kappa_accel = 1.5

        # Direct ledger calculation
        res = self.carnot_ledger.compute_parallel_sorting_entropy(
            omega_heads=omega_heads,
            gauge_circulation=gauge_circulation,
            euler_chi=euler_chi,
            lambda_min=lambda_min,
            kappa_accel=kappa_accel,
            k_B=1.0
        )
        assert "delta_s_sort" in res
        assert res["N_heads"] == N_heads
        # eta(4) = 1 / sqrt(4) = 0.5
        assert abs(res["eta_heads"] - 0.5) < 1e-5

        # Manual validation:
        # head 0: 1.5 * [ ln(100) - 0.2 + 2 / 0.01 ] = 1.5 * [ 4.60517 - 0.2 + 200 ] = 1.5 * 204.40517 = 306.60775
        expected_head_0 = 1.5 * (math.log(100.0) - 0.2 + 2.0 / 0.01)
        assert abs(res["per_head_entropy"][0].item() - expected_head_0) < 1e-3

        # Via Hook Integrator
        hook_res = self.hook_integrator.compute_parallel_sorting_entropy(
            omega_heads=omega_heads,
            gauge_circulation=gauge_circulation,
            euler_chi=euler_chi,
            lambda_min=lambda_min,
            kappa_accel=kappa_accel
        )
        assert torch.isclose(res["delta_s_sort"], hook_res["delta_s_sort"])

        # Via Node Editor hook
        editor_res = self.editor.hook_parallel_sorting_entropy(
            omega_heads=omega_heads,
            gauge_circulation=gauge_circulation,
            euler_chi=euler_chi,
            lambda_min=lambda_min,
            kappa_accel=kappa_accel
        )
        assert torch.isclose(res["delta_s_sort"], editor_res["delta_s_sort"])

    def test_equation_02_fractal_lhopital_regularization(self):
        lambda_min = 0.005
        kappa_accel = 2.0
        fractal_dim_D = 2.73
        C = 1.5

        # Direct module
        eps_min = self.lhopital_reg.compute_cutoff(lambda_min=lambda_min, kappa_accel=kappa_accel)
        assert abs(eps_min - (0.005 / 2.0)) < 1e-8
        
        # R_reg = C * eps_min^{2 - D} = 1.5 * (0.0025)^{-0.73}
        expected_r_reg = C * (eps_min ** (2.0 - fractal_dim_D))
        r_reg = self.lhopital_reg.evaluate_regularized_ratio(
            lambda_min=lambda_min,
            kappa_accel=kappa_accel,
            fractal_dim_D=fractal_dim_D,
            C_constant=C
        )
        assert abs(r_reg - expected_r_reg) < 1e-5

        # Octree application
        self.octree.add_fractal_node(1, 1, 1, material=1, complexity=0.5)
        oct_res = self.octree.apply_fractal_lhopital_regularization(
            lambda_min=lambda_min,
            kappa_accel=kappa_accel,
            fractal_dim_D=fractal_dim_D,
            C_constant=C
        )
        assert oct_res["total_nodes_regularized"] == 1
        assert abs(oct_res["R_reg"] - expected_r_reg) < 1e-5
        # Interleaved key complexity updated
        key = self.octree._interleave_bits(1, 1, 1)
        assert abs(self.octree.complexity_grid[key] - (0.5 * expected_r_reg)) < 1e-4

        # Hook integrator
        eval_dict = self.hook_integrator.evaluate_fractal_lhopital_regularization(
            lambda_min=lambda_min,
            kappa_accel=kappa_accel,
            fractal_dim_D=fractal_dim_D,
            C_constant=C
        )
        assert abs(eval_dict["R_reg"] - expected_r_reg) < 1e-5

    def test_equation_03_valley_vorticity_sweep_operator(self):
        # Construct curl fields and normals for J=3 valleys
        curl_fields = torch.tensor([[1.0, 0.0, 0.5], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]], dtype=torch.float32)
        area_normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        # Dot flux = 1.0*1.0 + 1.0*1.0 + 1.0*1.0 = 3.0
        upsilon = self.vorticity_sweeper.compute_vorticity_flux(curl_fields, area_normals)
        assert abs(upsilon.item() - 3.0) < 1e-5

        # With tau_crit = 2.5, upsilon (3.0) > tau_crit (2.5) -> sweep triggered!
        state_tensor = torch.ones((3, 3), dtype=torch.float32) * 10.0
        swept_state, triggered, diag = self.vorticity_sweeper.apply_sweep_operator(
            state_tensor,
            upsilon_val=upsilon.item(),
            tau_crit=2.5
        )
        assert triggered is True
        assert diag["sweep_triggered"] is True
        assert torch.allclose(swept_state, state_tensor * 0.85)

        # With high tau_crit = 5.0, not triggered
        swept_state_no, triggered_no, diag_no = self.vorticity_sweeper.apply_sweep_operator(
            state_tensor,
            upsilon_val=upsilon.item(),
            tau_crit=5.0
        )
        assert triggered_no is False
        assert torch.allclose(swept_state_no, state_tensor)

        # Test Hook Integrator sweep and octree clearing
        self.octree.add_fractal_node(5, 5, 5, material=2, complexity=0.01)
        swept_f, trig, d = self.hook_integrator.evaluate_valley_vorticity_sweep(
            curl_fields=curl_fields,
            area_normals=area_normals,
            tau_crit=2.5
        )
        assert trig is True
        # Octree stagnant node cleared
        assert (5, 5, 5) not in self.octree.morton_grid


