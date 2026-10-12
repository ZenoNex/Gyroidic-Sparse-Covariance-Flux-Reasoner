"""
Unit tests for the unification of Recursive Fractals & L'Hôpital's Rule with Voxelboxter:
- PointerlessOctree NonUniformGrid adaptive resolution scaling log(c+1)/c
- Dynamic depth compression via L'Hôpital limit
- RigidBody Gyroid zero-level set snapping force field F_snap(x)
- Non-Ergodic Kelly Defense hooked with snapping volatility
- FractalDiscontinuityEmbedding (phi: F -> R^m)
- BonfireNomadicRing P2P mesh state projection & topological sightline computation
- JourneyTopoRadar topological radar sweep across recursion depths
"""

import unittest
import math
import torch
import numpy as np

from src.ui.voxelboxter_simulation import PointerlessOctree, RigidBody, VoxelboxterEngine
from src.p2p.bonfire_consensus import FractalDiscontinuityEmbedding, BonfireNomadicRing
from src.p2p.freenet_ws_client import FreenetClient
from src.environment.cleanroom_mechanics import JourneyTopoRadar
from src.topology.gyroid_differentiation import GyroidFlowConstraint


class TestFractalLHopitalVoxelboxter(unittest.TestCase):

    def test_pointerless_octree_lhopital_scaling(self):
        """
        Verify that PointerlessOctree computes the L'Hôpital limit:
        lim_{c -> 0} log(c + 1) / c = 1.0,
        and compresses depth dynamically for high fractal complexity.
        """
        octree = PointerlessOctree(max_depth=8)
        
        # Limit at zero complexity should be 1.0
        scale_zero = octree.compute_local_scale(0.0)
        self.assertAlmostEqual(scale_zero, 1.0, places=5)
        
        # Limit at epsilon complexity should approach 1.0
        scale_eps = octree.compute_local_scale(1e-7)
        self.assertAlmostEqual(scale_eps, 1.0, places=5)
        
        # High complexity compresses scale factor: log(10+1)/10 ~= 2.3979 / 10 = 0.2398
        scale_high = octree.compute_local_scale(10.0)
        self.assertLess(scale_high, 0.3)
        self.assertGreater(scale_high, 0.2)
        
        # Fractional scaling order alpha
        frac_scale = octree.compute_fractional_scale(10.0, alpha=0.5)
        self.assertGreater(frac_scale, scale_high) # sqrt(scale) > scale for scale < 1
        
        # Dynamic depth adaptation: high complexity reduces depth to prevent compute explosion
        depth_zero = octree.get_adaptive_depth(0.0)
        depth_high = octree.get_adaptive_depth(10.0)
        self.assertEqual(depth_zero, 8)
        self.assertLess(depth_high, depth_zero)
        self.assertGreaterEqual(depth_high, 1)

    def test_pointerless_octree_adaptive_node_insertion(self):
        """
        Verify adaptive fractal node insertion into the Morton grid
        correctly stores dynamic depth and local complexity.
        """
        octree = PointerlessOctree(max_depth=8)
        
        # Add node with high complexity
        code1 = octree.add_adaptive_fractal_node(10.0, 20.0, 30.0, material=3, complexity=5.0)
        self.assertIn(code1, octree.morton_grid)
        self.assertEqual(octree.morton_grid[code1], 3)
        self.assertEqual(octree.complexity_grid[code1], 5.0)
        self.assertLess(octree.depth_grid[code1], 8)
        
        # Add node with zero complexity
        code2 = octree.add_adaptive_fractal_node(50.0, 50.0, 50.0, material=7, complexity=0.0)
        self.assertIn(code2, octree.morton_grid)
        self.assertEqual(octree.depth_grid[code2], 8)

    def test_rigidbody_gyroid_snapping_force_field(self):
        """
        Verify that RigidBody calculates restoring snapping forces
        F_snap(x) toward the Gyroid zero-level set G(x, y, z) = 0.
        """
        rb = RigidBody(mass=2.0, position=(1.0, 1.0, 1.0))
        
        # Evaluate snap force
        snap_f, g_viol = rb.compute_gyroid_snap_force()
        self.assertGreater(g_viol, 0.0)
        f_mag = math.sqrt(snap_f[0]**2 + snap_f[1]**2 + snap_f[2]**2)
        self.assertGreater(f_mag, 0.0)
        
        # Update Kelly growth with snapping constraint active
        initial_kelly = rb.kelly_growth_rate
        new_kelly = rb.update_kelly_growth(dt=0.05, apply_snap_constraint=True)
        
        # Verify that velocity was modified to steer toward Gyroid zero-level set
        v_mag = math.sqrt(rb.linear_velocity[0]**2 + rb.linear_velocity[1]**2 + rb.linear_velocity[2]**2)
        self.assertGreater(v_mag, 0.0)
        self.assertIsNotNone(rb.kelly_growth_rate)

    def test_fractal_discontinuity_embedding_and_sightlines(self):
        """
        Verify the embedding theorem phi: F -> R^m:
        m >= 2 * dim_H(F) + sum omega_j
        and that players situated at different recursive depths (e.g. depth 0 vs depth 2)
        can calculate consistent topological sightlines.
        """
        embedder = FractalDiscontinuityEmbedding(base_hausdorff_dim=2.7298, default_discontinuities=[0.25, 0.5])
        
        # Verify continuous embedding dimension m >= 6
        dim_m = embedder.compute_embedding_dimension(complexity=0.5)
        self.assertGreaterEqual(dim_m, 6)
        
        # Player A at macroscopic depth 0
        state_a = embedder.embed_state(position=(10.0, 20.0, 0.0), recursion_depth=0, complexity=0.1)
        # Player B at microscopic depth 2
        state_b = embedder.embed_state(position=(10.0, 20.0, 0.0), recursion_depth=2, complexity=0.8)
        
        # All peers must share the same ambient R^m for aligned sightline coordinates
        ambient_m = embedder.compute_ambient_dimension()
        self.assertGreaterEqual(ambient_m, dim_m)
        self.assertEqual(len(state_a["embedding"]), ambient_m)
        self.assertEqual(len(state_b["embedding"]), ambient_m)
        self.assertLessEqual(state_a["local_dimension"], ambient_m)
        self.assertLessEqual(state_b["local_dimension"], ambient_m)
        
        # Compute topological sightline
        sightline = embedder.compute_topological_sightline(state_a, state_b, horizon_radius=50.0)
        self.assertTrue(sightline["has_sightline"])
        self.assertGreater(sightline["transmission"], 0.0)
        self.assertEqual(sightline["scale_delta"], 2)
        self.assertAlmostEqual(sightline["complexity_delta"], 0.7, places=4)

        # Verify dynamic PrimeResonanceLadder scaling without hardcoded primes
        self.assertGreaterEqual(len(embedder.primes), 16)
        # Higher dimensional embedder with large discontinuity sum
        high_discont = [1.0] * 20
        high_embedder = FractalDiscontinuityEmbedding(default_discontinuities=high_discont)
        high_m = high_embedder.compute_ambient_dimension()
        high_state = high_embedder.embed_state((1.0, 2.0, 3.0), recursion_depth=1, complexity=0.2)
        self.assertEqual(len(high_state["embedding"]), high_m)
        self.assertGreater(high_m, 20)

    def test_bonfire_nomadic_ring_mesh_sightline_integration(self):
        """
        Verify BonfireNomadicRing broadcasts embedded states and resolves
        cross-scale mutual visibility between peers in the P2P mesh.
        """
        # Mock client to avoid network websockets
        class MockFreenet:
            def __init__(self):
                self.published = {}
                self.subscriptions = {}
            def subscribe(self, topic, cb):
                self.subscriptions[topic] = cb
            def publish(self, topic, data):
                self.published[topic] = data
                if topic in self.subscriptions:
                    self.subscriptions[topic](data)
                    
        freenet = MockFreenet()
        ring = BonfireNomadicRing(freenet_client=freenet)
        
        # Peer 1 broadcasts macroscopic state
        ring.broadcast_peer_embedded_state(
            local_peer_id="PEER_ALPHA",
            position=(5.0, 10.0, 15.0),
            recursion_depth=0,
            complexity=0.05
        )
        
        # Peer 2 broadcasts microscopic nested state
        ring.broadcast_peer_embedded_state(
            local_peer_id="PEER_BETA",
            position=(5.0, 10.0, 15.0),
            recursion_depth=3,
            complexity=1.2
        )
        
        # Verify both peers exist in embedded cache
        self.assertIn("PEER_ALPHA", ring.peer_embedded_states)
        self.assertIn("PEER_BETA", ring.peer_embedded_states)
        
        # Compute cross-scale sightline
        result = ring.compute_peer_sightline("PEER_ALPHA", "PEER_BETA", horizon_radius=50.0)
        self.assertTrue(result["has_sightline"])
        self.assertGreater(result["transmission"], 0.0)
        self.assertEqual(result["scale_delta"], 3)

    def test_journey_topo_radar_topological_sweep(self):
        """
        Verify JourneyTopoRadar evaluates entities across fractal recursion depths
        using the R^m continuous embedding and L'Hôpital scale transmission.
        """
        radar = JourneyTopoRadar(radar_radius=64)
        
        entities = [
            {"id": "near_entity", "pos": (2.0, 64.0, 3.0), "recursion_depth": 0, "complexity": 0.1},
            {"id": "deep_micro_entity", "pos": (2.0, 64.0, 3.0), "recursion_depth": 4, "complexity": 2.5},
            {"id": "far_entity", "pos": (500.0, 64.0, 500.0), "recursion_depth": 0, "complexity": 0.0}
        ]
        
        blips = radar.tick_radar_topological(
            entities=entities,
            player_pos=(0.0, 64.0, 0.0),
            player_depth=0,
            player_complexity=0.0,
            radar_radius=64
        )
        
        blip_ids = [b["id"] for b in blips]
        self.assertIn("near_entity", blip_ids)
        self.assertIn("deep_micro_entity", blip_ids) # Detected via R^m embedding despite scale jump
        self.assertNotIn("far_entity", blip_ids) # Exceeds horizon

    def test_gyroid_flow_constraint_lhopital_regularization(self):
        """
        Verify GyroidFlowConstraint correctly uses L'Hôpital regularized gradient
        when embedding does not have requires_grad.
        """
        constraint = GyroidFlowConstraint()
        residue = torch.zeros(1, 3)
        embedding = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=False)
        
        res = constraint(residue, embedding)
        self.assertIn("grad_flow", res)
        self.assertEqual(res["grad_flow"].shape, (1, 3))
        # Ensure no NaN or inf values
        self.assertFalse(torch.isnan(res["grad_flow"]).any())
        self.assertFalse(torch.isinf(res["grad_flow"]).any())


if __name__ == '__main__':
    unittest.main()
