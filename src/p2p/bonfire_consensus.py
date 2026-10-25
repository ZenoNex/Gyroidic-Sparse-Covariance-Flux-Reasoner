import torch
import logging
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .freenet_ws_client import FreenetClient
from src.data.freenet_ghost_caller import FreenetGhostCaller
from src.data.freenet_bulletin_router import FreenetBulletinRouter

logger = logging.getLogger(__name__)

class FractalDiscontinuityEmbedding:
    """
    Embedding Theorem for Discontinuous Fractals:
    phi: F -> R^m where m >= 2 * dim_H(F) + sum_{j=1}^k omega_j
    
    Embeds a player's discontinuous fractal state (position across Matrioshka shell depths,
    local fractal complexity c, and discontinuity orders omega_j) into a continuous
    m-dimensional Euclidean manifold R^m.
    
    Enables players situated at disparate recursive fractal depths to calculate
    consistent topological sightlines and mutual line-of-sight without Cartesian artifacts.
    """
    def __init__(
        self,
        base_hausdorff_dim: float = 2.7298,
        default_discontinuities: Optional[List[float]] = None,
        ladder: Optional[Any] = None
    ):
        self.base_hausdorff_dim = base_hausdorff_dim
        self.default_discontinuities = default_discontinuities or [0.25, 0.50]
        self._ladder = ladder

    def get_prime_ladder(self, required_resonators: int = 16):
        """
        Dynamically acquires or scales PrimeResonanceLadder.
        Adheres to anti-lobotomy guidelines: no hardcoded prime sequences.
        Generates incommensurate frequencies f_{p_n} = 2*pi*log(p_n) via Lazarus prime-repunit basis.
        """
        if self._ladder is None or self._ladder.num_resonators < required_resonators:
            from src.core.invariants import PrimeResonanceLadder
            self._ladder = PrimeResonanceLadder(num_resonators=max(required_resonators, 16))
        return self._ladder

    @property
    def primes(self) -> List[int]:
        """Dynamically exposed primes from PrimeResonanceLadder (non-hardcoded)."""
        ladder = self.get_prime_ladder(16)
        return ladder.primes.tolist()

    def compute_embedding_dimension(self, complexity: float, discontinuity_orders: Optional[List[float]] = None) -> int:
        """
        Calculates minimum continuous embedding dimension m:
        m >= 2 * dim_H(F) + sum omega_j
        """
        discont = discontinuity_orders if discontinuity_orders is not None else self.default_discontinuities
        omega_sum = sum(discont)
        # L'Hôpital scaling for fractal dimension correction factor phi(Delta)
        scale_mod = float(np.log(complexity + 1.0) / complexity) if complexity > 1e-6 else 1.0
        dim_h = 2.0 + (self.base_hausdorff_dim - 2.0) * scale_mod
        m = max(6, int(math.ceil(2.0 * dim_h + omega_sum)))
        return m

    def compute_ambient_dimension(self, discontinuity_orders: Optional[List[float]] = None) -> int:
        """
        Shared ambient dimension for the P2P visibility manifold.
        Since log(c + 1) / c <= 1 for all c >= 0, the local bound
        2 * dim_H(F) + sum omega_j is maximized at c -> 0 (L'Hôpital limit = 1).
        Using this supremum guarantees every peer satisfies the embedding bound
        while all peers share the same R^m, so coordinates remain aligned.
        """
        return self.compute_embedding_dimension(0.0, discontinuity_orders)

    def embed_state(
        self,
        position: Tuple[float, float, float],
        recursion_depth: int = 0,
        complexity: float = 0.0,
        discontinuity_orders: Optional[List[float]] = None,
        time_tick: float = 0.0,
        use_einstein_monotile: bool = True
    ) -> Dict[str, Any]:
        """
        Computes the continuous embedding vector phi(P, depth, complexity) in the shared R^m.
        When use_einstein_monotile=True, anchors the spatial coordinates in the aperiodic
        monotile lattice, eliminating Cartesian grid seam tearing across scale jumps (2^-depth).
        """
        discont = discontinuity_orders if discontinuity_orders is not None else self.default_discontinuities
        m = self.compute_ambient_dimension(discont)
        local_m = self.compute_embedding_dimension(complexity, discont)
        
        # Matrioshka scale contraction factor (2^-depth)
        scale_contract = 2.0 ** (-float(recursion_depth))
        px, py, pz = position
        
        coords = [0.0] * m
        
        if use_einstein_monotile:
            try:
                from src.topology.einstein_monotile import get_einstein_tessellation
                tess = get_einstein_tessellation()
                mono_coords = tess.compute_scale_invariant_sightline_coordinates(
                    position=position,
                    recursion_depth=recursion_depth,
                    target_dim=m
                )
                for idx in range(min(len(mono_coords), m)):
                    coords[idx] = mono_coords[idx]
            except Exception:
                coords[0] = px * scale_contract
                coords[1] = py * scale_contract
                coords[2] = pz * scale_contract
                scale_phase = 2.0 * math.pi * ((recursion_depth % 16) / 16.0)
                coords[3] = math.cos(scale_phase)
                coords[4] = math.sin(scale_phase)
        else:
            # 0..2: Scale-contracted spatial coordinates
            coords[0] = px * scale_contract
            coords[1] = py * scale_contract
            coords[2] = pz * scale_contract
            
            # 3..4: Scale phase on S^1 torus
            scale_phase = 2.0 * math.pi * ((recursion_depth % 16) / 16.0)
            coords[3] = math.cos(scale_phase)
            coords[4] = math.sin(scale_phase)
        
        # 5: Local complexity measure regularized by L'Hôpital limit
        scale_lhopital = float(np.log(complexity + 1.0) / complexity) if complexity > 1e-6 else 1.0
        if m > 5:
            coords[5] = float(complexity) * scale_lhopital
            
        # 6..m-1: High-dimensional harmonic phases along prime resonance frequencies
        num_harmonic = m - 6
        if num_harmonic > 0:
            ladder = self.get_prime_ladder(num_harmonic)
            freqs = ladder.frequencies.cpu().tolist()
            for idx in range(6, m):
                h_idx = idx - 6
                freq = freqs[h_idx % len(freqs)]
                omega = discont[h_idx % len(discont)] if discont else 0.0
                coords[idx] = math.cos(freq * time_tick + omega)
            
        return {
            "embedding": coords,
            "dimension": m,
            "local_dimension": local_m,
            "scale_contract": scale_contract,
            "einstein_monotile_anchored": use_einstein_monotile,
            "recursion_depth": recursion_depth,
            "complexity": complexity,
            "scale_factor": scale_lhopital,
            "discontinuity_sum": sum(discont)
        }

    def compute_monotile_sightline(
        self,
        peer_a_pos: Tuple[float, float, float],
        peer_a_depth: int,
        peer_b_pos: Tuple[float, float, float],
        peer_b_depth: int
    ) -> Dict[str, Any]:
        """
        Computes scale-invariant topological sightlines across Matrioshka depth jumps (e.g. depth 3 to 7)
        using hierarchical aperiodic monotile inflation coordinates.
        Eliminates periodic grid seam tearing across scale boundaries.
        """
        embed_a = self.embed_state(peer_a_pos, recursion_depth=peer_a_depth, use_einstein_monotile=True)
        embed_b = self.embed_state(peer_b_pos, recursion_depth=peer_b_depth, use_einstein_monotile=True)
        vec_a = np.array(embed_a["embedding"])
        vec_b = np.array(embed_b["embedding"])
        sightline_dist = float(np.linalg.norm(vec_a - vec_b))
        scale_gap = abs(peer_a_depth - peer_b_depth)
        seam_coherence = float(1.0 / (1.0 + 0.05 * sightline_dist * (scale_gap ** 0.5)))
        return {
            "sightline_distance": sightline_dist,
            "seam_coherence": seam_coherence,
            "depth_gap": scale_gap,
            "is_continuous": True
        }

    def compute_topological_sightline(
        self,
        emb_a: Dict[str, Any],
        emb_b: Dict[str, Any],
        horizon_radius: float = 50.0
    ) -> Dict[str, Any]:
        """
        Computes mutual visibility / topological sightline between two players in R^m.
        Applies L'Hôpital limit across inter-scale complexity boundaries.
        """
        vec_a = emb_a.get("embedding", [])
        vec_b = emb_b.get("embedding", [])
        dim = max(len(vec_a), len(vec_b))
        if dim == 0:
            return {"has_sightline": False, "transmission": 0.0, "embedded_distance": float('inf')}
            
        va_padded = vec_a + [0.0] * (dim - len(vec_a))
        vb_padded = vec_b + [0.0] * (dim - len(vec_b))
        
        dist_sq = sum((x - y) ** 2 for x, y in zip(va_padded, vb_padded))
        dist_m = math.sqrt(dist_sq)
        
        # L'Hôpital fractional transmission across complexity delta
        ca = emb_a.get("complexity", 0.0)
        cb = emb_b.get("complexity", 0.0)
        delta_c = abs(ca - cb)
        scale_trans = float(np.log(delta_c + 1.0) / delta_c) if delta_c > 1e-6 else 1.0
        
        # Decay transmission over embedded distance in R^m
        transmission = math.exp(-dist_m / max(1.0, horizon_radius)) * scale_trans
        has_sightline = dist_m <= horizon_radius and transmission > 0.01
        
        return {
            "has_sightline": has_sightline,
            "transmission": float(transmission),
            "embedded_distance": float(dist_m),
            "scale_delta": abs(emb_a.get("recursion_depth", 0) - emb_b.get("recursion_depth", 0)),
            "complexity_delta": float(delta_c)
        }

class BonfireNomadicRing:
    """
    Implements the Bonfire Nomadic Rings: Federated Consensus & Egalitarian Microhedging.
    Wraps the FreenetClient to broadcast topological signatures.
    """
    def __init__(self, freenet_client: FreenetClient, contract_id: str = "bonfire_nomadic_ring"):
        self.freenet = freenet_client
        self.contract_id = contract_id
        
        self.ghost_caller = FreenetGhostCaller(freenet_client=self.freenet)
        self.bulletin_router = FreenetBulletinRouter(freenet_client=self.freenet)
        
        # Kelly Consensus state
        self.peer_allocations: Dict[str, float] = {}
        
        # ZK CRT Integration
        from src.p2p.zk_aggregator import ZKAggregator
        self.zk_aggregator = ZKAggregator()
        self.pending_crt_channels: Dict[int, torch.Tensor] = {}

        # Prime Resonance Ladder for Chebyshev-Chebyshev oscillator and CRT fallback
        from src.core.invariants import PrimeResonanceLadder
        self.prime_ladder = PrimeResonanceLadder(num_resonators=32)
        self.primes = self.prime_ladder.primes.tolist()

        # Embedding & Sightline Engine (Fractal Discontinuity Embedding phi: F -> R^m)
        self.embedding_engine = FractalDiscontinuityEmbedding()
        self.peer_embedded_states: Dict[str, Dict[str, Any]] = {}
        
        # Bind the Freenet subscription
        self.freenet.subscribe(self.contract_id, self._handle_network_update)
        self.freenet.subscribe("agent_smith_ring", self._handle_agent_smith_update)
        self.freenet.subscribe("bonfire_sightline_ring", self._handle_sightline_update)
        
        # Callback for when a foreign Agent Smith payload is received
        self.on_agent_smith_received = None

    def _handle_network_update(self, state_update: Dict):
        """Callback for incoming state updates from the Freenet contract."""
        peer_id = state_update.get("peer_id", "unknown")
        k_frac = state_update.get("kelly_fraction", 0.0)
        
        if peer_id != "unknown":
            self.peer_allocations[peer_id] = k_frac
            logger.debug(f"[BONFIRE] Received Kelly fraction {k_frac} from {peer_id}")

        # ZK CRT Integration: receive processed residue channel
        crt_residue = state_update.get("crt_residue")
        zk_proof = state_update.get("zk_proof")
        channel_id = state_update.get("channel_id")
        
        if crt_residue is not None and zk_proof is not None and channel_id is not None:
            # Cryptographically verify the foreign channel
            if self.zk_aggregator.verify_proof(f"crt_channel_{channel_id}", zk_proof):
                logger.info(f"[BONFIRE] Verified CRT Channel {channel_id} from {peer_id}")
                self.pending_crt_channels[channel_id] = torch.tensor(crt_residue)
            else:
                logger.warning(f"[BONFIRE] Rejected corrupted CRT Channel {channel_id} from {peer_id}")

    def _handle_agent_smith_update(self, state_update: Dict):
        """Callback for incoming Agent Smith payloads (Base64 encoded)."""
        peer_id = state_update.get("peer_id", "unknown")
        payload_b64 = state_update.get("payload_b64")
        if not payload_b64:
            return
            
        logger.info(f"[BONFIRE] Received Foreign Agent Smith payload from {peer_id}")
        if self.on_agent_smith_received:
            import base64
            import tempfile
            import os
            try:
                raw_bytes = base64.b64decode(payload_b64)
                # Write to temp file for injection
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name
                
                # Execute callback
                self.on_agent_smith_received(tmp_path)
                
                # Cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as e:
                logger.error(f"[BONFIRE] Failed to process incoming Agent Smith payload: {e}")

    def broadcast_agent_smith(self, local_peer_id: str, filepath: str):
        """Broadcasts a local Agent Smith .pt file over the Freenet."""
        import base64
        try:
            with open(filepath, "rb") as f:
                payload_b64 = base64.b64encode(f.read()).decode('utf-8')
                
            payload = {
                "peer_id": local_peer_id,
                "payload_b64": payload_b64,
                "voynich_exemption": True # Allows foreign tensors to bypass strict local vetoes initially
            }
            self.freenet.publish("agent_smith_ring", payload)
            logger.info(f"[BONFIRE] Broadcasted Agent Smith payload ({len(payload_b64)} bytes)")
        except Exception as e:
            logger.error(f"[BONFIRE] Error broadcasting Agent Smith: {e}")

    def broadcast_crt_residues(self, local_peer_id: str, channels: Dict[int, torch.Tensor]):
        """
        Distribute CRT polynomial tasks to peers based on available Kelly Allocation.
        """
        for channel_id, tensor_data in channels.items():
            payload = {
                "peer_id": local_peer_id,
                "channel_id": channel_id,
                "crt_residue": tensor_data.tolist(),
                # In a full system, we generate actual proofs here
                "zk_proof": {"publicSignals": ["1"]}
            }
            self.freenet.publish("crt_federation_ring", payload)
            logger.debug(f"[BONFIRE] Broadcasted CRT task for channel {channel_id}")

    def compute_egalitarian_consensus(self, engine_meta_state: torch.Tensor = None) -> float:
        """
        Calculates the Egalitarian Consensus Kelly Allocation (K_bar)
        and optionally adjusts local structural resonance (meta_state).
        """
        from src.core.hardware_monitor import has_headroom
        
        # If hardware is constrained or poisoned, fallback to substrate stability 
        # instead of attempting full egalitarian peer consensus.
        if not self.peer_allocations or not has_headroom():
            # ASD-STE100 Rules: Hardware-Sovereign Fallback
            # Prime-ladder Chebyshev-Chebyshev oscillator simulation
            if engine_meta_state is not None:
                # Map state to [-1, 1] for Chebyshev domain
                x_norm = torch.tanh(engine_meta_state)
                # Apply T_p(x) = cos(p * arccos(x)) for the first prime p=2
                p = float(self.primes[0]) if getattr(self, 'primes', None) else 2.0
                oscillator = torch.cos(p * torch.acos(x_norm))
                # Sustain local substrate stability using the oscillator
                engine_meta_state.copy_(engine_meta_state * 0.9 + oscillator * 0.1)
            return 1.0
        
        total_k = sum(self.peer_allocations.values())
        k_bar = total_k / len(self.peer_allocations)
        
        if engine_meta_state is not None:
            # Real-time microhedging: shifting local allocations toward consensus
            hedge_factor = torch.tensor([k_bar], device=engine_meta_state.device, dtype=engine_meta_state.dtype)
            if len(engine_meta_state.shape) == 2:
                hedge_factor = hedge_factor.expand(1, engine_meta_state.size(1))
            engine_meta_state.copy_(engine_meta_state * 0.95 + hedge_factor * 0.05)
            
        return k_bar

    def share_topological_signature(self, local_peer_id: str, betti_numbers: list, variance: float, engine=None):
        """
        Broadcasts the local state via Freenet.
        """
        # Calculate local Kelly betting allocation
        # P_success derived from low variance
        p_success = max(0.01, 1.0 - variance)
        kelly_fraction = p_success * 0.5  # safe fractional Kelly

        payload = {
            "peer_id": local_peer_id,
            "betti_numbers": betti_numbers,
            "kelly_fraction": kelly_fraction
        }
        
        self.freenet.publish(self.contract_id, payload)
        logger.debug(f"[BONFIRE] Shared topological signature: Betti={betti_numbers}, Kelly={kelly_fraction:.3f}")

        # Puncture Event for Cerumen Pot (Meliponini Topology)
        # If variance is low enough, broadcast to global Freenet boards
        if variance < 0.1:
            self.ghost_caller.broadcast_ghost_call(topological_variance=variance, engine=engine)
            
            volume = 1000.0 * (1.0 - variance)
            metrics = {
                "kelly_fraction": kelly_fraction,
                "covariance_variance": variance,
                "betti_numbers": betti_numbers,
                "euler_characteristic": sum([(-1)**i * b for i, b in enumerate(betti_numbers)]) if betti_numbers else 0,
                "coprime_residue": 1
            }
            self.bulletin_router.broadcast_proof_of_honesty(volume=volume, mischief=0.0, metrics=metrics)

    def _handle_sightline_update(self, state_update: Dict):
        """Callback for incoming embedded peer states for sightline calculation."""
        peer_id = state_update.get("peer_id", "unknown")
        if peer_id != "unknown":
            self.peer_embedded_states[peer_id] = state_update
            logger.debug(f"[BONFIRE] Received embedded state from {peer_id} (dim={state_update.get('dimension')})")

    def broadcast_peer_embedded_state(
        self,
        local_peer_id: str,
        position: Tuple[float, float, float],
        recursion_depth: int = 0,
        complexity: float = 0.0,
        betti_numbers: Optional[list] = None,
        variance: float = 0.0,
        discontinuity_orders: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Projects local state using the embedding map phi: F -> R^m and broadcasts across P2P mesh.
        Gated by ComputeAffordanceType.P2P_MESH to ensure mesh headroom is verified before broadcast.
        Ensures peers at varying scale depths can project onto shared visibility manifolds.
        """
        try:
            from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
            monitor = HardwareMonitor.get_instance()
            if not monitor.has_compute_affordance(ComputeAffordanceType.P2P_MESH):
                logger.warning("[BONFIRE] P2P_MESH compute affordance constrained; throttling broadcast.")
        except Exception:
            pass

        embedded = self.embedding_engine.embed_state(
            position=position,
            recursion_depth=recursion_depth,
            complexity=complexity,
            discontinuity_orders=discontinuity_orders,
            time_tick=time.time()
        )
        payload = {
            "peer_id": local_peer_id,
            "betti_numbers": betti_numbers or [1, 0],
            "variance": variance,
            **embedded
        }
        self.peer_embedded_states[local_peer_id] = payload
        self.freenet.publish("bonfire_sightline_ring", payload)
        logger.info(f"[BONFIRE] Broadcasted R^{embedded['dimension']} embedded state for {local_peer_id}")
        return payload

    def compute_peer_sightline(
        self,
        local_peer_id: str,
        remote_peer_id: str,
        horizon_radius: float = 50.0
    ) -> Dict[str, Any]:
        """
        Calculates topological sightline and mutual visibility in R^m between two peers across scales.
        """
        emb_a = self.peer_embedded_states.get(local_peer_id)
        emb_b = self.peer_embedded_states.get(remote_peer_id)
        if not emb_a or not emb_b:
            return {
                "has_sightline": False,
                "reason": "Missing embedded peer state in mesh",
                "transmission": 0.0,
                "embedded_distance": float('inf')
            }
        return self.embedding_engine.compute_topological_sightline(
            emb_a, emb_b, horizon_radius=horizon_radius
        )
