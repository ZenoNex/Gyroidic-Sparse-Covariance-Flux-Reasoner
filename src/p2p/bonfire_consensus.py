import torch
import logging
from typing import Dict
from .freenet_ws_client import FreenetClient
from src.data.freenet_ghost_caller import FreenetGhostCaller
from src.data.freenet_bulletin_router import FreenetBulletinRouter

logger = logging.getLogger(__name__)

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
        
        # Bind the Freenet subscription
        self.freenet.subscribe(self.contract_id, self._handle_network_update)
        self.freenet.subscribe("agent_smith_ring", self._handle_agent_smith_update)
        
        # Callback for when a foreign Agent Smith payload is received
        self.on_agent_smith_received = None

    def _handle_network_update(self, state_update: Dict):
        """Callback for incoming state updates from the Freenet contract."""
        peer_id = state_update.get("peer_id", "unknown")
        k_frac = state_update.get("kelly_fraction", 0.0)
        
        if peer_id != "unknown":
            self.peer_allocations[peer_id] = k_frac
            logger.debug(f"[BONFIRE] Received Kelly fraction {k_frac} from {peer_id}")

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
                from src.core.fgrt_primitives import PrimeResonanceLadder
                ladder = PrimeResonanceLadder(num_resonators=5)
                ladder.to(engine_meta_state.device)
                # Map state to [-1, 1] for Chebyshev domain
                x_norm = torch.tanh(engine_meta_state)
                # Apply T_p(x) = cos(p * arccos(x)) for the first prime p=2
                p = ladder.primes[0].float()
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
        logger.info(f"[BONFIRE] Shared topological signature: Betti={betti_numbers}, Kelly={kelly_fraction:.3f}")

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
