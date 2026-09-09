import torch
import logging
from typing import Dict
from .freenet_ws_client import FreenetClient

logger = logging.getLogger(__name__)

class BonfireNomadicRing:
    """
    Implements the Bonfire Nomadic Rings: Federated Consensus & Egalitarian Microhedging.
    Wraps the FreenetClient to broadcast topological signatures.
    """
    def __init__(self, freenet_client: FreenetClient, contract_id: str = "bonfire_nomadic_ring"):
        self.freenet = freenet_client
        self.contract_id = contract_id
        
        # Kelly Consensus state
        self.peer_allocations: Dict[str, float] = {}
        
        # Bind the Freenet subscription
        self.freenet.subscribe(self.contract_id, self._handle_network_update)

    def _handle_network_update(self, state_update: Dict):
        """Callback for incoming state updates from the Freenet contract."""
        peer_id = state_update.get("peer_id", "unknown")
        k_frac = state_update.get("kelly_fraction", 0.0)
        
        if peer_id != "unknown":
            self.peer_allocations[peer_id] = k_frac
            logger.debug(f"[BONFIRE] Received Kelly fraction {k_frac} from {peer_id}")

    def compute_egalitarian_consensus(self, engine_meta_state: torch.Tensor = None) -> float:
        """
        Calculates the Egalitarian Consensus Kelly Allocation (K_bar)
        and optionally adjusts local structural resonance (meta_state).
        """
        if not self.peer_allocations:
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

    def share_topological_signature(self, local_peer_id: str, betti_numbers: list, variance: float):
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
