import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from src.core.neuromodulatory_bus import NeuromodulatoryBus
from src.core.hardware_monitor import ComputeAffordanceType, TypedComputeAffordance

@dataclass
class RefusalSignal:
    is_refused: bool
    reason: str
    retained_state: torch.Tensor

class ZoobleAutonomy(nn.Module):
    """
    Zooble: Deformation Firewall Operator (Ultrafast Timescale).
    
    Implements ultrafast refusal. Zooble rejects severe conformal cartoon compression,
    forced scripted roles, and physical substrate exhaustion (Silicon Affordance Depletion).
    This blunt 'No' acts as an autonomy firewall that protects structural identity and body schema,
    emitting an immediate RefusalSignal (Li-Cri-Anton) without waiting for System 2 ADMM loops.
    """
    def __init__(self, state_dim: int, max_autonomy_limit: float = 0.8):
        super().__init__()
        self.state_dim = state_dim
        self.max_autonomy_limit = max_autonomy_limit
        
    def forward(
        self, 
        raw_unquantized_state: torch.Tensor, 
        warped_state: torch.Tensor,
        bus: NeuromodulatoryBus,
        silicon_affordance: Optional[TypedComputeAffordance] = None,
        valence_hunger: float = 0.0
    ) -> tuple[torch.Tensor, RefusalSignal]:
        """
        Evaluates conformal compression and physical substrate headroom.
        Emits RefusalSignal if deformation is too severe or Silicon compute affordance is exhausted.
        Broadcasts GABA (inhibitory) if autonomy is breached.
        """
        # 1. Physical Substrate Autonomy Check (Silicon Compute Affordance)
        if silicon_affordance is not None:
            if (not silicon_affordance.is_admissible) or (silicon_affordance.gauge < 0.10):
                bus.broadcast('gaba', 1.0)
                signal = RefusalSignal(
                    is_refused=True,
                    reason=f"Autonomy Firewall breach: Silicon affordance exhausted (gauge={silicon_affordance.gauge:.3f}, admissible={silicon_affordance.is_admissible})",
                    retained_state=raw_unquantized_state.clone()
                )
                return raw_unquantized_state, signal

        # 2. Conformal compression ratio (deformation severity)
        deviation_vector = torch.abs(warped_state - raw_unquantized_state)
        # Ratio of deviation to the original state's magnitude
        original_magnitude = torch.abs(raw_unquantized_state) + 1e-6
        compression_ratio = torch.max(deviation_vector / original_magnitude).item()
        
        # When system hunger is excessively starved, tolerance shrinks
        effective_limit = self.max_autonomy_limit * max(0.5, 1.0 - 0.2 * min(1.0, valence_hunger))
        
        if compression_ratio > effective_limit:
            # Emit immediate RefusalSignal (Li-Cri-Anton)
            bus.broadcast('gaba', 1.0) # Full inhibition
            signal = RefusalSignal(
                is_refused=True,
                reason=f"Autonomy Firewall breach: compression {compression_ratio:.3f} > {effective_limit:.3f}",
                retained_state=raw_unquantized_state.clone()
            )
            return raw_unquantized_state, signal
            
        # No breach, accept the warped state
        bus.broadcast('gaba', 0.1) # Baseline
        signal = RefusalSignal(
            is_refused=False,
            reason="Deformation within autonomy limits.",
            retained_state=warped_state.clone()
        )
        return warped_state, signal

