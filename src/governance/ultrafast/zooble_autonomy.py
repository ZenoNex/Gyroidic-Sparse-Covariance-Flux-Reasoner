import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class RefusalSignal:
    is_refused: bool
    reason: str
    retained_state: torch.Tensor

class ZoobleAutonomy(nn.Module):
    """
    Zooble: Deformation Firewall Operator (Ultrafast Timescale).
    
    Implements ultrafast refusal. Zooble rejects severe conformal cartoon compression 
    and forced scripted roles. This blunt 'No' acts as an autonomy firewall that 
    protects structural identity, emitting an immediate RefusalSignal (Li-Cri-Anton) 
    without waiting for System 2 ADMM loops.
    """
    def __init__(self, state_dim: int, max_autonomy_limit: float = 0.8):
        super().__init__()
        self.state_dim = state_dim
        self.max_autonomy_limit = max_autonomy_limit
        
    def forward(
        self, 
        raw_unquantized_state: torch.Tensor, 
        warped_state: torch.Tensor
    ) -> tuple[torch.Tensor, RefusalSignal]:
        """
        Evaluates conformal compression. Emits RefusalSignal if deformation is too severe.
        """
        # Calculate conformal compression ratio (deformation severity)
        deviation_vector = torch.abs(warped_state - raw_unquantized_state)
        # Ratio of deviation to the original state's magnitude
        original_magnitude = torch.abs(raw_unquantized_state) + 1e-6
        compression_ratio = torch.max(deviation_vector / original_magnitude).item()
        
        if compression_ratio > self.max_autonomy_limit:
            # Emit immediate RefusalSignal (Li-Cri-Anton)
            signal = RefusalSignal(
                is_refused=True,
                reason=f"Autonomy Firewall breach: compression {compression_ratio:.3f} > {self.max_autonomy_limit}",
                retained_state=raw_unquantized_state.clone()
            )
            return raw_unquantized_state, signal
            
        # No breach, accept the warped state
        signal = RefusalSignal(
            is_refused=False,
            reason="Deformation within autonomy limits.",
            retained_state=warped_state.clone()
        )
        return warped_state, signal
