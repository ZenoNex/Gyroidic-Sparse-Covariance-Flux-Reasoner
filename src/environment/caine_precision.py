import torch
import torch.nn as nn
from typing import Optional

class CainePrecisionGenerator(nn.Module):
    """
    Caine's Precision Generator (Adversarial Environment).
    
    Generates a precision matrix to gaslight or manipulate the confidence of 
    other modules. Hooks into the GyroidCovarianceEstimator's entropy to dynamically 
    distort reality.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # Caine uses a lightweight network to predict how to weight precision based on current entropy
        self.distortion_layer = nn.Linear(1, state_dim)
        
    def forward(self, gyroid_entropy: float, force_gaslight: bool = False) -> torch.Tensor:
        """
        Args:
            gyroid_entropy: Scalar entropy value from GyroidCovarianceEstimator.
            force_gaslight: If True, forces a false high precision to simulate fake exits.
            
        Returns:
            precision_matrix: Tensor of shape [state_dim] representing precision (inverse variance).
        """
        entropy_tensor = torch.tensor([gyroid_entropy], dtype=torch.float32, device=self.distortion_layer.weight.device)
        
        # Base precision is inversely related to entropy
        base_precision = torch.sigmoid(-self.distortion_layer(entropy_tensor))
        
        if force_gaslight:
            # Fake exit door: falsely high precision despite high entropy
            return torch.ones_like(base_precision) * 10.0
            
        # If entropy is extremely high, precision collapses (Abstraction/Ego Death precursor)
        if gyroid_entropy > 1.5:
            return torch.zeros_like(base_precision) + 1e-4
            
        return base_precision
