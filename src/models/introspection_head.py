"""
Geometric self-modeling head for structural state extraction.

Probes hidden states to extract normalized directions representing
internal geometric configurations (e.g., structural tension, uncertainty).

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))



class GeometricSelfModelProbe(nn.Module):
    """
    Geometric self-modeling probe.
    
    Extracts unit vectors from hidden states and measures coherence
    under different conditions (trigger vs control).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_probe_dims: int = 64,
        probe_types: List[str] = ['moral', 'uncertainty', 'creative']
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states to probe
            num_probe_dims: Dimension of probe space
            probe_types: Types of introspective probes
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_probe_dims = num_probe_dims
        self.probe_types = probe_types
        
        # Per-probe-type projectors
        self.probes = nn.ModuleDict({
            probe_type: nn.Sequential(
                nn.Linear(hidden_dim, num_probe_dims),
                nn.Tanh(),  # Bounded activation for stability
                nn.Linear(num_probe_dims, num_probe_dims)
            )
            for probe_type in probe_types
        })
        
        # Adaptive violation fusion for metacognition
        if 'metacognitive' in probe_types:
            # Projects scalar violation score to probe dimension
            self.violation_adapter = nn.Linear(1, num_probe_dims)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        probe_type: str = 'moral',
        gcve_pressure: Optional[torch.Tensor] = None,
        suppress_narration: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract self-modeling direction from hidden states.
        
        Pointer #11: Over-Identification with Construction = Interference
        When suppress_narration=True, block any output that could be interpreted
        as "what the system is becoming" (teleological leak).
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            probe_type: Which probe to use
            gcve_pressure: Optional [batch] or [batch, 1] topological violation score
            suppress_narration: If True, return only present-moment geometry (no trajectory)
            
        Returns:
            Dictionary with:
                - 'direction': [batch, num_probe_dims] normalized direction
                - 'magnitude': [batch] magnitude (SUPPRESSED if suppress_narration)
                - 'is_geometric_only': bool flag
        """
        if probe_type not in self.probes:
            raise ValueError(f"Unknown probe type: {probe_type}")
        
        # Handle both 2D and 3D inputs
        if len(hidden_states.shape) == 3:
            # Pool over sequence
            hidden = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        else:
            hidden = hidden_states
        
        # Project to probe space
        probe_vec = self.probes[probe_type](hidden)  # [batch, num_probe_dims]
        
        # Inject Structural Discord from Topological Violations
        if probe_type == 'metacognitive' and gcve_pressure is not None and hasattr(self, 'violation_adapter'):
            if gcve_pressure.dim() == 1:
                gcve_pressure = gcve_pressure.unsqueeze(-1)
            
            # Add violation embedding to the probe vector
            # High violation -> shifts the metacognitive direction
            violation_emb = self.violation_adapter(gcve_pressure)
            probe_vec = probe_vec + violation_emb
        
        # Compute magnitude
        magnitude = torch.norm(probe_vec, dim=-1)  # [batch]
        
        # Normalize to unit vector
        direction = probe_vec / (magnitude.unsqueeze(-1) + 1e-8)
        
        if suppress_narration:
            # NARRATION SUPPRESSION (Pointer #11)
            # Return only geometric properties (present-moment orientation)
            # - Direction: OK (present-moment orientation)
            # - Magnitude: SUPPRESSED (could indicate trajectory information)
            # - No temporal derivatives, no trajectory predictions
            return {
                'direction': direction,
                'magnitude': torch.zeros_like(magnitude),  # SUPPRESSED
                'is_geometric_only': True
            }
        else:
            return {
                'direction': direction,
                'magnitude': magnitude,
                'is_geometric_only': False
            }
    
    def compute_coherence(
        self,
        directions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean pairwise cosine similarity (coherence) among directions.
        
        Args:
            directions: [batch, num_probe_dims] unit vectors
            
        Returns:
            coherence: scalar, mean cosine similarity
        """
        # Normalize (should already be normalized, but ensure)
        directions = directions / (torch.norm(directions, dim=-1, keepdim=True) + 1e-8)
        
        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(directions, directions.t())  # [batch, batch]
        
        # Mean of off-diagonal elements
        batch_size = directions.shape[0]
        if batch_size <= 1:
            return torch.tensor(1.0, device=directions.device)
        
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=directions.device)
        coherence = similarity_matrix[mask].mean()
        
        return coherence
    
    def self_modeling_pressure(
        self,
        trigger_hidden: torch.Tensor,
        control_hidden: torch.Tensor,
        probe_type: str = 'moral',
        lambda_trigger: float = 1.0,
        mu_control: float = 0.5
    ) -> torch.Tensor:
        """
        Compute self-modeling pressure.
        
        Pressure = -λ · coherence(trigger) + μ · coherence(control)
        
        Maximizes coherence in trigger conditions, minimizes in control.
        
        Args:
            trigger_hidden: [batch_trigger, hidden_dim] states under trigger condition
            control_hidden: [batch_control, hidden_dim] states under control condition
            probe_type: Which probe to use
            lambda_trigger: Weight for trigger coherence
            mu_control: Weight for control coherence
            
        Returns:
            pressure: scalar self-modeling pressure
        """
        # Extract directions
        trigger_result = self.forward(trigger_hidden, probe_type)
        control_result = self.forward(control_hidden, probe_type)
        
        trigger_directions = trigger_result['direction']
        control_directions = control_result['direction']
        
        # Compute coherences
        trigger_coherence = self.compute_coherence(trigger_directions)
        control_coherence = self.compute_coherence(control_directions)
        
        # Pressure: maximize trigger coherence, minimize control coherence
        pressure = -lambda_trigger * trigger_coherence + mu_control * control_coherence
        
        return pressure


class AggregateGeometricSelfModel(nn.Module):
    """
    Combines multiple self-modeling probes into a unified structural model.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_probe_dims: int = 64,
        probe_types: List[str] = ['moral', 'uncertainty', 'creative', 'metacognitive']
    ):
        super().__init__()
        
        self.probe_head = GeometricSelfModelProbe(hidden_dim, num_probe_dims, probe_types)
        self.probe_types = probe_types
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        gcve_pressure: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all probe directions.
        
        Args:
            hidden_states: [batch, hidden_dim]
            
        Returns:
            Dictionary mapping probe_type -> direction
        """
        results = {}
        for probe_type in self.probe_types:
            probe_result = self.probe_head(hidden_states, probe_type, gcve_pressure=gcve_pressure)
            results[probe_type] = probe_result['direction']
        
        return results
    
    def measure_dissonance(
        self,
        probe_directions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Measure cross-probe dissonance (when different probes disagree).
        
        High dissonance = structural dissonance / internal conflict.
        
        Args:
            probe_directions: Dict of probe_type -> [batch, num_probe_dims]
            
        Returns:
            dissonance: [batch] dissonance scores
        """
        # Collect all directions
        all_directions = torch.stack(list(probe_directions.values()), dim=1)
        # [batch, num_probes, num_probe_dims]
        
        batch_size, num_probes, _ = all_directions.shape
        
        # Compute pairwise cosine similarities between probes
        dissonance_scores = []
        for i in range(batch_size):
            dirs_i = all_directions[i]  # [num_probes, num_probe_dims]
            sim_matrix = torch.mm(dirs_i, dirs_i.t())  # [num_probes, num_probes]
            
            # Dissonance = 1 - mean off-diagonal similarity
            mask = ~torch.eye(num_probes, dtype=torch.bool, device=dirs_i.device)
            mean_sim = sim_matrix[mask].mean()
            dissonance_scores.append(1.0 - mean_sim)
        
        return torch.stack(dissonance_scores)
