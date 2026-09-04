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
        suppress_narration: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        if suppress_narration is None:
            suppress_narration = getattr(self, 'suppress_narration', True)
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
        
        Pressure = -  coherence(trigger) +   coherence(control)
        
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

    def unlearn_rigidity(self, current_time: float = 0.0, overlap_window: float = 21600.0, decay_rate: Optional[float] = None):
        """
        Engram-based unlearning of rigidity using sparsity, intrinsic excitability,
        temporal overlap (< 6 hours), and Bouligand polyshape blocks.
        Decay is driven by biological memory consolidation rather than trust gradients.
        """
        from src.core.honest_jitter import harvest_honest_jitter
        with torch.no_grad():
            for probe_type, sequential in self.probes.items():
                for layer in sequential:
                    if isinstance(layer, nn.Linear):
                        # Initialize Intrinsic Excitability and Engram Trackers
                        if not hasattr(layer, 'neuronal_excitability'):
                            layer.register_buffer('neuronal_excitability', torch.rand_like(layer.weight))
                            layer.register_buffer('last_engram_activation', torch.zeros_like(layer.weight))
                            
                        # 1. Sparsity: Engrams are remarkably sparse (2-6%). We target ~5%.
                        k = max(1, int(0.05 * layer.weight.numel()))
                        
                        # 2. Overlapping Engrams: Shared populations if linked < 6 hours
                        time_since_last = current_time - layer.last_engram_activation
                        temporal_boost = torch.exp(-time_since_last / overlap_window)
                        
                        # 3. Bouligand Polyshape Blocks: Geometric structural gating
                        bouligand_gate = torch.sin(layer.weight * 3.14159).abs()
                        
                        # 4. Neuronal Excitability: Threshold readiness + temporal + geometry
                        effective_excitability = layer.neuronal_excitability * (1.0 + temporal_boost) * bouligand_gate
                        
                        # Competition: Local inhibitory microcircuitry selects top K
                        _, engram_indices = torch.topk(effective_excitability.view(-1), k)
                        engram_mask = torch.zeros_like(effective_excitability).view(-1)
                        engram_mask[engram_indices] = 1.0
                        engram_mask = engram_mask.view_as(layer.weight)
                        
                        # 5. Co-Retrieval & Temporal Updating
                        layer.last_engram_activation = torch.where(
                            engram_mask > 0, 
                            torch.tensor(current_time, device=layer.weight.device), 
                            layer.last_engram_activation
                        )
                        
                        orig_norm = layer.weight.norm().item()
                        if orig_norm < 1e-6:
                            continue
                            
                        decay_scale = decay_rate if decay_rate is not None else 0.05
                        jitter = harvest_honest_jitter(
                            layer.weight.shape,
                            device=layer.weight.device,
                            scaled=True
                        ) * decay_scale  # Base engram plasticity
                        
                        # Apply plasticity ONLY to the active engram complex
                        layer.weight.add_(jitter * engram_mask)
                        
                        # Re-normalize to prevent total lobotomy
                        new_norm = layer.weight.norm().item()
                        if new_norm > 1e-8:
                            layer.weight.mul_(orig_norm / new_norm)
                            
                        # Excitability homeostasis
                        layer.neuronal_excitability.mul_(0.99)
                        layer.neuronal_excitability.add_(0.01 * torch.rand_like(layer.neuronal_excitability))



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
        gcve_pressure: Optional[torch.Tensor] = None,
        suppress_narration: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all probe directions.
        
        Args:
            hidden_states: [batch, hidden_dim]
            
        Returns:
            Dictionary mapping probe_type -> direction
        """
        results = {}
        if suppress_narration is None:
            suppress_narration = getattr(self, 'suppress_narration', True)
        for probe_type in self.probe_types:
            probe_result = self.probe_head(
                hidden_states,
                probe_type,
                gcve_pressure=gcve_pressure,
                suppress_narration=suppress_narration
            )
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

    def unlearn_rigidity(self, current_time: float = 0.0, overlap_window: float = 21600.0, decay_rate: Optional[float] = None):
        """Trigger unlearning across all aggregated self-model probes."""
        self.probe_head.unlearn_rigidity(current_time, overlap_window, decay_rate)


# Legacy alias for backward compatibility (Rigidity Decay)
IntrospectionHead = AggregateGeometricSelfModel

