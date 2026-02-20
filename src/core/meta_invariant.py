"""
Meta-Invariant: Non-Teleological Topology Expansion

Implements the meta-invariant that ensures topology never collapses
toward a single basin.

Mathematical Foundation:
    d/dt E_r[dim H_1(C_t)] >= 0
    
    Topology may expand or fracture, but must never collapse toward
    a single basin.

Phase 3: Advanced Constraints Implementation

Author: Implementation Documentation
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import networkx as nx


class MetaInvariant(nn.Module):
    """
    Meta-Invariant: d/dt E_r[dim H_1(C_t)] >= 0
    
    Monitors topology expansion to prevent collapse toward single basin.
    """
    
    def __init__(self, expansion_threshold: float = 0.0):
        """
        Args:
            expansion_threshold: Minimum allowed expansion rate (default: 0.0)
        """
        super().__init__()
        self.expansion_threshold = expansion_threshold
        
        # Track previous H_1 dimension
        self.register_buffer('prev_h1_dim', torch.tensor(0.0))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
    
    def compute_h1_dimension(
        self,
        graph: nx.Graph,
        node_coherence: Optional[Dict[int, float]] = None,
        node_violations: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """
        Compute dimension of H_1 (first homology group).
        
        H_1 dimension = number of independent cycles = |E| - |V| + |components|
        
        Args:
            graph: NetworkX graph
            node_coherence: Optional coherence scores
            node_violations: Optional violation scores
            
        Returns:
            h1_dim: Scalar dimension
        """
        if graph.number_of_nodes() == 0:
            return torch.tensor(0.0)
        
        num_vertices = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        num_components = nx.number_connected_components(graph)
        
        # H_1 dimension: β_1 = |E| - |V| + |components|
        h1_dim = max(0, num_edges - num_vertices + num_components)
        
        # Weight by coherence and violations if provided
        if node_coherence is not None or node_violations is not None:
            # Compute weighted average coherence/violation
            if node_coherence is not None:
                coherence_avg = sum(node_coherence.values()) / len(node_coherence) if node_coherence else 1.0
            else:
                coherence_avg = 1.0
            
            if node_violations is not None:
                violation_avg = sum(node_violations.values()) / len(node_violations) if node_violations else 0.0
            else:
                violation_avg = 0.0
            
            # Weight H_1 dimension by coherence (higher coherence = more stable topology)
            # and violation (higher violation = more complex topology)
            weight = coherence_avg * (1.0 + violation_avg)
            h1_dim = h1_dim * weight
        
        return torch.tensor(float(h1_dim), dtype=torch.float32)
    
    def compute_expected_h1_dimension(
        self,
        residue_distribution: torch.Tensor,
        graphs: Optional[list] = None,
        node_coherence: Optional[Dict[int, float]] = None,
        node_violations: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """
        Compute expected H_1 dimension over residue distribution.
        
        E_r[dim H_1(C_t)]
        
        Args:
            residue_distribution: [batch, ...] residue distribution
            graphs: Optional list of graphs (one per batch element)
            node_coherence: Optional coherence scores
            node_violations: Optional violation scores
            
        Returns:
            expected_h1: Expected H_1 dimension
        """
        if graphs is not None:
            # Compute H_1 for each graph
            h1_dims = []
            for graph in graphs:
                h1_dim = self.compute_h1_dimension(graph, node_coherence, node_violations)
                h1_dims.append(h1_dim)
            
            # Expected value: mean
            expected_h1 = torch.stack(h1_dims).mean()
        else:
            # Approximate from residue distribution
            # Use variance as proxy for topology complexity
            # Higher variance -> more complex topology -> higher H_1
            variance = torch.var(residue_distribution)
            expected_h1 = variance * 10.0  # Scaling factor (heuristic)
        
        return expected_h1
    
    def check_invariant(
        self,
        current_h1_dim: torch.Tensor,
        residue_distribution: Optional[torch.Tensor] = None
    ) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        """
        Check meta-invariant: d/dt dim H_1 >= 0
        
        Pointer #4: Coherence Loss Is a Signal, Not a Fault
        - Contraction (rate < 0) = COLLAPSE = violation
        - Stasis (rate ≈ 0) = acceptable
        - Expansion (rate > 0) = orthogonality gain = SIGNAL (log, don't penalize)
        
        Args:
            current_h1_dim: Current H_1 dimension (scalar or [batch])
            residue_distribution: Optional residue distribution for expected value
            
        Returns:
            is_satisfied: Boolean (True if invariant satisfied)
            rate: Expansion rate (d/dt)
            violation: Violation magnitude (if negative rate)
        """
        if self.step_count == 0:
            # First step: initialize
            if isinstance(current_h1_dim, torch.Tensor):
                if current_h1_dim.dim() == 0:
                    self.prev_h1_dim.data[0] = current_h1_dim.item()
                else:
                    self.prev_h1_dim.data[0] = current_h1_dim.mean().item()
            else:
                self.prev_h1_dim.data[0] = float(current_h1_dim)
            
            self.step_count += 1
            return True, torch.tensor(0.0), torch.tensor(0.0)
        
        # Compute expected H_1 dimension
        if isinstance(current_h1_dim, torch.Tensor):
            if current_h1_dim.dim() == 0:
                expected_h1 = current_h1_dim.item()
            else:
                expected_h1 = current_h1_dim.mean().item()
        else:
            expected_h1 = float(current_h1_dim)
        
        # Compute rate: d/dt = current - previous
        rate = expected_h1 - self.prev_h1_dim.item()
        
        # Pointer #4: Different handling for expansion vs contraction
        if rate < -self.expansion_threshold:
            # COLLAPSE: Topology contracting toward single basin
            # This is a VIOLATION - return violation magnitude
            is_satisfied = False
            violation = abs(rate)
        elif rate > self.expansion_threshold:
            # EXPANSION: Coherence "loss" = orthogonality gain
            # This is a SIGNAL, not a fault - log but NO violation pressure
            is_satisfied = True
            violation = 0.0
            self._log_expansion_event(rate, expected_h1)
        else:
            # STASIS: Acceptable, within threshold
            is_satisfied = True
            violation = 0.0
        
        # Update previous value
        self.prev_h1_dim.data[0] = expected_h1
        self.step_count += 1
        
        return is_satisfied, torch.tensor(rate), torch.tensor(violation)
    
    def _log_expansion_event(self, rate: float, new_dim: float):
        """
        Log expansion events for analysis (not enforcement).
        
        Expansion is allowed and informative - it indicates orthogonality growth.
        """
        if not hasattr(self, '_expansion_log'):
            self._expansion_log = []
        
        self._expansion_log.append({
            'rate': rate,
            'new_dim': new_dim,
            'step': self.step_count.item()
        })
        
        # Keep bounded history
        if len(self._expansion_log) > 100:
            self._expansion_log = self._expansion_log[-50:]
    
    def forward(
        self,
        current_h1_dim: torch.Tensor,
        residue_distribution: Optional[torch.Tensor] = None,
        graphs: Optional[list] = None,
        node_coherence: Optional[Dict[int, float]] = None,
        node_violations: Optional[Dict[int, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: check meta-invariant.
        
        Args:
            current_h1_dim: Current H_1 dimension
            residue_distribution: Optional residue distribution
            graphs: Optional list of graphs
            node_coherence: Optional coherence scores
            node_violations: Optional violation scores
            
        Returns:
            Dictionary with:
            - 'is_satisfied': Boolean
            - 'rate': Expansion rate
            - 'violation': Violation magnitude
            - 'current_h1': Current H_1 dimension
            - 'prev_h1': Previous H_1 dimension
        """
        # Compute expected H_1 if graphs provided
        if graphs is not None:
            expected_h1 = self.compute_expected_h1_dimension(
                residue_distribution, graphs, node_coherence, node_violations
            )
        else:
            if isinstance(current_h1_dim, torch.Tensor):
                expected_h1 = current_h1_dim.mean() if current_h1_dim.dim() > 0 else current_h1_dim
            else:
                expected_h1 = torch.tensor(float(current_h1_dim))
        
        is_satisfied, rate, violation = self.check_invariant(expected_h1, residue_distribution)
        
        return {
            'is_satisfied': is_satisfied,
            'rate': rate,
            'violation': violation,
            'current_h1': expected_h1,
            'prev_h1': self.prev_h1_dim.item()
        }
    
    def reset(self):
        """Reset tracking state."""
        self.prev_h1_dim.data[0] = 0.0
        self.step_count.data[0] = 0
