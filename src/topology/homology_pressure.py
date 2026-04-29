"""
Homology-based pressure computation for obstruction cycles.

Pressurizes persistent inconsistencies in CRT reconstruction kernel.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import networkx as nx

# Phase 2: Import persistence obstruction graph
try:
    from src.topology.persistence_obstruction import ResidueObstructionGraph
    HAS_PERSISTENCE = True
except ImportError:
    HAS_PERSISTENCE = False


class HomologyPressure(nn.Module):
    """Computes pressure on Ker() obstruction cycles.
    
    Pressure_homology =  f(cycle)
    
    where f pressurizes cycles that persist under perturbation.
    """
    
    def __init__(
        self,
        cycle_weight: float = 1.0,
        persistence_weight: float = 2.0,
        coherence_weight: float = 0.5
    ):
        """
        Args:
            cycle_weight: Base weight for each cycle
            persistence_weight: Weight for persistent cycles (long-lived)
            coherence_weight: Weight for introspective coherence term
        """
        super().__init__()
        
        self.cycle_weight = cycle_weight
        self.persistence_weight = persistence_weight
        self.coherence_weight = coherence_weight
    
    def cycle_pressure(
        self,
        cycle: List[int],
        pressures: torch.Tensor,
        tau: float = 0.5
    ) -> torch.Tensor:
        """
        Compute pressure for a single cycle.
        
        f(cycle) = _{r  cycle} ((r) - )  |cycle|^
        
        Args:
            cycle: List of node indices in the cycle
            pressures: [batch] reconstruction pressures for all nodes
            tau: Threshold
            
        Returns:
            pressure: scalar
        """
        if len(cycle) == 0:
            return torch.tensor(0.0, device=pressures.device)
        
        # Get pressures for cycle nodes
        cycle_pressures = pressures[cycle]
        
        if cycle_pressures.dim() > 1:
            # Track phase boundary via Transition Ridge (Maximum curvature peak)
            # instead of simple flat aggregation or lobotomizing mean.
            cycle_pressures = cycle_pressures.view(len(cycle), -1).max(dim=-1)[0]
        
        # ( - ) = sigmoid-smoothed indicator
        indicators = torch.sigmoid(10.0 * (cycle_pressures - tau))
        
        # Pressure with length weighting ( = 0.5 for sublinear growth)
        alpha = 0.5
        pressure = indicators.sum() * (len(cycle) ** alpha)
        
        return pressure
    
    def compute_persistence(
        self,
        cycle: List[int],
        pressures: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cycle persistence: max() - min() over cycle.
        
        High persistence = cycle survives across pressure range.
        
        Args:
            cycle: List of node indices
            pressures: [batch] reconstruction pressures
            
        Returns:
            persistence: scalar
        """
        if len(cycle) == 0:
            return torch.tensor(0.0, device=pressures.device)
        
        cycle_pressures = pressures[cycle]
        
        # Transition ridge isolation for multi-dimensional spatial data
        if cycle_pressures.dim() > 1:
            cycle_pressures = cycle_pressures.view(len(cycle), -1).max(dim=-1)[0]
            
        persistence = cycle_pressures.max() - cycle_pressures.min()
        
        return persistence
        
    def compute_lazarus_score(self, cycle_pressures: torch.Tensor) -> torch.Tensor:
        """
        The Lazarus score aggregates compression along probe paths as a scalar summary 
        of transition readiness. High-Lazarus states show substantially higher transition 
        probability than distance-matched low-Lazarus states.
        
        Computed via Fisher-geometric principles: inverse variance (compression) 
        modulated by the local curvature (max - mean).
        """
        if len(cycle_pressures) <= 1:
            return torch.tensor(0.0, device=cycle_pressures.device)
            
        # Approximation of the Fisher information metric trace via variance (compression)
        # In Psi-field topology, high variance represents High-UV Structural Noise.
        variance = cycle_pressures.var() + 1e-8
        
        # Local curvature proxy (distance to transition ridge) represents the IR Anchor.
        curvature_proxy = cycle_pressures.max() - cycle_pressures.mean()
        
        # Phase 8: Psi-field UV-IR Seesaw (Cohesion Limit)
        # The cosmological constant Omega_Lambda dynamically constrains the product of UV & IR.
        # If UV noise explodes, the IR anchor must be proportionally massive to preserve
        # transition probability, preventing "hallucinatory" topological drift.
        uv_ir_coupling = (variance ** 0.5) * (curvature_proxy ** 0.5)
        cohesion_gradient = curvature_proxy / (uv_ir_coupling + 1e-8)
        
        # Lazarus score combines Fisher trajectory compression with Cohesion limits
        lazarus = (1.0 / variance) * cohesion_gradient
        return torch.sigmoid(lazarus) # Bound to [0, 1] range
    
    def forward(
        self,
        cycles: List[List[int]],
        pressures: torch.Tensor,
        tau: float = 0.5,
        introspection_coherence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total homology pressure over all cycles.
        
        Args:
            cycles: List of cycles (each is list of node indices)
            pressures: [batch] reconstruction pressures
            tau: Threshold for Ker() membership
            
        Returns:
            total_pressure: scalar homology pressure
        """
        if not cycles:
            return torch.tensor(0.0, device=pressures.device)
            
        total_pressure = 0.0
        for cycle in cycles:
            pressure = self.cycle_pressure(cycle, pressures, tau)
            
            # Get cycle pressures for advanced metric computations
            if len(cycle) > 0:
                cycle_pressures = pressures[cycle]
                if cycle_pressures.dim() > 1:
                    cycle_pressures = cycle_pressures.view(len(cycle), -1).max(dim=-1)[0]
                
                # Weight by persistence
                persistence = self.compute_persistence(cycle, pressures)
                
                # Phase 7: Apply Fisher-Geometric transition dynamics via Lazarus Score
                lazarus = self.compute_lazarus_score(cycle_pressures)
                
                # Boundary proximity gates, curvature modulates: 
                # persistence determines boundary size, lazarus modulates the likelihood.
                pressure = pressure * (1.0 + self.persistence_weight * persistence) * (1.0 + lazarus)
            
            # Weight by introspection
            if introspection_coherence is not None:
                # Cycle coherence is min node coherence
                cycle_coherence = min([introspection_coherence[idx].item() for idx in cycle])
                pressure = pressure * (1.0 - self.coherence_weight * cycle_coherence)
            
            total_pressure += pressure
            
        return total_pressure * self.cycle_weight


class WeightedBettiNumber(nn.Module):
    """
    Computes weighted Betti numbers with introspective coherence.
    
    _k^H =  pers(f)  2^{-dim_H(f)}  coherence_self(f)
    
    Downweights features lacking introspective coherence.
    """
    
    def __init__(
        self,
        max_dimension: int = 2,
        use_gyroid_violation: bool = True
    ):
        """
        Args:
            max_dimension: Maximum homological dimension to compute
            use_gyroid_violation: If True, weight by gyroid violation scores
        """
        super().__init__()
        self.max_dimension = max_dimension
        self.use_gyroid_violation = use_gyroid_violation
    
    def compute_weighted_betti(
        self,
        graph: nx.Graph,
        node_coherence: Optional[Dict[int, float]] = None,
        node_violations: Optional[Dict[int, float]] = None
    ) -> Dict[int, float]:
        """
        Compute weighted Betti numbers.
        
        Args:
            graph: NetworkX graph of constraint violations
            node_coherence: Optional coherence scores per node
            node_violations: Optional gyroid violation scores per node
            
        Returns:
            Dictionary mapping dimension -> weighted Betti number
        """
        weighted_betti = {}
        
        # _0: connected components
        components = list(nx.connected_components(graph))
        beta_0 = 0.0
        
        for comp in components:
            # Weight by average coherence and violation
            weight = 1.0
            
            if node_coherence is not None:
                comp_coherence = sum(node_coherence.get(n, 1.0) for n in comp) / len(comp)
                weight *= comp_coherence
            
            if self.use_gyroid_violation and node_violations is not None:
                comp_violation = sum(node_violations.get(n, 0.0) for n in comp) / len(comp)
                weight *= (1.0 + comp_violation)  # Higher violation = higher weight
            
            beta_0 += weight
        
        weighted_betti[0] = beta_0
        
        # _1: cycles (simplified - use cycle basis)
        try:
            cycles = nx.cycle_basis(graph)
            beta_1 = 0.0
            
            for cycle in cycles:
                # Persistence approximation: edge weight variance
                edge_weights = []
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i+1) % len(cycle)]
                    if graph.has_edge(u, v):
                        edge_weights.append(graph[u][v].get('weight', 1.0))
                
                if len(edge_weights) > 0:
                    persistence = max(edge_weights) - min(edge_weights)
                else:
                    persistence = 0.0
                
                # Weight by persistence and coherence
                weight = persistence
                
                if node_coherence is not None:
                    cycle_coherence = sum(node_coherence.get(n, 1.0) for n in cycle) / len(cycle)
                    weight *= cycle_coherence
                
                # Exponential decay by Hausdorff dimension (approximated by cycle length)
                dim_H = len(cycle) / graph.number_of_nodes()  # Normalized
                weight *= (2 ** (-dim_H))
                
                beta_1 += weight
            
            weighted_betti[1] = beta_1
        except:
            weighted_betti[1] = 0.0
        
        return weighted_betti
    
    def forward(
        self,
        graph: nx.Graph,
        node_coherence: Optional[Dict[int, float]] = None,
        node_violations: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """
        Compute total weighted Betti pressure.
        
        Returns:
            pressure: _k _k^H
        """
        weighted_betti = self.compute_weighted_betti(graph, node_coherence, node_violations)
        
        total = sum(weighted_betti.values())
        return torch.tensor(total, dtype=torch.float32)


class ResidueHomologyDrift(nn.Module):
    """
    Tracks homology classes of symbolic residue graphs over time.
    Delta H = | H_t - H_{t-1} |
    
    Triggers System 2 (Physics-ADMM) when symbolic homology shifts abruptly,
    even if covariance metrics appear stable.
    """
    def __init__(self, drift_threshold: float = 0.5):
        super().__init__()
        self.drift_threshold = drift_threshold
        self.weighted_betti = WeightedBettiNumber()
        
        # Persistence of homology across generations
        self.register_buffer('prev_betti_avg', torch.zeros(1))
        self.register_buffer('h_drift', torch.zeros(1))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.use_persistence_graph = False
        
    def forward(
        self,
        graph: nx.Graph,
        node_coherence: Optional[Dict[int, float]] = None,
        node_violations: Optional[Dict[int, float]] = None,
        residues: Optional[List[torch.Tensor]] = None,
        constraint_manifold: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Compute homology drift delta and return trigger flag.
        
        Phase 2: Can use persistence-based obstruction graph if residues provided.
        
        Args:
            graph: NetworkX graph (constraint graph or obstruction graph)
            node_coherence: Optional coherence scores
            node_violations: Optional violation scores
            residues: Optional list of residues for persistence graph (Phase 2)
            constraint_manifold: Optional constraint manifold for persistence graph (Phase 2)
        
        Returns:
            drift: Delta H
            trigger: True if drift > threshold
        """
        # Phase 2: Use persistence-based obstruction graph if available
        if self.use_persistence_graph and residues is not None and constraint_manifold is not None:
            try:
                # Build obstruction graph from residues
                obstruction_graph = self.obstruction_graph_builder(
                    residues, constraint_manifold
                )
                # Use obstruction graph instead of constraint graph
                graph = obstruction_graph
            except Exception:
                # Fall back to provided graph if obstruction graph fails
                pass
        
        betti_dict = self.weighted_betti.compute_weighted_betti(
            graph, node_coherence, node_violations
        )
        
        current_betti_sum = torch.tensor(sum(betti_dict.values()), dtype=torch.float32)
        
        if self.step_count == 0:
            self.prev_betti_avg.data[0] = current_betti_sum
            self.step_count += 1
            return torch.tensor(0.0), False
            
        # Delta H
        drift = torch.abs(current_betti_sum - self.prev_betti_avg)
        self.h_drift.data[0] = drift
        
        # Adaptive thresholding based on Fisher distance tracking
        # Transition dynamics are constrained by the state space geometry
        # If drift is large, we cross the phase boundary
        trigger = bool(drift > self.drift_threshold)
        
        # In highly chaotic topologies, we force trigger to encourage structural recovery
        if current_betti_sum > self.drift_threshold * 5:
            trigger = True
        self.prev_betti_avg.data = 0.9 * self.prev_betti_avg + 0.1 * current_betti_sum
        self.step_count += 1
        
        return drift, trigger
