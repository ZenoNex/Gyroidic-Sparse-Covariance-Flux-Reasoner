"""
Persistence-Based Residue Obstruction Graph

Implements filtration-based persistent homology for residue interactions
and constructs obstruction graphs based on joint loop counts.

Mathematical Foundation:
    C_epsilon = {c in C | L(r, c) <= epsilon}
    PH_k(r) = H_k(C_epsilon) as epsilon increases
    
    G = (V, E) where:
    V = {r_i}
    E_{ij} <-> exists epsilon: beta_1^{ij}(epsilon) != 0
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import networkx as nx
import numpy as np


class ResidueFiltration(nn.Module):
    """
    Filtration: C_epsilon = {c in C | L(r, c) <= epsilon}
    
    Builds simplicial complexes at increasing epsilon values.
    """
    
    def __init__(
        self,
        residue: torch.Tensor,
        constraint_manifold: torch.Tensor,
        loss_fn: Optional[callable] = None
    ):
        """
        Args:
            residue: [batch, ...] residue tensor
            constraint_manifold: [batch, dim] constraint manifold points
            loss_fn: Optional function L(r, c) -> loss tensor
        """
        super().__init__()
        self.r = residue
        self.C = constraint_manifold
        self.loss_fn = loss_fn or self._default_loss_fn
    
    def _default_loss_fn(self, residue: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        """Default loss: L2 distance."""
        if residue.shape != constraint.shape:
            if residue.numel() == constraint.numel():
                residue = residue.reshape(constraint.shape)
            else:
                # Project to same dimension
                if residue.shape[-1] != constraint.shape[-1]:
                    proj = nn.Linear(residue.shape[-1], constraint.shape[-1], 
                                   device=residue.device)
                    residue = proj(residue)
        
        return torch.norm(residue - constraint, dim=-1)
    
    def filter_by_loss(
        self,
        epsilon: float
    ) -> torch.Tensor:
        """
        Filter constraint manifold: C_epsilon = {c | L(r, c) <= epsilon}
        
        Args:
            epsilon: Filtration parameter
            
        Returns:
            filtered_points: [num_points, dim] points in C_epsilon
        """
        # Compute losses
        losses = self.loss_fn(self.r, self.C)  # [batch]
        
        # Filter: keep points where loss <= epsilon
        mask = losses <= epsilon
        filtered_points = self.C[mask]
        
        return filtered_points
    
    def build_simplicial_complex(
        self,
        points: torch.Tensor,
        max_dimension: int = 2
    ) -> Dict[int, List[Tuple]]:
        """
        Build simplicial complex from point cloud.
        
        Uses Vietoris-Rips complex construction (distance-based).
        
        Args:
            points: [num_points, dim] point cloud
            max_dimension: Maximum simplex dimension
            
        Returns:
            complex: Dictionary mapping dimension -> list of simplices
        """
        if points.shape[0] == 0:
            return {0: [], 1: [], 2: []}
        
        num_points = points.shape[0]
        complex = {k: [] for k in range(max_dimension + 1)}
        
        # 0-simplices (vertices)
        complex[0] = [(i,) for i in range(num_points)]
        
        # 1-simplices (edges): connect points within distance threshold
        # Use adaptive threshold based on point cloud scale
        if num_points > 1:
            # Compute pairwise distances
            distances = torch.cdist(points, points)  # [num_points, num_points]
            
            # Threshold: median distance (or mean)
            threshold = torch.median(distances[distances > 0]) if num_points > 1 else 1.0
            
            # Add edges for pairs within threshold
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    if distances[i, j] <= threshold:
                        complex[1].append((i, j))
        
        # 2-simplices (triangles): complete triangles from edges
        if max_dimension >= 2 and len(complex[1]) > 0:
            # Build edge set for fast lookup
            edge_set = set(complex[1])
            
            # Check all triplets
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    for k in range(j + 1, num_points):
                        if (i, j) in edge_set and (j, k) in edge_set and (i, k) in edge_set:
                            complex[2].append((i, j, k))
        
        return complex
    
    def build_filtration(
        self,
        epsilon_values: torch.Tensor,
        max_dimension: int = 2
    ) -> List[Dict[int, List[Tuple]]]:
        """
        Build simplicial complexes at each epsilon value.
        
        Args:
            epsilon_values: [num_epsilons] filtration parameters
            max_dimension: Maximum simplex dimension
            
        Returns:
            complexes: List of simplicial complexes (one per epsilon)
        """
        complexes = []
        for eps in epsilon_values:
            C_eps = self.filter_by_loss(eps.item())
            complex = self.build_simplicial_complex(C_eps, max_dimension)
            complexes.append(complex)
        
        return complexes


class PersistentHomologyComputer(nn.Module):
    """
    Computes persistent homology from a filtration.
    
    PH_k(r) = H_k(C_epsilon) as epsilon increases.
    Returns birth-death pairs for each dimension.
    """
    
    def __init__(self, max_dimension: int = 1):
        """
        Args:
            max_dimension: Maximum homological dimension to compute
        """
        super().__init__()
        self.max_dimension = max_dimension
    
    def compute_betti_numbers(
        self,
        complex: Dict[int, List[Tuple]]
    ) -> Dict[int, int]:
        """
        Compute Betti numbers for a simplicial complex.
        
        Simplified computation using Euler characteristic and rank.
        
        Args:
            complex: Simplicial complex dictionary
            
        Returns:
            betti: Dictionary mapping dimension -> Betti number
        """
        betti = {}
        
        # β_0: number of connected components
        # Use graph connectivity
        if len(complex[0]) == 0:
            betti[0] = 0
        elif len(complex[1]) == 0:
            betti[0] = len(complex[0])  # All isolated vertices
        else:
            # Build graph and count components
            G = nx.Graph()
            G.add_nodes_from([v[0] for v in complex[0]])
            G.add_edges_from(complex[1])
            betti[0] = nx.number_connected_components(G)
        
        # β_1: number of independent cycles
        if len(complex[1]) == 0:
            betti[1] = 0
        else:
            # Build graph
            G = nx.Graph()
            G.add_nodes_from([v[0] for v in complex[0]])
            G.add_edges_from(complex[1])
            
            # β_1 = |E| - |V| + |components|
            num_edges = len(complex[1])
            num_vertices = len(complex[0])
            num_components = nx.number_connected_components(G)
            betti[1] = max(0, num_edges - num_vertices + num_components)
        
        return betti
    
    def compute_persistent_homology(
        self,
        filtration: List[Dict[int, List[Tuple]]],
        epsilon_values: torch.Tensor
    ) -> Dict[int, List[Tuple[float, float]]]:
        """
        Compute persistent homology from filtration.
        
        Args:
            filtration: List of simplicial complexes
            epsilon_values: [num_epsilons] corresponding epsilon values
            
        Returns:
            persistence_diagram: Dictionary mapping dimension -> list of (birth, death) pairs
        """
        persistence_diagram = {k: [] for k in range(self.max_dimension + 1)}
        
        # Track Betti numbers across filtration
        prev_betti = {k: 0 for k in range(self.max_dimension + 1)}
        
        for idx, complex in enumerate(filtration):
            epsilon = epsilon_values[idx].item()
            current_betti = self.compute_betti_numbers(complex)
            
            # Detect birth: Betti number increases
            for dim in range(self.max_dimension + 1):
                if current_betti.get(dim, 0) > prev_betti.get(dim, 0):
                    # Feature born at this epsilon
                    birth = epsilon
                    # Find death (when feature disappears)
                    death = self._find_death(filtration, idx, dim, current_betti[dim])
                    persistence_diagram[dim].append((birth, death))
            
            prev_betti = current_betti
        
        return persistence_diagram
    
    def _find_death(
        self,
        filtration: List[Dict[int, List[Tuple]]],
        birth_idx: int,
        dimension: int,
        target_betti: int
    ) -> float:
        """
        Find death time for a feature born at birth_idx.
        
        Simplified: look ahead in filtration for when Betti number decreases.
        """
        # Look ahead to find when feature dies
        for idx in range(birth_idx + 1, len(filtration)):
            betti = self.compute_betti_numbers(filtration[idx])
            if betti.get(dimension, 0) < target_betti:
                # Feature died - use epsilon from previous step
                return float('inf')  # Infinite persistence (simplified)
        
        return float('inf')  # Feature persists to infinity


class ResidueObstructionGraph(nn.Module):
    """
    Residue Obstruction Graph: G = (V, E)
    
    V = {r_i}
    E_{ij} <-> exists epsilon: beta_1^{ij}(epsilon) != 0
    """
    
    def __init__(
        self,
        num_epsilon_samples: int = 20,
        max_dimension: int = 1
    ):
        """
        Args:
            num_epsilon_samples: Number of epsilon values to sample
            max_dimension: Maximum homological dimension
        """
        super().__init__()
        self.num_epsilon_samples = num_epsilon_samples
        self.max_dimension = max_dimension
        self.ph_computer = PersistentHomologyComputer(max_dimension=max_dimension)
    
    def compute_joint_loop_count(
        self,
        residue_i: torch.Tensor,
        residue_j: torch.Tensor,
        constraint_manifold: torch.Tensor,
        epsilon_range: Optional[torch.Tensor] = None
    ) -> int:
        """
        Compute joint loop count beta_1^{ij}(epsilon) for residue pair.
        
        Args:
            residue_i: [batch_i, ...] first residue
            residue_j: [batch_j, ...] second residue
            constraint_manifold: [batch, dim] constraint manifold
            epsilon_range: Optional [num_epsilons] epsilon values
            
        Returns:
            beta_1_ij: Joint loop count (max over epsilon)
        """
        # Combine residues for joint filtration
        # Use mean or concatenation
        if residue_i.shape == residue_j.shape:
            joint_residue = (residue_i + residue_j) / 2.0
        else:
            # Concatenate if shapes differ
            joint_residue = torch.cat([residue_i, residue_j], dim=0)
        
        # Create filtration
        if epsilon_range is None:
            # Auto-generate epsilon range
            # Use loss values to determine range
            loss_fn = lambda r, c: torch.norm(r.reshape(-1, c.shape[-1]) - c, dim=-1)
            sample_losses = loss_fn(joint_residue, constraint_manifold)
            min_loss = sample_losses.min().item()
            max_loss = sample_losses.max().item()
            epsilon_range = torch.linspace(min_loss, max_loss, self.num_epsilon_samples,
                                         device=constraint_manifold.device)
        
        filtration_obj = ResidueFiltration(joint_residue, constraint_manifold)
        filtration = filtration_obj.build_filtration(epsilon_range, max_dimension=self.max_dimension)
        
        # Compute persistent homology
        persistence_diagram = self.ph_computer.compute_persistent_homology(filtration, epsilon_range)
        
        # Extract beta_1 (dimension 1) persistence
        beta_1_pairs = persistence_diagram.get(1, [])
        
        # Count persistent features (with significant persistence)
        # Use count of features with persistence > threshold
        beta_1_count = len([(b, d) for b, d in beta_1_pairs if d - b > 1e-6])
        
        return beta_1_count
    
    def build_graph(
        self,
        residues: List[torch.Tensor],
        constraint_manifold: torch.Tensor,
        epsilon_range: Optional[torch.Tensor] = None
    ) -> nx.Graph:
        """
        Build obstruction graph from residues.
        
        Args:
            residues: List of residue tensors
            constraint_manifold: [batch, dim] constraint manifold
            epsilon_range: Optional epsilon values for filtration
            
        Returns:
            G: NetworkX graph with edges where beta_1^{ij} > 0
        """
        G = nx.Graph()
        G.add_nodes_from(range(len(residues)))
        
        # Compute joint loop counts for all pairs
        for i in range(len(residues)):
            for j in range(i + 1, len(residues)):
                beta_1_ij = self.compute_joint_loop_count(
                    residues[i], residues[j], constraint_manifold, epsilon_range
                )
                
                if beta_1_ij > 0:
                    G.add_edge(i, j, weight=beta_1_ij)
        
        return G
    
    def forward(
        self,
        residues: List[torch.Tensor],
        constraint_manifold: torch.Tensor,
        epsilon_range: Optional[torch.Tensor] = None
    ) -> nx.Graph:
        """
        Forward pass: build obstruction graph.
        
        Returns:
            G: NetworkX obstruction graph
        """
        return self.build_graph(residues, constraint_manifold, epsilon_range)
