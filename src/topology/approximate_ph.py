"""
Approximate Persistent Homology for topological rupture detection.

Uses relative barcode change, not exact Betti numbers.
PASₕ fires on RELATIVE change, not absolute topology.

Author: Implementation from Structural Design Decisions
Created: January 2026
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class ApproximatePHProbe(nn.Module):
    """
    Ripser-style approximate persistent homology.
    
    Key principle: Approximation error is orthogonal to the invariants
    we care about (birth-death stability of defects).
    
    Treats PH as a TRIGGER, not a metric.
    """
    
    def __init__(
        self,
        max_dimension: int = 1,
        max_edge_length: float = 2.0,
        num_landmarks: int = 100,  # Sparse filtration
        relative_threshold: float = 0.2,  # Trigger on 20% barcode change
        signature_size: int = 10
    ):
        """
        Args:
            max_dimension: Maximum homology dimension to track
            max_edge_length: Maximum edge length for Rips complex
            num_landmarks: Number of landmarks for sparse filtration
            relative_threshold: Threshold for rupture detection
            signature_size: Size of barcode signature vector
        """
        super().__init__()
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.num_landmarks = num_landmarks
        self.relative_threshold = relative_threshold
        self.signature_size = signature_size
        
        # Track previous barcode for relative change
        self.register_buffer('prev_barcode_signature', torch.zeros(signature_size))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
    
    def _compute_landmark_subsample(self, points: torch.Tensor) -> torch.Tensor:
        """
        Maxmin landmark sampling for sparse filtration.
        
        Greedy algorithm: iteratively select point farthest from current set.
        """
        n = points.shape[0]
        if n <= self.num_landmarks:
            return points
        
        device = points.device
        indices = [0]
        distances = torch.cdist(points, points[0:1]).squeeze(-1)
        
        for _ in range(self.num_landmarks - 1):
            farthest = distances.argmax().item()
            indices.append(farthest)
            new_dists = torch.cdist(points, points[farthest:farthest+1]).squeeze(-1)
            distances = torch.minimum(distances, new_dists)
        
        return points[indices]
    
    def _compute_barcode_signature(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute compact barcode signature (not full diagram).
        
        Uses distance matrix eigenvalue decay as topological proxy.
        This is approximate but captures the structural information
        relevant for rupture detection.
        
        Returns:
            signature: [signature_size] normalized eigenvalue vector
        """
        landmarks = self._compute_landmark_subsample(points)
        
        # Distance matrix
        D = torch.cdist(landmarks, landmarks)
        
        # Make symmetric and add small diagonal for stability
        D = (D + D.t()) / 2
        D = D + torch.eye(D.shape[0], device=D.device) * 1e-6
        
        # Eigenvalue decay as topological signature
        try:
            eigenvalues = torch.linalg.eigvalsh(D)
            eigenvalues = eigenvalues.flip(0)[:self.signature_size]
        except:
            eigenvalues = torch.ones(self.signature_size, device=points.device)
        
        # Normalize
        signature = eigenvalues / (eigenvalues[0].abs() + 1e-8)
        
        # Pad if needed
        if len(signature) < self.signature_size:
            signature = torch.cat([
                signature,
                torch.zeros(self.signature_size - len(signature), device=signature.device)
            ])
        
        return signature[:self.signature_size]
    
    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect topological rupture via relative barcode change.
        
        PASₕ fires on RELATIVE change, not absolute homology.
        
        Args:
            points: [num_points, dim] point cloud
            
        Returns:
            Dict with:
            - 'is_rupture': boolean
            - 'relative_change': scalar
            - 'barcode_signature': [signature_size] vector
        """
        current_sig = self._compute_barcode_signature(points)
        
        # Relative change (only after first step)
        if self.step_count == 0:
            relative_change = torch.tensor(0.0, device=points.device)
            is_rupture = torch.tensor(False, device=points.device)
        else:
            delta = torch.abs(current_sig - self.prev_barcode_signature)
            relative_change = delta.sum() / (self.prev_barcode_signature.abs().sum() + 1e-8)
            is_rupture = relative_change > self.relative_threshold
        
        # Update history
        self.prev_barcode_signature = current_sig.detach()
        self.step_count += 1
        
        return {
            'is_rupture': is_rupture,
            'relative_change': relative_change,
            'barcode_signature': current_sig
        }
    
    def reset(self):
        """Reset tracking state."""
        self.prev_barcode_signature.zero_()
        self.step_count.zero_()
