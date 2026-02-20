"""
Tripartite Evidence Modules: Data-derived + Architectural + Adversarial

E_α ⊂ C such that ∂E_α ≠ ∅ ∧ E_α ⊄ span(E_{β≠α})

If evidence modules become mutually predictable, PASₕ goes blind.

Author: Implementation from Structural Design Decisions
Created: January 2026
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import warnings


class TripartiteEvidenceModule(nn.Module):
    """
    Heterogeneous evidence that must remain mutually unpredictable.
    
    Three types:
    1. Data-derived: Covariance anomalies, spectral outliers
    2. Architectural: Predefined failure mode probes
    3. Adversarial: Injected contradictions, impossible cycles
    
    If evidence modules become predictable, PASₕ goes blind.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_clusters: int = 4,
        blindness_threshold: float = 0.7
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_clusters: Number of covariance clusters for data evidence
            blindness_threshold: Correlation threshold for blindness detection
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.blindness_threshold = blindness_threshold
        
        # Type 1: Data-derived (anomaly detectors)
        self.covariance_probe = nn.Linear(hidden_dim, num_clusters)
        
        # Type 2: Architectural priors (known failure modes)
        self.failure_mode_detectors = nn.ModuleDict({
            'mode_collapse': nn.Linear(hidden_dim, 1),
            'over_compression': nn.Linear(hidden_dim, 1),
            'symmetry_lock': nn.Linear(hidden_dim, 1)
        })
        
        # Type 3: Adversarial (synthetic contradictions)
        self.contradiction_generator = nn.Linear(hidden_dim, hidden_dim)
    
    def compute_data_evidence(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Type 1: Covariance anomalies, spectral outliers.
        
        Returns entropy-based anomaly score (high = doesn't fit clusters).
        """
        # Flatten if needed
        if hidden.dim() > 2:
            hidden = hidden.mean(dim=1)
        
        cluster_logits = self.covariance_probe(hidden)
        probs = torch.softmax(cluster_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        return entropy
    
    def compute_architectural_evidence(
        self,
        hidden: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Type 2: Predefined failure mode probes.
        
        Returns scores for known failure modes.
        """
        if hidden.dim() > 2:
            hidden = hidden.mean(dim=1)
        
        results = {}
        for name, detector in self.failure_mode_detectors.items():
            score = torch.sigmoid(detector(hidden)).squeeze(-1)
            results[name] = score
        return results
    
    def compute_adversarial_evidence(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Type 3: Inject contradictions.
        
        Generates constraint orthogonal to current representation.
        """
        if hidden.dim() > 2:
            hidden = hidden.mean(dim=1)
        
        contradiction = self.contradiction_generator(hidden)
        
        # Make orthogonal via Gram-Schmidt
        projection = (hidden * contradiction).sum(dim=-1, keepdim=True)
        hidden_norm_sq = (hidden.norm(dim=-1, keepdim=True) ** 2) + 1e-8
        orthogonal = contradiction - (projection / hidden_norm_sq) * hidden
        
        # Contradiction strength = norm of orthogonal component
        strength = orthogonal.norm(dim=-1)
        return strength
    
    def check_mutual_predictability(
        self,
        data_evidence: torch.Tensor,
        arch_evidence: Dict[str, torch.Tensor],
        adv_evidence: torch.Tensor
    ) -> bool:
        """
        Check if evidence modules have become mutually predictable.
        
        If yes, PASₕ is blind - issue warning.
        """
        # Flatten all evidence to 1D
        data_flat = data_evidence.flatten()
        arch_flat = torch.stack(list(arch_evidence.values())).mean(dim=0).flatten()
        adv_flat = adv_evidence.flatten()
        
        # Match lengths for correlation
        min_len = min(len(data_flat), len(arch_flat), len(adv_flat))
        if min_len < 2:
            return False
        
        data_flat = data_flat[:min_len]
        arch_flat = arch_flat[:min_len]
        adv_flat = adv_flat[:min_len]
        
        # Correlations
        try:
            corr_data_arch = torch.corrcoef(
                torch.stack([data_flat, arch_flat])
            )[0, 1]
            corr_data_adv = torch.corrcoef(
                torch.stack([data_flat, adv_flat])
            )[0, 1]
            
            # Handle NaN
            corr_data_arch = torch.nan_to_num(corr_data_arch, nan=0.0)
            corr_data_adv = torch.nan_to_num(corr_data_adv, nan=0.0)
        except:
            return False
        
        is_predictable = (
            corr_data_arch.abs() > self.blindness_threshold or
            corr_data_adv.abs() > self.blindness_threshold
        )
        
        if is_predictable:
            warnings.warn(
                f"EVIDENCE BLINDNESS: Evidence modules are mutually predictable "
                f"(data-arch corr: {corr_data_arch:.2f}, data-adv corr: {corr_data_adv:.2f}). "
                f"PASₕ cannot distinguish failure modes.",
                UserWarning
            )
        
        return bool(is_predictable.item()) if hasattr(is_predictable, 'item') else bool(is_predictable)
    
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all three evidence types and check for blindness.
        """
        data_ev = self.compute_data_evidence(hidden)
        arch_ev = self.compute_architectural_evidence(hidden)
        adv_ev = self.compute_adversarial_evidence(hidden)
        
        is_blind = self.check_mutual_predictability(data_ev, arch_ev, adv_ev)
        
        return {
            'data_evidence': data_ev,
            'architectural_evidence': arch_ev,
            'adversarial_evidence': adv_ev,
            'is_blind': torch.tensor(is_blind, device=hidden.device)
        }
