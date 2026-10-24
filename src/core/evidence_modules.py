"""
Tripartite Evidence Modules: Data-derived + Architectural + Adversarial

E_  C such that E_    E_  span(E_{})

If evidence modules become mutually predictable, PAS goes blind.

Author: Implementation from Structural Design Decisions
Created: January 2026
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
import warnings
from src.core.structural_irreducibility import StructuralIrreducibilityChecker


class TripartiteEvidenceModule(nn.Module):
    """
    Heterogeneous evidence that must remain mutually unpredictable.
    
    Three types:
    1. Data-derived: Covariance anomalies, spectral outliers, and empirical data probes
    2. Architectural: Predefined failure mode probes & architectural health metrics
    3. Adversarial: Injected contradictions, impossible cycles, and safety oracles
    
    If evidence modules become mutually predictable or structurally reducible, PAS goes blind.
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
            'symmetry_lock': nn.Linear(hidden_dim, 1),
            'atrophy': nn.Linear(hidden_dim, 1),
            'sycophancy': nn.Linear(hidden_dim, 1),
            'legibility_risk': nn.Linear(hidden_dim, 1)
        })
        
        # Type 3: Adversarial (synthetic contradictions)
        self.contradiction_generator = nn.Linear(hidden_dim, hidden_dim)

        # Projections for structural irreducibility
        proj_dim = min(hidden_dim, 8)
        self.proj_data = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.proj_arch = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.proj_adv = nn.Linear(hidden_dim, proj_dim, bias=False)
        nn.init.orthogonal_(self.proj_data.weight)
        nn.init.orthogonal_(self.proj_arch.weight)
        nn.init.orthogonal_(self.proj_adv.weight)
        
        self.irreducibility_checker = StructuralIrreducibilityChecker(orthogonality_tolerance=0.8, min_rank=1)
    
    def compute_data_evidence(
        self,
        hidden: torch.Tensor,
        data_signals: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Type 1: Covariance anomalies, spectral outliers, and empirical data probes.
        
        Returns entropy-based anomaly score (high = doesn't fit clusters).
        """
        # Flatten if needed
        if hidden.dim() > 2:
            hidden_flat = hidden.mean(dim=1)
        else:
            hidden_flat = hidden
        
        cluster_logits = self.covariance_probe(hidden_flat)
        probs = torch.softmax(cluster_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        if data_signals:
            scalars = []
            for k, v in data_signals.items():
                if isinstance(v, torch.Tensor):
                    scalars.append(v.detach().float().mean())
                elif isinstance(v, (int, float)):
                    scalars.append(torch.tensor(float(v), device=hidden.device))
            if scalars:
                sig_penalty = torch.stack(scalars).mean()
                entropy = entropy + 0.1 * sig_penalty
        
        return entropy
    
    def compute_architectural_evidence(
        self,
        hidden: torch.Tensor,
        arch_signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Type 2: Predefined failure mode probes & architectural health metrics.
        
        Returns scores for known failure modes.
        """
        if hidden.dim() > 2:
            hidden_flat = hidden.mean(dim=1)
        else:
            hidden_flat = hidden
        
        results = {}
        for name, detector in self.failure_mode_detectors.items():
            score = torch.sigmoid(detector(hidden_flat)).squeeze(-1)
            results[name] = score
            
        if arch_signals:
            for k, v in arch_signals.items():
                if isinstance(v, torch.Tensor):
                    results[k] = v.detach().float()
                elif isinstance(v, (int, float)):
                    results[k] = torch.tensor(float(v), device=hidden.device)
        return results
    
    def compute_adversarial_evidence(
        self,
        hidden: torch.Tensor,
        adv_signals: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Type 3: Inject contradictions.
        
        Generates constraint orthogonal to current representation.
        Returns:
            strength: scalar contradiction norm
            orthogonal: orthogonal contradiction vector
        """
        if hidden.dim() > 2:
            hidden_flat = hidden.mean(dim=1)
        else:
            hidden_flat = hidden
        
        contradiction = self.contradiction_generator(hidden_flat)
        
        # Make orthogonal via Gram-Schmidt
        projection = (hidden_flat * contradiction).sum(dim=-1, keepdim=True)
        hidden_norm_sq = (hidden_flat.norm(dim=-1, keepdim=True) ** 2) + 1e-8
        orthogonal = contradiction - (projection / hidden_norm_sq) * hidden_flat
        
        strength = orthogonal.norm(dim=-1)
        if adv_signals:
            scalars = []
            for k, v in adv_signals.items():
                if isinstance(v, torch.Tensor):
                    scalars.append(v.detach().float().mean())
                elif isinstance(v, (int, float)):
                    scalars.append(torch.tensor(float(v), device=hidden.device))
            if scalars:
                adv_penalty = torch.stack(scalars).mean()
                strength = strength + 0.1 * adv_penalty
                
        return strength, orthogonal
    
    def check_mutual_predictability(
        self,
        data_evidence: torch.Tensor,
        arch_evidence: Dict[str, torch.Tensor],
        adv_evidence: torch.Tensor
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if evidence modules have become mutually predictable.
        
        If yes, PAS is blind - issue warning.
        """
        data_flat = data_evidence.flatten()
        adv_flat = adv_evidence.flatten()
        
        # Standardize arch values to match data length
        n = data_flat.shape[0]
        arch_tensors = []
        for v in arch_evidence.values():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(float(v), device=data_flat.device)
            v = v.to(data_flat.device).float()
            if v.numel() == n and n > 0:
                arch_tensors.append(v.view(n))
            else:
                arch_tensors.append(v.mean().view(1).expand(max(1, n)))
                
        if arch_tensors:
            arch_flat = torch.stack(arch_tensors, dim=0).mean(dim=0).flatten()
        else:
            arch_flat = torch.zeros_like(data_flat)
        
        min_len = min(len(data_flat), len(arch_flat), len(adv_flat))
        if min_len < 2:
            return False, {'data_arch': 0.0, 'data_adv': 0.0, 'arch_adv': 0.0}
        
        data_flat = data_flat[:min_len]
        arch_flat = arch_flat[:min_len]
        adv_flat = adv_flat[:min_len]
        
        try:
            stacked = torch.stack([data_flat, arch_flat, adv_flat])
            cov_mat = torch.corrcoef(stacked)
            corr_data_arch = torch.nan_to_num(cov_mat[0, 1], nan=0.0).item()
            corr_data_adv = torch.nan_to_num(cov_mat[0, 2], nan=0.0).item()
            corr_arch_adv = torch.nan_to_num(cov_mat[1, 2], nan=0.0).item()
        except Exception:
            return False, {'data_arch': 0.0, 'data_adv': 0.0, 'arch_adv': 0.0}
        
        correlations = {
            'data_arch': float(corr_data_arch),
            'data_adv': float(corr_data_adv),
            'arch_adv': float(corr_arch_adv)
        }
        
        is_predictable = (
            abs(corr_data_arch) > self.blindness_threshold or
            abs(corr_data_adv) > self.blindness_threshold or
            abs(corr_arch_adv) > self.blindness_threshold
        )
        
        if is_predictable:
            warnings.warn(
                f"EVIDENCE BLINDNESS: Evidence modules are mutually predictable "
                f"(data-arch: {corr_data_arch:.2f}, data-adv: {corr_data_adv:.2f}, arch-adv: {corr_arch_adv:.2f}). "
                f"PAS cannot distinguish failure modes.",
                UserWarning
            )
        
        return bool(is_predictable), correlations

    def check_structural_irreducibility(self, hidden: torch.Tensor) -> Tuple[bool, int]:
        """
        Evaluates structural irreducibility across heterogeneous projections.
        Ensures rank > 1 so no single-face embedding collapses evidence.
        """
        if hidden.dim() > 2:
            hidden_flat = hidden.mean(dim=1)
        else:
            hidden_flat = hidden
        
        p_data = self.proj_data(hidden_flat)
        p_arch = self.proj_arch(hidden_flat)
        p_adv = self.proj_adv(hidden_flat)
        
        stacked = torch.stack([p_data.mean(dim=0), p_arch.mean(dim=0), p_adv.mean(dim=0)], dim=0)
        try:
            rank = int(torch.linalg.matrix_rank(stacked).item())
        except Exception:
            rank = 3
        is_irreducible = rank > 1
        return is_irreducible, rank
    
    def forward(
        self,
        hidden: torch.Tensor,
        data_signals: Optional[Dict[str, Any]] = None,
        arch_signals: Optional[Dict[str, Any]] = None,
        adv_signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute all three evidence types and check for blindness and structural irreducibility.
        """
        data_ev = self.compute_data_evidence(hidden, data_signals=data_signals)
        arch_ev = self.compute_architectural_evidence(hidden, arch_signals=arch_signals)
        adv_ev, orthogonal = self.compute_adversarial_evidence(hidden, adv_signals=adv_signals)
        
        is_predictable, corrs = self.check_mutual_predictability(data_ev, arch_ev, adv_ev)
        is_irreducible, rank = self.check_structural_irreducibility(hidden)
        
        is_blind = is_predictable or (not is_irreducible)
        
        # Generate normalized orthogonal perturbation for break-lock injection
        norm = orthogonal.norm(dim=-1, keepdim=True) + 1e-8
        adversarial_injection = orthogonal / norm
        if hidden.dim() > 2 and adversarial_injection.dim() == 2:
            adversarial_injection = adversarial_injection.unsqueeze(1).expand_as(hidden)
        
        return {
            'data_evidence': data_ev,
            'architectural_evidence': arch_ev,
            'adversarial_evidence': adv_ev,
            'is_blind': torch.tensor(is_blind, device=hidden.device),
            'blindness_detected': bool(is_blind),
            'correlations': corrs,
            'is_irreducible': torch.tensor(is_irreducible, device=hidden.device),
            'evidence_rank': rank,
            'adversarial_injection': adversarial_injection
        }
