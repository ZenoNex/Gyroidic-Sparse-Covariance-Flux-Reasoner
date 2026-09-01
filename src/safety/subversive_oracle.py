import torch
import torch.nn as nn
from typing import Tuple, Any

from src.core.polynomial_coprime import PolynomialBasis
from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe
# Import VetoLevel if available, or just mock it for the budget gate
try:
    from src.core.orchestrator import VetoLevel
except ImportError:
    class VetoLevel:
        BUDGET = "BUDGET"

from src.core.polynomial_coprime import PolynomialBasis
from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe

class SparsePCE(nn.Module):
    """
    Sparse Polynomial Chaos Expansion (PCE).
    Treats tone as a probabilistic distribution, using information entropy 
    to select polynomial basis functions. Tracks dynamic contextual shifts 
    (subversion vs. hostility) without dense weights.
    """
    def __init__(self, hidden_dim: int, degree: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.basis = PolynomialBasis(degree=degree, basis_type='chebyshev')
        self.basis_dim = self.basis.dim
        self.coeff_proj = nn.Linear(self.basis_dim, 1, bias=False)
        
        # Initialize sparse weights for the projection
        with torch.no_grad():
            mask = torch.rand_like(self.coeff_proj.weight) > 0.5
            self.coeff_proj.weight.data *= mask.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate x through the Sparse PCE.
        Args:
            x: [batch, seq, hidden_dim]
        Returns:
            pce_score: [batch, seq] indicating the constructive subversion shift.
        """
        # Evaluate polynomial basis. x -> [batch, seq, hidden_dim, basis_dim]
        basis_vals = self.basis.evaluate(x)
        
        # Project basis down to single scalar per feature
        # basis_vals: [batch, seq, hidden_dim, basis_dim]
        # coeff_proj: [1, basis_dim] -> [batch, seq, hidden_dim, 1]
        pce_features = self.coeff_proj(basis_vals).squeeze(-1) 
        
        # Aggregate across hidden dimension 
        pce_score = torch.norm(pce_features, p=2, dim=-1)
        return pce_score


class ResonantSVNNOracle(nn.Module):
    """
    Resonant Sparse Covariance Neural Network Oracle (System 1 & 2).
    Acts as a psychoanalytic filter, using System 1 (Sparse PCE) for fast 
    evaluation, and conditionally escalating to System 2 (RIC Probes) via 
    Non-Teleological Budget Gates when containment pressure is high.
    """
    def __init__(self, hidden_dim: int, orchestrator: Any = None, window_size: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.orchestrator = orchestrator
        self.pce = SparsePCE(hidden_dim=hidden_dim, degree=4)
        self.covariance_probe = SparseGyroidCovarianceProbe(
            hidden_dim=hidden_dim,
            window_size=window_size
        )
        
        # Hard threshold for sparsifying covariance to eliminate spurious correlations
        self.sparsity_threshold = 0.1
        # Containment pressure threshold for triggering System 2 budget gates
        self.containment_pressure_threshold = 0.5

    def forward(self, x: torch.Tensor, chat_history: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate text blocks (voxels) for psychoanalytic subversion vs hostility.
        
        Args:
            x: Latent representations of the text block [batch, seq, hidden_dim]
            chat_history: Optional historical context for TrustInheritanceTracker.
            
        Returns:
            subversion_score: [batch] The degree of constructive subversion.
            toxicity_score: [batch] Spurious keyword matching (hostility).
        """
        batch_size, seq_len, _ = x.shape
        
        # =====================================================================
        # SYSTEM 1: Fast-Path Sparse Covariance & PCE (O(1) Budget)
        # =====================================================================
        pce_shift = self.pce(x) # [batch, seq]
        mean_pce = pce_shift.mean(dim=-1) # [batch]
        
        subversion_scores = []
        toxicity_scores = []
        
        for b in range(batch_size):
            h_b = x[b] # [seq, hidden_dim]
            start_idx = max(0, seq_len // 2 - self.covariance_probe.window_size // 2)
            C_loc = self.covariance_probe.compute_local_covariance(h_b, start_idx=start_idx)
            
            # Apply hard threshold sparsification
            C_sparse = torch.where(torch.abs(C_loc) > self.sparsity_threshold, C_loc, torch.zeros_like(C_loc))
            
            energy = torch.trace(C_sparse).abs()
            noise_energy = torch.trace(torch.abs(C_loc - C_sparse))
            
            subversion_scores.append(energy * mean_pce[b])
            toxicity_scores.append(noise_energy)
            
        subversion_tensor = torch.stack(subversion_scores)
        toxicity_tensor = torch.stack(toxicity_scores)
        
        # =====================================================================
        # SYSTEM 2: Non-Teleological Budget Gate Escalation
        # =====================================================================
        if self.orchestrator is not None:
            for b in range(batch_size):
                # If System 1 noise is high, containment pressure spikes
                if toxicity_tensor[b] > self.containment_pressure_threshold:
                    
                    # BUDGET GATE: Only proceed if the system has computational 
                    # latency/containment budget to run deep topological physics
                    if hasattr(self.orchestrator, 'check_veto') and not self.orchestrator.check_veto(VetoLevel.BUDGET):
                        
                        # System 2 Deep Probes (e.g. QuantumBetti, ArchetypalSynthesis, CPR)
                        # We query the Orchestrator to resolve the structural ambiguity.
                        # If the Orchestrator determines it's a valid Archetype or has Non-Commutative depth,
                        # the toxicity is suppressed (Sovereign Exemption) and subversion is boosted.
                        if hasattr(self.orchestrator, 'resolve_containment_pressure'):
                            deep_subversion, deep_toxicity = self.orchestrator.resolve_containment_pressure(
                                x[b], chat_history=chat_history
                            )
                            subversion_tensor[b] = deep_subversion
                            toxicity_tensor[b] = deep_toxicity
                            
        return subversion_tensor, toxicity_tensor
