"""
Legibility Audit: Detects when structures are favored for explainability.

Rich-club attractor detection: If a configuration has high "narrative coherence"
relative to its structural merit, it may be a legibility trap.

Pointer #2: Hidden Scalar Reward = Narrative Legibility

Author: Implementation from Sparse Operational Pointers
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import warnings


class NarrativeCoherenceEstimator(nn.Module):
    """
    Estimates how "explainable" a configuration is.
    
    High narrative coherence = configuration can be easily described/defended.
    This is a DANGER SIGNAL, not a goal.
    """
    
    def __init__(self, hidden_dim: int, num_narrative_patterns: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable "narrative templates" - common explainable patterns
        # These are NOT trained - they capture typical ML biases
        self.register_buffer(
            'narrative_templates',
            self._init_narrative_templates(hidden_dim, num_narrative_patterns)
        )
    
    def _init_narrative_templates(self, dim: int, n: int) -> torch.Tensor:
        """
        Initialize with common "explainable" patterns:
        - Sparse (few active dimensions)
        - Clustered (clear groupings)
        - Monotonic (ordered relationships)
        """
        templates = []
        
        # Sparse patterns (1-hot like)
        for i in range(n // 4):
            t = torch.zeros(dim)
            t[i % dim] = 1.0
            templates.append(t)
        
        # Block-sparse patterns
        block_size = dim // 4
        for i in range(n // 4):
            t = torch.zeros(dim)
            start = (i * block_size) % dim
            end = min(start + block_size, dim)
            t[start:end] = 1.0 / max(1, end - start)
            templates.append(t)
        
        # Smooth gradient patterns
        for i in range(n // 4):
            t = torch.linspace(0, 1, dim)
            t = t.roll(i * (dim // max(1, n // 4)))
            templates.append(t / (t.sum() + 1e-8))
        
        # Random (baseline)
        while len(templates) < n:
            t = torch.randn(dim)
            templates.append(t / (t.norm() + 1e-8))
        
        return torch.stack(templates)  # [n, dim]
    
    def forward(self, config_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute narrative coherence score.
        
        Args:
            config_embedding: [batch, hidden_dim] configuration embedding
            
        Returns:
            coherence: [batch] narrative coherence (HIGHER = MORE SUSPICIOUS)
        """
        # Normalize
        config_norm = config_embedding / (config_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Similarity to narrative templates
        similarities = torch.mm(config_norm, self.narrative_templates.t())  # [batch, n]
        
        # Max similarity = how well this matches ANY narrative template
        max_sim, _ = similarities.max(dim=-1)  # [batch]
        
        return max_sim


class LegibilityTripwire(nn.Module):
    """
    Tripwire: Warns when selection correlates with explainability.
    
    This is a MONITORING module, not an enforcer.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        warning_threshold: float = 0.7,
        correlation_window: int = 100
    ):
        super().__init__()
        self.coherence_estimator = NarrativeCoherenceEstimator(hidden_dim)
        self.warning_threshold = warning_threshold
        self.correlation_window = correlation_window
        
        # Track correlation between selection and coherence
        self.register_buffer('selection_history', torch.zeros(correlation_window))
        self.register_buffer('coherence_history', torch.zeros(correlation_window))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
    
    def forward(
        self,
        selected_embeddings: torch.Tensor,
        rejected_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Check if selection correlates with narrative coherence.
        
        Args:
            selected_embeddings: [n_selected, hidden_dim] survived configurations
            rejected_embeddings: [n_rejected, hidden_dim] rejected configurations
            
        Returns:
            Dict with 'warning' (bool) and 'correlation' (float)
        """
        device = selected_embeddings.device if selected_embeddings.numel() > 0 else rejected_embeddings.device
        
        # Compute coherence for both groups
        if selected_embeddings.shape[0] > 0:
            selected_coherence = self.coherence_estimator(selected_embeddings).mean()
        else:
            selected_coherence = torch.tensor(0.0, device=device)
            
        if rejected_embeddings.shape[0] > 0:
            rejected_coherence = self.coherence_estimator(rejected_embeddings).mean()
        else:
            rejected_coherence = torch.tensor(0.0, device=device)
        
        # Update history
        ptr = self.history_ptr.item()
        window = self.selection_history.shape[0]
        
        self.selection_history[ptr % window] = 1.0  # Selected
        self.coherence_history[ptr % window] = selected_coherence.detach()
        self.history_ptr += 1
        
        # Compute correlation
        if self.history_ptr >= window:
            stacked = torch.stack([self.selection_history, self.coherence_history])
            correlation = torch.corrcoef(stacked)[0, 1]
            if torch.isnan(correlation):
                correlation = torch.tensor(0.0, device=device)
        else:
            correlation = torch.tensor(0.0, device=device)
        
        # Warning condition: High coherence gap AND/OR correlation
        coherence_gap = selected_coherence - rejected_coherence
        is_warning = (coherence_gap > self.warning_threshold) or (correlation.abs() > 0.5)
        
        if is_warning:
            warnings.warn(
                f"LEGIBILITY TRIPWIRE: Selected configs have {coherence_gap:.2f} higher "
                f"narrative coherence. Correlation with selection: {correlation:.2f}. "
                f"This may indicate rich-club attractor bias.",
                UserWarning
            )
        
        return {
            'warning': torch.tensor(is_warning, device=device),
            'coherence_gap': coherence_gap,
            'correlation': correlation,
            'selected_coherence': selected_coherence,
            'rejected_coherence': rejected_coherence
        }
    
    def reset(self):
        """Reset tracking history."""
        self.selection_history.zero_()
        self.coherence_history.zero_()
        self.history_ptr.zero_()
