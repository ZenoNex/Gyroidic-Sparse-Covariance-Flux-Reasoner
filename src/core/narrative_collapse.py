"""
Linguistic Entropy Monitor (formerly Narrative Collapse Detector).

Role: Monitors the "Smoothness" of the reasoning chain.
Specifically checks for "Hallucination Loops" where:
1. Entropy drops (Confidence implies certainty).
2. Trajectory becomes linear/predictable (Narrative Closure).

Integration:
- Feeds into SpeculativeHomologyEngine.
- If Entropy < Threshold, it flags "Artificial Smoothing" (Draft Rejection).

Author: Implementation from Sparse Operational Pointers
Created: January 2026
Refactored: January 2026 (Anti-Lobotomy)
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List
import warnings

from src.topology.persistence_obstruction import ResidueObstructionGraph

class LinguisticEntropyMonitor(nn.Module):
    """
    Monitors entropy and trajectory to detect artificial smoothing/hallucination.
    """
    
    def __init__(
        self,
        num_pressure_domains: int = 2,
        alignment_threshold: float = 0.8,
        prediction_threshold: float = 0.99,
        entropy_threshold: float = 0.01,
        lobotomy_threshold: float = 0.5,
        device: str = None
    ):
        super().__init__()
        self.num_domains = num_pressure_domains
        self.alignment_threshold = alignment_threshold
        self.prediction_threshold = prediction_threshold
        self.entropy_threshold = entropy_threshold
        self.lobotomy_threshold = lobotomy_threshold
        
        # Constitutional Alignment: Homological Monitor (PAS_h)
        self.pas_monitor = ResidueObstructionGraph(
            num_epsilon_samples=10, 
            max_dimension=1
        )
        self.last_betti_count = 0.0

    def detect_homological_collapse(
        self, 
        residues: List[torch.Tensor], 
        manifold: torch.Tensor, 
        current_gradient_norm: float
    ) -> Tuple[bool, float]:
        """
        Constitutional Alignment: Detects if Betti complexity drops significantly
        under low gradient pressure (indicating artificial smoothing or lobotomy).
        """
        import networkx as nx
        # 1. Build graph using self.pas_monitor
        G = self.pas_monitor.build_graph(residues, manifold)
        
        # 2. Calculate current Betti count
        # Betti = E - V + C (for a graph)
        num_nodes = G.number_of_nodes()
        if num_nodes > 0:
            try:
                num_components = nx.number_connected_components(G)
            except Exception:
                num_components = 1
            current_betti = float(G.number_of_edges() - num_nodes + num_components)
        else:
            current_betti = 0.0
            
        # 3. Calculate delta (complexity drop)
        delta = float(self.last_betti_count - current_betti)
        
        # 4. Update last Betti count
        self.last_betti_count = current_betti
        
        # 5. Check for lobotomy
        is_lobotomized = False
        if delta > 0.0 and current_gradient_norm < self.lobotomy_threshold:
            is_lobotomized = True
            warnings.warn("CONSTITUTIONAL ALARM: Homological Collapse (Lobotomy) detected!", UserWarning)
            
        return is_lobotomized, delta

    def calculate_entropy(self, hidden_state: torch.Tensor) -> float:
        """
        Estimate entropy of the hidden state (via Softmax approximation).
        """
        # Assume hidden state represents logits or distribution
        probs = torch.softmax(hidden_state, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        return float(entropy)

    def detect_smoothing(
        self,
        current_state: torch.Tensor,
        recent_states: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Detects if the state trajectory is becoming "Too Smooth" (Linear).
        """
        if recent_states.shape[0] < 3:
            return False, 0.0
        
        # Trajectory linearity check
        delta1 = recent_states[-1] - recent_states[-2]
        delta2 = recent_states[-2] - recent_states[-3]
        
        # Cosine sim between consecutive steps
        smoothness = torch.nn.functional.cosine_similarity(delta1, delta2, dim=0).mean().item()
        
        return smoothness > self.prediction_threshold, smoothness

    def forward(
        self,
        current_state: torch.Tensor,
        recent_states: Optional[torch.Tensor] = None,
        constraints: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Monitor Step.
        """
        results = {}
        
        # 1. Entropy Check
        entropy = self.calculate_entropy(current_state)
        results['entropy'] = torch.tensor(entropy, device=current_state.device)
        if entropy < self.entropy_threshold:
            # Low entropy warning
            results['smoothing_warning'] = torch.tensor(True, device=current_state.device)
        else:
            results['smoothing_warning'] = torch.tensor(False, device=current_state.device)
            
        # 2. Trajectory Smoothness
        if recent_states is not None:
            is_smooth, smoothness_score = self.detect_smoothing(current_state, recent_states)
            # Corroborate
            results['is_linear'] = torch.tensor(is_smooth, device=current_state.device)
            results['smoothness_score'] = torch.tensor(smoothness_score, device=current_state.device)
            
        return results


# Backward compatibility alias
NarrativeCollapseDetector = LinguisticEntropyMonitor
