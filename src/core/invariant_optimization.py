"""
Invariant Optimization and Lexicographical Hierarchy.

This module enforces "The Semiotic Hierarchy" — a mathematical logic 
shielding System 2 by assigning Invariant Admissibility as the primary index,
and System 1's heuristic speed (the "Horse") as the secondary index. 

A global optimizer cannot sacrifice a System 2 invariant to gain a System 1 
performance boost, just as you cannot change the first letter of a word to 
make it a 'better' word.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SemioticState:
    """Represents a composed evaluation of a state."""
    system_2_admissibility: float  # The 'First Letter'. Lower is better (0 = perfectly admissible)
    system_1_heuristic: float      # The 'Second Letter'. Higher is better (fast heuristic)
    state_tensor: torch.Tensor
    
class LexicographicalOrderingDispatcher:
    """
    Enforces the Dictionary Order for state acceptance.
    System 2 Invariant Admissibility strictly dominates System 1 heuristic performance.
    """
    
    def __init__(self, eps: float = 1e-5):
        self.eps = eps # Epsilon for checking strict equality of the primary index
        
    def compare(self, state_a: SemioticState, state_b: SemioticState) -> int:
        """
        Returns:
            -1 if state_a is strictly preferred over state_b
             1 if state_b is strictly preferred over state_a
             0 if indistinguishable
        """
        # Primary Index: System 2 Admissibility (lower represents better adherence to invariants)
        diff = state_a.system_2_admissibility - state_b.system_2_admissibility
        
        if diff < -self.eps:
            return -1 # A is more admissible
        elif diff > self.eps:
            return 1  # B is more admissible
            
        # If the primary index is identical (within eps), System 1 heuristic speed is evaluated
        h_diff = state_a.system_1_heuristic - state_b.system_1_heuristic
        
        if h_diff > self.eps:
            return -1 # A is faster/better heuristically
        elif h_diff < -self.eps:
            return 1  # B is faster
            
        return 0

    def select_best(self, proposals: List[SemioticState]) -> SemioticState:
        """
        Filters a list of proposed states, returning the strictly best one
        according to the Lexicographical Hierarchy.
        """
        if not proposals:
            raise ValueError("No proposals provided.")
            
        best = proposals[0]
        for p in proposals[1:]:
            if self.compare(p, best) < 0:
                best = p
                
        return best
