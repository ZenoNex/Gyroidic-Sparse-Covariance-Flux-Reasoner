"""
Situational Batching: Co-arising and Attachment without Symmetry.

Implements batches as "Situationships" (L_ij) where relational indices
are grouped based on their historical entanglement/conflict scores.

Indices (i, j) are relationally distinct but not ontologically sealed.
"""

import torch
from torch.utils.data import Sampler
from typing import List, Iterator, Optional, Dict
import numpy as np


class SituationalBatchSampler(Sampler[List[int]]):
    """
    Samples batches based on an evolving Relational Entanglement Matrix.
    
    Instead of i.i.d. sampling, we follow the "scars" of interaction.
    If indices i and j have high conflict (pressure), they are entangled.
    
    Logic:
    1. Pick seed index i.
    2. Sample neighbors j from the entanglement graph L_ij.
    3. Include occasional "Play" samples (random) to prevent collapse.
    """
    
    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        play_ratio: float = 0.2,
        contrastive_ratio: float = 0.2,
        decay: float = 0.99,
        boundary_threshold: float = 0.5,
        device: str = None
    ):
        """
        Args:
            num_samples: Total number of items in dataset
            batch_size: Number of items per batch
            play_ratio: Fraction of batch that is sampled randomly (non-entangled)
            contrastive_ratio: Fraction of batch that is 'offending' (high current pressure potential)
            decay: Rate at which entanglement scars fade
            boundary_threshold: T_ij for paradoxical refusal
        """
        super().__init__(None)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.play_ratio = play_ratio
        self.contrastive_ratio = contrastive_ratio
        self.decay = decay
        self.boundary_threshold = boundary_threshold
        self.device = device
        
        # 1. Resonance Matrix R_ij: "Oscillation frequency" coherence
        self.R = torch.zeros((num_samples, num_samples), device=device)
        
        # 2. Mischief Matrix M_ij: Chaotic/Playful affinity
        self.M = torch.zeros((num_samples, num_samples), device=device)
        
        # 3. Offending Potential O_i (Legacy L)
        self.O = torch.zeros(num_samples, device=device)

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generates batches based on Resonance and Mischief.
        """
        seeds = torch.randperm(self.num_samples).tolist()
        consumed = set()
        
        for i in seeds:
            if i in consumed:
                continue
                
            batch = [i]
            consumed.add(i)
            
            # 1. Mischievous & Resonant Selection (Co-arising)
            num_coupled = self.batch_size - int(self.batch_size * self.play_ratio) - 1
            if num_coupled > 0:
                # Combined score: R_ij + M_ij
                coupling = (self.R[i] + self.M[i])
                mask = torch.ones_like(coupling)
                mask[list(consumed)] = 0.0
                
                # Softmax over coupling (Wattsian play)
                effective_probs = torch.softmax(coupling * 5.0, dim=0) * mask
                if effective_probs.sum() > 0:
                    probs = effective_probs / effective_probs.sum()
                    neighbors = torch.multinomial(probs, min(num_coupled, int(mask.sum().item())), replacement=False).tolist()
                    batch.extend(neighbors)
                    for n in neighbors:
                        consumed.add(n)
            
            # 2. Playful Sampling (Non-dual exploration)
            num_play = self.batch_size - len(batch)
            if num_play > 0:
                remaining = list(set(range(self.num_samples)) - consumed)
                if remaining:
                    idx_to_sample = min(num_play, len(remaining))
                    play_samples = np.random.choice(remaining, idx_to_sample, replace=False).tolist()
                    batch.extend(play_samples)
                    for p in play_samples:
                        consumed.add(p)
            
            if len(batch) > 0:
                yield batch

    def update_love_invariant(self, indices: List[int], pressure: torch.Tensor, mischief_scores: torch.Tensor):
        """
        Update R_ij, M_ij, and O_i.
        
        Indices are relationally distinct (situationship) but not ontologically sealed.
        """
        self.R *= self.decay
        self.M *= self.decay
        self.O *= self.decay
        
        # Broadcast scalar pressure if necessary
        p_val = pressure if pressure.dim() > 0 else pressure.unsqueeze(0).repeat(len(indices))
        m_val = mischief_scores if mischief_scores.dim() > 0 else mischief_scores.unsqueeze(0).repeat(len(indices))

        for i, idx_a in enumerate(indices):
            self.O[idx_a] += p_val[i].item()
            
            for j, idx_b in enumerate(indices):
                if i != j:
                    # 1. Resonant Attraction (Pu): Co-emergent coupling
                    res = (p_val[i] * p_val[j]).sqrt().item()
                    
                    # 2. Refusal as Affirmation (Li-Cri-Anton)
                    # Paradoxically amplify resonance if boundary B_ij is met
                    # We treat high local pressure as a boundary signal
                    b_ij = (p_val[i] + p_val[j]).item() / 2.0
                    f_ij = 1.5 if b_ij > self.boundary_threshold else 1.0
                    
                    self.R[idx_a, idx_b] += res * f_ij
                    
                    # 3. Mischievous Affinity (Sa): Chaotic surprise
                    self.M[idx_a, idx_b] += m_val[i].item() + m_val[j].item()

    def update_pusafiliacrimonto(self, indices: List[int], pressure: torch.Tensor, mischief_scores: torch.Tensor):
        """Deprecated alias for update_love_invariant."""
        return self.update_love_invariant(indices, pressure, mischief_scores)

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def get_summary(self) -> Dict[str, float]:
        return {
            'max_resonance': self.R.max().item(),
            'max_mischief': self.M.max().item(),
            'mean_offending': self.O.mean().item()
        }

