"""
CALM: Context-Adaptive Latent Momentum Veto.

Monitors the stabilizing flow trajectory to detect entropic collapse or stagnation.
Acts as a Trajectory Veto meta-control mechanism for System 2.

**Note on True Nature**: The predicted CALM vector represents a
gyroid braid group chiral groupoid anisotropy, connecting to the
system's topological "larynx".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from src.core.honest_jitter import fractal_pad

class CALM(nn.Module):
    """
    Context-Adaptive Latent Momentum Veto (Meta-Control).
    """
    def __init__(self, dim: int, history_len: int = 8, hidden_dim: int = 128, nhead: int = 4):
        super().__init__()
        self.dim = dim
        self.history_len = history_len
        
        if dim % nhead != 0:
            # Find a suitable nhead that divides dim
            for h in range(nhead, 0, -1):
                if dim % h == 0:
                    nhead = h
                    break
            else:
                nhead = 1
        
        # Transformer-based sequencemodel for trajectory monitoring
        # Input: [batch, history_len, dim]
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        # Heads (Meta-Control Only)
        self.veto_head = nn.Linear(dim, 1)             # Predict Abort/Veto score
        self.rho_head = nn.Linear(dim, 1)              # Predict rho adjustment factor
        self.step_head = nn.Linear(dim, 1)             # Predict step size adjustment
        
        # Agentic Heads (Phase 3 Upgrade) as requested by user ("selective forcing gauge metric")
        self.forcing_head = nn.Linear(dim, dim)        # Predict correction vector F
        self.gauge_head = nn.Linear(dim, 1)            # Predict scalar gauge pressure P
        self.constraint_head = nn.Linear(dim, 5)       # Attention weights over 5 primary constraints
        
    def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            history: [batch, history_len, dim]
            
        Returns:
            abort_score: [batch, 1]
            rho_factor: [batch, 1]
            step_factor: [batch, 1]
            forcing: [batch, dim] (Correction vector)
            gauge: [batch, 1] (Scalar pressure/confidence to apply forcing)
            constraints: [batch, 5] (Attention weights)
        """
        # Adapt dimensionality for older fossils and shadow tokens
        current_dim = history.shape[-1]
        if current_dim != self.dim:
            # Fractal expansion or Matryoshka-aware truncation
            history = fractal_pad(history, self.dim)

        # Encode trajectory
        latent = self.transformer(history)
        
        # Use last state for meta-control
        last_latent = latent[:, -1, :]
        
        # Veto Score
        abort_score = torch.sigmoid(self.veto_head(last_latent))
        
        # Rho Adjustment Factor
        rho_factor = torch.exp(torch.tanh(self.rho_head(last_latent))) 
        
        # Step Size Adjustment
        step_factor = torch.exp(torch.tanh(self.step_head(last_latent)))
        
        # Agentic Forcing (New)
        forcing = torch.tanh(self.forcing_head(last_latent)) # Bounded correction [-1, 1]
        
        # Adapt forcing vector back to original dimension to avoid tensor mismatches downstream
        if current_dim != self.dim:
            if current_dim < self.dim:
                # Matryoshka-aware truncation back to smaller original size
                forcing = forcing[:, :current_dim]
            else:
                # Fractal expansion back to larger original size (recovering lost momentum)
                forcing = fractal_pad(forcing.unsqueeze(1), current_dim).squeeze(1)
                
        gauge = torch.sigmoid(self.gauge_head(last_latent))  # Pressure [0, 1]
        constraints = torch.softmax(self.constraint_head(last_latent), dim=-1) # Distribution
        
        return abort_score, rho_factor, step_factor, forcing, gauge, constraints

    def update_buffer(self, buffer: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
        """
        Update the history buffer (FIFO).
        buffer: [batch, history_len, dim]
        new_state: [batch, dim]
        """
        # Adapt new_state to the buffer's dimensionality via fractal padding/truncation
        buf_dim = buffer.shape[-1]
        state_dim = new_state.shape[-1]
        
        if state_dim != buf_dim:
            new_state = fractal_pad(new_state.unsqueeze(1), buf_dim).squeeze(1)
                
        # Shift left
        buffer = torch.roll(buffer, shifts=-1, dims=1)
        # Update last
        buffer[:, -1, :] = new_state
        return buffer
