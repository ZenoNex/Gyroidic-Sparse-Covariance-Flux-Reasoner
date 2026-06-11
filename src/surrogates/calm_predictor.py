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
from typing import Tuple, List, Dict, Optional, Union
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

    def functional_forward(self, history: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute forward pass. If params is provided, temporarily load them to perform
        the forward pass, then restore the original parameters.
        """
        if params is None:
            return self.forward(history)
            
        orig_params = {k: v.data.clone() for k, v in self.named_parameters()}
        try:
            for name, param in self.named_parameters():
                if name in params:
                    param.data.copy_(params[name].data)
            return self.forward(history)
        finally:
            for name, param in self.named_parameters():
                if name in orig_params:
                    param.data.copy_(orig_params[name])

    def adapt(self, support_history: torch.Tensor, support_targets: Union[torch.Tensor, Dict[str, torch.Tensor]], steps: int = 1, lr: float = 0.01, entropy: Optional[torch.Tensor] = None) -> 'CALM':
        """
        Perform online inner-loop adaptation (MAML step) on support data.
        Returns an adapted instance of CALM.
        
        Dynamic LR: scaled by (1.0 + entropy.mean()) if entropy is provided.
        """
        def clone_module(module):
            import copy
            clone = copy.copy(module)
            clone._parameters = {}
            for k, v in module._parameters.items():
                if v is not None:
                    clone._parameters[k] = nn.Parameter(v.clone(), requires_grad=v.requires_grad)
                else:
                    clone._parameters[k] = None
            clone._buffers = {k: v.clone() if v is not None else None for k, v in module._buffers.items()}
            clone._modules = {}
            for k, v in module._modules.items():
                if v is not None:
                    clone._modules[k] = clone_module(v)
                else:
                    clone._modules[k] = None
            return clone

        adapted_model = clone_module(self)
        
        effective_lr = lr
        if entropy is not None:
            entropy_val = entropy.mean().item()
            effective_lr = lr * (1.0 + abs(entropy_val))
            
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=effective_lr)
        
        for p in adapted_model.parameters():
            p.requires_grad_(True)
            
        adapted_model.train()
        for step in range(steps):
            optimizer.zero_grad()
            abort_score, rho_factor, step_factor, forcing, gauge, constraints = adapted_model(support_history)
            
            loss = torch.tensor(0.0, device=support_history.device)
            if isinstance(support_targets, dict):
                if 'forcing' in support_targets:
                    t_forcing = support_targets['forcing']
                    if t_forcing.shape != forcing.shape:
                        t_forcing = t_forcing.view(forcing.shape)
                    loss = loss + F.mse_loss(forcing, t_forcing)
                if 'abort_score' in support_targets:
                    t_abort = support_targets['abort_score']
                    if t_abort.shape != abort_score.shape:
                        t_abort = t_abort.view(abort_score.shape)
                    loss = loss + F.mse_loss(abort_score, t_abort)
                if 'rho_factor' in support_targets:
                    t_rho = support_targets['rho_factor']
                    if t_rho.shape != rho_factor.shape:
                        t_rho = t_rho.view(rho_factor.shape)
                    loss = loss + F.mse_loss(rho_factor, t_rho)
                if 'step_factor' in support_targets:
                    t_step = support_targets['step_factor']
                    if t_step.shape != step_factor.shape:
                        t_step = t_step.view(step_factor.shape)
                    loss = loss + F.mse_loss(step_factor, t_step)
                if 'gauge' in support_targets:
                    t_gauge = support_targets['gauge']
                    if t_gauge.shape != gauge.shape:
                        t_gauge = t_gauge.view(gauge.shape)
                    loss = loss + F.mse_loss(gauge, t_gauge)
                if 'constraints' in support_targets:
                    t_constraints = support_targets['constraints']
                    if t_constraints.shape != constraints.shape:
                        t_constraints = t_constraints.view(constraints.shape)
                    loss = loss + F.mse_loss(constraints, t_constraints)
            else:
                t_forcing = support_targets
                if t_forcing.shape != forcing.shape:
                    try:
                        t_forcing = t_forcing.view(forcing.shape)
                    except Exception:
                        pass
                if t_forcing.shape[-1] == forcing.shape[-1]:
                    loss = loss + F.mse_loss(forcing, t_forcing)
                else:
                    loss = loss + F.mse_loss(forcing[:, :t_forcing.shape[-1]], t_forcing)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                
        adapted_model.eval()
        return adapted_model
