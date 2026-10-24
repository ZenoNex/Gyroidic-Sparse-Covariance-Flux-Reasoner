"""
Muon (Momentum Orthogonalized by Newton-Schulz) & Ricci Flow Hybrid Optimization.

Implements matrix-aware optimization that respects the structural geometry of 2D weight
matrices via quintic Newton-Schulz polar decomposition, equalizing singular values to 1.0.
Hybridizes with Ricci Flow (Chern-Simons gasket tension) for non-commutative topological
chiral dynamics and 1D parameter stabilization.

References:
- Keller Jordan et al. (Muon Optimizer: Zeropower via Newton-Schulz quintic iteration)
- Parallax Architecture: Non-standard attention & higher effective rank representations
- Chern-Simons Gasket & Split-Beam Interfactorization (Non-teleological invariant optimization)
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Iterable, Any
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.core.invariants import PHI, PHI_RECIPROCAL
from src.core.hardware_monitor import has_headroom, has_compute_affordance, ComputeAffordanceType
from src.optimization.ricci_flow_optimizer import SplitBeamInterfactorization, RicciFlowOptimizer


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: Optional[int] = None,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration to compute the zeroth power (polar factor) of matrix G.
    Equalizes all non-zero singular values to approximately 1.0 without costly SVD,
    yielding an isometric matrix update that spreads information across internal dimensions.

    Coefficients (a=3.4445, b=-4.7750, c=2.0315) are optimized to maximize derivative at 0.

    Dynamically gates iteration count (5 steps under full headroom, 3 under constrained affordance).
    """
    if G.dim() != 2:
        raise ValueError(f"zeropower_via_newtonschulz5 requires a 2D matrix, got shape {tuple(G.shape)}")

    # Affordance-aware step adaptation
    if steps is None:
        if has_compute_affordance(ComputeAffordanceType.COMPUTE_BATCH) and has_headroom():
            steps = 5
        else:
            steps = 3

    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_dtype = G.dtype

    # Float32 or BFloat16 precision for stable quintic polynomials
    if G.is_cuda and orig_dtype in (torch.float16, torch.bfloat16):
        X = G.to(torch.bfloat16)
    else:
        X = G.float()

    norm = torch.norm(X)
    X = X / (norm + eps)

    # Transpose so that inner dimension is minimal for cheaper Gram matrix (X @ X.T)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = torch.matmul(X, X.T)
        AA = torch.matmul(A, A)
        B = b * A + c * AA
        X = a * X + torch.matmul(B, X)

    if transposed:
        X = X.T

    return X.to(dtype=orig_dtype)


class MuonOptimizer(Optimizer):
    """
    Momentum Orthogonalized by Newton-Schulz (Muon) Optimizer.

    Specialized matrix-aware optimizer for 2D weight matrices (linear layers, projections).
    Maintains first-moment momentum buffers, orthogonalizes momentum updates via Newton-Schulz,
    and applies spectral scaling max(1, sqrt(rows / cols)).
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: Optional[int] = 5,
        weight_decay: float = 0.0,
        scale_mode: str = "spectral"
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            scale_mode=scale_mode
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']
            scale_mode = group['scale_mode']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonOptimizer does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)

                state['step'] += 1
                buf = state['momentum_buffer']

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # First moment update
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update_matrix = grad.add(buf, alpha=momentum)
                else:
                    update_matrix = buf

                # Polar decomposition via Newton-Schulz iteration
                if p.dim() == 2:
                    ortho_update = zeropower_via_newtonschulz5(update_matrix, steps=ns_steps)
                    if scale_mode == "spectral":
                        # Standard Muon spectral norm calibration
                        scale = max(1.0, math.sqrt(p.size(0) / max(1, p.size(1))))
                    elif scale_mode == "frobenius":
                        scale = math.sqrt(max(p.size(0), p.size(1)))
                    else:
                        scale = 1.0
                    p.add_(ortho_update, alpha=-lr * scale)
                else:
                    # Fallback for non-2D tensors if accidentally included
                    p.add_(update_matrix, alpha=-lr)

        return loss


class MuonRicciHybridOptimizer(Optimizer):
    """
    Unifying Hybrid Optimizer: Muon + Ricci Flow.

    Architecture:
    1. 2D Weight Matrices (min_dim >= 8): Optimized via Muon (Newton-Schulz polar updates).
       Optionally coupled with Chern-Simons Gasket tension to preserve non-commutative
       chiral orientation alongside singular-value equalization.
    2. 1D Parameters / Biases / Small Weights: Optimized via Ricci Flow (Split-Beam Interfactorization)
       or Adam-style second-moment stabilization.
    """
    def __init__(
        self,
        params: Iterable[Union[torch.nn.Parameter, Dict[str, Any]]],
        lr: float = 0.02,
        lr_1d: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: Optional[int] = 5,
        weight_decay: float = 0.0,
        seam_width: float = 0.1,
        fuse_chern_simons: bool = True,
        engine: Optional[Any] = None,
        min_2d_dim: int = 8
    ):
        self.fuse_chern_simons = fuse_chern_simons
        self.split_beam = SplitBeamInterfactorization(engine=engine)
        self.min_2d_dim = min_2d_dim

        # Categorize parameters into Muon 2D group and Ricci 1D group
        param_list = list(params)
        muon_params = []
        ricci_params = []

        for item in param_list:
            if isinstance(item, dict):
                p_items = item.get('params', [])
            else:
                p_items = [item]

            for p in p_items:
                if not isinstance(p, torch.Tensor) or not p.requires_grad:
                    continue
                if p.dim() == 2 and min(p.shape) >= self.min_2d_dim:
                    muon_params.append(p)
                else:
                    ricci_params.append(p)

        groups = [
            {
                'name': 'muon_2d',
                'params': muon_params,
                'lr': lr,
                'momentum': momentum,
                'nesterov': nesterov,
                'ns_steps': ns_steps,
                'weight_decay': weight_decay,
                'seam_width': seam_width,
                'is_muon': True
            },
            {
                'name': 'ricci_1d',
                'params': ricci_params,
                'lr': lr_1d,
                'momentum': 0.9,
                'beta2': 0.999,
                'eps': 1e-8,
                'weight_decay': weight_decay * 0.1,
                'seam_width': seam_width,
                'is_muon': False
            }
        ]

        defaults = dict(
            lr=lr,
            lr_1d=lr_1d,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            seam_width=seam_width
        )
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            is_muon = group.get('is_muon', False)
            lr = group['lr']
            weight_decay = group.get('weight_decay', 0.0)
            seam_width = group.get('seam_width', 0.1)

            if is_muon:
                momentum = group['momentum']
                nesterov = group['nesterov']
                ns_steps = group['ns_steps']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['momentum_buffer'] = torch.zeros_like(p)

                    state['step'] += 1
                    buf = state['momentum_buffer']

                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)

                    # Non-Commutative Chern-Simons Tension fusion
                    if self.fuse_chern_simons:
                        if p.shape[-1] == p.shape[-2]:
                            flux_cov = p
                        else:
                            flux_cov = torch.matmul(p, p.transpose(-1, -2))
                        seam_tension = self.split_beam.compute_chern_simons_tension(flux_cov, seam_width)
                        if p.shape[-1] == p.shape[-2]:
                            twist = torch.diag_embed(seam_tension)
                        else:
                            twist = seam_tension.unsqueeze(-1).expand_as(p)
                        # Add topological twist to raw gradient pressure
                        effective_grad = grad + twist
                    else:
                        effective_grad = grad

                    # First moment accumulation
                    buf.mul_(momentum).add_(effective_grad)

                    if nesterov:
                        u = effective_grad.add(buf, alpha=momentum)
                    else:
                        u = buf

                    # Newton-Schulz polar update
                    ortho_update = zeropower_via_newtonschulz5(u, steps=ns_steps)
                    scale = max(1.0, math.sqrt(p.size(0) / max(1, p.size(1))))
                    p.add_(ortho_update, alpha=-lr * scale)

            else:
                # Ricci 1D / Fallback AdamW branch for biases and normalization layers
                beta1 = group.get('momentum', 0.9)
                beta2 = group.get('beta2', 0.999)
                eps = group.get('eps', 1e-8)

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']

                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)

                    # Topological 1D tension
                    if p.dim() == 1:
                        flux_cov = torch.outer(p, p)
                        tension = self.split_beam.compute_chern_simons_tension(flux_cov, seam_width)
                        effective_grad = grad + tension
                    else:
                        effective_grad = grad

                    exp_avg.mul_(beta1).add_(effective_grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(effective_grad, effective_grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
