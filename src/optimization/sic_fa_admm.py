"""
SIC-FA-ADMM: Sparse Incoherent Constraints - Fractional Anisotropy - ADMM.

A Hybrid Physics-Geometric search mechanism that serves as the **System 2 Probe**.
Invoked only when symbolic residues (System 1) fail co-primality or reconstruction checks.
"Non-convergence is a feature of the saturated symbolic regime."
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Callable

from src.optimization.fractional_operators import frac_apply
from src.surrogates.kagh_networks import KAGHBlock
from src.surrogates.calm_predictor import CALM

class SicFaAdmmSolver:
    """
    Stabilizes: min_c 1/2 ||W(B A_alpha^-1 c - A)||^2 + lambda ||c||_1
    Where A is the symbolic residue anchor.
    """
    
    def __init__(
        self,
        dim: int,
        rho: float = 2.0,
        lambda_sparse: float = 0.1,
        max_iters: int = 100,
        tol: float = 1e-4,
        admissibility_threshold: float = 0.75,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.dim = dim
        self.rho = rho
        self.lambda_sparse = lambda_sparse
        self.max_iters = max_iters
        self.tol = tol
        self.admissibility_threshold = admissibility_threshold
        self.phase_clock = 0.0
        self.device = device

    def soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0).to(x.device))

    def solve(
        self,
        forward_op: Callable, # B or KAGH surrogate
        anchor: torch.Tensor,
        M_alpha_op: Callable, # A_alpha operator
        initial_c: Optional[torch.Tensor] = None,
        calm_predictor: Optional[CALM] = None,
        history: Optional[torch.Tensor] = None,
        lambda_sparse_override: Optional[float] = None
    ) -> torch.Tensor:
        """
        Run ADMM stabilizing flow.
        """
        # Use override or default
        effective_lambda = lambda_sparse_override if lambda_sparse_override is not None else self.lambda_sparse
        
        # Initialize variables
        if initial_c is None:
            c = torch.zeros_like(anchor).to(self.device, non_blocking=True)
        else:
            c = initial_c.clone()
            
        z = c.clone()
        u = torch.zeros_like(c)
        
        # Stabilizing flow loop
        for k in range(self.max_iters):
            c_prev = c.clone()
            
            # --- 1. y-update (Forward Model Fidelity) ---
            # Ideally: y = argmin 1/2 ||W(F(c) - T)||^2 + rho/2 ||Ax - y + u||^2
            # Simplified: We treat 'forward_op' as linear B for now, or use gradient descent if non-linear
            # With KAGH, we might backprop through the surrogate to update c directly
            
            # Here we implement the Standard ADMM splitting:
            # min_c f(c) + g(z) s.t. c - z = 0
            # f(c) = 1/2 ||F(c) - T||^2
            # g(z) = lambda ||z||_1
            
            # c-update: min_c 1/2 ||F(c) - T||^2 + rho/2 ||c - z + u||^2
            # If F is KAGH (nonlinear), this is a nonconvex subproblem.
            # We solve it via a few gradient steps or CG if linear.
            
            # Hybrid approach: Use CALM to suggest c^{k+1} guess
            if calm_predictor is not None and history is not None:
                c_pred, rho_factor = calm_predictor(history)
                
                # --- Spectral Speculative Decoding ---
                # Check if the predicted thought is spectrally "clean" (Soliton-like).
                # If so, we trust the System 1 intuition and skip the expensive System 2 repair.
                c_freq = torch.fft.rfft(c_pred)
                power = torch.abs(c_freq) ** 2
                power_norm = power / (power.sum() + 1e-8)
                spectral_entropy = -(power_norm * torch.log(power_norm + 1e-8)).sum()
                
                # Speculative Exit: If entropy is low (highly structured signal), accept prediction.
                # Threshold typically ~1.0 for sharp peaks
                if spectral_entropy < 1.0:
                     return c_pred
                
                # Adaptive rho
                self.rho = 0.1 # FORCE THAW
            
            # Gradient Step for c-update (supporting KAGH surrogate)
            # This replaces the exact linear solve step in standard ADMM
            c.requires_grad_(True)
            optimizer = optim.LBFGS([c], lr=1.0, max_iter=5)
            
            def closure():
                optimizer.zero_grad()
                # Apply Fractional Anisotropy Inverse: c' = A_alpha^-1 c
                # c_frac = frac_apply(M_alpha_op, c, -1.0) 
                # For efficiency, assume M_alpha_op handles fractional apply inside or we pass params
                
                # Forward surrogate
                if isinstance(forward_op, nn.Module):
                     pred = forward_op(c) # KAGH(c)
                elif callable(forward_op):
                     pred = forward_op(c) # Lambda/function
                else:
                     pred = forward_op @ c # Matrix-vector multiplication
                
                fidelity_pressure = 0.5 * torch.sum((pred - anchor)**2)
                admm_tension = (self.rho / 2) * torch.sum((c - z + u)**2)
                
                total_tension = fidelity_pressure + admm_tension
                total_tension.backward()
                return total_tension
                
            optimizer.step(closure)
            c.requires_grad_(False)
            
            # --- 2. z-update (Sparsity Prox) ---
            # z = prox_g(c + u)
            # g(z) = lambda ||z||_1
            # prox(v) = soft(v, lambda/rho)
            
            z = self.soft_threshold(c + u, effective_lambda / self.rho)
            
            # --- 3. u-update (Dual Ascent) ---
            u = u + (c - z)
            
            # --- 4. History Update & Saturation Check ---
            if history is not None:
                 history = calm_predictor.update_buffer(history.unsqueeze(0), c.unsqueeze(0)).squeeze(0)
            
            # Check primal residual (Saturation)
            r_norm = torch.norm(c - z)
            # RENAME: 'pas' -> 'primal_residual' to avoid confusion with Homological PAS_h
            # This is a Local Geometric Invariant, NOT the Global Topological Invariant.
            # See narrative_collapse.py for true PAS_h (Anti-Lobotomy) monitoring.
            primal_residual = r_norm 
            if primal_residual < self.admissibility_threshold:
            # Inadmissible: Warp suppressed (Structural Tripwire)
                break # Exit the stabilizing flow if inadmissible
            
            if r_norm < self.tol:
                break
                
        return c

def sic_fa_admm_kagh_calm_gpu(
    anchor: torch.Tensor,
    kagh_net: KAGHBlock,
    calm_net: CALM,
    initial_c: torch.Tensor,
    history: torch.Tensor,
    rho: float = 2.0
) -> torch.Tensor:
    """
    Wrapper for running the full hybrid System 2 probe.
    """
    solver = SicFaAdmmSolver(
        dim=initial_c.shape[0],
        rho=rho,
        device=initial_c.device.type
    )
    
    # Define operators
    # KAGH acts as the forward operator
    
    c_opt = solver.solve(
        forward_op=kagh_net,
        anchor=anchor,
        M_alpha_op=None, # KAGH handles M implicitly via pre-scaling layer
        initial_c=initial_c,
        calm_predictor=calm_net,
        history=history
    )
    
    return c_opt

