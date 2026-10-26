"""
Local Actuation QP Operator (P_QP)

Encapsulates quadratic programming and convex actuation constraints as a 
Local Constraint Probe Operator:
    P_QP: r -> argmin_{c in C_QP} L_QP(r, c)
    subject to u_min <= c <= u_max, A c <= b

Adheres strictly to the Four Non-Negotiable Laws and Implementation Integrity:
1. Feasibility Probe Only: Evaluates whether physical actuation bounds are satisfied.
2. Discrete Failure Token Emission: On infeasibility or rupture, emits a discrete FailureToken (bot).
3. Atemporal / Non-Teleological: Internal cost parameters evolve via Ricci Flow or 
   Evolutionary Trust Selection, NEVER through end-to-end backpropagation.
4. Silent Failure Contract: Exposes no intermediate gradients or near-miss scalars to System 1.
"""

import math
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.optimization.constraint_probe import ConstraintProbeOperator
from src.core.failure_token import FailureToken, FailureTokenType
from src.core.honest_jitter import harvest_honest_jitter


class LocalActuationQPOperator(ConstraintProbeOperator):
    """
    Local Constraint Probe Operator for Quadratic Programming (P_QP).
    Solves local physical control actuation feasibility without global optimization goals.
    """

    def __init__(
        self,
        constraint_index: int = 0,
        actuation_dim: int = 4,
        u_min: Optional[torch.Tensor] = None,
        u_max: Optional[torch.Tensor] = None,
        A_ineq: Optional[torch.Tensor] = None,
        b_ineq: Optional[torch.Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        dev = torch.device(device) if device is not None else torch.device("cpu")

        # Initialize positive-definite sparse covariance Sigma_k from honest hardware jitter
        w_init = harvest_honest_jitter((actuation_dim, actuation_dim), device=dev) * 0.05
        sigma_k = w_init @ w_init.t() + torch.eye(actuation_dim, device=dev) * 0.1

        super().__init__(
            constraint_index=constraint_index,
            sparse_covariance=sigma_k,
            device=dev,
        )

        self.actuation_dim = actuation_dim

        # Control surface / actuation bounds
        if u_min is None:
            u_min = torch.full((actuation_dim,), -1.0, device=dev)
        if u_max is None:
            u_max = torch.full((actuation_dim,), 1.0, device=dev)

        self.register_buffer("u_min", u_min.to(dev))
        self.register_buffer("u_max", u_max.to(dev))

        # Optional linear inequality constraints: A_ineq @ c <= b_ineq
        if A_ineq is not None and b_ineq is not None:
            self.register_buffer("A_ineq", A_ineq.to(dev))
            self.register_buffer("b_ineq", b_ineq.to(dev))
            self.has_ineq = True
        else:
            self.register_buffer("A_ineq", torch.empty((0, actuation_dim), device=dev))
            self.register_buffer("b_ineq", torch.empty((0,), device=dev))
            self.has_ineq = False

        # Internal QP cost matrix Q and linear bias q.
        # These evolve exclusively via Ricci Flow or Evolutionary Trust Selection.
        self.register_buffer("Q", sigma_k.clone())
        self.register_buffer("q_bias", torch.zeros(actuation_dim, device=dev))
        self.register_buffer("trust_scalar", torch.tensor(1.0, device=dev))
        self.register_buffer("consecutive_failures", torch.tensor(0, dtype=torch.long, device=dev))

    def compute_gyroid_admissibility(self, c: torch.Tensor) -> torch.Tensor:
        """
        Evaluates Gyroid zero-level set violation on first 3 coordinates:
        psi(c) = |sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)|
        """
        if c.shape[-1] < 3:
            return torch.zeros(c.shape[0] if c.dim() > 1 else 1, device=c.device)

        cx = c[..., 0]
        cy = c[..., 1]
        cz = c[..., 2]
        g_val = torch.sin(cx) * torch.cos(cy) + torch.sin(cy) * torch.cos(cz) + torch.sin(cz) * torch.cos(cx)
        return torch.abs(g_val)

    def solve_local_feasibility(
        self,
        residue: torch.Tensor,
        max_iters: int = 20,
        rho_relax: float = 0.08,
        tolerance: float = 1e-4,
    ) -> Tuple[torch.Tensor, Optional[FailureToken], Dict[str, Any]]:
        """
        Solves the local QP feasibility problem:
            c* = argmin_{u_min <= c <= u_max} 0.5 c^T Q c + c^T (q_bias - Phi(r)) + psi_gyroid(c) + ineq_penalty
        
        Returns:
            optimal_c: Actuation tensor satisfying constraints
            failure_token: None if feasible; FailureToken if ruptured or infeasible
            diagnostics: Numeric metrics
        """
        phi_r = self.embedding_fn(residue).detach()
        is_single = False
        if phi_r.dim() == 1:
            phi_r = phi_r.unsqueeze(0)
            is_single = True

        batch_size = phi_r.shape[0]
        # Match dimensions if needed
        if phi_r.shape[-1] != self.actuation_dim:
            if phi_r.shape[-1] < self.actuation_dim:
                pad = torch.zeros(batch_size, self.actuation_dim - phi_r.shape[-1], device=phi_r.device)
                phi_r = torch.cat([phi_r, pad], dim=-1)
            else:
                phi_r = phi_r[..., :self.actuation_dim]

        # Warm start projected onto box bounds
        c = torch.clamp(phi_r.clone(), self.u_min, self.u_max)
        converged = False
        final_iter = 0
        step_norm = 0.0

        with torch.enable_grad():
            c = c.detach().requires_grad_(True)
            for it in range(max_iters):
                # 1. Quadratic cost: 0.5 c^T Q c + c^T (q_bias - phi_r)
                quad_term = 0.5 * torch.sum(c * (c @ self.Q), dim=-1)
                lin_term = torch.sum(c * (self.q_bias.unsqueeze(0) - phi_r), dim=-1)

                # 2. Gyroid admissibility filter
                gyroid_viol = self.compute_gyroid_admissibility(c)

                # 3. Linear inequalities penalty: relu(A c - b)
                ineq_viol = torch.zeros_like(quad_term)
                if self.has_ineq and self.A_ineq.shape[0] > 0:
                    ineq_slack = torch.matmul(c, self.A_ineq.t()) - self.b_ineq.unsqueeze(0)
                    ineq_viol = torch.sum(F.relu(ineq_slack) ** 2, dim=-1)

                total_loss = quad_term + lin_term + gyroid_viol + 10.0 * ineq_viol

                grad = torch.autograd.grad(total_loss.sum(), c, create_graph=False)[0]

                c_next = torch.clamp(c.detach() - rho_relax * grad, self.u_min, self.u_max)
                step_norm = torch.norm(c_next - c.detach(), dim=-1).max().item()
                c = c_next.clone().requires_grad_(True)
                final_iter = it + 1
                if step_norm < tolerance:
                    converged = True
                    break

        with torch.no_grad():
            final_gyroid = self.compute_gyroid_admissibility(c).mean().item()
            final_ineq = 0.0
            if self.has_ineq and self.A_ineq.shape[0] > 0:
                final_ineq = F.relu(torch.matmul(c, self.A_ineq.t()) - self.b_ineq.unsqueeze(0)).max().item()

            # Local feasibility check: non-divergent step, physical box bounds, inequality satisfaction, and gyroid admissibility
            is_feasible = (
                bool(torch.isfinite(c).all())
                and (final_gyroid < 1.5)
                and (final_ineq < 1e-2)
                and (converged or step_norm < 0.2)
            )

            diagnostics = {
                "converged": converged,
                "iterations": final_iter,
                "gyroid_violation": final_gyroid,
                "ineq_violation": final_ineq,
                "is_feasible": is_feasible,
                "trust_scalar": float(self.trust_scalar.item()),
            }

            if not is_feasible:
                self.consecutive_failures += 1
                token = FailureToken(
                    FailureTokenType.BOT,
                    constraint_index=self.k,
                    residue=phi_r.squeeze(0) if is_single else phi_r,
                )
                res_c = phi_r.squeeze(0) if is_single else phi_r
                return res_c, token, diagnostics

            self.consecutive_failures.zero_()
            res_c = c.detach().squeeze(0) if is_single else c.detach()
            return res_c, None, diagnostics

    def forward(
        self,
        residue: torch.Tensor,
        lambda_k: Optional[torch.Tensor] = None,
        initial_constraint: Optional[torch.Tensor] = None,
        max_probe_iters: int = 20,
        probe_step_size: float = 0.08,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ConstraintProbeOperator interface compliance:
        Returns optimal constraint state c and local strain loss.
        """
        c_opt, failure_token, _ = self.solve_local_feasibility(
            residue, max_iters=max_probe_iters, rho_relax=probe_step_size
        )

        with torch.no_grad():
            strain = self.compute_local_strain(residue, c_opt)
            if failure_token is not None:
                strain = strain + 100.0
        return c_opt, strain

    def evolve_parameters_ricci_flow(self, curvature_tensor: torch.Tensor, dt: float = 0.01):
        """
        Evolves internal QP cost parameters via geometric Ricci Flow:
            dQ/dt = -2 * Ric(Q)
        Preserves symmetry and positive semi-definiteness on the manifold.
        """
        with torch.no_grad():
            curv = curvature_tensor.to(self.Q.device)
            if curv.shape != self.Q.shape:
                if curv.numel() == 1:
                    curv = torch.eye(self.actuation_dim, device=self.Q.device) * curv.item()
                else:
                    curv = torch.eye(self.actuation_dim, device=self.Q.device) * curv.mean().item()

            ricci_step = -2.0 * curv * dt
            self.Q.add_(ricci_step)

            # Symmetrize
            self.Q.copy_((self.Q + self.Q.t()) * 0.5)

            # Eigenvalue clamping to ensure strictly positive definite cone (Q >= 0.01 I)
            evals, evecs = torch.linalg.eigh(self.Q)
            evals_clamped = torch.clamp(evals, min=0.01)
            self.Q.copy_(evecs @ torch.diag(evals_clamped) @ evecs.t())

    def evolve_trust_selection(self, survivorship_streak: int, survival_threshold: int = 5, decay: float = 0.98):
        """
        Non-teleological evolutionary trust adaptation.
        Expands or contracts trust scalars based on survival streaks.
        """
        with torch.no_grad():
            if survivorship_streak >= survival_threshold:
                self.trust_scalar.mul_(1.02)
            else:
                self.trust_scalar.mul_(decay)
            self.trust_scalar.clamp_(min=0.1, max=10.0)
