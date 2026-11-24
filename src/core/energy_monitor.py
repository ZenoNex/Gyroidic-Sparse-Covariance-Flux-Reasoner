"""
Structural Energy Monitor: Formalizing Pressure as Energy.

This module implements the mapping from the Energy-Based Models (EBM) 
framework to the Project's topological manifold.

Energy E(Y, X) <=> Structural Pressure Phi
Free Energy F <=> Topological Free Energy F_topo
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple


class StructuralEnergyMonitor(nn.Module):
    """
    Monitors the 'Energy' (Pressure) of the structural manifold.
    
    In the EBM ontology:
    - E_correct (E_C): The pressure of the current symbolic lock.
    - E_offending (E_I): The pressure of alternative, unstable configurations.
    - F_topo: The log-partition function (Free Energy) of the manifold.
    """
    
    def __init__(
        self,
        margin: float = 0.1,
        beta_init: float = 1.0,
        tau_decay: float = 10.0,
        lambda_min_epsilon: float = 1e-6,
        device: str = None
    ):
        super().__init__()
        self.margin = margin
        self.tau_decay = tau_decay
        self.lambda_min_epsilon = lambda_min_epsilon
        self.device = device
        
        # Internal state
        self.register_buffer('E_correct', torch.tensor(0.0, device=device))
        self.register_buffer('E_offending', torch.tensor(0.0, device=device))
        self.register_buffer('beta', torch.tensor(beta_init, device=device))
        self.register_buffer('free_energy', torch.tensor(0.0, device=device))

    def compute_v_m(self, V: torch.Tensor, h_mischief: float, lambda_min: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        """
        Compute formally the Mischief Violation Score (V_m).
        V_m = V + H_mischief / tau_decay - lambda_min / tr(C_loc)
        """
        flatness_penalty = lambda_min / (trace + self.lambda_min_epsilon)
        v_m = V + (h_mischief / self.tau_decay) - flatness_penalty
        return v_m

    def update(self, current_pressure: torch.Tensor, alternative_pressures: Optional[torch.Tensor] = None):
        """
        Update energy metrics based on current manifold state.
        
        Args:
            current_pressure: Scalar or [batch] pressure of the accepted state.
            alternative_pressures: [batch, K] pressures of K alternative configurations.
        """
        # E_correct is the mean pressure of the current state
        self.E_correct.copy_(current_pressure.mean().detach())
        
        if alternative_pressures is not None:
            # E_offending is the 'most offending' incorrect answer (lowest alternative pressure)
            # or the expected energy of alternatives.
            # Following EBM tutorial (Definition 1), Y_bar is the lowest energy incorrect answer.
            min_alt = alternative_pressures.min(dim=1)[0]
            self.E_offending.copy_(min_alt.mean().detach())
            
            # Compute Topological Free Energy (F_topo)
            # F = -1/beta * log(sum(exp(-beta * E_i)))
            # We treat the current state and all alternatives as the ensemble.
            combined_energies = torch.cat([current_pressure.unsqueeze(-1), alternative_pressures], dim=1)
            self.free_energy.copy_(self.compute_free_energy(combined_energies))

    def compute_free_energy(self, energies: torch.Tensor) -> torch.Tensor:
        """
        F = -1/beta * log_sum_exp(-beta * E)
        """
        # Using logsumexp for numerical stability
        # -1/beta * log(sum(exp(-beta * E)))
        scaled_energies = -self.beta * energies
        lse = torch.logsumexp(scaled_energies, dim=1)
        f_topo = - (1.0 / (self.beta + 1e-8)) * lse
        return f_topo.mean()

    def set_temperature(self, dt_ratio: float):
        """
        Link temperature/beta to the coordinate time step ratio.
        As dt shrinks (Seriousness), beta increases (Cooling).
        """
        # beta is inverse temperature. 
        # When dt_ratio = 1.0 (base), beta = beta_init.
        # When dt_ratio -> 0, beta -> infinity.
        self.beta.copy_(torch.tensor(1.0 / (dt_ratio + 1e-8), device=self.device))

    def get_gap(self) -> float:
        """Measure the Energy Gap: E_offending - E_correct."""
        return (self.E_offending - self.E_correct).item()

    def get_metrics(self) -> Dict[str, float]:
        return {
            'E_correct': self.E_correct.item(),
            'E_offending': self.E_offending.item(),
            'energy_gap': self.get_gap(),
            'free_energy': self.free_energy.item(),
            'temperature_inv': self.beta.item(),
            'thermal_runaway': float(self.check_topological_thermal_runaway())
        }

    def check_topological_thermal_runaway(self, beta_crit: float = 0.05, free_energy_crit: float = 100.0) -> bool:
        """
        Evaluates whether the topological free energy or coordinate temperature has entered thermal runaway.
        When coordinate time accelerates (dt_ratio >> 1) or energy diverges, beta drops below beta_crit.
        Broadcasts to HardwareMonitor to throttle compute affordance.
        """
        is_runaway = bool(self.beta.item() < beta_crit or abs(self.free_energy.item()) > free_energy_crit)
        if is_runaway:
            try:
                from src.core.hardware_monitor import HardwareMonitor
                p_waste_est = float(abs(self.free_energy.item()) * 10.0)
                HardwareMonitor().update_thermodynamics(
                    p_waste=p_waste_est,
                    p_cool_max=500.0,
                    thermal_runaway=True,
                    source="structural_energy_monitor"
                )
            except Exception:
                pass
        return is_runaway

    def compute_ebm_loss_functionals(
        self,
        e_correct: torch.Tensor,
        e_offending: torch.Tensor,
        all_energies: Optional[torch.Tensor] = None,
        lvq2_alpha: float = 0.25,
        sq_exp_gamma: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Implements the complete family of Contrastive Energy-Based Learning (EBM) loss functionals
        and Condition 3 margin verification from LeCun, Chopra, Hadsell, Ranzato, Huang (2006)
        (docs/.../06_reports_and_inspirations/A Tutorial on Energy-Based Learning.txt, Table 1 & Eqs. 6-27, 54-57):
          - E_C = E(W, Y^i, X^i) (correct answer energy / structural pressure)
          - E_I = E(W, Y_bar^i, X^i) (most offending incorrect answer energy, Def. 1)
          - Condition 3 (No-Collapse Margin Condition, Eq. 54): E_C + m < E_I.
        """
        ec = e_correct.float().reshape(-1)
        ei = e_offending.float().reshape(-1)
        m = float(self.margin)
        beta_val = float(self.beta.item()) if isinstance(self.beta, torch.Tensor) else float(self.beta)
        beta_safe = max(1e-6, beta_val)

        # Eq. 6: Energy loss L_energy = E_C
        l_energy = ec.mean()

        # Eq. 7: Generalized Perceptron loss L_perceptron = E_C - min_Y E(W, Y, X)
        e_min = torch.minimum(ec, ei) if all_energies is None else all_energies.float().min(dim=-1)[0].reshape(-1)
        l_perceptron = (ec - e_min).mean()

        # Eq. 11: Generalized Margin / Hinge loss L_hinge = max(0, m + E_C - E_I)
        l_hinge = torch.clamp(m + ec - ei, min=0.0).mean()

        # Eq. 12: Log loss (soft hinge) L_log = log(1 + exp(E_C - E_I))
        l_log = torch.nn.functional.softplus(ec - ei).mean()

        # Eq. 13: LVQ2 loss L_lvq2 = min(max(0, E_C - E_I), delta * E_I) where delta = lvq2_alpha
        l_lvq2 = torch.minimum(
            torch.clamp(ec - ei, min=0.0),
            float(lvq2_alpha) * torch.abs(ei)
        ).mean()

        # Eq. 15: Minimum Classification Error (MCE) loss L_mce = sigmoid(E_C - E_I)
        l_mce = torch.sigmoid(ec - ei).mean()

        # Eq. 16: Square-Square loss L_sq_sq = E_C^2 + (max(0, m - E_I))^2
        l_sq_sq = (ec.pow(2) + torch.clamp(m - ei, min=0.0).pow(2)).mean()

        # Eq. 17: Square-Exponential loss L_sq_exp = E_C^2 + gamma * exp(-E_I)
        l_sq_exp = (ec.pow(2) + float(sq_exp_gamma) * torch.exp(-torch.clamp(ei, min=-20.0, max=20.0))).mean()

        # Eq. 23 & Eq. 27: NLL / Helmholtz Free Energy loss & Minimum Empirical Error (MEE)
        if all_energies is not None:
            ens = all_energies.float()
            if ens.dim() == 1:
                ens = ens.unsqueeze(0)
        else:
            ens = torch.stack([ec, ei], dim=-1)

        # F_beta = -1/beta * log sum_y exp(-beta * E(W, y, X))
        lse = torch.logsumexp(-beta_safe * ens, dim=-1)
        f_helmholtz = -(1.0 / beta_safe) * lse
        # Eq. 23: L_nll = E_C - F_beta = E_C + (1/beta) * log sum_y exp(-beta E(y))
        l_nll = (ec - f_helmholtz).mean()

        # Eq. 27: L_mee = 1 - P(Y^i | X^i) = 1 - exp(-beta * E_C - lse)
        p_correct = torch.exp(torch.clamp(-beta_safe * ec - lse, max=0.0))
        l_mee = (1.0 - p_correct).mean()

        # Condition 3 (Eq. 54-57): E_C + m < E_I guarantees flat-energy collapse is prevented
        energy_gap = (ei - ec).mean()
        condition_3_satisfied = bool((ec + m < ei).float().mean().item() > 0.5)

        return {
            "energy_loss": float(l_energy.item()),
            "perceptron_loss": float(l_perceptron.item()),
            "hinge_loss": float(l_hinge.item()),
            "log_loss": float(l_log.item()),
            "lvq2_loss": float(l_lvq2.item()),
            "mce_loss": float(l_mce.item()),
            "square_square_loss": float(l_sq_sq.item()),
            "square_exp_loss": float(l_sq_exp.item()),
            "nll_helmholtz_loss": float(l_nll.item()),
            "mee_loss": float(l_mee.item()),
            "helmholtz_free_energy": float(f_helmholtz.mean().item()),
            "energy_gap": float(energy_gap.item()),
            "margin": m,
            "condition_3_no_collapse_satisfied": condition_3_satisfied,
            "tensors": {
                "hinge": l_hinge,
                "log": l_log,
                "lvq2": l_lvq2,
                "mce": l_mce,
                "square_square": l_sq_sq,
                "square_exp": l_sq_exp,
                "nll": l_nll,
                "mee": l_mee,
            },
        }

    def marginalize_latent_free_energy(self, joint_energies_z: torch.Tensor) -> Dict[str, Any]:
        """
        Latent-Variable EBM Redefinition & Marginalization (Tutorial Section 5, Eqs. 46-47):
          - Zero-temperature elimination: E_min(Y, X) = min_{z in Z} E(z, Y, X) (Eq. 46)
          - Finite-temperature Helmholtz free energy over latent Z:
            F_beta(Z) = -1/beta * log sum_{z in Z} exp(-beta * E(z, Y, X)) (Eq. 47)
        """
        ez = joint_energies_z.float()
        if ez.dim() == 1:
            ez = ez.unsqueeze(0)
        beta_val = float(self.beta.item()) if isinstance(self.beta, torch.Tensor) else float(self.beta)
        beta_safe = max(1e-6, beta_val)

        e_min_z, best_z_idx = torch.min(ez, dim=-1)
        f_beta_z = -(1.0 / beta_safe) * torch.logsumexp(-beta_safe * ez, dim=-1)
        latent_entropy = (e_min_z - f_beta_z) * beta_safe

        return {
            "zero_temp_min_energy": e_min_z.mean(),
            "helmholtz_latent_free_energy": f_beta_z.mean(),
            "optimal_latent_index": best_z_idx,
            "latent_entropic_advantage": float(latent_entropy.mean().item()),
        }


