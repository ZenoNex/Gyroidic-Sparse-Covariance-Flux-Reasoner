
#Spectral Structural Trainer: Non-Teleological Optimization via Spectral Coherence.

#Integrates Ricci Flow, Polynomial ADMR, and SIC-FA-ADMM into a spectral 
#stabilization loop. 


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.optimization.ricci_flow_optimizer import RicciFlowOptimizer, BouligandWillmoreGasket
from src.core.fgrt_primitives import GyroidManifold, BerryPhaseTracker
from src.optimization.sic_fa_admm import SicFaAdmmSolver
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.admr_solver import PolynomialADMRSolver
from src.core.codes_constraint_framework import CODESConstraintFramework
from src.models.resonance_cavity import ResonanceCavity
from src.core.invariants import PhaseAlignmentInvariant
from src.core.birkhoff_projection import project_to_birkhoff

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))



class SpectralStructuralTrainer:
    """
    Trainer that uses Spectral Speculative Decoding and Ricci Flow to align
    states with the polynomial manifold.
    """
    def __init__(
        self,
        model: nn.Module,
        poly_config: PolynomialCoprimeConfig,
        lr: float = 1e-4,
        seam_width: float = 0.1,
        spectral_threshold: float = 1.0
    ):
        self.model = model
        self.config = poly_config
        self.optimizer = RicciFlowOptimizer(
            model.parameters(), 
            lr=lr, 
            seam_width=seam_width
        )
        self.willmore = BouligandWillmoreGasket()
        self.phase_tracker = BerryPhaseTracker()
        self.gyroid = GyroidManifold()
        self.pas_metric = PhaseAlignmentInvariant(degree=poly_config.degree)
        
        # 1. Polynomial ADMR for state reconciliation
        self.admr = PolynomialADMRSolver(
            poly_config=poly_config,
            state_dim=model.dim 
        )
        
        # 2. System 2 Probe: SIC-FA-ADMM
        self.system2_probe = SicFaAdmmSolver(
            dim=model.dim, # state_dim
            max_iters=50,
            admissibility_threshold=spectral_threshold
        )
        
        # 3. Formal System 2 Constraints (RIC/CODES)
        # These are used for calculating the formal Survivorship Pressure (6.3)
        self.ric = ResonanceCavity(hidden_dim=model.dim, num_modes=64, poly_config=poly_config)
        self.codes = CODESConstraintFramework(state_dim=model.dim)
        
        # Seed CODES with standard formal constraints
        self.codes.add_constraint(0, constraint_type='quadratic')
        self.codes.add_constraint(1, constraint_type='harmonic')
        self.codes.add_constraint(2, constraint_type='polynomial_coprime')
        
      
        self.prev_output = None

    def train_step(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Performs a non-teleological training step with spectral gating.

        Architecture follows PHYSICS_ADMM.md §2.1 Cyclic Constraint Traversal:
            For each constraint k: P_k: r -> argmin_{c in C_k} L_k(r, c)
        Each probe is isolated — no cross-domain gradient contamination.

        Mischief is a local strain tolerance modifier (NonDualProbe §5.1),
        NOT a negative loss term. It gates how tightly coherence is enforced.
        This is the anti-scalarization mandated by INVARIANT_OPTIMIZATION.md Tripwire 3.
        """

        # ------------------------------------------------------------------ #
        # 1. System 1: Heuristic Proposal                                     #
        # ------------------------------------------------------------------ #
        # Forward pass — zero_grad happens per-probe below
        self.optimizer.zero_grad()
        output = self.model(input_data)

        # ------------------------------------------------------------------ #
        # 2. Spectral Speculative Check (Wager #4, EFFICIENCY doc)            #
        # Does the proposal exhibit Soliton structure?                         #
        # ------------------------------------------------------------------ #
        output_freq = torch.fft.rfft(output)
        power = torch.abs(output_freq) ** 2
        power_norm = power / (power.sum(dim=-1, keepdim=True) + 1e-8)
        spectral_entropy = -(power_norm * torch.log(power_norm + 1e-8)).sum(dim=-1).mean()

        # Group Relative Sparsity baseline
        l1_norms = torch.norm(output, p=1, dim=-1)
        batch_avg_l1 = l1_norms.mean().item() + 1e-8

        # ------------------------------------------------------------------ #
        # 3. System 2 Gate: Trust System 1 or invoke geometric repair         #
        # ------------------------------------------------------------------ #
        proposal = output
        if spectral_entropy > self.system2_probe.admissibility_threshold:
            lambda_grs = self.system2_probe.lambda_sparse * (l1_norms.mean() / batch_avg_l1)
            repaired_output = self.system2_probe.solve(
                forward_op=lambda x: x,
                anchor=output.detach(),
                M_alpha_op=None,
                lambda_sparse_override=lambda_grs.item()
            )
            recon_loss = F.mse_loss(proposal, repaired_output.detach())
            output = repaired_output
        else:
            recon_loss = torch.tensor(0.0, device=output.device, requires_grad=False)

        # ------------------------------------------------------------------ #
        # 4. Compute Invariants (read-only diagnostics, no grad)              #
        # ------------------------------------------------------------------ #
        pas_h = self.pas_metric(
            output.unsqueeze(1) if output.dim() == 2 else output
        ).mean().item()

        # Mischief: measure of novelty relative to RIC resonant baseline.
        # Per NonDualProbe §5.1: mischief is a TOLERANCE MODIFIER, not a loss.
        # High mischief -> loosen coherence enforcement; low mischief -> tighten.
        # It is NOT subtracted from a scalar aggregate — that would be scalarization.
        with torch.no_grad():
            resonance_data = self.ric.query(proposal.detach())
            coherence_scalar = resonance_data['resonance_scores'].mean().item()
            mischief_tolerance = (1.0 - coherence_scalar) * spectral_entropy.item()
        # beta_mischief governs how much mischief expands the coherence tolerance
        beta_mischief = 0.05
        alpha_coh = 0.1

        # ------------------------------------------------------------------ #
        # 5. CYCLIC CONSTRAINT PROBES — each probe is a sovereign domain      #
        #                                                                      #
        # Probe ordering follows PHYSICS_ADMM.md §2.2:                        #
        #   k=0: Reconstruction (association inaccuracy)                       #
        #   k=1: Coherence (NonDualProbe — mischief gates tolerance)          #
        #   k=2: CODES formal constrainment energy                             #
        #   k=3: Topological curvature (optional)                              #
        #                                                                      #
        # Each probe uses a detached anchor for its internal computation.      #
        # Gradients flow only through the probe's own forward—no cross-leak.  #
        # ------------------------------------------------------------------ #

        probe_log = {}  # For diagnostics

        # --- Probe k=0: Reconstruction / Association Inaccuracy ---
        # P_0: r -> argmin_{c in C_0} |output - repaired|^2
        # Enforces that System 1 output stays close to the repair anchor.
        if recon_loss.requires_grad:
            self.optimizer.zero_grad()
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        probe_log['recon_loss'] = recon_loss.item()

        # Birkhoff projection after each probe step (mandatory per GDPO pattern)
        with torch.no_grad():
            for p in self.model.parameters():
                if p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.copy_(project_to_birkhoff(p.data))

        # --- Probe k=1: Coherence — gated by mischief tolerance (NonDualProbe) ---
        # Per PHYSICS_ADMM.md §5.1: penalty = max(strain - beta * H_mischief, 0)
        # This makes mischief a *strain tolerance* gate, not a scalar reward to subtract.
        # Re-query coherence with fresh graph on current proposal (not detached)
        self.optimizer.zero_grad()
        
        # Fresh forward pass on updated parameters to avoid in-place graph conflict
        with torch.enable_grad():
            new_proposal = self.model(input_data)
            
        resonance_data_live = self.ric.query(new_proposal)
        coherence_live = resonance_data_live['resonance_scores'].mean()
        raw_coherence_strain = alpha_coh * (1.0 - coherence_live)
        # Apply mischief tolerance gate: allow more strain if system is mischievous
        coherence_probe_pressure = torch.clamp(
            raw_coherence_strain - beta_mischief * mischief_tolerance, min=0.0
        )
        if coherence_probe_pressure.requires_grad:
            coherence_probe_pressure.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
        # Birkhoff projection after each probe step (mandatory per GDPO pattern)
        with torch.no_grad():
            for p in self.model.parameters():
                if p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.copy_(project_to_birkhoff(p.data))
