
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

from src.optimization.ricci_flow_optimizer import RicciFlowOptimizer, WillmoreEnergy
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
        torsion_weight: float = 0.1,
        spectral_threshold: float = 1.0
    ):
        self.model = model
        self.config = poly_config
        self.optimizer = RicciFlowOptimizer(
            model.parameters(), 
            lr=lr, 
            torsion_weight=torsion_weight
        )
        self.willmore = WillmoreEnergy()
        self.phase_tracker = BerryPhaseTracker()
        self.gyroid = GyroidManifold()
        self.pas_metric = PhaseAlignmentInvariant(degree=poly_config.degree)
        
        # 1. Polynomial ADMR for state reconciliation
        self.admr = PolynomialADMRSolver(
            poly_config=poly_config,
            state_dim=256 
        )
        
        # 2. System 2 Probe: SIC-FA-ADMM
        self.system2_probe = SicFaAdmmSolver(
            dim=256, # state_dim
            max_iters=50,
            admissibility_threshold=spectral_threshold
        )
        
        # 3. Formal System 2 Constraints (RIC/CODES)
        # These are used for calculating the formal Survivorship Pressure (6.3)
        self.ric = ResonanceCavity(hidden_dim=256, num_modes=64, poly_config=poly_config)
        self.codes = CODESConstraintFramework(state_dim=256)
        
        # Seed CODES with standard formal constraints
        self.codes.add_constraint(0, constraint_type='quadratic')
        self.codes.add_constraint(1, constraint_type='harmonic')
        self.codes.add_constraint(2, constraint_type='polynomial_coprime')
        
        self.register_buffer('prev_output', None, persistent=False)

    def train_step(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Performs a non-teleological training step with spectral gating."""
        self.optimizer.zero_grad()
        
        # 1. System 1: Heuristic Proposal
        output = self.model(input_data)
        
        # 2. Spectral Speculative Check
        # Does the proposal exhibit 'Soliton' structure?
        output_freq = torch.fft.rfft(output)
        power = torch.abs(output_freq) ** 2
        power_norm = power / (power.sum(dim=-1, keepdim=True) + 1e-8)
        spectral_entropy = -(power_norm * torch.log(power_norm + 1e-8)).sum(dim=-1).mean()
        
        # --- Group Relative Sparsity (GRS) Logic ---
        # Calculate batch-relative sparsity baseline to keep encodings efficient
        l1_norms = torch.norm(output, p=1, dim=-1)
        batch_avg_l1 = l1_norms.mean().item() + 1e-8
        
        # 3. Decision Logic: Trust System 1 or Invoke System 2?
        proposal = output
        if spectral_entropy > self.system2_probe.admissibility_threshold:
            # System 2: Geometric Repair (SIC-FA-ADMM)
            # Use GRS to dampen or amplify sparsity pressure based on batch density
            # If current thought is much denser than average, increase lambda
            lambda_grs = self.system2_probe.lambda_sparse * (l1_norms.mean() / batch_avg_l1)
            
            repaired_output = self.system2_probe.solve(
                forward_op=lambda x: x,
                anchor=output.detach(),
                M_alpha_op=None,
                lambda_sparse_override=lambda_grs.item()
            )
            # Add reconciliation pressure: model should have predicted the repair
            recon_loss = F.mse_loss(proposal, repaired_output.detach())
            output = repaired_output # Following metrics use repaired state
        else:
            recon_loss = torch.tensor(0.0, device=output.device)
        
        # 4. Compute Invariants
        pas_h = self.pas_metric(output.unsqueeze(1) if output.dim() == 2 else output).mean().item()
        
        # 5. Compute formal Survivorship Pressure (6.3 TAT Unified)
        # Survivorship_Pressure = Association_Inaccuracy +   (1.0 - Coherence) -   Mischief
        # - Association_Inaccuracy (recon_loss): Pressure to find the correct manifold.
        # - Coherence Penalty (1.0 - coherence): Pressure to maintain temporal stability.
        # - Mischief Reward (mischief): Reward for novel topological exploration (15.2).
        
        alpha_coh = 0.1
        beta_mischief = 0.05
        
        # Formal Coherence via Resonance Cavity (RIC)
        # We query the cavity to see how much the proposal resonates with known patterns.
        resonance_data = self.ric.query(proposal)
        coherence = resonance_data['resonance_scores'].mean()
        
        # Formal Mischief via High-Order Resonance (15.2)
        # Mischief is the measure of novelty relative to the resonant baseline.
        # It's high when the proposal is structurally valid but 'surprising' to the RIC.
        mischief = (1.0 - coherence) * spectral_entropy
        
        # Formal Survivorship Pressure ensuring non-negative bias for accuracy/coherence
        # and a reward for 'Play' (mischief).
        survivorship_pressure = recon_loss + alpha_coh * (1.0 - coherence) - beta_mischief * mischief
        
        # Formal Constrainment Energy via CODES
        # This replaces the Willmore heuristic for the local manifold curvature.
        formal_energy = self.codes.compute_total_energy(proposal).mean()
        
        # Total Non-Teleological Energy
        energy = formal_energy + survivorship_pressure
        
        # 6. Topological Curvature Modulation
        # f_topo = f * (1 + gamma * K)
        if proposal.shape[-1] >= 3:
            k_gaussian = self.gyroid.gaussian_curvature(proposal[..., :3])
            # High negative curvature (holes) increases the local "functional potential"
            # preventing the system from flattening the topological features.
            curvature_pressure = torch.mean(torch.abs(k_gaussian) * proposal.pow(2).mean(dim=-1))
            energy = energy + 0.1 * curvature_pressure
            violation = self.gyroid(output[..., :3]).mean()
        else:
            violation = torch.tensor(0.0)
            
        # 7. Backward Pass & Ricci Step
        # Ricci Flow: g_{ij}(t+1) = g_{ij}(t) - 2 * R_{ij}
        # In our case, we use the energy gradient to 'warp' the parameters.
        energy.backward()
        self.optimizer.step()

        # --- MANDATORY BIRKHOFF MANIFOLD PROJECTION ---
        with torch.no_grad():
            for p in self.model.parameters():
                if p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.copy_(project_to_birkhoff(p.data))
        # --- END BIRKHOFF PROJECTION ---
        
        # 8. Tracker updates
        if self.prev_output is not None and self.prev_output.shape == output.shape:
             self.phase_tracker.update(self.prev_output, output)
        self.prev_output = output.detach()
        
        return {
            "willmore_energy": energy.item(),
            "spectral_entropy": spectral_entropy.item(),
            "pas_h": pas_h,
            "gyroid_violation": violation.item(),
            "berry_phase": self.phase_tracker.running_phase.item()
        }

    def register_buffer(self, name, tensor, persistent=True):
        """Helper for non-parameter buffers."""
        self.model.register_buffer(name, tensor, persistent=persistent)
        setattr(self, name, tensor)
