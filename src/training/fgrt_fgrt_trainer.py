
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
from src.core.invariants import PhaseAlignmentInvariant

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
            state_dim=64 # Assuming default dimension
        )
        
        # 2. System 2 Probe: SIC-FA-ADMM
        self.system2_probe = SicFaAdmmSolver(
            dim=64, # state_dim
            max_iters=50,
            admissibility_threshold=spectral_threshold
        )
        
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
        
        # 5. Compute Non-Teleological Energy (Willmore + Curvature)
        # Willmore energy preserves 'smoothness', Curvature preserves 'roughness'.
        # We compute this on THE PROPOSAL to train the model to be smooth.
        energy = self.willmore(proposal) + recon_loss
        
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
            from src.core.birkhoff_projection import project_to_birkhoff
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
