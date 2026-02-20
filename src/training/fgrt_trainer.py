"""
FGRT Structural Trainer: Non-Teleological Optimization.

Integrates Ricci Flow and FGRT manifolds into the training loop.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from src.optimization.ricci_flow_optimizer import RicciFlowOptimizer, WillmoreEnergy
from src.core.fgrt_primitives import GyroidManifold, BerryPhaseTracker
from src.core.gluing_operator import GluingOperator
from src.core.orchestrator import UniversalOrchestrator
from src.core.unknowledge_flux import NostalgicLeakFunctional
from src.core.invariants import PhaseAlignmentInvariant, ImplicationInvariant
from src.topology.gyroid_differentiation import GyroidFlowConstraint, ForbiddenSmoothingChecker

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class FGRTStructuralTrainer:
    """
    Trainer that uses Ricci Flow and Manifold alignment instead of standard SGD.
    """
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        torsion_weight: float = 0.1,
        gluing_dim: int = 4
    ):
        self.model = model
        self.optimizer = RicciFlowOptimizer(
            model.parameters(), 
            lr=lr, 
            torsion_weight=torsion_weight
        )
        self.willmore = WillmoreEnergy()
        try:
            self.gluer = GluingOperator(gluing_dim)
        except:
            # Fallback if dim mismatch
            self.gluer = None
        self.phase_tracker = BerryPhaseTracker()
        
        # 3. Universal Orchestrator & Dynamics
        self.orchestrator = UniversalOrchestrator(dim=gluing_dim)
        self.leak = NostalgicLeakFunctional(fossil_dim=gluing_dim)
        self.implication_guard = ImplicationInvariant()
        
        # Gyroidic Constraints
        self.flow_constraint = GyroidFlowConstraint()
        self.smoothing_checker = ForbiddenSmoothingChecker()
        self.pas_metric = PhaseAlignmentInvariant(degree=3) # Consistent with poly_degree
        
        # 4. State History (For Berry Phase & Coherence)
        self.prev_output = None

    def train_step(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Performs a non-teleological training step."""
        # 1. Forward Pass (Initial Heuristic Proposal)
        with torch.enable_grad():
            output = self.model(input_data)
            # Ensure it has functional dimensions [B, K, D] for PAS_h
            # If standard embedding, we might need a mapping.
            # Assuming model outputs [B, hidden] or [B, K*D]
            if output.dim() == 2:
                # Attempt to view as [B, K, D] for invariant checks
                try:
                    coeffs = output.view(output.shape[0], -1, 4) # Assuming D=4
                except:
                    coeffs = output.unsqueeze(1) # Fallback
            else:
                coeffs = output
            
            # 2. Compute Real Invariants (No Placeholders)
            # PAS_h measures the harmonic synchronization
            pas_h_tensor = self.pas_metric(coeffs)
            pas_h = pas_h_tensor.mean().item()
            
            # 3. Compute Real Spectral Coherence
            # We measure alignment across functional groups
            # normalized: [B, K, D]
            normalized = coeffs / (torch.norm(coeffs, dim=-1, keepdim=True) + 1e-8)
            # mean alignment across K: [B, D]
            coherence_vec = normalized.mean(dim=1)
            # scalar coherence: [B]
            coherence = torch.norm(coherence_vec, dim=-1)
            
            # 4. Compute Real Pressure Gradient (No Proxy)
            # We use the gradient of the Willmore energy w.r.t the state
            energy_tmp = self.willmore(output)
            pressure_grad = torch.autograd.grad(energy_tmp, output, retain_graph=True)[0]

        # 5. Universal Orchestration (Logical Primitives & Asymptotics)
        orchestrated_state, regime, routing = self.orchestrator(
            state=output,
            pressure_grad=pressure_grad,
            pas_h=pas_h,
            coherence=coherence
        )
        
        # orchestrated_state is the 'Topological Twist' (Glued state)
        # We use it to 'see' chirality and calculate invariants.
        output = orchestrated_state
        
        # 3. Apply Nostalgic Leak (ψ_l)
        # Preserves 'unknowledge' as high-frequency solitons
        leak_signal = self.leak(output)
        output = output + 0.1 * leak_signal
        
        # 4. Compute Non-Teleological Energy (Willmore)
        energy = self.willmore(output)
        
        # 5. Compute Gyroid Flow Violation
        # We check if the flow (w.r.t residue=input) matches Gyroid geometry
        # Note: If output is not 3D, GyroidFlowConstraint handles padding/truncation internally
        flow_stats = self.flow_constraint(input_data, output)
        if not flow_stats['is_satisfied'].all():
             # Add penalty for violation
             # penalty ~ |dot_product|
             violation = flow_stats['dot_product'].abs().mean()
             energy = energy + 1.0 * violation
        
        # Check Forbidden Smoothing (Periodically or always if cheap)
        if self.prev_output is not None and self.prev_output.shape == output.shape:
            # Check smooth path from prev to current
            # This is expensive, so maybe we only do it if violation is high or randomly?
            # For now, we compute it but don't hard-block, just add to energy
            smoothing_stats = self.smoothing_checker(
                input_data, input_data, # Residues (dummy same for now)
                self.prev_output, output
            )
            if smoothing_stats['is_forbidden'].any():
                 energy = energy + 0.5 * smoothing_stats['is_forbidden'].float().mean()
            
        # 6. Backward Pass
        energy.backward()
        
        # 7. Anti-Lobotomy Check (Implication Invariant)
        # Check if internal interaction produces a significant implication
        if input_data.requires_grad:
            # gradient of orchestrated state w.r.t input
            # We check if 'phi' primitive preserved the agency
            violation_imp, preservation_score = self.implication_guard(input_data, output)
            if violation_imp.any():
                # "Interaction without Implication is a lobotomy."
                # We could zero out the gradient or add a penalty
                energy = energy + 10.0 * violation_imp.mean()
        
        # 8. Update Berry Phase with Real State Transition
        if self.prev_output is not None and self.prev_output.shape == output.shape:
            self.phase_tracker.update(self.prev_output, output)
        self.prev_output = output.detach()
        
        # 8. Step Ricci Flow
        self.optimizer.step()

        # --- MANDATORY BIRKHOFF MANIFOLD PROJECTION ---
        with torch.no_grad():
            from src.core.birkhoff_projection import project_to_birkhoff
            for p in self.model.parameters():
                if p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.copy_(project_to_birkhoff(p.data))
        # --- END BIRKHOFF PROJECTION ---
        with torch.no_grad():
            from src.core.birkhoff_projection import project_to_birkhoff
            for p in self.model.parameters():
                if p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.copy_(project_to_birkhoff(p.data))

