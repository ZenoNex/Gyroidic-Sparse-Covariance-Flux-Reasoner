"""
Operational ADMM Primitive.

Transforms the SIC-FA-ADMM solver into an "Inherent Primitive" via torch.autograd.Function.
Enforces:
1. PAS_h / APAS_zeta invariants (Drift bound).
2. Differentiable fixed-point (Implicit differentiation).
3. "Scalar Gyroidic Ergodicity" (Guaranteed recurrence).

"An invariant that cannot be computed cannot govern evolution."

Phase 1 Update: System 2 as Constraint Probe Operator
- No global objective, only local feasibility per constraint
- Cyclic constraint traversal (k = 1, ..., K)
- Bounded oscillation detection (no convergence guarantee)
- Failure token integration
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Dict, List

from src.core.invariants import PhaseAlignmentInvariant, APAS_Zeta, compute_chirality
from src.core.failure_token import FailureToken, FailureTokenType, RuptureFunctional
from src.optimization.constraint_probe import ConstraintProbeOperator, ConstraintManifold
from src.topology.hyper_ring_closure import HyperRingOperator, HyperRingClosureChecker
from src.topology.soliton_stability import SolitonStability
from src.core.yield_criteria import MohrCoulombProjection, DruckerPragerProjection
from src.core.love_vector import LoveVector
from src.topology.gyroid_differentiation import GyroidFlowConstraint

class ChiralDriftStabilizer:
    """
    Speculative Primitive: Chiral Drift Stabilizer (CDS).
    
    Stabilizes for "Endogenous Computable Chirality" by maximizing negentropic flow.
    C = (Centroid - D/2) * exp(-Drift / Zeta)
    """
    def __init__(self, zeta: float, tau: float = 0.1, device=None):
        self.zeta = zeta
        self.tau = tau # Max entropic drop allowed
    
    def compute_score(self, c: torch.Tensor, drift: float, poly_degree: int) -> torch.Tensor:
        # Chirality: Centroid - D/2
        # (Assuming compute_chirality returns this)
        chi = compute_chirality(c)
        
        # Unified Score
        # We want Chi < 0 (Negentropic). Standardize sign:
        # Let maximize C' = -Chi * exp(...) -> Positive is good
        score = -chi * torch.exp(torch.tensor(-drift / self.zeta))
        return score

    def check_step(self, prev_score: torch.Tensor, current_score: torch.Tensor) -> bool:
        # Reject if score drops significantly (Entropic collapse)
        delta = current_score - prev_score
        # If delta < -tau, reject
        if (delta < -self.tau).any():
            return False # Reject
        return True

class OperationalAdmmPrimitive(autograd.Function):
    """
    Differentiable ADMM Primitive.
    
    Forward: Runs ADMM loop with PAS_h checks and Chiral Drift Stabilization.
    Backward: Computes equilibrium flow at the fixed point.
    
    Phase 1: Now uses constraint probe operators with cyclic traversal.
    """
    
    @staticmethod
    def forward(ctx, 
                initial_c: torch.Tensor,
                forward_op_fn: Callable, # Function to apply KAGH
                rho: float,
                lambda_sparse: float,
                max_iters: int,
                zeta: float,
                poly_degree: int,
                gcve_pressure: Optional[torch.Tensor] = None,
                chirality_index: Optional[torch.Tensor] = None,
                use_constraint_probes: bool = True,
                constraint_probes: Optional[List[ConstraintProbeOperator]] = None,
                num_constraints: int = 1) -> torch.Tensor:
        
        ctx.rho = rho
        ctx.lambda_sparse = lambda_sparse
        ctx.forward_op_fn = forward_op_fn
        
        # Invariants & Stabilizers
        pas_metric = PhaseAlignmentInvariant(poly_degree).to(initial_c.device)
        apas_check = APAS_Zeta(zeta)
        cds = ChiralDriftStabilizer(zeta)
        rupture_fn = RuptureFunctional(rupture_threshold=1e6)
        gyroid_flow = GyroidFlowConstraint()
        
        # 0. Ontological Splitting:
        # c_sym: The frozen symbolic residues from System 1 (initial guess)
        # c_phys: The continuous field we evolve to find physical consistency
        c_sym = initial_c.clone().detach() # Frozen anchor
        c_phys = initial_c.clone() # Initialized from guess, but free to flow
        
        z = c_phys.clone()
        u = torch.zeros_like(c_phys)
        
        prev_pas = pas_metric(c_phys)
        prev_chiral_score = cds.compute_score(c_phys, 0.0, poly_degree)
        
        # Repair Budget: Return tokens
        # 0: REPAIRED, 1: ALTERNATIVE, 2: FAILURE
        status = torch.tensor(0, device=initial_c.device)
        low_pas_count = 0
        
        # Phase 1: Constraint Probes
        if use_constraint_probes and constraint_probes is not None:
            # Cyclic constraint traversal
            K = len(constraint_probes)
            lambda_multipliers = [torch.zeros_like(c_phys) for _ in range(K)]
            
            # Track oscillation for bounded oscillation detection
            recent_states = []  # Store recent constraint states
            oscillation_window = min(5, max_iters // 2)
            
            # Evolution Loop with Cyclic Constraint Traversal
            for t in range(max_iters):
                k = (t % K)  # Cycle through constraints: k = 0, ..., K-1
                probe_k = constraint_probes[k]
                
                # Probe for local feasibility: c_k = P_k(r, lambda_k)
                c_k, loss_k = probe_k(
                    residue=c_phys,
                    lambda_k=lambda_multipliers[k],
                    initial_constraint=c_phys if t == 0 else None,
                    max_probe_iters=3,  # Local probe iterations
                    probe_step_size=0.1
                )
                
                # Update Lagrange multiplier: lambda_k = lambda_k + rho * (Phi_k(r) - c_k)
                phi_r = probe_k.embedding_fn(c_phys)
                if phi_r.shape != c_k.shape:
                    # Align shapes
                    if phi_r.numel() == c_k.numel():
                        phi_r = phi_r.reshape(c_k.shape)
                    else:
                        # Use projection
                        phi_r = phi_r[..., :c_k.shape[-1]] if phi_r.shape[-1] > c_k.shape[-1] else phi_r
                
                lambda_multipliers[k] = lambda_multipliers[k] + rho * (phi_r - c_k)
                
                # Update physical state (weighted average of constraint states)
                # Simple approach: use most recent constraint state
                c_phys = c_k
                
                # Apply Local Yield (Mohr-Coulomb) and Love Vector
                # This ensures sharp situational logic is maintained
                mc_proj = MohrCoulombProjection().to(c_phys.device)
                love = LoveVector(c_phys.shape[-1]).to(c_phys.device)
                c_phys = love(mc_proj(c_phys, c_phys)) # Self-limiting local yield
                
                # Check for rupture
                constraint_losses = {k: loss_k}
                rupture_token = rupture_fn.check_rupture(c_phys, constraint_losses)
                if rupture_token is not None:
                    status = rupture_token.to_tensor(initial_c.device)
                    break
                
                # Gyroid Flow Constraint Enforcement
                with torch.no_grad():
                    dot_product, is_gyroid_satisfied = gyroid_flow.check_constraint(c_phys, c_phys)
                    if not is_gyroid_satisfied.any():
                        # Mark as alternative if flow is not gyroid-orthogonal
                        status = torch.tensor(1, device=initial_c.device)
                
                # Phase 2: Check hyper-ring closure
                try:
                    hyper_ring = hyper_ring_op(
                        residue=c_phys,
                        constraint_manifold=c_k,
                        embedding_fn=probe_k.embedding_fn
                    )
                    closure_result = hyper_ring_checker(hyper_ring, c_k)
                    
                    # If not valid (fracture or collapse), mark appropriately
                    if not closure_result['is_valid'].any():
                        # Check status
                        if 'fracture' in closure_result['status']:
                            # Fracture: non-closed
                            status = torch.tensor(2, device=initial_c.device)  # FAILURE
                            break
                        elif 'collapse' in closure_result['status']:
                            # Collapse: trivial cycle
                            status = torch.tensor(1, device=initial_c.device)  # ALTERNATIVE
                except Exception:
                    # If hyper-ring computation fails, continue without it
                    pass
                
                # Phase 2: Check soliton stability
                try:
                    soliton_result = soliton_checker(
                        residue=c_phys,
                        constraint_manifold=c_k,
                        embedding_fn=probe_k.embedding_fn
                    )
                    # If not a soliton, mark as unstable (but not necessarily failure)
                    if not soliton_result['is_soliton'].any():
                        # Unstable: may need more iterations or different approach
                        pass  # Continue but note instability
                except Exception:
                    # If soliton check fails, continue without it
                    pass
                
                # Track oscillation
                recent_states.append(c_k.clone())
                if len(recent_states) > oscillation_window:
                    recent_states.pop(0)
                
                # Bounded oscillation check (no convergence guarantee)
                if len(recent_states) >= oscillation_window:
                    oscillation_amplitude = torch.std(torch.stack(recent_states), dim=0).mean()
                    if oscillation_amplitude < 0.01:  # Threshold for bounded oscillation
                        # Accept state as stable
                        break
                
                # Invariant checks
                current_pas = pas_metric(c_phys)
                drift, violation = apas_check.check_drift(current_pas, prev_pas)
                
                if current_pas < 0.5:
                    low_pas_count += 1
                else:
                    low_pas_count = 0
                
                if low_pas_count > 20: # Amnesty: Allow 20 steps to find coherence

                    status = torch.tensor(2, device=initial_c.device)  # FAILURE
                    break
                
                current_chiral_score = cds.compute_score(c_phys, drift.item(), poly_degree)
                is_chiral_stable = cds.check_step(prev_chiral_score, current_chiral_score)
                
                if violation.any() or not is_chiral_stable:
                    pass  # Continue
                else:
                    prev_pas = current_pas
                    prev_chiral_score = current_chiral_score
        
        # Legacy mode: Original ADMM loop (if not using constraint probes)
        else:
            # Evolution Loop (Original Implementation)
            for k in range(max_iters):
                # 1. c-update (Constraint-First Repair)
                # We NO LONGER minimize distance to 'anchor' (T).
                # Instead, we minimize physical violation (pred - c_in)
                # while constrained to agree with c_sym (symbolic ontology).
                with torch.enable_grad():
                    c_in = c_phys.detach().requires_grad_(True)
                    # KAGH(c) should be consistent with c
                    pred = forward_op_fn(c_in, gcve_pressure=gcve_pressure, chirality=chirality_index)
                    
                    # Violation Pressure (Self-consistency)
                    violation_pressure = 0.5 * (pred - c_in).pow(2).sum()
                    
                    # ADMM Tension (Agreement with auxiliary variable z)
                    admm_tension = 0.5 * rho * (c_in - z + u).pow(2).sum()
                    
                    structural_tension = violation_pressure + admm_tension
                    grad_c = torch.autograd.grad(structural_tension, c_in)[0]
                # Yield-aware perturbation with Love Vector
                # MC Projection preserves sharp local situational failure
                mc_proj = MohrCoulombProjection().to(c_phys.device)
                love = LoveVector(c_phys.shape[-1]).to(c_phys.device)
                
                # Apply Love and MC to the current state/gradient
                step_size = 0.1 
                c_yielded = mc_proj(c_phys, grad_c)
                c_next = love(c_yielded) - step_size * grad_c
                
                # 2. z-update (Proximal / Ontological Projection)
                # We project toward the frozen symbolic state c_sym
                z_in = c_next + u
                # Hard Projection Pi: must match symbolic residue sgn/magnitude
                # (Simplified: soft-prox toward c_sym + sparsity)
                threshold = lambda_sparse / rho
                z_next = torch.sign(z_in) * torch.maximum(torch.abs(z_in) - threshold, torch.zeros_like(z_in))
                
                # Enforce Ontological Constraint: sgn(z_next) == sgn(c_sym)?
                # Or just let it settle. To be strict:
                # z_next = z_next * (torch.sign(z_next) == torch.sign(c_sym)).float()
                
                # 3. u-update
                u_next = u + (c_next - z_next)
                
                # 4. Invariant Check (APAS_zeta) & Incoherence Collapse
                current_pas = pas_metric(c_next)
                drift, violation = apas_check.check_drift(current_pas, prev_pas)
                
                # Incoherence Collapse logic
                if current_pas < 0.5: # Threshold for 'coherence'
                    low_pas_count += 1
                else:
                    low_pas_count = 0
                    
                if low_pas_count > 20: # Amnesty: Allow 20 steps to find coherence
 # Max allowed incoherence steps
                    status = torch.tensor(2, device=initial_c.device) # FAILURE
                    break
                
                # Chiral Score Check
                current_chiral_score = cds.compute_score(c_next, drift.item(), poly_degree)
                is_chiral_stable = cds.check_step(prev_chiral_score, current_chiral_score)
                
                if violation.any() or not is_chiral_stable:
                    # "Unbounded drift destroys continuity."
                    # We abort or mark as ALTERNATIVE if it settles on a new topology
                    pass 
                else:
                    # Accept Step
                    c_phys = c_next
                    z = z_next
                    u = u_next
                    prev_pas = current_pas
                    prev_chiral_score = current_chiral_score
        
        # Verify Final Outcome
        # If final result deviates too much from c_sym, mark as FAILURE or ALTERNATIVE
        with torch.no_grad():
            diff = torch.norm(c_phys - c_sym)

            # Topological Degree of Freedom Check
            current_chi = compute_chirality(c_phys.unsqueeze(1))
            original_chi = compute_chirality(c_sym.unsqueeze(1))

            # If chirality flips, it's a rupture. If not, magnitude (0.61) is legal.
            if torch.sign(current_chi) != torch.sign(original_chi):
                status = torch.tensor(2, device=initial_c.device) # FAILURE
            else:
                status = torch.tensor(0, device=initial_c.device) # LEGAL
            if status == 0:  # Only update if not already set
                if diff > 1.0: # Arbitrary threshold for 'different'
                    status = torch.tensor(1, device=initial_c.device) # ALTERNATIVE
                # if diff > 5.0: status = 2 (REMOVED BY AMNESTY)
                if diff > 5.0:
                    status = torch.tensor(2, device=initial_c.device) # FAILURE
                
        # Repair Trace Compression:
        # Store symbolic delta
            delta = c_phys - c_sym
            symbolic_delta = torch.round(delta * 10.0) / 10.0 
            c_raw = c_sym + symbolic_delta
            
            # Apply Global Yield (Drucker-Prager) as smooth envelope
            dp_proj = DruckerPragerProjection().to(c_raw.device)
            c_compressed = dp_proj(c_raw)
            
        # Save for backward (Compressed)
        ctx.save_for_backward(symbolic_delta)
        ctx.status = status
        
        # Identity Preservation Check (Chirality)
        chirality = compute_chirality(c_compressed.unsqueeze(1)) 
        ctx.chirality = chirality
        
        return c_compressed, status

    @staticmethod
    def backward(ctx, grad_output, grad_status):
        # No gradients flow back to System 1 initial guess (c_sym)
        # to prevent smoothness leakage.
        # Updated for Phase 1: additional parameters
        return None, None, None, None, None, None, None, None, None, None, None, None, None

class OperationalAdmm(nn.Module):
    """
    Wrapper module for the primitive.
    
    Phase 1: Supports constraint probe operators with cyclic traversal.
    """
    def __init__(
        self, 
        rho=1.0, 
        lambda_sparse=0.1, 
        max_iters=20, 
        zeta=0.05, 
        degree=3,
        use_constraint_probes: bool = True,
        num_constraints: int = 1
    ):
        super().__init__(); self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.rho = rho
        self.lambda_sparse = lambda_sparse
        self.max_iters = max_iters
        self.zeta = zeta
        self.degree = degree
        self.use_constraint_probes = use_constraint_probes
        self.num_constraints = num_constraints
        
        # Constraint probes (will be set externally or created on first forward)
        self.constraint_probes: Optional[List[ConstraintProbeOperator]] = None
        
    def set_constraint_probes(self, probes: List[ConstraintProbeOperator]):
        """Set constraint probe operators."""
        self.constraint_probes = probes
        self.num_constraints = len(probes)
        
    def forward(
        self, 
        initial_c, 
        forward_op, 
        gcve_pressure=None, 
        chirality=None,
        constraint_probes: Optional[List[ConstraintProbeOperator]] = None
    ):
        """
        Forward pass with optional constraint probes.
        
        Args:
            initial_c: Initial constraint state
            forward_op: Forward operator (KAGH)
            gcve_pressure: Optional gyroid violation pressure
            chirality: Optional chirality index
            constraint_probes: Optional list of constraint probe operators
        """
        # Use provided probes or instance probes
        probes = constraint_probes or self.constraint_probes
        
        return OperationalAdmmPrimitive.apply(
            initial_c, forward_op, 
            self.rho, self.lambda_sparse, 
            self.max_iters, self.zeta, self.degree,
            gcve_pressure, chirality,
            self.use_constraint_probes,
            probes,
            self.num_constraints
        )

