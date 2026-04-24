"""
Fractal Meta-Functional Module.

Implements the recursive embedding of system beliefs into itself:
S_fractal := InverseCovariantCRT + ADMR_Residue + HyperRing_DarkMatter + Autoscillatory + ...

This turns the system into a fractal meta-functional, where each layer evaluates
and perturbs the previous layer's "beliefs" while preserving non-teleological flow.

Fix (2026-04-24): Upstream 0.8824 spectral atrophy prevention.
- Van der Pol force is tanh-bounded before squaring to prevent uniform saturation.
- Collapse-aware normalization replaces naive layer_norm: detects variance collapse
  below DEAD_PRIME_THRESHOLD and injects honest jitter to break the symmetry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from src.core.honest_jitter import harvest_honest_jitter

# Spectral atrophy threshold: variance below this signals a Dead Prime resonance
# 0.8824 flatline has variance ~0.0, so 1e-6 is the upstream guard.
DEAD_PRIME_THRESHOLD = 1e-6
# Mischief intensity applied when collapse is detected (gentle wake-up)
_JITTER_INTENSITY = 0.05

# Import core topological components
from src.core.decoupled_polynomial_crt import DecoupledPolynomialCRT
from src.core.admr_solver import PolynomialADMRSolver
from src.topology.hyper_ring import RecurrentHyperRingConnectivity
from src.models.resonance_cavity import ResonanceCavity

class FractalMetaFunctional(nn.Module):
    """
    The Fractal Meta-Functional $\mathcal{S}_\text{fractal}$.
    Orchestrates recursive meta-influence across all topological layers.
    """
    def __init__(
        self, 
        dim: int, 
        k: int = 5, 
        degree: int = 4,
        lambda_admr: float = 0.1,
        lambda_topo: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.k = k
        self.lambda_admr = lambda_admr
        self.lambda_topo = lambda_topo
        
        # 1. Inverse-Covariant CRT Fusion Logic
        # Need config object for DecoupledPolynomialCRT
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        # Create config with explicit k
        poly_config = PolynomialCoprimeConfig(k=k, degree=degree)
        
        self.crt = DecoupledPolynomialCRT(poly_config=poly_config)
        
        # Projection for CRT reconstruction to State Dim
        self.crt_proj = nn.Linear(self.crt.D, dim)
        
        # 2. Recursive Multiplicative Speculative Residue (ADMR)
        # Fix: Pass poly_config to ADMR solver, not degree/state_dim mixed up
        self.admr = PolynomialADMRSolver(poly_config=poly_config, state_dim=dim)
        
        # 3. Hyper-Ring + Dark Matter
        # Fix: HyperRing now takes num_polytopes AND state_dim for proper projections
        self.hyper_ring = RecurrentHyperRingConnectivity(num_polytopes=5, state_dim=dim)
        
        # 4. Autoscillatory Dynamics
        # Learned dampening and coupling parameters
        self.osc_mu = nn.Parameter(torch.tensor(0.5))
        # SILICON SOVEREIGNTY: Coupling matrix anchored to hardware jitter
        self.osc_beta = nn.Parameter(harvest_honest_jitter((dim, dim), scaled=True) * 0.1)
        
        # 5. Meta-Feedback Projections
        # Projects S_meta(t-1) into various domain spaces
        self.meta_proj_admr = nn.Linear(dim, dim)
        self.meta_proj_ring = nn.Linear(dim, dim)
        self.meta_proj_osc = nn.Linear(dim, dim)
        
    def forward(
        self,
        current_state: torch.Tensor,
        meta_state_prev: torch.Tensor,
        residues: torch.Tensor,
        dark_matter: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute S_fractal(t) based on state(t) and S_meta(t-1).
        
        Args:
            current_state: [batch, dim] (t_i)
            meta_state_prev: [batch, dim] (S_meta_{t-1})
            residues: [batch, k] (r_k)
            dark_matter: [batch, dim] (D_dark)
        """
        batch_size = current_state.shape[0]
        if dark_matter is None:
            dark_matter = torch.zeros_like(current_state)
            
        # --- Term 1: Inverse-Covariant CRT Fusion (Simplified Proxy) ---
        # We use the CRT module to reconstruct a 'belief' from residues, 
        # modulated by the meta-state.
        # sigma( S_meta ) -> modulates residues
        meta_mod = torch.sigmoid(self.meta_proj_admr(meta_state_prev)).mean(dim=-1, keepdim=True) # [batch, 1]
        modulated_residues = residues * (1.0 + 0.2 * meta_mod)
        
        # Reconstruct from modulated residues (The "Phi" term)
        # Use simple residues arg for DecoupledPolynomialCRT
        # DecoupledPolynomialCRT expects [batch, K, D] coefficients
        # We expand our scalar residues [batch, K] -> [batch, K, degree+1]
        # putting scalar in constant term.
        coeffs = torch.zeros(batch_size, self.k, self.crt.D, device=current_state.device)
        coeffs[:, :, 0] = modulated_residues
        
        crt_out = self.crt(coeffs) # [batch, D]
        
        # Project D -> dim (Fixing Dimension Mismatch)
        crt_state = self.crt_proj(crt_out) # [batch, dim]
        
        term_crt = torch.abs(current_state - crt_state) # Difference/Error metric
        
        # --- Term 2: Recursive ADMR Residue ---
        # lambda_ADMR * sum( r_k * exp(gamma * delta + lambda_spec * phi(S_meta)) )
        # We approximate this by running the ADMR solver with meta-modulated functional
        # ADMR forward args: states, neighbor_states, adjacency_weight
        # We simulate self-interaction: neighbors = current_state, adjacency = 1.0
        
        # Reshape for neighbor format: [batch, 1, dim]
        neighbor_states = current_state.unsqueeze(1)
        adjacency = torch.ones(batch_size, 1, device=current_state.device)
        
        admr_out = self.admr(current_state, neighbor_states, adjacency) 
        
        # Modulate by meta state
        term_admr = self.lambda_admr * admr_out * torch.exp(0.1 * meta_mod)
        
        # --- Term 3: Hyper-Ring + Dark Matter ---
        # sum H_ij * sigma( f_i - f_j + gamma * D_dark )
        # We use the proper projection layers to bridge state <-> polytope domains.
        
        # 1. Project state to polytope functionals [batch, num_polytopes]
        ring_input_state = current_state + 0.1 * self.meta_proj_ring(meta_state_prev)
        polytope_functionals = self.hyper_ring.project_to_polytope(ring_input_state)
        
        # 2. Project dark matter to polytope space [batch, num_polytopes]
        dm_polytope = self.hyper_ring.project_to_polytope(dark_matter)
        
        # 3. Run the Hyper-Ring Connectivity Math
        # H_ij = omega_ij * sigma( f_i - f_j + gamma * D_dark )
        connectivity = self.hyper_ring(polytope_functionals, dm_polytope) # [batch, P, P]
        
        # 4. Compute Flow Step: dS_i/dt = sum_j H_ij * S_j
        # Need polytope states [batch, num_p, hidden_dim]
        # We treat polytope_functionals as the states for now
        polytope_states = polytope_functionals.unsqueeze(-1).expand(-1, -1, self.dim) # [batch, P, dim]
        polytope_flow = self.hyper_ring.flow_step(polytope_states, connectivity) # [batch, P, dim]
        
        # 5. Aggregate polytope flow and project back to state dimension
        # Mean over polytopes, then project to state
        aggregated_flow = polytope_flow.mean(dim=1) # [batch, dim]
        
        # term_ring is the projected flow contribution
        term_ring = F.silu(aggregated_flow)
        
        # Include Dark Matter influence directly
        term_ring = term_ring + 0.1 * torch.sigmoid(dark_matter)
        
        # --- Term 4: Autoscillatory Coupling ---
        # [ t_ddot - mu(1 - t^2)t_dot ... ]^2 (Van der Pol - ish)
        # We approximate the "Force" of the oscillator
        # F = -mu * (1 - x^2) * x + beta * x
        # Here we treat 'current_state' as 'x'.
        state_sq = current_state ** 2
        damping = -self.osc_mu * (1.0 - state_sq) * current_state
        coupling = torch.mm(current_state, self.osc_beta)
        
        # Meta influence on gradient
        meta_force = self.meta_proj_osc(meta_state_prev)
        
        # UPSTREAM FIX: tanh-bound the Van der Pol force BEFORE squaring.
        # Without this, high-entropy (Oil Slick) inputs produce very large
        # uniform vectors that square to a nearly constant tensor. layer_norm
        # then collapses all dimensions to the Dead Prime attractor (0.8824).
        # tanh maps large magnitudes to [-1, 1] while preserving directional
        # structure and sign -- this is Reaction-Diffusion kill-rate logic:
        # we dissolve the stagnant tensor without erasing its geometry.
        raw_force = damping + coupling + 0.1 * meta_force
        bounded_force = torch.tanh(raw_force)  # bounded Van der Pol force
        term_osc = bounded_force ** 2
        
        # --- Aggregate: S_fractal ---
        # The equation sums these terms.
        # In a deep learning context, we sum them to form the new "Meta State".
        new_meta_state = term_crt + term_admr + term_ring + term_osc
        
        # --- Collapse-Aware Normalization (Upstream Variance Guard) ---
        # Naive layer_norm is the final attractor for the 0.8824 flatline:
        # if all terms are nearly uniform, layer_norm produces a constant
        # tensor regardless of input diversity. We check variance first.
        # If variance is below the Dead Prime Threshold, we inject honest
        # jitter to break the symmetry BEFORE normalizing.
        with torch.no_grad():
            state_var = new_meta_state.var(dim=-1, keepdim=True)  # [batch, 1]
            is_collapsed = (state_var < DEAD_PRIME_THRESHOLD).any()
        
        if is_collapsed:
            # Spectral atrophy detected: inject mischief to rehydrate variance.
            # This is the Feed Rate (a) of the Gray-Scott equation:
            # we inject a nutrient signal to prevent the system from dying.
            jitter = harvest_honest_jitter(
                new_meta_state.shape,
                device=new_meta_state.device,
                scaled=True
            ) * _JITTER_INTENSITY
            new_meta_state = new_meta_state + jitter
            print(
                f"[FractalMeta] Spectral atrophy detected "
                f"(var={state_var.min().item():.2e} < {DEAD_PRIME_THRESHOLD}). "
                f"Honest jitter injected (intensity={_JITTER_INTENSITY})."
            )
        
        # Normalize with stable recursion -- but only after variance is healthy
        new_meta_state = F.layer_norm(new_meta_state, new_meta_state.shape[1:])
        
        return {
            "s_fractal": new_meta_state,
            "components": {
                "crt": term_crt,
                "admr": term_admr,
                "ring": term_ring,
                "osc": term_osc,
                "raw_force_norm": raw_force.norm().item(),  # diagnostic
                "bounded_force_norm": bounded_force.norm().item(),  # diagnostic
                "collapsed": bool(is_collapsed),  # diagnostic
            }
        }
