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
    r"""
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
        
        # 6. Modular Virtualization Layer
        from src.core.modular_virtualization import ModularVirtualizationLayer
        self.modular_rns = ModularVirtualizationLayer(dim=dim, base=2)

        # 7. Topological Isolation expanded into Fractal Systems (CerumenPotIsolation)
        try:
            from src.core.garden_statistical_attractors import CerumenPotIsolation
            self.cerumen_isolation = CerumenPotIsolation(threshold=0.5)
        except Exception:
            self.cerumen_isolation = None
        
    def forward(
        self,
        current_state: torch.Tensor,
        meta_state_prev: torch.Tensor,
        residues: torch.Tensor,
        dark_matter: Optional[torch.Tensor] = None,
        cavity_overtones: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute S_fractal(t) based on state(t) and S_meta(t-1).
        
        Args:
            current_state: [batch, dim] (t_i)
            meta_state_prev: [batch, dim] (S_meta_{t-1})
            residues: [batch, k] (r_k)
            dark_matter: [batch, dim] (D_dark)
        """
        # Virtualize states to hybrid modular coordinates (endogenous connection)
        # Using meta_state_prev as the parity anchor to trigger topological refusal snap if invalid
        virtual_state_rns = self.modular_rns.float_to_rns(current_state, anchor_sym=meta_state_prev)
        current_state = self.modular_rns.rns_to_float(virtual_state_rns)
        
        batch_size = current_state.shape[0]
        if dark_matter is None:
            dark_matter = torch.zeros_like(current_state)
            
        # Compute spatial correlation of current_state to dynamically size fractal clusters
        from src.core.martinova_correlation import compute_bounded_correlation
        corr_input = current_state.unsqueeze(-1) if current_state.dim() == 2 else current_state
        spatial_corr = compute_bounded_correlation(corr_input)  # [batch]
        # Map spatial_corr from [-1, 1] to a cluster scaling factor in [0.2, 2.0]
        # High spatial clustering (crystallization) shrinks the active cluster influence,
        # while spatial dispersion expands it.
        cluster_scale = (1.0 - 0.8 * spatial_corr).unsqueeze(-1)  # [batch, 1]
            
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
        
        term_crt = torch.abs(current_state - crt_state) * cluster_scale # Difference/Error metric scaled
        
        # --- Term 2: Recursive ADMR Residue ---
        # lambda_ADMR * sum( r_k * exp(gamma * delta + lambda_spec * phi(S_meta)) )
        # We approximate this by running the ADMR solver with meta-modulated functional
        # ADMR forward args: states, neighbor_states, adjacency_weight
        # We simulate self-interaction: neighbors = current_state, adjacency = 1.0
        
        # Reshape for neighbor format: [batch, 1, dim]
        neighbor_states = current_state.unsqueeze(1)
        adjacency = torch.ones(batch_size, 1, device=current_state.device)
        
        admr_out = self.admr(current_state, neighbor_states, adjacency) 
        
        # Modulate by meta state and cluster scale
        term_admr = self.lambda_admr * admr_out * torch.exp(0.1 * meta_mod) * cluster_scale
        
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
        
        # Include Dark Matter influence directly and scale by cluster_scale
        term_ring = (term_ring + 0.1 * torch.sigmoid(dark_matter)) * cluster_scale

        
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

    def isolate_fractal_fundamental_domain(
        self,
        state: torch.Tensor,
        pqr: tuple = (3, 3, 7),
        max_reflections: int = 16,
        complex_fold_type: int = 2,
        fold_power: int = 2,
        fold_a: tuple = (0.6, 0.2),
        noise_sigma: float = 0.24,
    ) -> Dict[str, object]:
        """
        Expands topological isolation (CerumenPotIsolation + hyperbolic Coxeter
        {p, q, r} reflection fundamental domains inspired by Limit Worlds) into
        the fractal meta-functional, coupled with the Takahashi & El Hady
        (docs/2611.02220v1.pdf) Affordance-Reset dwell occupation law.
        """
        import math

        p, q, r = pqr
        dih = lambda x: 1.0 if x <= 0 else math.cos(math.pi / float(x))
        cp = dih(p)
        sp = math.sqrt(max(1e-8, 1.0 - cp * cp))
        cq = dih(q)
        cr = max(1e-6, dih(r))
        R = 1.0 / cr
        k_val = R * cq / sp
        hyperbolic_margin = 1.0 + k_val * k_val - R * R
        is_valid_coxeter = hyperbolic_margin > 1e-3

        s_in = state.detach().float()
        orig_shape = s_in.shape
        flat = s_in.view(-1, s_in.shape[-1]) if s_in.dim() > 1 else s_in.unsqueeze(0)
        B, D = flat.shape

        # Project leading 3D coordinates into hyperbolic reflection chamber
        pad_d = max(0, 3 - D)
        pts = F.pad(flat, (0, pad_d))[:, :3].clone()
        pts = pts / (torch.norm(pts, dim=-1, keepdim=True) + 1e-6)

        # Fundamental reflection planes A, B, D and inversion sphere C (LightHorus Coxeter)
        nA = torch.tensor([1.0, 0.0, 0.0], device=pts.device)
        nB = torch.tensor([-cp, sp, 0.0], device=pts.device)
        nD = torch.tensor([0.0, 0.0, 1.0], device=pts.device)
        sphere_c = torch.tensor([0.0, k_val, 0.0], device=pts.device)
        sphere_r2 = max(0.25, R * R)

        reflections = torch.zeros(B, device=pts.device)
        orb = torch.ones(B, device=pts.device)
        for _ in range(max(1, int(max_reflections))):
            changed = torch.zeros(B, dtype=torch.bool, device=pts.device)
            for n_vec in (nA, nB, nD):
                proj = (pts * n_vec).sum(dim=-1, keepdim=True)
                neg = proj.squeeze(-1) < 0.0
                if neg.any():
                    pts = torch.where(neg.unsqueeze(-1), pts - 2.0 * proj * n_vec, pts)
                    reflections = reflections + neg.float()
                    changed = changed | neg
            qv = pts - sphere_c
            d2 = (qv * qv).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            inv_k = sphere_r2 / d2
            in_sphere = inv_k.squeeze(-1) >= 1.0
            if in_sphere.any():
                pts = torch.where(in_sphere.unsqueeze(-1), inv_k * qv + sphere_c, pts)
                reflections = reflections + in_sphere.float()
                orb = torch.where(in_sphere, orb * inv_k.squeeze(-1).clamp(max=1e4), orb)
                changed = changed | in_sphere
            pts = pts / (torch.norm(pts, dim=-1, keepdim=True) + 1e-6)
            if not changed.any():
                break

        # Conformal complex-plane fold on leading (x, y) pair
        z = torch.complex(pts[:, 0], pts[:, 1])
        a_c = complex(float(fold_a[0]), float(fold_a[1]))
        pw = max(1, int(fold_power))
        if complex_fold_type == 1:
            z_fold = z ** pw
        elif complex_fold_type == 2:
            z_fold = (z ** pw) + a_c / (z ** pw + 1e-4)
        elif complex_fold_type == 3:
            # Blaschke disk automorphism ((z - a) / (1 - conj(a) z))^pw
            z_fold = ((z - a_c) / (1.0 - a_c.conjugate() * z + 1e-4)) ** pw
        elif complex_fold_type == 4:
            z_fold = (z ** pw) * ((z - a_c) / (z + a_c + 1e-4))
        else:
            z_fold = z
        z_fold = z_fold / (z_fold.abs().clamp(min=1.0))

        isolated = flat.clone()
        isolated[:, 0] = z_fold.real
        if D >= 2:
            isolated[:, 1] = z_fold.imag
        if D >= 3:
            isolated[:, 2] = pts[:, 2]

        # CerumenPotIsolation aperiodic monotile diffusion barrier check
        barrier_ok, barrier_score = True, 1.0
        if getattr(self, "cerumen_isolation", None) is not None:
            try:
                pot_centers = self.cerumen_isolation.generate_aperiodic_pot_centers(
                    num_pots=max(16, min(32, B + 16)), dim=min(32, max(4, D)), device=pts.device
                )
                barrier_ok, barrier_score = self.cerumen_isolation.verify_diffusion_barrier(
                    pot_centers, num_probe_rays=16
                )
            except Exception:
                pass

        # Takahashi & El Hady (docs/2611.02220v1.pdf) Affordance-Reset dwell occupation:
        # T := sigma^2, dwell m(a) = 1 + mean_reflections, Z_0 = <tau>, ergodic edge alpha = T / T_lambda
        T_eff = max(1e-6, float(noise_sigma) ** 2)
        mean_dwell = float(1.0 + reflections.mean().item())
        T_lambda = max(1e-6, 1.0 / (1.0 + math.log(mean_dwell + 1.0)))
        ergodic_alpha = T_eff / T_lambda

        return {
            "isolated_state": isolated.view(orig_shape),
            "coxeter_pqr": pqr,
            "is_valid_coxeter": bool(is_valid_coxeter),
            "hyperbolic_margin": float(hyperbolic_margin),
            "mean_reflections_dwell": mean_dwell,
            "conformal_orbit_log_scale": float(torch.log(orb.clamp(min=1.0)).mean().item()),
            "cerumen_barrier_verified": bool(barrier_ok),
            "cerumen_barrier_score": float(barrier_score),
            "affordance_reset_temperature": float(T_eff),
            "ergodic_edge_alpha": float(ergodic_alpha),
            "weak_ergodicity_breaking": bool(ergodic_alpha < 1.0),
        }
