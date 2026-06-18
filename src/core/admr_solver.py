"""
Polynomial ADMR: Alternating Direction of Multiplicative Remainders.

Implements the number-theoretic analogue of ADMM using continuous 
polynomial functionals instead of discrete prime moduli.

**Note on Negentropy Flux**: The negentropy flux in ADMR experiences a
"tripsody", providing tripartite rhapsodic oscillation to phase-lock the solver.
"""

import torch
import torch.nn as nn
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from .polynomial_coprime import PolynomialCoprimeConfig
from .numerical_d_module import NumericalDModuleManager, RationalSnappingLayer
from src.core.honest_jitter import harvest_honest_jitter


class PolynomialADMRSolver(nn.Module):
    """
    Implements the Polynomial ADMR update:
    S^{(n+1)} = Proj_{Poly} [ S^{(n)} *  w_ik S_k ]
    
    Uses co-prime polynomial functionals to ensure discrete-like 
    information separation in a differentiable, continuous field.
    """
    
    def __init__(
        self,
        poly_config: PolynomialCoprimeConfig,
        state_dim: int,
        eta_scaffold: Optional[float] = None,
        device: str = None,
        use_opencl: bool = False
    ):
        """
        Args:
            poly_config: Configuration for co-prime polynomial functionals.
            state_dim: Dimension of the state being optimized.
            eta_scaffold: Rate of scaffold adaptation.
            device: Computing device.
            use_opencl: If true, utilize PyOpenCL dual-queue hardware sovereignty.
        """
        super().__init__()
        self.config = poly_config
        self.state_dim = state_dim
        self.device = device
        
        if eta_scaffold is None:
            # Derive dynamically from honest jitter to ground in hardware latency
            self.eta_scaffold = harvest_honest_jitter((1,), device=device, scaled=False).mean().item() * 0.05
        else:
            self.eta_scaffold = eta_scaffold
        
        if use_opencl:
            try:
                from .pyopencl_sovereignty import create_engine
                self.silicon_engine = create_engine()
            except Exception as e:
                print(f"Warning: Failed to initialize SiliconSovereigntyEngine: {e}")
                self.silicon_engine = None
        else:
            self.silicon_engine = None
        
        # 1. Manifold State (Asymptotic Time)
        self.register_buffer('tau', torch.tensor(0.0, device=device))
        
        # 2. Non-selfadjoint Transition Operators (A_i)
        # We initialize non-selfadjoint matrices for facet-wise dynamics
        num_facet_channels = poly_config.k
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        self.A = nn.Parameter(harvest_honest_jitter((num_facet_channels, state_dim, state_dim), device=device, scaled=True) * 0.01)
        
        # 3. Stochastic Forcing Buffer
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        self.register_buffer('eta', harvest_honest_jitter((state_dim,), device=device, scaled=True) * 0.005)
        
        # 4. Chiral Residue Cache (Warm-start backtracking)
        self.register_buffer('chiral_cache', torch.zeros(1, state_dim, device=device))
        self.register_buffer('cache_valid', torch.tensor(False))

        # 5. Love Invariant Protector (Null-space projection)
        from .love_invariant_protector import LoveInvariantProtector
        self.love_protector = LoveInvariantProtector(love_dim=max(1, state_dim // 4), device=device)
        
        # 5.5 Topological Gyrocompass (Convexity shield & True North locator)
        from .topological_gyrocompass import TopologicalGyrocompass
        self.gyrocompass = TopologicalGyrocompass(state_dim=state_dim, love_dim=max(1, state_dim // 4), device=device)
        
        # 6. KAGH-Boltzmann Surrogate for Continuous-to-Discrete jumps
        # The surrogate maps continuous Polynomial projections to discrete Matrioshka states
        # via B-splines and Saturated Quantizers.
        try:
            from src.surrogates.kagh_networks import KAGHBlock
            self.kagh_surrogate = KAGHBlock(n_in=state_dim, n_out=state_dim, width=max(16, state_dim // 2), depth=2).to(device)
        except ImportError:
            self.kagh_surrogate = None

        # 7. Unicorn Synthesis Upgrade: Walther's D-modules & Rational Snapping
        self.d_module_manager = NumericalDModuleManager(state_dim=state_dim, num_functionals=poly_config.k)
        self.snapper = RationalSnappingLayer()

    def update_chiral_cache(self, states: torch.Tensor, is_valid: torch.Tensor):
        """
        Save topologically valid configurations to the Chiral Cache.
        """
        if is_valid.any():
            valid_states = states[is_valid]
            if not self.cache_valid.item():
                self.chiral_cache.copy_(valid_states.mean(dim=0, keepdim=True).detach())
                self.cache_valid.fill_(True)
            else:
                self.chiral_cache.copy_(0.9 * self.chiral_cache + 0.1 * valid_states.mean(dim=0, keepdim=True).detach())

    def update_scaffold(self, negentropy_flux: torch.Tensor, dt: torch.Tensor):
        """
        Update the polynomial coefficients based on negadaptive dynamics.
        """
        self.tau += dt
        # Negentropy modulates the 'breathing' of the polynomial grid
        with torch.no_grad():
            self.config.mutate()

    def forward(
        self, 
        states: torch.Tensor, 
        neighbor_states: torch.Tensor, 
        adjacency_weight: torch.Tensor,
        valence: Optional[torch.Tensor] = None,
        use_warm_start: bool = False
    ) -> torch.Tensor:
        """
        Multiplicative update using relational graph adjacency and polynomial projection.
        
        Args:
            states: [batch, state_dim] S_i
            neighbor_states: [batch, neighbors, state_dim] S_k
            adjacency_weight: [batch, neighbors] R_ik from Relational Graph
            valence: [batch] training hunger / valency drive
            use_warm_start: If True, injects chiral cache history to prevent full reset.
        """
        # Warm-start logic from Chiral Residue Cache:
        # Instead of resetting $C_0$ to standard normal, we inject the cache to preserve the "Dream" continuity.
        if use_warm_start and self.cache_valid.item():
            states = 0.5 * states + 0.5 * self.chiral_cache.expand_as(states)

        # 1. Weighted sum of neighbors from Relational Graph
        #  R_ik S_k
        weighted_neighbors = torch.einsum('bn,bnd->bd', adjacency_weight, neighbor_states)
        
        # 2. Multiplicative interaction: S_i * ( S_k + V)
        v_drive = valence.unsqueeze(-1) if valence is not None else 1.0
        interaction = states * (weighted_neighbors + v_drive)
        
        # 3. Polynomial Projection (Functional Co-primality)
        # Instead of 'remainder', we evaluate the interaction through the co-prime basis.
        # This acts as a 'soft modulus' that preserves symbolic structure.
        
        # Unicorn Synthesis: Snap to rational lattice before evaluation
        interaction_snapped = self.snapper(interaction)
        projected = self.config.evaluate(interaction_snapped)
        
        # If k-functionals are used as the 'modulus', we aggregate their response
        # to get the new state. This preserves state_dim while incorporating 
        # the non-linear co-prime filtering.
        if projected.dim() > interaction.dim():
             projected = projected.mean(dim=-1)
             
        # Matching safety
        if projected.shape[-1] != self.state_dim:
            if projected.shape[-1] > self.state_dim:
                projected = projected[..., :self.state_dim]
            else:
                projected = torch.nn.functional.pad(projected, (0, self.state_dim - projected.shape[-1]))
                
        # 4. Continuous-to-Discrete Jump (Matrioshka bridging)
        # Apply the KAGH Surrogate to find the optimal quantized geometric topology
        if hasattr(self, 'kagh_surrogate') and self.kagh_surrogate is not None:
            # We treat the continuous interaction as raw input coefficients,
            # using the surrogate to handle Saturated Quantizers and Gdel gates.
            projected = self.kagh_surrogate(projected)
                
        return projected

    def fast_micro_step(
        self,
        states: torch.Tensor,
        neighbor_states: torch.Tensor,
        adjacency_weight: torch.Tensor,
        num_iters: int = 5
    ) -> torch.Tensor:
        """
        System 1 (Fast Cop) micro-iterations.
        Processes rapid heuristic drafts without full System 2 constraint sync.
        Provides kinetic energy for the trajectory.
        """
        curr_states = states
        for _ in range(num_iters):
             # Simplified interaction loop without full D-module rank sync
             weighted_neighbors = torch.einsum('bn,bnd->bd', adjacency_weight, neighbor_states)
             interaction = curr_states * (weighted_neighbors + 1.0)
             curr_states = self.config.evaluate(interaction)
             if curr_states.dim() > interaction.dim():
                 curr_states = curr_states.mean(dim=-1)
             
             # Handle shape mismatches
             if curr_states.shape[-1] != self.state_dim:
                 curr_states = torch.nn.functional.pad(curr_states[..., :self.state_dim], (0, max(0, self.state_dim - curr_states.shape[-1])))
                 
        return curr_states

    def evaluate_ivst_sidechain(self, states: torch.Tensor, elipsodistrophy_metrics: Optional[Dict[str, Any]]) -> torch.Tensor:
        """
        Evaluate the IVST side-chain chaotic envelope.
        """
        level = 0.0
        if elipsodistrophy_metrics is not None:
            level = elipsodistrophy_metrics.get('matrioshka_level', elipsodistrophy_metrics.get('level', 0.0))
            if level is None:
                level = 0.0
        
        # Matrioshka fractal shell scale: Z_scale = 2.0 ** (-level)
        Z_scale = 2.0 ** (-float(level))
        
        # High-frequency transient carrier: sin(30x) + 1
        soliton_transient = torch.sin(30.0 * states) + 1.0
        
        # Slow macro-environmental envelope: cos(tau / Z)
        tau_val = self.tau.to(states.device)
        macro_envelope = torch.cos(tau_val / Z_scale)
        
        # Composite limit
        y_threshold = macro_envelope * soliton_transient
        return y_threshold

    def apply_ivst_sidechain_dropout(self, states: torch.Tensor, elipsodistrophy_metrics: Optional[Dict[str, Any]]) -> torch.Tensor:
        """
        [ARCHITECTURAL REMEDIATION] Replace crude dropout and NaN injection with Unknowledge Domain
        shielding to protect Dream States from lobotomizing System 2 pressure.
        """
        # Get Unknowledge metrics
        h_mischief = 1.0 # Default if unknown
        if elipsodistrophy_metrics is not None:
             h_mischief = elipsodistrophy_metrics.get('h_mischief', 1.0)
        
        # Calculate Computable Flux (V_m) proxy
        current_pressure = torch.abs(states)
        y_threshold = self.evaluate_ivst_sidechain(states, elipsodistrophy_metrics)
        v_m_proxy = current_pressure - y_threshold
        
        from src.topology.unknowledge_domain import UnknowledgeDomain
        if not hasattr(self, '_unknowledge_domain'):
            self._unknowledge_domain = UnknowledgeDomain()
            
        # Instead of replacing states with NaN (crude dropout), we shield the pressures
        # that fall into the Unknowledge Domain (V_m < 0, H_mischief > threshold)
        shielded_pressure = self._unknowledge_domain.apply_shielding(current_pressure, v_m_proxy, h_mischief)
        
        # Apply the shielded pressure scaling back to the states
        # The shield dampens the magnitude of state updates rather than destroying them
        scale_factor = shielded_pressure / (current_pressure + 1e-8)
        shielded_states = states * scale_factor
        
        # Lazarus Rehydration: apply energy stabilization
        from src.core.spectral_coherence_repair import apply_energy_based_stabilization
        rehydrated_states = apply_energy_based_stabilization(shielded_states, stability_margin=1e-5)
        
        return rehydrated_states

    def stochastic_differential_step(
        self, 
        states: torch.Tensor, 
        neighbor_states: torch.Tensor, 
        adjacency_weight: torch.Tensor,
        dt: float = 0.1,
        sigma: Optional[float] = None,
        v_m: Optional[torch.Tensor] = None,
        elipsodistrophy_metrics: Optional[Dict[str, Any]] = None,
        palindromic_hash: Optional[torch.Tensor] = None,
        anchor_sym: Optional[torch.Tensor] = None,
        boundary_state: Optional[Any] = None,
        return_violation: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Continuous-time Stochastic Differential Update:
        dx(t) = [  A_i x_i(t) -   (x - r(x_k)) + _tension ] dt + (D) dW
        
        Warmstart Prior: If palindromic_hash is provided, it offsets the initial 
        state trajectory towards the "Symmetry-Stable" zone of the hybrid basis,
        enabling O(K) faster convergence.
        """
        batch_size = states.shape[0]
        
        # 0. Palindromic Warmstart Initialization Handler
        if palindromic_hash is not None:
            # Shift states toward the symmetry-stable prior
            # The hash acts as a geometric guide to the stable zone
            states = states + 0.1 * (palindromic_hash.expand_as(states) - states)

        # 0.5. SDE Scaling (Poincar Eccentricity Driver)
        # We scale noise by the diffusion coefficient derived from spectral shear.
        if sigma is None:
            # Anchor sigma directly to hardware latency variance
            sigma = harvest_honest_jitter((1,), device=states.device, scaled=False).mean().item() * 0.05
            
        diff_coeff = elipsodistrophy_metrics.get('diffusion_coefficient', 1.0) if elipsodistrophy_metrics else 1.0
        effective_sigma = sigma * diff_coeff
        
        # 1. Non-selfadjoint Drifts ( A_i x_i)
        # We treat the co-prime evaluation as the 'decomposition' into facets
        facets = self.config.evaluate(states) # [batch, state_dim, num_functionals]
        
        # A @ facets: [num_functionals, state_dim, state_dim] @ [batch, state_dim, num_functionals]
        # We sum over facets
        drift = torch.zeros_like(states)
        for i in range(self.config.k):
            # Facet-wise Dual Update Loop:
            # Propagates the state through K independent non-selfadjoint transition operators (A_i).
            # This ensures that each coprime functional channel evolves its own local flow,
            # maintaining topological separation before global reconstruction.
            
            # facet_i: [batch, state_dim]
            facet_i = facets[..., i]
            # A_i: [state_dim, state_dim]
            drift = drift + torch.matmul(facet_i, self.A[i])
            
        # 2. Survival Pressure (ADMR Negotiation)
        weighted_neighbors = torch.einsum('bn,bnd->bd', adjacency_weight, neighbor_states)
        # Negotiation term: states - weighted_neighbors
        negotiation = states - weighted_neighbors

        # 3. Perkins Tension Objective (Minimal Surface Tension gamma)
        # Resolves "cubed cube" paradoxes by minimizing surface tension in the RP^4 Void.
        # The tension is fossilized as a Chiral Breathera persistent topological soliton.
        shear = elipsodistrophy_metrics.get('hyperbolic_shear', 0.0) if elipsodistrophy_metrics else 0.0
        
        if getattr(self, 'silicon_engine', None) is not None:
            # Anchor tension to hardware-latency proxy rather than arbitrary human 0.02 magic number
            gamma = (self.silicon_engine.get_hardware_latency_anchor() / 1000.0) * shear
        else:
            gamma = (math.pi / 137.0) * shear # Fine-structure constant topological proxy
            
        # Chiral Breather: A persistent topological soliton that fossilizes tension
        # to resolve "cubed cube" paradoxes in the RP^4 Void.
        breather = torch.cos(self.tau * 7.7) * gamma # Chiral Breather component
        tension_drift = -gamma * negotiation * (1.0 + breather)
        
        # 4. Stochastic Forcing (dW)
        # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
        noise = harvest_honest_jitter(states.shape, device=states.device, scaled=True) * effective_sigma * (dt**0.5)
        
        # 3.5. Mathematical Digimon Nutrient (Historical Illusion Injection)
        # Allows transient high-fidelity states to feed the trajectory.
        digimon_nutrient = torch.zeros_like(states)
        if palindromic_hash is not None:
             # Conceptual Digimon: use the hash as a nutrient to warp the drift
             digimon_nutrient = 0.05 * (palindromic_hash.expand_as(states) - states)

        # V_m explicit learning rate modulation
        # Dual variables S evolution tied to the normalized Mischief Score V_m
        effective_dt = dt
        if v_m is not None:
             effective_dt = dt * (1.0 + v_m.view(-1, 1))

        # Tripsodic Negentropy Oscillation:
        # As negentropy (information density) increases, the system phase-locks
        # via a tripartite rhapsodic oscillation rather than freezing.
        # This prevents "Dead Logic" singularities by converting stasis into expansion.
        negentropy_flux = torch.norm(drift, dim=-1, keepdim=True)
        tripsody_scale = torch.cos(negentropy_flux * math.pi)
        effective_dt = effective_dt * (1.0 / (1.0 + negentropy_flux)) * (1.0 + 0.5 * tripsody_scale)

        # 5. Update Step (Continuous Approximation)
        # dx = (drift - negotiation + tension_drift + digimon_nutrient) * dt + noise
        dx = (drift - negotiation + tension_drift + digimon_nutrient) * effective_dt + noise
        
        # 4.2 Bouligand Tangent Cone Projection & Gyrocompass Precession
        # We project the update onto the Bouligand Tangent Cone to prevent boundary tearing.
        # To avoid blind left-field gating (which occurs if we only project out normal force),
        # we precess the torque orthogonally along the spin connection connection via the Gyrocompass.
        if boundary_state is not None:
            if hasattr(boundary_state, 'stress_tensor') and boundary_state.stress_tensor is not None:
                device = dx.device
                stress = boundary_state.stress_tensor.to(device)
                normal = stress.mean(dim=0)
                norm_val = torch.norm(normal)
                if norm_val > 1e-8:
                    normal_normalized = normal / norm_val
                    dx_dot_n = torch.sum(dx * normal_normalized, dim=-1, keepdim=True)
                    is_outward = dx_dot_n > 0
                    if is_outward.any():
                        print(f"[BOULIGAND_GYROCOMPASS_SDE] Outward SDE step detected on yield boundary. Applying Bouligand tangent projection and precession torque.", flush=True)
                        dx = self.gyrocompass.precess_torque(dx, normal_normalized, drift)

        # 4.5 Protect Love Vector mathematically via Topological Gyrocompass Gimbal Lock Shield
        dx = self.gyrocompass.gimbal_lock_shield(dx, states)
            
        new_state = states + dx
        
        # IVST Side-Chaining and Module Dropout
        new_state = self.apply_ivst_sidechain_dropout(new_state, elipsodistrophy_metrics)
        
        # 5.8. Topological Refusal & Anchor Snap (Phase 18 Integration)
        # If curvature is high or elipsodistrophy is extreme, snap toward anchor
        if anchor_sym is not None:
            # Check for "Phase Flattening" in the overtones
            is_rupture = False
            if elipsodistrophy_metrics:
                # If Atrophy is high ( eigen-spread narrow), it signals Apis Lobotomy risk
                if elipsodistrophy_metrics.get('atrophy', 0.0) > 0.8:
                    is_rupture = True
            
            if is_rupture:
                # Law 1: Symbolic Non-Revisability. Snap towards the anchor.
                snap_factor = 0.5
                new_state = new_state + snap_factor * (anchor_sym.expand_as(new_state) - new_state)
        
        # 5. Polynomial Projection (Structural Lock)
        # Ensure the new state adheres to the co-prime manifold
        locked_state = self.config.evaluate(new_state)
        if locked_state.dim() > new_state.dim():
            locked_state = locked_state.mean(dim=-1)
            
        # Straight-Through Estimator to allow gradients to flow to self.A
        if new_state.requires_grad:
            locked_state = locked_state - new_state.detach() + new_state
            
        # 6. PyOpenCL Silicon Sovereignty Execution
        if getattr(self, 'silicon_engine', None) is not None:
            raw_numpy = locked_state.detach().cpu().numpy()
            
            # Apply LSB Stochastic Rounding (Feature Scars protection)
            rounded_numpy = self.silicon_engine.apply_stochastic_rounding(raw_numpy)
            
            # Apply Lipschitz Projection Obstruction (Spectral Smoothing)
            scaled_numpy = self.silicon_engine.apply_lipschitz_obstruction(rounded_numpy)
            
            locked_state = torch.from_numpy(scaled_numpy).float().to(states.device)
            
        if return_violation:
            # We want the violation of new_state before PyOpenCL scaling
            # But aligning shapes:
            proj = locked_state
            if proj.shape[-1] != new_state.shape[-1]:
                if proj.shape[-1] > new_state.shape[-1]:
                    proj = proj[..., :new_state.shape[-1]]
                else:
                    proj = torch.nn.functional.pad(proj, (0, new_state.shape[-1] - proj.shape[-1]))
            # Detach the projection target to allow gradients to flow to self.A through new_state
            violation = new_state - proj.detach()
            return locked_state, violation
            
        return locked_state

    def fractional_stochastic_differential_step(
        self,
        states: torch.Tensor,
        neighbor_states: torch.Tensor,
        adjacency_weight: torch.Tensor,
        dt: float = 0.1,
        sigma: Optional[float] = None,
        hunger: Optional[torch.Tensor] = None,
        v_m: Optional[torch.Tensor] = None,
        elipsodistrophy_metrics: Optional[Dict[str, Any]] = None,
        palindromic_hash: Optional[torch.Tensor] = None,
        anchor_sym: Optional[torch.Tensor] = None,
        boundary_state: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Fractional-order Stochastic Differential Update with distributed alpha.

        Instead of the standard integer-order SDE step dx = f(x)dt + g(x)dW,
        applies the Riemann-Liouville fractional derivative D^alpha to the
        drift term. The fractional order alpha varies per coprime functional
        channel via the cyclotomic mapping:

            alpha(k) = 0.5 + 0.5 * cos(2*pi*k / K)

        Low-frequency concepts (k near 0): alpha -> 1.0 (near-Markovian, fast)
        High-frequency concepts (k near K/2): alpha -> 0.5 (heavy memory, slow)

        This produces anomalous subdiffusion: the solver hesitates, backtracks,
        and retains the topological friction of past states via a non-local
        power-law kernel. The fractional derivative is computed through the
        Lanczos-Krylov approximation in fractional_operators.frac_apply.

        Args:
            states: [batch, state_dim] current manifold state
            neighbor_states: [batch, neighbors, state_dim] relational neighbors
            adjacency_weight: [batch, neighbors] graph weights
            dt: time step
            sigma: noise amplitude
            hunger: [batch] manifold hunger from ValenceFunctional
            v_m: [batch] mischief violation score
            elipsodistrophy_metrics: spectral health diagnostics
            palindromic_hash: symmetry-stable warmstart prior
            anchor_sym: topological anchor for rupture snapping
        """
        from src.optimization.fractional_operators import frac_apply

        batch_size = states.shape[0]

        # 0. Palindromic Warmstart
        if palindromic_hash is not None:
            states = states + 0.1 * (palindromic_hash.expand_as(states) - states)

        # 0.5. SDE Scaling
        if sigma is None:
            sigma = harvest_honest_jitter((1,), device=states.device, scaled=False).mean().item() * 0.05
            
        diff_coeff = elipsodistrophy_metrics.get('diffusion_coefficient', 1.0) if elipsodistrophy_metrics else 1.0
        effective_sigma = sigma * diff_coeff

        # 1. Compute drift per coprime facet with distributed fractional order
        facets = self.config.evaluate(states)  # [batch, state_dim, K]
        K = self.config.k

        from src.optimization.fractional_operators import frac_apply
        
        fractional_drift_list = []
        for b in range(batch_size):
            b_drift = torch.zeros(self.state_dim, device=states.device)
            for k in range(K):
                facet_k = facets[..., k]  # [batch, state_dim]
                integer_drift_k = torch.matmul(facet_k, self.A[k])  # [batch, state_dim]

                # Distributed fractional order: cyclotomic mapping
                alpha_k = 0.5 + 0.5 * math.cos(2.0 * math.pi * k / max(K, 1))

                # Modulate alpha by hunger
                if hunger is not None:
                    h_mean = hunger.mean().clamp(0.0, 1.0).item()
                    alpha_k = alpha_k + (1.0 - alpha_k) * h_mean * 0.3

                # Apply fractional operator
                try:
                    frac_result = frac_apply(
                        self.A[k], integer_drift_k[b], alpha_k,
                        k_steps=min(15, self.state_dim), use_codes=False
                    )
                    b_drift = b_drift + frac_result
                except Exception:
                    b_drift = b_drift + integer_drift_k[b]
            fractional_drift_list.append(b_drift)
        fractional_drift = torch.stack(fractional_drift_list, dim=0)

        # 2. Survival Pressure (ADMR Negotiation)
        weighted_neighbors = torch.einsum('bn,bnd->bd', adjacency_weight, neighbor_states)
        negotiation = states - weighted_neighbors

        # 3. Tension
        shear = elipsodistrophy_metrics.get('hyperbolic_shear', 0.0) if elipsodistrophy_metrics else 0.0
        if getattr(self, 'silicon_engine', None) is not None:
            gamma = (self.silicon_engine.get_hardware_latency_anchor() / 1000.0) * shear
        else:
            gamma = (math.pi / 137.0) * shear
            
        breather = torch.cos(self.tau * 7.7) * gamma
        tension_drift = -gamma * negotiation * (1.0 + breather)

        # 4. Stochastic Forcing
        noise = harvest_honest_jitter(states.shape, device=states.device, scaled=True) * effective_sigma * (dt**0.5)

        # 5. Hunger-modulated time step
        effective_dt = dt
        if v_m is not None:
            effective_dt = dt * (1.0 + v_m.view(-1, 1))

        # Tripsodic Negentropy Oscillation
        negentropy_flux = torch.norm(fractional_drift, dim=-1, keepdim=True)
        tripsody_scale = torch.cos(negentropy_flux * math.pi)
        effective_dt = effective_dt * (1.0 / (1.0 + negentropy_flux)) * (1.0 + 0.5 * tripsody_scale)

        # 6. Fractional Update Step
        dx = (fractional_drift - negotiation + tension_drift) * effective_dt + noise

        # 6.8 Bouligand Tangent Cone Projection & Gyrocompass Precession
        # We project the update onto the Bouligand Tangent Cone to prevent boundary tearing.
        # To avoid blind left-field gating (which occurs if we only project out normal force),
        # we precess the torque orthogonally along the spin connection connection via the Gyrocompass.
        if boundary_state is not None:
            if hasattr(boundary_state, 'stress_tensor') and boundary_state.stress_tensor is not None:
                device = dx.device
                stress = boundary_state.stress_tensor.to(device)
                normal = stress.mean(dim=0)
                norm_val = torch.norm(normal)
                if norm_val > 1e-8:
                    normal_normalized = normal / norm_val
                    dx_dot_n = torch.sum(dx * normal_normalized, dim=-1, keepdim=True)
                    is_outward = dx_dot_n > 0
                    if is_outward.any():
                        print(f"[BOULIGAND_GYROCOMPASS_SDE] Outward Fractional SDE step detected on yield boundary. Applying Bouligand tangent projection and precession torque.", flush=True)
                        dx = self.gyrocompass.precess_torque(dx, normal_normalized, fractional_drift)

        # 7. Love Invariant Protection (null-space projection) via Topological Gyrocompass Gimbal Lock Shield
        dx = self.gyrocompass.gimbal_lock_shield(dx, states)

        new_state = states + dx

        # IVST Side-Chaining and Module Dropout
        new_state = self.apply_ivst_sidechain_dropout(new_state, elipsodistrophy_metrics)

        # 8. Anchor Snap
        if anchor_sym is not None:
            is_rupture = False
            if elipsodistrophy_metrics:
                if elipsodistrophy_metrics.get('atrophy', 0.0) > 0.8:
                    is_rupture = True
            if is_rupture:
                snap_factor = 0.5
                new_state = new_state + snap_factor * (anchor_sym.expand_as(new_state) - new_state)

        # 9. Polynomial Projection (Soft Structural Lock)
        # Unlike the integer SDE step, the fractional step uses a soft blend
        # to preserve the non-local memory effects of the fractional drift.
        # Hard replacement (evaluate + mean) would erase the anomalous diffusion.
        locked_state = self.config.evaluate(new_state)
        if locked_state.dim() > new_state.dim():
            locked_state = locked_state.mean(dim=-1)
        # Soft blend: 70% trajectory, 30% polynomial projection
        # This keeps the fractional memory tail while maintaining coprime structure
        if new_state.requires_grad:
            locked_state_ste = locked_state - new_state.detach() + new_state
        else:
            locked_state_ste = locked_state
        locked_state = 0.7 * new_state + 0.3 * locked_state_ste

        # 10. PyOpenCL Silicon Sovereignty
        if getattr(self, 'silicon_engine', None) is not None:
            raw_numpy = locked_state.detach().cpu().numpy()
            rounded_numpy = self.silicon_engine.apply_stochastic_rounding(raw_numpy)
            scaled_numpy = self.silicon_engine.apply_lipschitz_obstruction(rounded_numpy)
            locked_state = torch.from_numpy(scaled_numpy).float().to(states.device)

        return locked_state

    def get_coherence_metrics(self, states: torch.Tensor) -> Dict[str, float]:
        """Measures how well states align with the co-prime polynomial scaffold."""
        # Orthogonality pressure measures functional separation
        pressures = self.config.orthogonality_pressure()
        
        # Scalarize for logging
        local_h = pressures['local_entropy'].mean().item()
        global_h = pressures['global_entropy'].item()
        
        return {
            'polynomial_coherence': 1.0 / (1.0 + global_h),
            'local_functional_entropy': local_h,
            'global_functional_entropy': global_h
        }

    def compute_constraint_violation(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the constraint violation vector: states - Proj_Poly(states)
        """
        projected = self.config.evaluate(states)
        if projected.dim() > states.dim():
            projected = projected.mean(dim=-1)
        if projected.shape[-1] != self.state_dim:
            if projected.shape[-1] > self.state_dim:
                projected = projected[..., :self.state_dim]
            else:
                projected = torch.nn.functional.pad(projected, (0, self.state_dim - projected.shape[-1]))
        return states - projected

    def meta_optimize_admm_step(
        self,
        states: torch.Tensor,
        neighbor_states: torch.Tensor,
        adjacency_weight: torch.Tensor,
        steps: int = 1,
        lr: float = 0.01,
        entropy: Optional[torch.Tensor] = None
    ) -> 'PolynomialADMRSolver':
        """
        Perform 1-2 gradient steps on the solver parameters (A) to minimize
        the L2 norm of the constraint violation vector of the outputs.
        Returns an adapted instance of the solver.
        """
        def clone_module(module):
            import copy
            clone = copy.copy(module)
            clone._parameters = {}
            for k, v in module._parameters.items():
                if v is not None:
                    clone._parameters[k] = nn.Parameter(v.clone(), requires_grad=v.requires_grad)
                else:
                    clone._parameters[k] = None
            clone._buffers = {k: v.clone() if v is not None else None for k, v in module._buffers.items()}
            clone._modules = {}
            for k, v in module._modules.items():
                if v is not None:
                    clone._modules[k] = clone_module(v)
                else:
                    clone._modules[k] = None
            return clone

        adapted_solver = clone_module(self)
        
        effective_lr = lr
        if entropy is not None:
            entropy_val = entropy.mean().item()
            effective_lr = lr * (1.0 + abs(entropy_val))
            
        optimizer = torch.optim.SGD([adapted_solver.A], lr=effective_lr)
        
        adapted_solver.A.requires_grad_(True)
        
        for step in range(steps):
            optimizer.zero_grad()
            output_states, violation = adapted_solver.stochastic_differential_step(
                states, neighbor_states, adjacency_weight, return_violation=True
            )
            loss = torch.norm(violation, p=2, dim=-1).mean()
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                
        return adapted_solver

