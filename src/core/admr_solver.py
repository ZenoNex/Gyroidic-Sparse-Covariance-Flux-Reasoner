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


class PolynomialADMRSolver(nn.Module):
    """
    Implements the Polynomial ADMR update:
    S^{(n+1)} = Proj_{Poly} [ S^{(n)} * Σ w_ik S_k ]
    
    Uses co-prime polynomial functionals to ensure discrete-like 
    information separation in a differentiable, continuous field.
    """
    
    def __init__(
        self,
        poly_config: PolynomialCoprimeConfig,
        state_dim: int,
        eta_scaffold: float = 0.01,
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
        self.eta_scaffold = eta_scaffold
        self.device = device
        
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
        self.A = nn.Parameter(torch.randn(num_facet_channels, state_dim, state_dim, device=device) * 0.01)
        
        # 3. Stochastic Forcing Buffer
        self.register_buffer('eta', torch.randn(state_dim, device=device) * 0.005)
        
        # 4. Chiral Residue Cache (Warm-start backtracking)
        self.register_buffer('chiral_cache', torch.zeros(1, state_dim, device=device))
        self.register_buffer('cache_valid', torch.tensor(False))

        # 5. Love Invariant Protector (Null-space projection)
        from .love_invariant_protector import LoveInvariantProtector
        self.love_protector = LoveInvariantProtector(love_dim=max(1, state_dim // 4), device=device)
        
        # 6. KAGH-Boltzmann Surrogate for Continuous-to-Discrete jumps
        # The surrogate maps continuous Polynomial projections to discrete Matrioshka states
        # via B-splines and Saturated Quantizers.
        try:
            from src.surrogates.kagh_networks import KAGHBlock
            self.kagh_surrogate = KAGHBlock(n_in=state_dim, n_out=state_dim, width=max(16, state_dim // 2), depth=2).to(device)
        except ImportError:
            self.kagh_surrogate = None

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
        # Σ R_ik S_k
        weighted_neighbors = torch.einsum('bn,bnd->bd', adjacency_weight, neighbor_states)
        
        # 2. Multiplicative interaction: S_i * (Σ S_k + V)
        v_drive = valence.unsqueeze(-1) if valence is not None else 1.0
        interaction = states * (weighted_neighbors + v_drive)
        
        # 3. Polynomial Projection (Functional Co-primality)
        # Instead of 'remainder', we evaluate the interaction through the co-prime basis.
        # This acts as a 'soft modulus' that preserves symbolic structure.
        projected = self.config.evaluate(interaction)
        
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
            # using the surrogate to handle Saturated Quantizers and Gödel gates.
            projected = self.kagh_surrogate(projected)
                
        return projected

    def stochastic_differential_step(
        self, 
        states: torch.Tensor, 
        neighbor_states: torch.Tensor, 
        adjacency_weight: torch.Tensor,
        dt: float = 0.1,
        sigma: float = 0.01,
        v_m: Optional[torch.Tensor] = None,
        elipsodistrophy_metrics: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Continuous-time Stochastic Differential Update:
        dx(t) = [ Σ A_i x_i(t) - ρ Σ (x - r(x_k)) + Γ_tension ] dt + σ(D) dW
        
        If v_m is provided, learning rate of dual variables is tied strictly to it.
        Diffusion σ is scaled by the dynamic Hyperbolic Shear D.
        """
        batch_size = states.shape[0]
        
        # 0. SDE Scaling (Poincaré Eccentricity Driver)
        # We scale noise by the diffusion coefficient derived from spectral shear.
        diff_coeff = elipsodistrophy_metrics.get('diffusion_coefficient', 1.0) if elipsodistrophy_metrics else 1.0
        effective_sigma = sigma * diff_coeff
        
        # 1. Non-selfadjoint Drifts (Σ A_i x_i)
        # We treat the co-prime evaluation as the 'decomposition' into facets
        facets = self.config.evaluate(states) # [batch, state_dim, num_functionals]
        
        # A @ facets: [num_functionals, state_dim, state_dim] @ [batch, state_dim, num_functionals]
        # We sum over facets
        drift = torch.zeros_like(states)
        for i in range(self.config.k):
            # facet_i: [batch, state_dim]
            facet_i = facets[..., i]
            # A_i: [state_dim, state_dim]
            drift += torch.matmul(facet_i, self.A[i])
            
        # 2. Survival Pressure (ADMR Negotiation)
        weighted_neighbors = torch.einsum('bn,bnd->bd', adjacency_weight, neighbor_states)
        # Negotiation term: states - weighted_neighbors
        negotiation = states - weighted_neighbors

        # 3. Perkins Tension Objective (Minimal Surface Tension gamma)
        # Resolves "cubed cube" paradoxes by minimizing surface tension in the RP^4 Void.
        # The tension is fossilized as a Chiral Breather—a persistent topological soliton.
        shear = elipsodistrophy_metrics.get('hyperbolic_shear', 0.0) if elipsodistrophy_metrics else 0.0
        gamma = 0.02 * shear # Surface tension coefficient
        breather = torch.cos(self.tau * 7.7) * gamma # Chiral Breather component
        tension_drift = -gamma * negotiation * (1.0 + breather)
        
        # 4. Stochastic Forcing (dW)
        noise = torch.randn_like(states) * effective_sigma * (dt**0.5)
        
        # V_m explicit learning rate modulation
        # Dual variables S evolution tied to the normalized Mischief Score V_m
        effective_dt = dt
        if v_m is not None:
             effective_dt = dt * (1.0 + v_m.view(-1, 1))

        # Tripsodic Negentropy Oscillation:
        # As negentropy (information density) increases, the system phase-locks
        # via a tripartite rhapsodic oscillation rather than freezing.
        # This creates expansion at singularities instead of stasis.
        negentropy_flux = torch.norm(drift, dim=-1, keepdim=True)
        tripsody_scale = torch.cos(negentropy_flux * math.pi)
        effective_dt = effective_dt * (1.0 / (1.0 + negentropy_flux)) * (1.0 + 0.5 * tripsody_scale)

        # 5. Update Step (Continuous Approximation)
        # dx = (drift - negotiation + tension_drift) * dt + noise
        dx = (drift - negotiation + tension_drift) * effective_dt + noise
        
        # 4.5 Protect Love Vector mathematically by projecting update to null-space of ownership operator
        ownership_op = self.love_protector.compute_ownership_operator(states)
        null_proj = self.love_protector.compute_null_space_projection(ownership_op)
        
        love_dim = self.love_protector.love_dim
        if dx.shape[-1] == love_dim:
            dx = torch.matmul(dx, null_proj.T)
        elif dx.shape[-1] > love_dim:
            dx_subset = dx[..., :love_dim]
            dx[..., :love_dim] = torch.matmul(dx_subset, null_proj.T)
            
        new_state = states + dx
        
        # 5. Polynomial Projection (Structural Lock)
        # Ensure the new state adheres to the co-prime manifold
        locked_state = self.config.evaluate(new_state)
        if locked_state.dim() > new_state.dim():
            locked_state = locked_state.mean(dim=-1)
            
        # 6. PyOpenCL Silicon Sovereignty Execution
        if getattr(self, 'silicon_engine', None) is not None:
            raw_numpy = locked_state.detach().cpu().numpy()
            
            # Apply LSB Stochastic Rounding (Feature Scars protection)
            rounded_numpy = self.silicon_engine.apply_stochastic_rounding(raw_numpy)
            
            # Apply Lipschitz Projection Obstruction (Spectral Smoothing)
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

