import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from src.surrogates.kagh_networks import InputConvexNeuralNetwork
from src.core.honest_jitter import harvest_honest_jitter

class ConjugateMomentTransport(nn.Module):
    """
    Conjugate Moment Measure Factorization via Convex Potentials.
    Implements optimal transport for highly concentrated states (converged/collapsed)
    using an Input Convex Neural Network (ICNN) to model the potential psi.
    
    Includes OKLab Moment Field projections for perceptual color mappings, 
    teased out of the optimal transport math.
    """
    def __init__(self, dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.icnn = InputConvexNeuralNetwork(dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # CC-THEORY-001 Integration: Mandelbulb Substrate Clock (96-unit tied by CRT)
        from src.core.fgrt_primitives import PrimeResonanceLadder
        self.clock = PrimeResonanceLadder(num_resonators=96)
        
    def nabla_psi_star(self, y: torch.Tensor, max_iters: int = 20) -> torch.Tensor:
        r"""
        Computes the gradient of the Legendre transform \nabla \psi^*(y).
        By Envelope Theorem, \nabla \psi^*(y) = argmax_x <x, y> - \psi(x).
        This maps the source (noise) to the target.
        """
        # Initialize x at y as a warm start, detaching y to prevent multi-backward graph conflicts
        y_detached = y.detach()
        x = y_detached.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([x], lr=1.0, max_iter=max_iters, line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            # Maximize <x, y> - \psi(x) => Minimize \psi(x) - <x, y>
            psi_x = self.icnn(x).squeeze(-1)
            dot_xy = torch.sum(x * y_detached, dim=-1)
            loss = (psi_x - dot_xy).mean()
            loss.backward()
            return loss
            
        optimizer.step(closure)
        return x.detach()

    def langevin_prior_drift(self, z: torch.Tensor, gamma: float = 0.01, steps: int = 5, coh: float = 0.5) -> torch.Tensor:
        """
        Unadjusted Langevin Monte Carlo (ULA) drift to sample from the base prior \\mu_{W_\\theta}.
        z_{k+1} = z_k - \\gamma \\nabla W_\\theta(z_k) + \\sqrt{2\\gamma} \\eta_k
        Where \\eta_k is sourced from physical silicon anomalies via harvest_honest_jitter.
        The jitter is modulated by the PrimeResonanceLadder tied to the `coh` scalar
        to natively control sheet-spacing and breathing.
        """
        z_k = z.clone().detach().requires_grad_(True)
        
        # Use coherence to scale the amplitude of the resonance clock
        clock_signal = torch.sin(self.clock.frequencies[0]) * coh
        breathing_factor = 1.0 + clock_signal.item()
        
        for step_idx in range(steps):
            # Evaluate convex potential W_\\theta(z_k)
            W_z = self.icnn(z_k).sum()
            
            # Compute \\nabla W_\\theta(z_k)
            grad_W = torch.autograd.grad(W_z, z_k)[0]
            
            # Harvest honest jitter for \\eta_k
            eta_k = harvest_honest_jitter(z_k.shape, device=z_k.device, scaled=True)
            
            # Modulate jitter by the substrate clock breathing factor
            clock_modulator = torch.sin(self.clock.frequencies[step_idx % self.clock.num_resonators]).item()
            eta_k = eta_k * (breathing_factor + clock_modulator * (1.0 - coh))
            
            # Langevin update step
            with torch.no_grad():
                z_k -= gamma * grad_W
                z_k += math.sqrt(2 * gamma) * eta_k
                
        return z_k.detach()

    def rgb_to_oklab(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Transforms linear RGB into the OKLab perceptual moment field space.
        Uses the standard M1 and M2 matrices for LMS cone response.
        """
        # Linear RGB to LMS
        M1 = torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]
        ], dtype=rgb.dtype, device=rgb.device)
        
        lms = torch.matmul(rgb, M1.T)
        lms_non_linear = torch.sign(lms) * torch.pow(torch.abs(lms), 1.0/3.0)
        
        # LMS to OKLab
        M2 = torch.tensor([
            [ 0.2104542553,  0.7936177850, -0.0040720468],
            [ 1.9779984951, -2.4285922050,  0.4505937099],
            [ 0.0259040371,  0.7827717662, -0.8086757660]
        ], dtype=rgb.dtype, device=rgb.device)
        
        return torch.matmul(lms_non_linear, M2.T)

    def extract_oklab_moment_field(
        self, 
        rgb_points: torch.Tensor, 
        spatial_coords: Optional[torch.Tensor] = None,
        use_log_cholesky: bool = True
    ) -> torch.Tensor:
        """
        Calculates the first and second moments (Mean and Covariance structure) 
        of a point cloud's color palette in OKLab space.
        Returns a compressed moment field descriptor suitable for transport.
        
        When spatial_coords is provided:
        Interleaves spatial coordinates along the 3D Morton Z-order curve
        using morton_encode_3d to preserve metric locality and prevent CPU/GPU
        cache misses before computing local moment statistics.
        
        When use_log_cholesky is True:
        Decomposes the 3x3 covariance matrix into its lower-triangular Cholesky factor L:
            Sigma = L L^T
            v_cholesky = [log(L_11), log(L_22), log(L_33), L_21, L_31, L_32] in R^6
        This maps S_{++}^3 diffeomorphically to Euclidean space R^6, preserving matrix Riemannian
        distances and guaranteeing positive-definiteness by construction without determinant swelling.
        Moment field shape: (L, a, b) [3] + 6 (cholesky) = 9-dim Riemannian invariant vector.
        
        When use_log_cholesky is False:
        Returns flattened 3x3 covariance: 3 (mean) + 9 (cov) = 12-dim vector (legacy format).
        """
        # If spatial coordinates provided, sort points along Morton Z-order curve
        if spatial_coords is not None:
            from src.core.fgrt_primitives import morton_encode_3d
            c_min = spatial_coords.min(dim=-2, keepdim=True)[0]
            c_max = spatial_coords.max(dim=-2, keepdim=True)[0]
            c_range = (c_max - c_min).clamp(min=1e-5)
            norm_coords = ((spatial_coords - c_min) / c_range * 1023.0).clamp(0, 1023).long()
            
            morton_keys = morton_encode_3d(norm_coords[..., 0], norm_coords[..., 1], norm_coords[..., 2])
            sort_indices = torch.argsort(morton_keys, dim=-1)
            rgb_points = torch.gather(rgb_points, -2, sort_indices.unsqueeze(-1).expand_as(rgb_points))

        oklab = self.rgb_to_oklab(rgb_points)
        # First moment (Mean)
        mu = oklab.mean(dim=-2, keepdim=True)
        # Second moment (Covariance proxy via outer product trace)
        centered = oklab - mu
        cov = torch.matmul(centered.transpose(-1, -2), centered) / max(1, centered.shape[-2])
        
        if use_log_cholesky:
            # Regularize with numerical epsilon for strict positive-definiteness
            eps = 1e-6
            eye = torch.eye(3, device=cov.device, dtype=cov.dtype)
            cov_reg = cov + eye * eps
            
            # Lower-triangular Cholesky factor L
            L = torch.linalg.cholesky(cov_reg)
            
            # Logarithm of positive diagonals: log(L_11), log(L_22), log(L_33)
            log_diag = torch.log(torch.diagonal(L, dim1=-2, dim2=-1))
            
            # Lower off-diagonal elements: L_21, L_31, L_32
            L_21 = L[..., 1, 0:1]
            L_31 = L[..., 2, 0:1]
            L_32 = L[..., 2, 1:2]
            lower_off_diag = torch.cat([L_21, L_31, L_32], dim=-1)
            
            # 6-dim Log-Cholesky SPD Riemannian vector
            cholesky_vec = torch.cat([log_diag, lower_off_diag], dim=-1)
            
            # Concatenate: (L, a, b) [3] + 6 cholesky = 9-dim Riemannian moment field
            moment_field = torch.cat([mu.view(-1, 3), cholesky_vec.view(-1, 6)], dim=-1)
        else:
            # Flatten and concatenate the moments into a target state vector
            # (L, a, b) + 9 covariance elements = 12-dim moment field
            moment_field = torch.cat([mu.view(-1, 3), cov.view(-1, 9)], dim=-1)
            
        return moment_field

    @staticmethod
    def reconstruct_covariance_from_cholesky(cholesky_vec: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the 3x3 positive-definite covariance matrix from the 6-dim Log-Cholesky vector:
        v = [log(L_11), log(L_22), log(L_33), L_21, L_31, L_32]
        """
        batch_shape = cholesky_vec.shape[:-1]
        log_diag = cholesky_vec[..., :3]
        L_21 = cholesky_vec[..., 3]
        L_31 = cholesky_vec[..., 4]
        L_32 = cholesky_vec[..., 5]
        
        diag = torch.exp(log_diag)
        L = torch.zeros(*batch_shape, 3, 3, device=cholesky_vec.device, dtype=cholesky_vec.dtype)
        L[..., 0, 0] = diag[..., 0]
        L[..., 1, 1] = diag[..., 1]
        L[..., 2, 2] = diag[..., 2]
        L[..., 1, 0] = L_21
        L[..., 2, 0] = L_31
        L[..., 2, 1] = L_32
        
        return torch.matmul(L, L.transpose(-1, -2))

    @staticmethod
    def reconstruct_from_moment_field(
        moment_field: torch.Tensor, 
        use_log_cholesky: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstructs the mean (L, a, b) and 3x3 covariance matrix from a moment field vector.
        Returns:
            mu: [..., 3] mean color in OKLab space
            cov: [..., 3, 3] reconstructed positive-definite covariance matrix
        """
        mu = moment_field[..., :3]
        if use_log_cholesky:
            cholesky_vec = moment_field[..., 3:9]
            cov = ConjugateMomentTransport.reconstruct_covariance_from_cholesky(cholesky_vec)
        else:
            cov = moment_field[..., 3:12].view(*moment_field.shape[:-1], 3, 3)
        return mu, cov

    def trigonometric_unfolding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ternary Branch Connection for Casus Irreducibilis (cubic degeneracy).
        If the state hits a singularity where the polynomial roots degenerate,
        we unfold into 3 chiral branches and pick the one with highest negentropy (variance).
        """
        # Construct the three ternary angles: theta, theta + 2pi/3, theta - 2pi/3
        # We proxy this by splitting the vector phase.
        norm = x.norm(dim=-1, keepdim=True) + 1e-8
        direction = x / norm
        
        # Branch 1: Identity
        b1 = x
        
        # Branch 2 & 3: Gyroidic chiral twists
        # Using a simple block rotation proxy for the twist
        twist = torch.zeros_like(x)
        half = self.dim // 2
        twist[..., :half] = -direction[..., half:2*half]
        twist[..., half:2*half] = direction[..., :half]
        
        # 120 degrees (2pi/3) and -120 degrees
        cos_120 = -0.5
        sin_120 = 0.866025
        
        b2 = norm * (direction * cos_120 + twist * sin_120)
        b3 = norm * (direction * cos_120 - twist * sin_120)
        
        # Pick the branch that maximizes structural negentropy (using variance as proxy)
        vars = torch.stack([b1.var(dim=-1), b2.var(dim=-1), b3.var(dim=-1)], dim=-1)
        best_branch = torch.argmax(vars, dim=-1) # [batch]
        
        # Gather best branches
        result = torch.zeros_like(x)
        for i in range(x.shape[0]):
            if best_branch[i] == 0:
                result[i] = b1[i]
            elif best_branch[i] == 1:
                result[i] = b2[i]
            else:
                result[i] = b3[i]
                
        return result

    def compute_context_adaptive_monge_ampere_loss(
        self, 
        source: torch.Tensor, 
        target: torch.Tensor, 
        pas_anisotropy: torch.Tensor, 
        shell_depth: int
    ) -> torch.Tensor:
        r"""
        Computes the Context-Adaptive Monge-Ampère Loss.
        Scales the potential by the per-axis Phase Alignment Score (pas_anisotropy)
        and the Matrioshka shell depth.
        
        Using the dual formulation of W2 optimal transport:
        L = E_x[\psi(x)] + E_y[\psi^*(y)]
        """
        # Forward pass on source
        psi_x = self.icnn(source).squeeze(-1) # [batch]
        
        # To compute \psi^*(y), we need \sup_x <x,y> - \psi(x)
        x_star = self.nabla_psi_star(target)
        psi_star_y = torch.sum(x_star * target, dim=-1) - self.icnn(x_star).squeeze(-1)
        
        # Scale by shell depth and PAS anisotropy
        # Depth scaling: deeper shells (larger depth) should have lower loss weight 
        # as they are finer adjustments.
        depth_scale = 1.0 / (2 ** shell_depth)
        
        # PAS scaling: average anisotropy across dimensions
        pas_scale = pas_anisotropy.mean(dim=-1)
        
        # Context-adaptive loss
        loss = (psi_x + psi_star_y) * depth_scale * pas_scale
        return loss.mean()
        
    def project_to_cayley_surface(self, state: torch.Tensor, is_honeybee_mode: bool = False) -> torch.Tensor:
        """
        Projects the first 3 dimensions onto the Cayley Cubic surface: x^2 + y^2 + z^2 - xyz = 4.
        Bypassed if state has fewer than 3 dimensions, or if we are under high computational load
        ('Honeybee' collective computing mode) where curvature collapse is permitted.
        """
        if state.shape[-1] < 3 or is_honeybee_mode:
            return state
            
        coords = state[..., :3]
        # Evaluate Cayley polynomial V = x^2 + y^2 + z^2 - xyz - 4
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        V = x**2 + y**2 + z**2 - x*y*z - 4.0
        
        # Single Newton-Raphson step to push the state back to V = 0
        # grad V = (2x - yz, 2y - xz, 2z - xy)
        grad_x = 2*x - y*z
        grad_y = 2*y - x*z
        grad_z = 2*z - x*y
        
        grad_norm_sq = grad_x**2 + grad_y**2 + grad_z**2 + 1e-8
        
        # Newton step: delta = -V / |grad V|^2 * grad V
        step = -V / grad_norm_sq
        x_new = x + step * grad_x
        y_new = y + step * grad_y
        z_new = z + step * grad_z
        
        projected = state.clone()
        projected[..., 0] = x_new
        projected[..., 1] = y_new
        projected[..., 2] = z_new
        
        return projected
        
    def is_parabolic_singularity(self, state: torch.Tensor, tol: float = 0.1) -> torch.Tensor:
        """
        Detect if the state is anchored at one of the four A_1 singularities (+/-2, +/-2, +/-2).
        Returns a boolean mask of shape [batch].
        """
        if state.shape[-1] < 3:
            return torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
            
        coords = state[..., :3]
        singularities = torch.tensor([
            [2.0, 2.0, 2.0],
            [2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, -2.0, 2.0]
        ], dtype=torch.float32, device=state.device)
        
        # distance to closest singularity
        dist = torch.cdist(coords.view(-1, 3), singularities) # [batch, 4]
        min_dist = torch.min(dist, dim=-1)[0]
        return min_dist < tol

    def forward(self, state: torch.Tensor, is_honeybee_mode: bool = False) -> torch.Tensor:
        """
        Execute the conjugate transport mapping on the state.
        If anchored at a parabolic singularity, applies Trigonometric Unfolding first 
        to break the non-diagonalizable Jordan block.
        Then projects to Cayley surface if vector shape allows and not in Honeybee mode.
        """
        # Phase 2: Parabolic Warm-Start
        parabolic_mask = self.is_parabolic_singularity(state)
        if parabolic_mask.any():
            # Apply unfolding specifically to break out of singularity
            unfolded = self.trigonometric_unfolding(state)
            state = torch.where(parabolic_mask.unsqueeze(-1), unfolded, state)
            
        # Map using \nabla \psi^*
        transported = self.nabla_psi_star(state)
        
        # Phase 1: Cayley-Constrained Transport (Hard Projection)
        transported_cayley = self.project_to_cayley_surface(transported, is_honeybee_mode=is_honeybee_mode)
        
        # Apply ternary chiral branch unfolding for any other degeneracies
        transported_unfolded = self.trigonometric_unfolding(transported_cayley)
        
        return transported_unfolded
