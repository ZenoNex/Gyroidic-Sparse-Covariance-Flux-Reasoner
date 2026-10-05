import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        
    def nabla_psi_star(self, y: torch.Tensor, max_iters: int = 20) -> torch.Tensor:
        r"""
        Computes the gradient of the Legendre transform \nabla \psi^*(y).
        By Envelope Theorem, \nabla \psi^*(y) = argmax_x <x, y> - \psi(x).
        This maps the source (noise) to the target.
        """
        # Initialize x at y as a warm start
        x = y.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([x], lr=1.0, max_iter=max_iters, line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            # Maximize <x, y> - \psi(x) => Minimize \psi(x) - <x, y>
            psi_x = self.icnn(x).squeeze(-1)
            dot_xy = torch.sum(x * y, dim=-1)
            loss = (psi_x - dot_xy).mean()
            loss.backward()
            return loss
            
        optimizer.step(closure)
        return x.detach()

    def langevin_prior_drift(self, z: torch.Tensor, gamma: float = 0.01, steps: int = 5) -> torch.Tensor:
        """
        Unadjusted Langevin Monte Carlo (ULA) drift to sample from the base prior \\mu_{W_\\theta}.
        z_{k+1} = z_k - \\gamma \\nabla W_\\theta(z_k) + \\sqrt{2\\gamma} \\eta_k
        Where \\eta_k is sourced from physical silicon anomalies via harvest_honest_jitter.
        """
        z_k = z.clone().detach().requires_grad_(True)
        
        for _ in range(steps):
            # Evaluate convex potential W_\\theta(z_k)
            W_z = self.icnn(z_k).sum()
            
            # Compute \\nabla W_\\theta(z_k)
            grad_W = torch.autograd.grad(W_z, z_k)[0]
            
            # Harvest honest jitter for \\eta_k
            eta_k = harvest_honest_jitter(z_k.shape, device=z_k.device, scaled=True)
            
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

    def extract_oklab_moment_field(self, rgb_points: torch.Tensor) -> torch.Tensor:
        """
        Calculates the first and second moments (Mean and Covariance structure) 
        of a point cloud's color palette in OKLab space.
        Returns a compressed moment field descriptor suitable for transport.
        """
        oklab = self.rgb_to_oklab(rgb_points)
        # First moment (Mean)
        mu = oklab.mean(dim=-2, keepdim=True)
        # Second moment (Covariance proxy via outer product trace)
        centered = oklab - mu
        cov = torch.matmul(centered.transpose(-1, -2), centered) / max(1, centered.shape[-2])
        
        # Flatten and concatenate the moments into a target state vector
        # (L, a, b) + 9 covariance elements = 12-dim moment field
        moment_field = torch.cat([mu.view(-1, 3), cov.view(-1, 9)], dim=-1)
        return moment_field

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
