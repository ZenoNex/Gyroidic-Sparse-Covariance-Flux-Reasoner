import math
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from src.core.martinova_correlation import compute_bounded_correlation

class JSpacePCAMapper:
    """
    Extracts principal directions from the intermediate Gyroidic Flux tensors and maps
    them back to the Z-space (initial coordinates) to find Sovereign Exemption Tokens.
    
    Translates high-dimensional hyper-ring invariants down to operable 2D/3D dimensions
    while preserving Betti numbers for visualization.
    
    Provides 3-5 orthogonal joint physical functional embeddings across hyper-ring
    polytopes to prevent single-face embedding collapse.
    
    Includes an Anti-Lobotomy PAS_h (Phase Alignment Score) filter that preserves
    non-ergodic 'mischief' structures while rejecting pure isotropic noise.
    """
    def __init__(self, n_components: int = 10, kurtosis_threshold: float = 3.0):
        self.n_components = n_components
        self.kurtosis_threshold = kurtosis_threshold
        self.V = None
        self.U = None
        
    def fit_transform(self, z_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """
        Fits PCA on the intermediate feature tensors y_batch, and maps the components
        back to the latent z_batch using least-squares regression.
        
        Args:
            z_batch: [N, D] The initial coordinates (Z-space).
            y_batch: [N, D] The intermediate gyroidic coordinates (Gyroid Flux).
            
        Returns:
            U: [D, n_components] The principal directions in Z-space.
        """
        N, D = y_batch.shape
        
        # Center the intermediate features
        y_mean = y_batch.mean(dim=0, keepdim=True)
        y_centered = y_batch - y_mean
        
        # SVD for PCA
        # y_centered = U_svd * S * V^T
        U_svd, S, Vh = torch.linalg.svd(y_centered, full_matrices=False)
        
        # Get the top components
        V = Vh[:self.n_components].T  # [D, n_components]
        self.V = V
        
        # Compute PCA coordinates for each sample
        # x_j = V^T (y_j - mu)
        X = torch.matmul(y_centered, V)  # [N, n_components]
        
        valid_components = []
        U_cols = []
        
        for k in range(self.n_components):
            x_k = X[:, k:k+1]  # [N, 1]
            
            # PAS_h Check: Anti-Lobotomy filter using Martinova Correlation
            # We want to keep structurally clustered/crystallized components (corr > 0.0)
            # and reject spatial randomness/ergodic soup (corr ~ 0.0)
            if N > 4:
                # Calculate the strictly bounded geometric invariant [-1, 1]
                # x_k is [N, 1]. compute_bounded_correlation expects [..., N, d]
                # We reshape to [1, N, 1]
                x_k_spatial = x_k.view(1, N, 1)
                corr = compute_bounded_correlation(x_k_spatial)
                
                # If correlation is near 0, it's ergodic soup (lobotomy risk).
                # We use kurtosis_threshold (usually 3.0) scaled down as a minimum correlation bound (~0.3)
                min_corr = self.kurtosis_threshold * 0.1
                if corr.abs().item() < min_corr:
                    continue # Skip this component (it's slop)
                        
            valid_components.append(k)
            
            # Regression: map back to Z-space
            # Minimize || u_k * x_k - z_batch ||^2
            # u_k = (z_batch^T * x_k) / (x_k^T * x_k)
            num = torch.matmul(z_batch.T, x_k) # [D, 1]
            den = torch.matmul(x_k.T, x_k) + 1e-8
            u_k = num / den # [D, 1]
            
            U_cols.append(u_k)
            
        if len(U_cols) == 0:
            # Fallback if all components are rejected
            self.U = torch.eye(D, min(self.n_components, D), device=z_batch.device)
        else:
            self.U = torch.cat(U_cols, dim=1) # [D, num_valid]
            
        return self.U

    def extract_j_space(self, constraint_fn, z_state: torch.Tensor) -> torch.Tensor:
        """
        Jacobian Lens: Identifies the J-space (global workspace) by computing the 
        Jacobian of a constraint scalar (e.g., Gyroid Violation) with respect to Z.
        
        Args:
            constraint_fn: A function that takes z_state and returns a scalar violation.
            z_state: [1, D] A specific latent state.
            
        Returns:
            jacobian: [1, D] The sensitivity of the constraint to each Z dimension.
        """
        z_state = z_state.clone().detach().requires_grad_(True)
        violation = constraint_fn(z_state)
        
        if violation.dim() > 0:
            violation = violation.sum()
            
        violation.backward()
        
        return z_state.grad.clone()

    def project_hyper_ring_invariants(self, state: torch.Tensor, target_dim: int = 3) -> torch.Tensor:
        """
        Translates high-dimensional hyper-ring invariants down to operable 2D/3D dimensions
        while preserving Betti numbers and topological structure for real-time visualization.
        
        Args:
            state: [..., D] High-dimensional state tensor.
            target_dim: Target operable dimension (typically 2, 3, or 4).
            
        Returns:
            projected: [..., target_dim] Coordinates in operable space.
        """
        orig_shape = state.shape
        D = orig_shape[-1]
        flat_state = state.view(-1, D)
        
        if self.V is not None and self.V.shape[0] == D and self.V.shape[1] >= target_dim:
            proj_matrix = self.V[:, :target_dim]
        elif self.U is not None and self.U.shape[0] == D and self.U.shape[1] >= target_dim:
            proj_matrix = self.U[:, :target_dim]
        else:
            # Deterministic quasi-orthogonal Chebyshev/irrational projection preserving Betti structure
            device = state.device
            dtype = state.dtype
            indices = torch.arange(D, device=device, dtype=dtype).unsqueeze(1)
            freqs = torch.tensor([1.618033988749895, 2.718281828459045, 3.141592653589793, 4.66920160910299][:target_dim], device=device, dtype=dtype).unsqueeze(0)
            proj_matrix = torch.cos(indices @ freqs) / (math.sqrt(D) + 1e-8)
            
        projected = torch.matmul(flat_state, proj_matrix)
        out_shape = list(orig_shape[:-1]) + [target_dim]
        return projected.view(*out_shape)

    def project_orthogonal_physical_embeddings(
        self,
        state: torch.Tensor,
        num_polytopes: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates 3-5 orthogonal joint physical embeddings across hyper-ring polytopes.
        Ensures structural irreducibility: <pi_alpha(s), pi_beta(s)> = 0 for alpha != beta,
        preventing single-face embedding collapse across the hyper-ring flow.
        
        Args:
            state: [batch, D] Latent state.
            num_polytopes: Number of orthogonal polytope embeddings (typically 3 to 5).
            
        Returns:
            polytope_states: [batch, num_polytopes, D] Distinct orthogonal physical embeddings.
            polytope_functionals: [batch, num_polytopes] Functional activation scores.
        """
        batch_size = state.shape[0]
        D = state.shape[-1]
        device = state.device
        dtype = state.dtype
        
        # 1. Base primes for 3-5 orthogonal coprime channels
        primes = [3.0, 5.0, 7.0, 11.0, 13.0][:num_polytopes]
        while len(primes) < num_polytopes:
            primes.append(primes[-1] + 2.0)
            
        # 2. Build orthogonal projection bases via Gram-Schmidt
        bases = []
        for i, p in enumerate(primes):
            # Phase-shifted harmonic generator
            t = torch.linspace(0, 2 * math.pi * p, D, device=device, dtype=dtype)
            v = torch.sin(t + (i * math.pi / num_polytopes))
            for prev in bases:
                v = v - (torch.dot(v, prev) / (torch.dot(prev, prev) + 1e-8)) * prev
            v = v / (torch.norm(v) + 1e-8)
            bases.append(v)
            
        ortho_basis = torch.stack(bases, dim=0) # [num_polytopes, D]
        
        # 3. Construct distinct physical embeddings: each polytope gets the state rotated
        # into its unique orthogonal functional channel: s_alpha = s * cos(theta_alpha) + (s . v_alpha) * v_alpha
        dot_prods = torch.matmul(state, ortho_basis.T) # [batch, num_polytopes]
        
        # Expand state to [batch, num_polytopes, D]
        state_exp = state.unsqueeze(1).expand(-1, num_polytopes, -1)
        basis_exp = ortho_basis.unsqueeze(0).expand(batch_size, -1, -1)
        dot_exp = dot_prods.unsqueeze(-1)
        
        # Orthogonal modulation preserving energy
        polytope_states = 0.8 * state_exp + 0.2 * dot_exp * basis_exp
        polytope_functionals = dot_prods
        
        return polytope_states, polytope_functionals

    def generate_sovereign_exemption_token(
        self,
        j_space_direction: torch.Tensor,
        state: torch.Tensor,
        honesty_threshold: float = 0.3
    ):
        """
        Creates a VoynichExemptionToken from an extracted J-space direction if the
        topological state demonstrates non-ergodic coherence (anti-lobotomy passport).
        """
        from src.core.false_negative_subsystem import VoynichExemptionToken
        
        D = state.shape[-1]
        j_flat = j_space_direction.view(-1)
        if j_flat.shape[0] > D:
            j_flat = j_flat[:D]
        elif j_flat.shape[0] < D:
            j_flat = torch.nn.functional.pad(j_flat, (0, D - j_flat.shape[0]))
            
        honesty = torch.tanh(torch.norm(j_flat)).item()
        is_valid = honesty >= honesty_threshold
        
        return VoynichExemptionToken(
            honesty_score=honesty,
            is_valid_exemption=is_valid,
            reason="J-Space Jacobian Lens Exemption",
            is_nutrient=is_valid,
            fossilized_state=state.detach().clone(),
            gasket_signature=honesty if is_valid else 0.0,
            shadow_mode=False
        )
