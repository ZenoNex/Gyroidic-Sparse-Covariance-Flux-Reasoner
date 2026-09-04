import torch
import numpy as np
from typing import Tuple, List
from src.core.martinova_correlation import compute_bounded_correlation

class JSpacePCAMapper:
    """
    Extracts principal directions from the intermediate Gyroidic Flux tensors and maps
    them back to the Z-space (initial coordinates) to find Sovereign Exemption Tokens.
    
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
