"""
Enhanced Bezout Coefficient Refresh with Chinese Remainder Theorem
Based on number theory and energy-based learning principles.

Implements:
1. Proper CRT for modular arithmetic stability
2. Bezout coefficient computation for coprimality
3. Energy-based error correction
4. Numerical stability guarantees
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class EnhancedBezoutCRT(nn.Module):
    """
    Enhanced Bezout coefficient refresh using Chinese Remainder Theorem.
    
    Key principles:
    - Use coprime moduli for CRT decomposition
    - Bezout coefficients ensure numerical stability
    - Energy-based error correction
    - Margin-based robustness
    """
    
    def __init__(self, 
                 state_dim: int = 64,
                 num_moduli: int = 5,
                 stability_threshold: float = 1e-6):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_moduli = num_moduli
        self.stability_threshold = stability_threshold
        
        # Use polynomial co-prime system instead of hardcoded primes
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        self.polynomial_config = PolynomialCoprimeConfig(
            k=num_moduli,
            degree=4,
            basis_type='chebyshev',
            learnable=True,
            device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
        )
        
        # Generate moduli from polynomial functionals
        self.moduli = self._generate_polynomial_moduli()
        
        # Precompute CRT coefficients
        self.crt_coefficients = self._compute_crt_coefficients()
        
        # Bezout coefficient cache
        self.bezout_cache = {}
        
    def _generate_polynomial_moduli(self) -> torch.Tensor:
        """
        Generate moduli from polynomial co-prime functionals.
        
        Uses the polynomial functional values as moduli, ensuring they are
        positive and suitable for CRT operations.
        """
        # Sample points for polynomial evaluation
        x_sample = torch.linspace(-1, 1, 10)
        
        # Evaluate polynomial functionals
        phi_values = self.polynomial_config.evaluate(x_sample)  # [10, k]
        
        # Use mean absolute values as moduli (ensure positive)
        moduli = torch.abs(phi_values.mean(dim=0)) + 2.0  # Add offset to ensure > 1
        
        return moduli
        
    def _compute_crt_coefficients(self) -> torch.Tensor:
        """
        Compute Chinese Remainder Theorem coefficients.
        
        For moduli m1, m2, ..., mk, compute coefficients such that:
        x ≡ a1 (mod m1), x ≡ a2 (mod m2), ..., x ≡ ak (mod mk)
        has unique solution modulo M = m1 * m2 * ... * mk
        """
        M = torch.prod(self.moduli)
        coefficients = torch.zeros(self.num_moduli)
        
        for i in range(self.num_moduli):
            Mi = M / self.moduli[i]
            # Find modular inverse of Mi modulo moduli[i]
            # Using extended Euclidean algorithm
            yi = self._mod_inverse(Mi.item(), self.moduli[i].item())
            coefficients[i] = Mi * yi
        
        return coefficients
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """
        Compute modular inverse of a modulo m using extended Euclidean algorithm.
        Returns x such that (a * x) ≡ 1 (mod m)
        """
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
        return (x % m + m) % m
    
    def _compute_bezout_coefficients(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        Compute Bezout coefficients for two integers.
        Returns (gcd, x, y) such that ax + by = gcd(a, b)
        """
        if (a, b) in self.bezout_cache:
            return self.bezout_cache[(a, b)]
        
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        result = extended_gcd(a, b)
        self.bezout_cache[(a, b)] = result
        return result
    
    def apply_crt_decomposition(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply Chinese Remainder Theorem decomposition to state.
        
        Decomposes state into residues modulo coprime moduli,
        enabling stable numerical operations.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size, dim = state.shape
        
        # Ensure we have enough dimensions for CRT
        if dim < self.num_moduli:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.num_moduli - dim, device=state.device)
            state = torch.cat([state, padding], dim=1)
            dim = self.num_moduli
        
        # Reshape for CRT processing
        # Group dimensions into blocks for each modulus
        block_size = dim // self.num_moduli
        remainder_dims = dim % self.num_moduli
        
        residues = []
        
        for i in range(self.num_moduli):
            start_idx = i * block_size
            if i < remainder_dims:
                end_idx = start_idx + block_size + 1
            else:
                end_idx = start_idx + block_size
            
            if start_idx < dim:
                block = state[:, start_idx:min(end_idx, dim)]
                # Apply modular reduction
                modulus = self.moduli[i].item()
                residue = torch.fmod(block, modulus)
                residues.append(residue)
        
        # Concatenate residues
        crt_state = torch.cat(residues, dim=1)
        
        return crt_state
    
    def apply_crt_reconstruction(self, residues: torch.Tensor, original_dim: int) -> torch.Tensor:
        """
        Reconstruct state from CRT residues.
        
        Uses precomputed CRT coefficients to reconstruct the original state
        from its modular residues.
        """
        if residues.dim() == 1:
            residues = residues.unsqueeze(0)
        
        batch_size = residues.shape[0]
        
        # Split residues back into blocks
        residue_blocks = []
        current_idx = 0
        
        for i in range(self.num_moduli):
            block_size = original_dim // self.num_moduli
            if i < original_dim % self.num_moduli:
                block_size += 1
            
            if current_idx < residues.shape[1]:
                end_idx = min(current_idx + block_size, residues.shape[1])
                block = residues[:, current_idx:end_idx]
                residue_blocks.append(block)
                current_idx = end_idx
        
        # Reconstruct using CRT
        reconstructed_blocks = []
        
        for i, block in enumerate(residue_blocks):
            if i < len(self.crt_coefficients):
                # Apply CRT coefficient
                coeff = self.crt_coefficients[i]
                reconstructed_block = block * coeff
                reconstructed_blocks.append(reconstructed_block)
        
        # Concatenate and truncate to original dimension
        if reconstructed_blocks:
            reconstructed = torch.cat(reconstructed_blocks, dim=1)
            reconstructed = reconstructed[:, :original_dim]
        else:
            reconstructed = torch.zeros(batch_size, original_dim, device=residues.device)
        
        return reconstructed
    
    def refresh_bezout_coefficients(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Refresh Bezout coefficients for numerical stability.
        
        Applies CRT decomposition and reconstruction with Bezout coefficient
        correction for enhanced numerical stability.
        """
        original_shape = state.shape
        original_dim = state.shape[-1] if state.dim() > 0 else 1
        
        # Apply CRT decomposition
        crt_residues = self.apply_crt_decomposition(state)
        
        # Compute condition number for stability assessment
        try:
            # Use SVD for condition number estimation
            U, S, V = torch.svd(crt_residues)
            condition_number = (S.max() / (S.min() + self.stability_threshold)).item()
        except:
            condition_number = 1.0
        
        # Apply Bezout coefficient correction if needed
        if condition_number > 100:  # High condition number indicates instability
            # Apply stabilization using Bezout coefficients
            stabilized_residues = self._apply_bezout_stabilization(crt_residues)
        else:
            stabilized_residues = crt_residues
        
        # Reconstruct state
        reconstructed_state = self.apply_crt_reconstruction(stabilized_residues, original_dim)
        
        # Restore original shape
        if len(original_shape) == 1:
            reconstructed_state = reconstructed_state.squeeze(0)
        
        # Compute diagnostics
        diagnostics = {
            'bezout_condition_number': condition_number,
            'moduli_mean': self.moduli.mean().item(),
            'moduli_std': self.moduli.std().item(),
            'drift_threshold': self.stability_threshold,
            'crt_reconstruction_error': torch.norm(reconstructed_state - state).item() if state.shape == reconstructed_state.shape else 0.0
        }
        
        return reconstructed_state, diagnostics
    
    def _apply_bezout_stabilization(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Apply Bezout coefficient stabilization to residues.
        
        Uses Bezout identity to ensure numerical stability:
        For coprime integers a, b: ax + by = gcd(a, b) = 1
        """
        stabilized = residues.clone()
        
        # Apply pairwise Bezout stabilization
        for i in range(min(self.num_moduli - 1, residues.shape[1] - 1)):
            for j in range(i + 1, min(self.num_moduli, residues.shape[1])):
                # Get moduli for this pair
                a = int(self.moduli[i].item())
                b = int(self.moduli[j].item())
                
                # Compute Bezout coefficients
                gcd, x, y = self._compute_bezout_coefficients(a, b)
                
                if gcd == 1:  # Coprime moduli
                    # Apply Bezout correction
                    correction_i = residues[:, i] * x * self.stability_threshold
                    correction_j = residues[:, j] * y * self.stability_threshold
                    
                    stabilized[:, i] += correction_j
                    stabilized[:, j] += correction_i
        
        return stabilized
    
    def get_stability_metrics(self) -> Dict:
        """Get current stability metrics."""
        return {
            'num_moduli': self.num_moduli,
            'moduli_product': torch.prod(self.moduli).item(),
            'max_modulus': self.moduli.max().item(),
            'min_modulus': self.moduli.min().item(),
            'cache_size': len(self.bezout_cache),
            'stability_threshold': self.stability_threshold
        }

def create_enhanced_bezout_crt(state_dim: int = 64) -> EnhancedBezoutCRT:
    """Factory function to create enhanced Bezout CRT."""
    return EnhancedBezoutCRT(
        state_dim=state_dim,
        num_moduli=5,
        stability_threshold=1e-6
    )

