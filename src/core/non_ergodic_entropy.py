"""
Non-Ergodic Fractal Entropy Decomposition.

Optimizes the Russian Doll (Fractal) entropy computation using non-ergodic
intra-domain methods. Preserves soliton structure instead of ergodic mixing.

Key innovations:
1. Band-separated entropy (ergodic/transitional/soliton)
2. Adaptive block partitioning via spectral coherence
3. Dominant mode representatives (not mean)

Author: Implementation from Fractal Entropy Optimization Plan
Created: January 2026
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class NonErgodicEntropyEstimator(nn.Module):
    """
    Entropy estimation that preserves non-ergodic (soliton) structure.
    
    Standard entropy: H(X) = -Σ p(x) log p(x)  [ergodic mixing]
    Non-ergodic: Decompose into spectral bands, compute entropy per band,
    preserve soliton entropy separately.
    """
    
    def __init__(self, num_bands: int = 3, trust_threshold: float = 2.0):
        """
        Args:
            num_bands: Number of spectral bands
            trust_threshold: Min peak-to-mean ratio to trust as soliton (noise suppression)
        """
        super().__init__()
        self.num_bands = max(2, num_bands)
        self.trust_threshold = trust_threshold
    
    def forward(self, phi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute band-separated entropy.
        
        Args:
            phi: [batch, K] functional outputs
            
        Returns:
            Dict with ergodic_entropy, soliton_entropy, band entropies
        """
        batch_size, K = phi.shape
        device = phi.device
        
        if K < 2:
            return {
                'ergodic_entropy': torch.tensor(0.0, device=device),
                'soliton_entropy': torch.tensor(0.0, device=device),
                'total_bands': torch.zeros(self.num_bands, device=device)
            }
        
        # 1. Spectral decomposition with Windowed Projection Quantization (Safeguard)
        # "Restrict operator spectrum to physical acceptance window"
        # We assume phi lives in a dense/fractal space. Truncating FFT effectively 
        # acts as the windowing function W in frequency domain.
        
        # Max dimension (safety cutoff)
        max_dim = 256 
        if K > max_dim:
             # Windowing: Truncate to acceptance window
             phi_windowed = phi[:, :max_dim] # Simple rectangular window
             phi_freq = torch.fft.rfft(phi_windowed, dim=-1)
        else:
             phi_freq = torch.fft.rfft(phi, dim=-1)
             
        freq_len = phi_freq.shape[-1]
        
        if freq_len < self.num_bands:
            # Fallback for small K
            power = torch.abs(phi_freq) ** 2
            power_norm = power / (power.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(power_norm * torch.log(power_norm + 1e-8)).sum(dim=-1)
            return {
                'ergodic_entropy': entropy.mean(),
                'soliton_entropy': torch.tensor(0.0, device=device),
                'total_bands': entropy.unsqueeze(0)
            }
        
        # 2. Band splitting
        band_size = freq_len // self.num_bands
        band_entropies = []
        
        for i in range(self.num_bands):
            start = i * band_size
            end = start + band_size if i < self.num_bands - 1 else freq_len
            
            band_power = torch.abs(phi_freq[:, start:end]) ** 2
            band_total = band_power.sum(dim=-1, keepdim=True) + 1e-8
            band_power_norm = band_power / band_total
            
            # Shannon entropy of power distribution within band
            entropy = -(band_power_norm * torch.log(band_power_norm + 1e-8)).sum(dim=-1)
            band_entropies.append(entropy.mean())
        
        # 3. Soliton-specific entropy: Peak persistence
        soliton_start = (self.num_bands - 1) * band_size
        soliton_band = torch.abs(phi_freq[:, soliton_start:])
        
        if soliton_band.shape[-1] > 0:
            peak_val = soliton_band.max(dim=-1)[0]
            mean_val = soliton_band.mean(dim=-1) + 1e-8
            peak_persistence = peak_val / mean_val
            
            # NOISE MITIGATION: Trust Threshold
            # Only count as soliton if peak is significantly above mean
            trust_mask = (peak_persistence > self.trust_threshold).float()
            
            soliton_entropy = (torch.log(peak_persistence + 1) * trust_mask).mean()
        else:
            soliton_entropy = torch.tensor(0.0, device=device)
        
        return {
            'ergodic_entropy': band_entropies[0],
            'transitional_entropy': band_entropies[1] if len(band_entropies) > 2 else torch.tensor(0.0, device=device),
            'soliton_entropy': soliton_entropy,
            'total_bands': torch.stack(band_entropies)
        }


class AdaptiveFractalPartitioner(nn.Module):
    """
    Adaptive block sizing based on non-ergodic coherence.
    
    Blocks should group functions with similar spectral signatures.
    Splitting at coherence boundaries preserves soliton structure.
    """
    
    def __init__(self, min_block: int = 2, max_block: int = 8, coherence_threshold: float = 0.5):
        """
        Args:
            min_block: Minimum block size
            max_block: Maximum block size
            coherence_threshold: Threshold for splitting
        """
        super().__init__()
        self.min_block = min_block
        self.max_block = max_block
        self.coherence_threshold = coherence_threshold
    
    def _compute_spectral_coherence(
        self,
        phi_i: torch.Tensor,
        phi_j: torch.Tensor
    ) -> torch.Tensor:
        """Coherence between two signals in frequency domain."""
        # Handle different input shapes
        if phi_i.dim() == 0:
            phi_i = phi_i.unsqueeze(0)
        if phi_j.dim() == 0:
            phi_j = phi_j.unsqueeze(0)
        
        freq_i = torch.fft.rfft(phi_i.flatten())
        freq_j = torch.fft.rfft(phi_j.flatten())
        
        cross_power = freq_i * freq_j.conj()
        power_i = (torch.abs(freq_i) ** 2).sum()
        power_j = (torch.abs(freq_j) ** 2).sum()
        
        coherence = torch.abs(cross_power.sum()) / (torch.sqrt(power_i * power_j) + 1e-8)
        return coherence
    
    def partition(self, phi: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Partition phi into blocks based on spectral coherence.
        
        Args:
            phi: [batch, K] functional outputs
            
        Returns:
            List of (start, end) tuples for blocks
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        batch_size, K = phi.shape
        
        if K <= self.min_block:
            return [(0, K)]
        
        # Compute pairwise coherence (use first batch element)
        coherences = []
        for i in range(K - 1):
            coh = self._compute_spectral_coherence(phi[0, i], phi[0, i + 1])
            coherences.append(coh.item())
        
        # Find block boundaries at coherence drops
        blocks = []
        start = 0
        
        for i, coh in enumerate(coherences):
            block_len = i + 1 - start
            
            # Split if: at max size OR coherence drops below threshold (and min size met)
            if block_len >= self.max_block:
                blocks.append((start, i + 1))
                start = i + 1
            elif coh < self.coherence_threshold and block_len >= self.min_block:
                blocks.append((start, i + 1))
                start = i + 1
        
        # Final block
        if start < K:
            blocks.append((start, K))
        
        return blocks
    
    def forward(self, phi: torch.Tensor) -> List[Tuple[int, int]]:
        """Forward pass returns partition."""
        return self.partition(phi)


class NonErgodicFractalEntropy(nn.Module):
    """
    Fractal Entropy Decomposition optimized for non-ergodic dynamics.
    
    Replaces standard HypergraphOrthogonalityPressure with:
    1. Adaptive block sizing via spectral coherence
    2. Soliton-preserving entropy within blocks
    3. Non-mixing representatives for global coupling
    
    Key principle: High-frequency soliton structure must be preserved,
    not averaged away by ergodic mixing.
    """
    
    def __init__(
        self,
        k_order: int = 3,
        num_bands: int = 3,
        min_block: int = 2,
        max_block: int = 8
    ):
        """
        Args:
            k_order: Order for subset entropy calculation
            num_bands: Number of spectral bands for entropy
            min_block: Minimum adaptive block size
            max_block: Maximum adaptive block size
        """
        super().__init__()
        self.k_order = k_order
        self.partitioner = AdaptiveFractalPartitioner(
            min_block=min_block,
            max_block=max_block
        )
        self.entropy_estimator = NonErgodicEntropyEstimator(num_bands=num_bands)
    
    def _compute_block_representative(self, block_phi: torch.Tensor) -> torch.Tensor:
        """
        Non-ergodic representative: Preserve soliton peaks instead of averaging.
        
        Use dominant mode (highest energy) instead of mean.
        """
        if block_phi.dim() == 1:
            return block_phi.unsqueeze(-1)
        
        if block_phi.shape[-1] == 0:
            return block_phi
        
        # Find dominant mode (highest energy across batch)
        energies = block_phi.pow(2).sum(dim=0)
        dominant_idx = energies.argmax()
        
        # Use dominant mode as representative
        return block_phi[:, dominant_idx:dominant_idx + 1]
    
    def forward(self, phi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute non-ergodic fractal entropy pressure.
        
        Args:
            phi: [batch, K] functional outputs
            
        Returns:
            Dict with local/global entropies and soliton preservation metric
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        
        batch_size, K = phi.shape
        device = phi.device
        
        if K < 2:
            return {
                'local_entropy': torch.tensor([0.0], device=device),
                'global_entropy': torch.tensor(0.0, device=device),
                'soliton_preserved': torch.tensor(0.0, device=device),
                'num_blocks': 1
            }
        
        # 1. Adaptive partitioning
        blocks = self.partitioner.partition(phi)
        
        # 2. Local entropy per block (non-ergodic)
        local_entropies = []
        soliton_entropies = []
        block_reps = []
        
        for start, end in blocks:
            if end <= start:
                continue
            
            block_phi = phi[:, start:end]
            
            # Non-ergodic entropy estimation
            entropy_results = self.entropy_estimator(block_phi)
            
            # Combine ergodic and soliton for local pressure
            local_pressure = entropy_results['ergodic_entropy'] + entropy_results['soliton_entropy']
            local_entropies.append(local_pressure)
            soliton_entropies.append(entropy_results['soliton_entropy'])
            
            # Non-ergodic representative (dominant mode, not mean)
            rep = self._compute_block_representative(block_phi)
            block_reps.append(rep)
        
        # 3. Global coupling with soliton preservation
        if len(block_reps) > 1:
            global_phi = torch.cat(block_reps, dim=1)
            global_result = self.entropy_estimator(global_phi)
            global_entropy = global_result['ergodic_entropy'] + global_result['soliton_entropy']
        else:
            global_entropy = local_entropies[0] if local_entropies else torch.tensor(0.0, device=device)
        
        # 4. Soliton preservation metric
        if soliton_entropies:
            soliton_preserved = torch.stack(soliton_entropies).mean()
        else:
            soliton_preserved = torch.tensor(0.0, device=device)
        
        # 5. Stack local entropies
        if local_entropies:
            local_entropy_tensor = torch.stack(local_entropies)
        else:
            local_entropy_tensor = torch.tensor([0.0], device=device)
        
        return {
            'local_entropy': local_entropy_tensor,
            'global_entropy': global_entropy,
            'soliton_preserved': soliton_preserved,
            'num_blocks': len(blocks)
        }




class HybridLassoQuantizer(nn.Module):
    """
    Hybrid "LAS + Oblite" Quantization System.
    
    Combines:
    1. LAS (Lattice Adaptive Shrinkage / Lasso): L1 Sparsity to silence weak signals.
    2. Obligatory Bitrate: Meta Polytope quantization for strong signals.
    
    Logic:
        - If signal is weak (|x| < threshold), it is silenced (Lasso).
        - If signal is strong, it is snapped to the Lattice (Quantization).
    """
    def __init__(
        self, 
        dim: int, 
        prime_basis: Optional[List[int]] = None, 
        levels: int = 16,
        lasso_lambda: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.lasso_lambda = lasso_lambda
        
        # Meta Polytope Basis (Polynomial Lattice) - NO PRIME GENERATION (anti-lobotomy)
        if prime_basis is None:
            # Generate polynomial-based lattice using Chebyshev polynomial coefficients
            import math
            def get_polynomial_basis(n):
                """Generate polynomial basis coefficients instead of primes."""
                basis = []
                for k in range(n):
                    # Use Chebyshev polynomial T_k evaluated at multiple points
                    # and take the coefficient magnitudes
                    x = 0.3 + 0.1 * k  # Varying evaluation points
                    if k == 0:
                        coeff = 1.0
                    elif k == 1:
                        coeff = x
                    else:
                        # T_k(x) = 2*x*T_{k-1}(x) - T_{k-2}(x)
                        t_prev2 = 1.0
                        t_prev1 = x
                        for j in range(2, k + 1):
                            t_curr = 2 * x * t_prev1 - t_prev2
                            t_prev2 = t_prev1
                            t_prev1 = t_curr
                        coeff = t_prev1
                    
                    # Scale to positive values suitable for lattice
                    basis_val = abs(coeff * 10) + 1
                    basis.append(basis_val)
                
                return basis
            
            self.polynomial_basis = get_polynomial_basis(dim)
        else:
            self.polynomial_basis = prime_basis
            
        self.register_buffer('lattice_basis', torch.tensor(self.polynomial_basis, dtype=torch.float32))
        
        # Quantization grid
        self.register_buffer('codebook', torch.linspace(-1, 1, levels))
        
    def forward(
        self, 
        x: torch.Tensor, 
        hardening_factor: float = 1.0, 
        spectral_window: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hybrid Forward Pass: Sparsify then Quantize.
        
        Args:
            x: Input tensor
            hardening_factor: Multiplier for lasso_lambda (Asymptotic Hardening)
            spectral_window: Optional frequency-domain mask (Windowing)
        """
        # 0. Spectral Windowing (Optional)
        if spectral_window is not None:
            # Transform to freq, mask, transform back
            # Assuming x is compatible with the window shape
            if x.shape[-1] == spectral_window.shape[-1]:
                 x_freq = torch.fft.rfft(x, dim=-1)
                 x_freq_masked = x_freq * spectral_window
                 x = torch.fft.irfft(x_freq_masked, n=x.shape[-1], dim=-1)
        
        # 1. LAS (Lasso) / Soft Thresholding with Hardening
        # Asymptotically, hardening_factor -> infinity for fossilized blocks
        effective_lambda = self.lasso_lambda * hardening_factor
        
        x_abs = x.abs()
        mask = (x_abs > effective_lambda).float()
        x_sparse = torch.sign(x) * (x_abs - effective_lambda) * mask
        
        # 2. Obligatory Lattice Projection (Quantization)
        # We only quantize the surviving non-zero elements
        x_q = self._polytope_project(x_sparse)
        
        # Straight-Through Estimator (STE)
        # Gradient sees the Lasso+Quantization effect relative to input
        return (x_q - x).detach() + x
        
    def _polytope_project(self, x: torch.Tensor) -> torch.Tensor:
        # Project onto Codebook
        shape = x.shape
        x_flat = x.view(-1, 1) # [N, 1]
        codes = self.codebook.view(1, -1) # [1, L]
        
        dist = (x_flat - codes).abs()
        idx = dist.argmin(dim=1)
        x_q = self.codebook[idx].view(shape)
        
        # Force exact zero if Lasso silenced it (Codebook might not have exact 0 if levels are weird)
        # Ensure 0 is represented
        zero_mask = (x.abs() < 1e-6)
        x_q[zero_mask] = 0.0
        
        return x_q


# Backwards compatibility alias
class HypergraphOrthogonalityPressureNonErgodic(NonErgodicFractalEntropy):
    """Alias for backwards compatibility with HypergraphOrthogonalityPressure API."""
    pass
