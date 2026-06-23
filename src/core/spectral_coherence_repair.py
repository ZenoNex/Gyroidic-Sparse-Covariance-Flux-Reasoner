"""
Spectral Coherence Repair: Fixes the consonant clustering issue.

Implements the spectral coherence correction () to merge Soliton Band 
with Ergodic Band and prevent vowel starvation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from src.core.honest_jitter import harvest_honest_jitter, fractal_pad


from .energy_based_soliton_healer import EnergyBasedSolitonHealer
from .codes_constraint_framework import CODESConstraintFramework

def apply_energy_based_stabilization(state: torch.Tensor, 
                                   energy_threshold: float = 10.0,
                                   stability_margin: float = 1e-6) -> torch.Tensor:
    """
    Apply energy-based numerical stabilization.
    
    Based on energy-based learning principles:
    - Clamp values to prevent energy explosion
    - Apply soft normalization to maintain energy balance
    - Use margin-based stabilization for robustness
    """
    # Check for NaN/inf values
    if torch.isnan(state).any() or torch.isinf(state).any():
        print("state = apply_energy_based_stabilization(state)")
        # Replace NaN/inf with small random values
        # SILICON SOVEREIGNTY: Replaced PRNG noise with honest jitter
        state = torch.where(torch.isnan(state) | torch.isinf(state), 
                          harvest_honest_jitter(state.shape, device=state.device, scaled=True) * stability_margin, 
                          state)
    
    # Energy-based clamping
    state_energy = torch.norm(state, p=2, dim=-1, keepdim=True)
    if (state_energy > energy_threshold).any():
        # Soft normalization to preserve direction but limit energy
        normalization_factor = energy_threshold / (state_energy + stability_margin)
        normalization_factor = torch.clamp(normalization_factor, max=1.0)
        state = state * normalization_factor
    
    # Final safety clamp
    state = torch.clamp(state, -energy_threshold, energy_threshold)
    
    return state


from .enhanced_bezout_crt import EnhancedBezoutCRT
from .number_theoretic_stabilizer import NumberTheoreticStabilizer

from .admr_solver import PolynomialADMRSolver
from .polynomial_crt import PolynomialCRT, PolynomialCRTKernelDetector
from .decoupled_polynomial_crt import DecoupledPolynomialCRT
from ..optimization.operational_admm import OperationalAdmm
from .polynomial_coprime import PolynomialCoprimeConfig

class SpectralCoherenceCorrector(nn.Module):
    """
    Fixes spectral fragmentation by dynamically adjusting coherence threshold.
    
    The Problem: _coherence too high  Soliton Band isolation  consonant clustering
    The Solution: Adaptive threshold that allows vowel resonance to merge back
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.7,
        min_threshold: float = 0.1,
        adaptation_rate: float = 0.1,
        device: str = None
    ):
        super().__init__()
        self.min_threshold = min_threshold
        self.adaptation_rate = adaptation_rate
        self.device = device
        
        # Dynamic coherence threshold
        self.register_buffer('theta_coherence', torch.tensor(initial_threshold, device=device))
        
        # Spectral band tracking
        self.register_buffer('soliton_energy', torch.tensor(0.0, device=device))
        self.register_buffer('ergodic_energy', torch.tensor(0.0, device=device))
        
        # 4. Acoustic Resonance Parameters
        # Resonant frequencies (omega_i) for each functional facet
        # These are learned/adapted during spectral repair
        self.omega = nn.Parameter(torch.linspace(200, 4000, 256, device=device)) # Standard speech range (Hz)
        self.phi = nn.Parameter(torch.zeros(256, device=device)) # Phase offsets

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle shape mismatches for spectral buffers via fractal alignment."""
        for name in ['omega', 'phi']:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                target_shape = getattr(self, name).shape
                if val.shape != target_shape:
                    print(f" [ADAPTIVE] Aligning {name}: {val.shape} -> {target_shape}")
                    state_dict[key] = fractal_pad(val, target_shape[-1])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def detect_consonant_clustering(self, output_text: str) -> bool:
        """
        Detect if output shows consonant clustering (vowel starvation).
        
        Args:
            output_text: Generated text to analyze
            
        Returns:
            True if consonant clustering detected
        """
        if not output_text or len(output_text) < 5:
            return False
            
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
        vowel_count = sum(1 for c in output_text if c in vowels)
        consonant_count = sum(1 for c in output_text if c in consonants)
        
        if consonant_count == 0:
            return False
            
        vowel_ratio = vowel_count / (vowel_count + consonant_count)
        
        # Normal English has ~40% vowels, clustering shows <20%
        return vowel_ratio < 0.2
    
    def compute_spectral_bands(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose signal into Soliton Band and Ergodic Band using
        a semoid-semi-passive three-band bandpass filter architecture.
        
        Args:
            signal: Input signal [batch, seq_len, dim] or [batch, dim]
            
        Returns:
            soliton_band: High-frequency components (Phaser modulated)
            ergodic_band: Low-frequency + Mid-frequency components (Ring & Delay modulated)
        """
        import math
        
        if signal.dim() == 2:
            signal_for_fft = signal
            fft_dim = -1
        else:
            signal_for_fft = signal
            fft_dim = 1
            
        # 1. FFT-based spectral decomposition
        fft_signal = torch.fft.fft(signal_for_fft, dim=fft_dim)
        T_len = signal_for_fft.shape[fft_dim]
        freqs = torch.fft.fftfreq(T_len, device=signal.device)
        
        # Virtual sample rate to map speech range: 80 kHz
        fs = 80000.0
        f_hz = freqs * fs
        w = 2.0 * math.pi * f_hz
        
        # Constants
        Q = 4.32
        K = 1.0
        
        # Center frequencies (geometric means of the bands)
        f_r_low = math.sqrt(45.0 * 355.0)       # ~126.4 Hz
        f_r_mid = math.sqrt(355.0 * 3550.0)     # ~1122.9 Hz
        f_r_high = math.sqrt(3550.0 * 35500.0)  # ~11229.4 Hz
        
        # Helper to compute bandpass transfer function H
        def get_bandpass_H(f_r):
            w0 = 2.0 * math.pi * f_r
            num = 1j * K * (w0 / Q) * w
            denom = (w0**2 - w**2) + 1j * (w0 / Q) * w
            return num / (denom + 1e-8)
            
        H_low = get_bandpass_H(f_r_low)
        H_mid = get_bandpass_H(f_r_mid)
        H_high = get_bandpass_H(f_r_high)
        
        # Shape alignment for broadcasting H
        # H is [T_len], we need to broadcast it to [batch, T_len, dim] or [batch, dim]
        if fft_dim == -1:
            H_low = H_low.view(1, -1)
            H_mid = H_mid.view(1, -1)
            H_high = H_high.view(1, -1)
        else:
            H_low = H_low.view(1, -1, 1)
            H_mid = H_mid.view(1, -1, 1)
            H_high = H_high.view(1, -1, 1)
            
        # 2. Decompose into the three filter channels
        low_fft = fft_signal * H_low
        mid_fft = fft_signal * H_mid
        high_fft = fft_signal * H_high
        
        # 3. Apply Phaser modulation in frequency domain for High Band
        # Phaser shifts phases of frequencies dynamically
        phi_omega = 0.5 * torch.sin(2.0 * math.pi * f_hz / 1000.0)
        if fft_dim == -1:
            phi_omega = phi_omega.view(1, -1)
        else:
            phi_omega = phi_omega.view(1, -1, 1)
        high_fft = high_fft * torch.exp(1j * phi_omega)
        
        # Reconstruct signals
        low_band = torch.fft.ifft(low_fft, dim=fft_dim).real
        mid_band = torch.fft.ifft(mid_fft, dim=fft_dim).real
        soliton_band = torch.fft.ifft(high_fft, dim=fft_dim).real
        
        # 4. Apply Ring Modulation to Low Band
        t = torch.arange(T_len, device=signal.device, dtype=signal.dtype)
        if fft_dim == -1:
            ring = 1.0 + 0.5 * torch.sin(2.0 * math.pi * 30.0 * t / T_len)
            ring = ring.view(1, -1)
        else:
            ring = 1.0 + 0.5 * torch.sin(2.0 * math.pi * 30.0 * t / T_len)
            ring = ring.view(1, -1, 1)
        low_band = low_band * ring
        
        # 5. Apply Delay (Circular Shift) to Mid Band
        mid_band = torch.roll(mid_band, shifts=2, dims=fft_dim)
        
        # Merge Low and Mid into Ergodic Band for two-band compatibility
        ergodic_band = low_band + mid_band
        
        return soliton_band, ergodic_band
    
    def adaptive_coherence_correction(
        self, 
        signal: torch.Tensor,
        output_text: Optional[str] = None,
        categories: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply adaptive coherence correction to merge spectral bands.
        
        Args:
            signal: Input signal to correct
            output_text: Optional output text for clustering detection
            categories: Optional category activation tensor to monitor for clustering
            
        Returns:
            Corrected signal with merged spectral bands
        """
        # Detect consonant clustering
        clustering_detected = False
        if output_text:
            clustering_detected = self.detect_consonant_clustering(output_text)
        
        # Compute spectral bands
        soliton_band, ergodic_band = self.compute_spectral_bands(signal)
        
        # Update energy tracking
        self.soliton_energy = torch.norm(soliton_band).detach()
        self.ergodic_energy = torch.norm(ergodic_band).detach()
        
        # Adaptive threshold adjustment
        if clustering_detected or self.soliton_energy > 2 * self.ergodic_energy:
            # Lower threshold to allow more merging
            self.theta_coherence = torch.clamp(
                self.theta_coherence - self.adaptation_rate,
                min=self.min_threshold
            )
        
        # Monitor categories for high-frequency peak clustering and trigger Schizo dispersion event
        dispersion_triggered = False
        if categories is not None:
            from src.core.martinova_correlation import compute_bounded_correlation
            cat_corr = compute_bounded_correlation(categories)
            if (cat_corr >= 0.99).any():
                dispersion_triggered = True
                # Trigger Schizo Band dispersion event: force the threshold to minimum and inject jitter
                self.theta_coherence.copy_(torch.tensor(self.min_threshold, device=self.theta_coherence.device))
                jitter = harvest_honest_jitter(signal.shape, device=signal.device, scaled=True)
                signal = signal + 0.15 * jitter
                print("[SCHIZO] Category clustering reached 1.0. Triggered Schizo Band dispersion event.")

        # Monitor high-frequency peak clustering in soliton_band directly
        from src.core.martinova_correlation import compute_bounded_correlation
        soliton_corr_input = soliton_band
        if soliton_corr_input.dim() == 2:
            soliton_corr_input = soliton_corr_input.unsqueeze(-1)
        soliton_corr = compute_bounded_correlation(soliton_corr_input)
        if (soliton_corr >= 0.99).any() and not dispersion_triggered:
            # Trigger high-frequency peak clustering dispersion event
            self.theta_coherence.copy_(torch.tensor(self.min_threshold, device=self.theta_coherence.device))
            jitter = harvest_honest_jitter(signal.shape, device=signal.device, scaled=True)
            signal = signal + 0.1 * jitter
            print("[SCHIZO] High-frequency peak clustering reached 1.0 in soliton band. Triggered dispersion event.")
        
        # Coherence-based merging
        coherence = torch.cosine_similarity(
            soliton_band.flatten(1), 
            ergodic_band.flatten(1), 
            dim=1
        ).mean()
        
        if coherence < self.theta_coherence:
            # Merge bands with adaptive weighting
            merge_weight = (self.theta_coherence - coherence) / self.theta_coherence
            corrected_signal = (1 - merge_weight) * signal + merge_weight * (soliton_band + ergodic_band)
        else:
            corrected_signal = signal
        
        return corrected_signal


    def project_to_acoustic_resonance(self, facet_activations: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Project manifold facet activations to acoustic resonant frequencies:
        s(t) =  <v_i, x_i(t)> cos(_i t + _i)
        
        Args:
            facet_activations: [batch, num_facets] result of config.evaluate()
            time_steps: [seq_len] or [1] current time
            
        Returns:
            acoustic_signal: [batch, seq_len] synthetic resonant speech
        """
        # Ensure num_facets matches omega
        num_facets = facet_activations.shape[-1]
        omega = self.omega[:num_facets]
        phi = self.phi[:num_facets]
        
        # Resonant carrier: cos(_i t + _i)
        # time_steps: [T] -> omega * t: [num_facets, T]
        t_mesh = time_steps.unsqueeze(0) * omega.unsqueeze(1) # [num_facets, T]
        carriers = torch.cos(t_mesh + phi.unsqueeze(1)) # [num_facets, T]
        
        # Acoustic output: sum over facets
        # facet_activations: [batch, num_facets]
        # carriers: [num_facets, T]
        # output: [batch, T]
        acoustic_signal = torch.matmul(facet_activations, carriers)
        
        return acoustic_signal
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get current spectral diagnostics."""
        return {
            'theta_coherence': self.theta_coherence.item(),
            'soliton_energy': self.soliton_energy.item(),
            'ergodic_energy': self.ergodic_energy.item(),
            'energy_ratio': (self.soliton_energy / (self.ergodic_energy + 1e-8)).item()
        }


class BezoutCoefficientRefresh(nn.Module):
    """
    Refreshes Bezout coefficients to fix CRT modulus drift.
    
    The Problem: Stale residues causing wrong prime-index lattice reconstruction
    The Solution: Dynamic Bezout coefficient updates for CRT realignment
    """
    
    def __init__(self, num_functionals=5, poly_degree=12, device=None):
        from src.core.device_utils import DEVICE
        super().__init__()
        self.device = device if device is not None else DEVICE
        self.K = num_functionals
        self.D = poly_degree + 1

        # Bezout coefficient matrix [K, K]
        self.register_buffer('bezout_matrix', torch.eye(self.K, device=self.device))

        # Modulus tracking - initialize with dynamically generated primes to avoid trivial 1.0 collapse
        primes = []
        candidate = 2
        while len(primes) < self.K:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
                if p * p > candidate:
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        
        self.register_buffer('moduli', torch.tensor(primes, dtype=torch.float32, device=self.device))

        # Drift detection
        self.register_buffer('last_residues', torch.zeros(self.K, self.D, device=self.device))
        self.drift_threshold = 0.5

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle shape mismatches for Bezout buffers via fractal alignment."""
        for name in ['bezout_matrix', 'moduli', 'last_residues']:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                target_param = getattr(self, name, None)
                
                # Intercept sterile moduli (all 1.0) loaded from checkpoint and replace with the correct prime moduli
                if name == 'moduli' and (val == 1.0).all() and target_param is not None:
                    print(f" [ADAPTIVE] Intercepted sterile moduli {val} from state dict, preserving prime moduli.")
                    state_dict[key] = target_param.clone()
                    val = state_dict[key]
                
                if target_param is not None and val.shape != target_param.shape:
                    print(f" [ADAPTIVE] Aligning {name}: {val.shape} -> {target_param.shape}")
                    new_val = val
                    for i in range(len(target_param.shape)):
                        if i < len(new_val.shape) and new_val.shape[i] != target_param.shape[i]:
                            # Transpose dimension to last, pad, transpose back
                            if i != len(target_param.shape) - 1:
                                dims = list(range(len(target_param.shape)))
                                dims[i], dims[-1] = dims[-1], dims[i]
                                new_val = new_val.permute(*dims)
                                new_val = fractal_pad(new_val, target_param.shape[i])
                                new_val = new_val.permute(*dims)
                            else:
                                new_val = fractal_pad(new_val, target_param.shape[i])
                    state_dict[key] = new_val
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def update_moduli(self, residues: torch.Tensor):
        """
        Adaptive update for moduli based on residue flux.
        m_j(t+1) = m_j(t) + eta * Delta H_j
        """
        # Delta H_j is the mean magnitude of residues in that channel
        delta_h = residues.abs().mean(dim=(0, 2)) if residues.dim() == 3 else residues.abs().mean(dim=0)
        # Scale update to maintain stability (eta = 0.01)
        self.moduli.data = self.moduli.data + 0.01 * delta_h
    
    def detect_modulus_drift(self, current_residues: torch.Tensor) -> bool:
        """
        Detect if residues have drifted from their expected modulus.
        
        Args:
            current_residues: Current residue tensor [batch, K, D]
            
        Returns:
            True if significant drift detected
        """
        if self.last_residues.sum() == 0:
            # First run, initialize
            self.last_residues = current_residues.mean(dim=0).detach()
            return False
        
        # Compute drift magnitude
        current_mean = current_residues.mean(dim=0)
        drift = torch.norm(current_mean - self.last_residues, dim=1)
        max_drift = drift.max()
        
        # Update tracking
        self.last_residues = current_mean.detach()
        
        return max_drift > self.drift_threshold
    
    def refresh_bezout_coefficients(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Refresh Bezout coefficients using current residue statistics.
        
        Args:
            residues: Current residues [batch, K, D]
            
        Returns:
            Updated Bezout coefficient matrix
        """
        batch_size = residues.shape[0]
        
        # Compute pairwise residue correlations
        residue_flat = residues.contiguous().view(batch_size, -1)  # [batch, K*D]
        # corrcoef requires at least 2 samples; with batch=1 DoF is zero.
        # Fall back to identity correlation (zero cross-correlation prior) for single-sample batches.
        if batch_size >= 2:
            correlation_matrix = torch.corrcoef(residue_flat.T)  # [K*D, K*D]
        else:
            kd = residue_flat.shape[1]
            correlation_matrix = torch.eye(kd, device=self.device)

        
        # Aggregate to functional level [K, K]
        func_correlations = torch.zeros(self.K, self.K, device=self.device)
        for i in range(self.K):
            for j in range(self.K):
                i_start, i_end = i * self.D, (i + 1) * self.D
                j_start, j_end = j * self.D, (j + 1) * self.D
                func_correlations[i, j] = correlation_matrix[i_start:i_end, j_start:j_end].mean()
        
        # Update Bezout matrix (inverse correlation for independence)
        self.bezout_matrix.copy_(torch.inverse(func_correlations + 1e-6 * torch.eye(self.K, device=self.device)))
        
        return self.bezout_matrix
    
    def apply_crt_correction(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Apply CRT correction using refreshed Bezout coefficients.
        
        Args:
            residues: Input residues [batch, K, D]
            
        Returns:
            Corrected residues with proper modulus alignment
        """
        # Update moduli based on residue flux
        self.update_moduli(residues)
        
        # Check for drift
        if self.detect_modulus_drift(residues):
            self.refresh_bezout_coefficients(residues)
        
        # Apply Bezout correction
        batch_size = residues.shape[0]
        residue_vectors = residues.contiguous().view(batch_size, self.K, -1)  # [batch, K, D]
        
        # Matrix multiplication with Bezout coefficients
        corrected_vectors = torch.einsum('kj,bjd->bkd', self.bezout_matrix, residue_vectors)
        
        return corrected_vectors.view_as(residues)
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get current Bezout coefficient diagnostics."""
        # Compute condition number using SVD (more compatible)
        try:
            U, S, V = torch.svd(self.bezout_matrix)
            condition_number = (S.max() / (S.min() + 1e-8)).item()
        except:
            condition_number = 1.0  # Fallback
            
        return {
            'bezout_condition_number': condition_number,
            'moduli_mean': self.moduli.mean().item(),
            'moduli_std': self.moduli.std().item(),
            'drift_threshold': self.drift_threshold
        }

