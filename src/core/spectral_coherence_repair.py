"""
Spectral Coherence Repair: Fixes the consonant clustering issue.

Implements the spectral coherence correction (γ) to merge Soliton Band 
with Ergodic Band and prevent vowel starvation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


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
        state = torch.where(torch.isnan(state) | torch.isinf(state), 
                          torch.randn_like(state) * stability_margin, 
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
    
    The Problem: θ_coherence too high → Soliton Band isolation → consonant clustering
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
        Decompose signal into Soliton Band (high-freq) and Ergodic Band (low-freq).
        
        Args:
            signal: Input signal [batch, seq_len, dim] or [batch, dim]
            
        Returns:
            soliton_band: High-frequency components
            ergodic_band: Low-frequency components
        """
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)  # Add seq dimension
        
        # FFT-based spectral decomposition
        fft_signal = torch.fft.fft(signal, dim=1)
        freqs = torch.fft.fftfreq(signal.shape[1], device=signal.device)
        
        # Split at median frequency
        median_freq = torch.median(torch.abs(freqs))
        high_freq_mask = torch.abs(freqs) > median_freq
        low_freq_mask = ~high_freq_mask
        
        # Separate bands
        soliton_fft = fft_signal.clone()
        soliton_fft[:, low_freq_mask, :] = 0
        
        ergodic_fft = fft_signal.clone()
        ergodic_fft[:, high_freq_mask, :] = 0
        
        soliton_band = torch.fft.ifft(soliton_fft, dim=1).real
        ergodic_band = torch.fft.ifft(ergodic_fft, dim=1).real
        
        return soliton_band, ergodic_band
    
    def adaptive_coherence_correction(
        self, 
        signal: torch.Tensor,
        output_text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Apply adaptive coherence correction to merge spectral bands.
        
        Args:
            signal: Input signal to correct
            output_text: Optional output text for clustering detection
            
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
        s(t) = Σ <v_i, x_i(t)> cos(ω_i t + φ_i)
        
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
        
        # Resonant carrier: cos(ω_i t + φ_i)
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
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.K = num_functionals
        self.D = poly_degree + 1
        self.device = device

        # Bezout coefficient matrix [K, K]
        self.register_buffer('bezout_matrix', torch.eye(self.K, device=self.device))

        # Modulus tracking
        self.register_buffer('moduli', torch.ones(self.K, device=self.device))

        # Drift detection
        self.register_buffer('last_residues', torch.zeros(self.K, self.D, device=self.device))
        self.drift_threshold = 0.5
    
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
        correlation_matrix = torch.corrcoef(residue_flat.T)  # [K*D, K*D]
        
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

