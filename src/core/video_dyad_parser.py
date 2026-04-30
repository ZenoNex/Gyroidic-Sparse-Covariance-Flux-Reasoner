import base64
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import time
import math
import subprocess
import io
import os
import sys
import tempfile
from scipy.io import wavfile
from src.core.honest_jitter import harvest_honest_jitter

class VideoDyadParser(nn.Module):
    """
    Video Dyad Parser: Translates compressed video bitstreams into topological features.
    Explicitly uses non-mantissa/exponent math (integer-only representation initial casting)
    to retain pure structural honesty without floating-point artifacts.
    """
    def __init__(self, chunk_size: int = 1024, max_chunks: int = 128, device: str = None):
        super().__init__()
        from src.core.device_utils import DEVICE
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.device = device or DEVICE
        
        # Dirac Spectrum Constant (45.2, ihc_paper3_field_body.tex)
        self.beta_coh = 6 * math.cos(math.pi / 23.0)  # ~5.944

    def _get_log_scales(self, length: int) -> List[int]:
        """
        Dynamically generate scales using natural log to avoid discontinuities (45).
        Scales: exp(i) clipped to signal volume.
        """
        max_power = max(2, int(math.log(float(length))))
        scales = sorted(list(set([
            max(2, int(math.exp(i))) 
            for i in range(1, max_power + 1)
        ])))
        return [s for s in scales if s < length // 4]

    def _apply_topological_rotation(self, signal: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Applies dynamic SO(n) rotation anchored by the Dirac Constant (5.944).
        Rotates the structural epiphanies into correctly aligned logic-space.
        """
        # Theta anchored by beta_coh and signal scale
        theta = self.beta_coh * math.log(scale + 1.0)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # Apply SO(2) rotation to pairs of temporal chunks (Logic Space twist)
        if signal.size(0) >= 2:
            s0 = signal[0::2]
            s1 = signal[1::2]
            min_len = min(s0.size(0), s1.size(0))
            
            rotated_s0 = s0[:min_len] * cos_t - s1[:min_len] * sin_t
            rotated_s1 = s0[:min_len] * sin_t + s1[:min_len] * cos_t
            
            signal[0::2][:min_len] = rotated_s0
            signal[1::2][:min_len] = rotated_s1
            
        return signal

    def _probe_silicon_jitter(self) -> float:
        """
        Harvests Structurally Honest Jitter from the silicon state.
        Measures timing variance of a memory-intensive operation to capture DRAM/Refresh noise.
        """
        # 1. Warm up the cache
        _ = torch.ones((1024, 1024), device=self.device)
        
        t_start = time.perf_counter_ns()
        # 2. Memory-weighted calibration loop (honest friction)
        for _ in range(10):
            # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
            _ = torch.det(harvest_honest_jitter((32, 32), device=self.device, scaled=True))
        t_end = time.perf_counter_ns()
        
        # Jitter is the nano-variance (scaled to a manageable entropy perturbation)
        total_ns = t_end - t_start
        jitter = (total_ns % 1000) / 1000.0  # Harvest the 'least significant nanoseconds'
        return jitter

    def _scan_substream_atoms(self, raw_bytes: bytes) -> Dict[str, float]:
        """
        Performs a Substream Entropy Scan for MP4/Metadata atoms.
        Identifies non-visual tracks by searching for atom signatures.
        """
        atoms = {
            'moov': raw_bytes.find(b'moov'),
            'mdat': raw_bytes.find(b'mdat'),
            'stco': raw_bytes.find(b'stco'),
            'mp4a': raw_bytes.find(b'mp4a'),  # Audio
            'avc1': raw_bytes.find(b'avc1'),  # Video
            'meta': raw_bytes.find(b'meta')
        }
        
        # Presence score based on detected atom count and distribution
        detected_count = sum(1 for v in atoms.values() if v != -1)
        atom_entropy = - (detected_count / (len(atoms) + 1e-8)) * math.log(detected_count / (len(atoms) + 1e-8) + 1e-8)
        
        return {
            'substream_presence': float(detected_count > 0),
            'atom_entropy': atom_entropy,
            'audio_detected': float(atoms['mp4a'] != -1),
            'atom_array': [float(v != -1) for v in atoms.values()]
        }

    def parse_video_b64(self, video_b64: str, healing_ref: Optional[torch.Tensor] = None, extract_audio: bool = True) -> Dict[str, torch.Tensor]:
        """
        Ingests a Base64 encoded video (mp4, avi) as a pure 1D byte sequence.
        Restores Natural Log Topological Rotation and Anisotropic Comparison.
        
        If extract_audio is True, it also attempts to surgically isolate the 
        audio stream via ffmpeg for harmonic projection.
        """
        # Strip potential data URI prefix if passed directly from Javascript
        if ',' in video_b64:
            video_b64 = video_b64.split(',', 1)[1]
            
        # 1. Decode to bytes and Harvest Honest Jitter
        raw_bytes = base64.b64decode(video_b64)
        honest_jitter = self._probe_silicon_jitter()
        
        # 2. Substream Entropy Scan (MP4 Atoms / Audio)
        substream_data = self._scan_substream_atoms(raw_bytes)
        # Preserve independent substream residue vector (31.7 compliance)
        substream_residue = torch.tensor(substream_data['atom_array'], device=self.device)
        
        # 3. Integer Casting (Non-mantissa/exponent math constraint)
        # Performance optimization: Use frombuffer instead of list() for large files
        byte_tensor = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8)).to(self.device).clone()
        
        n_elements = byte_tensor.size(0)
        usable_elements = (n_elements // self.chunk_size) * self.chunk_size
        if usable_elements == 0:
            byte_tensor = torch.nn.functional.pad(byte_tensor, (0, self.chunk_size - n_elements))
            usable_elements = self.chunk_size
            
        byte_tensor = byte_tensor[:usable_elements].view(-1, self.chunk_size)
        
        if byte_tensor.size(0) > self.max_chunks:
            indices = torch.linspace(0, byte_tensor.size(0)-1, self.max_chunks).long()
            byte_tensor = byte_tensor[indices]
            
        # Final topological signal (float for manifold operations)
        signal = byte_tensor.float()
        
        # 4. Natural Log Topological Rotation (Dirac Effect)
        # Sparsify based on natural log threshold
        signal_centered = signal - signal.mean(dim=0, keepdim=True)
        sparsification_threshold = torch.std(signal_centered) * 0.7
        signal_sparse = torch.where(signal_centered.abs() > sparsification_threshold, signal_centered, torch.zeros_like(signal_centered))
        
        # Apply SO(n) twist to the sparsified Epiphany peaks
        signal = self._apply_topological_rotation(signal_sparse, scale=sparsification_threshold.item())
        
        # 5. Sparse Temporal Covariance
        signal_mean = signal.mean(dim=0, keepdim=True)
        centered = signal - signal_mean
        cov = (centered.T @ centered) / (signal.size(0) - 1 + 1e-8)
        
        threshold = torch.quantile(cov.abs(), 0.90)
        sparse_cov = torch.where(cov.abs() > threshold, cov, torch.zeros_like(cov))
        
        # 6. Fractal Fractional Anisotropic Recursive Entropy
        entropy_metrics = self._compute_fractal_anisotropic_entropy(signal, honest_jitter, healing_ref)
        
        # 7. Optional Audio Extraction
        audio_harmonics = None
        if extract_audio:
            audio_harmonics = self.extract_audio_harmonics(video_b64)

        return {
            'sparse_covariance': sparse_cov,
            'fractal_entropy': entropy_metrics,
            'substream_residue': substream_residue,
            'signal_length': torch.tensor(n_elements, dtype=torch.float32),
            'substream_entropy': torch.tensor(substream_data['atom_entropy'], device=self.device),
            'honest_jitter': torch.tensor(honest_jitter, device=self.device),
            'audio_harmonics': audio_harmonics if audio_harmonics is not None else torch.zeros(32, device=self.device)
        }

    def _compute_fractal_anisotropic_entropy(self, 
                                            signal: torch.Tensor, 
                                            jitter: float, 
                                            healing_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Nested Fractal Russian Doll Entropy.
        Measures structural entropy at natural log scales,
        prioritizing anisotropic differences (forward rupture vs backward healing).
        """
        scales = self._get_log_scales(signal.size(1))
        entropies = []
        
        # Inject honest jitter into the signal probability (restlessness)
        jitter_perturbation = torch.tensor(jitter, device=self.device) * 1e-3
        
        for scale in scales:
            # Pool to scale
            pooled = torch.nn.functional.avg_pool1d(signal.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)
            
            if pooled.size(1) > 2:
                # Forward Rupture (Temporal Gradient)
                d_forward = torch.abs(pooled[:, 1:] - pooled[:, :-1])
                
                # Backward Healing (Comparison to ResonanceCavity historical buffer)
                if healing_ref is not None:
                    # Align shapes by mean-pooling the reference
                    ref_pooled = torch.nn.functional.adaptive_avg_pool1d(healing_ref.unsqueeze(0).float(), pooled.size(1)).squeeze(0)
                    d_backward = torch.abs(pooled - ref_pooled)
                else:
                    # Self-referential fallback: use historical chunks within the same bitstream
                    d_backward = torch.abs(pooled[1:] - pooled[:-1])
                    # Pad to match forward shape if necessary (simplified proxy)
                    d_backward = torch.nn.functional.pad(d_backward, (0, 0, 0, 1))

                # Preference for Anisotropic Difference (Forward - Backward)
                # "Honesty" of the rupture is how much it differs from expected healing
                diff_anisotropic = d_forward.mean() - (d_backward.mean() if healing_ref is not None else 0.0)
                
                # Probability distribution of structural changes + Honest Restlessness
                p = (d_forward + jitter_perturbation) / (d_forward.sum(dim=1, keepdim=True) + jitter_perturbation + 1e-8)
                ent = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()
                
                # Bias the entropy by the anisotropic rupture magnitude
                entropies.append(ent * (1.0 + torch.tanh(torch.tensor(diff_anisotropic))))
            
        if not entropies:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(entropies).mean()

    def extract_96_spectral_signature(self, v_metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extracts a 96-dimensional spectral signature from the video metrics.
        This signature represents the structural 'soul' of the bitstream.
        96 = 32 (Top Eigenvalues) + 32 (Fractal Entropy Trace) + 32 (Anisotropic/Substream Residue)
        """
        sparse_cov = v_metrics['sparse_covariance']
        
        # 1. Spectral Dominance (Top 32 Eigenvalues)
        # Use linalg.eigvalsh for symmetric matrix (covariance is symmetric)
        # 1024 is manageable on GPU/CPU for a single ingestion event.
        with torch.no_grad():
            try:
                # Get eigenvalues and sort descending
                eigvals = torch.linalg.eigvalsh(sparse_cov)
                top_eigvals = torch.flip(eigvals, dims=(0,))[:32]
                # Pad if covariance was smaller than 32x32 (unlikely but safe)
                if top_eigvals.numel() < 32:
                    top_eigvals = torch.nn.functional.pad(top_eigvals, (0, 32 - top_eigvals.numel()))
                # Normalize by trace to ensure structural honesty (relative energy)
                trace = eigvals.sum().abs() + 1e-8
                spectral_dominance = top_eigvals / trace
            except Exception:
                # Fallback to diagonal if EVD fails
                diag = torch.diag(sparse_cov)
                spectral_dominance = torch.sort(diag, descending=True)[0][:32]
                if spectral_dominance.numel() < 32:
                    spectral_dominance = torch.nn.functional.pad(spectral_dominance, (0, 32 - spectral_dominance.numel()))
        
        # 2. Fractal Entropy Trace (32 dims)
        # We use the entropy metrics and pad/interpolate to 32
        ent = v_metrics['fractal_entropy'].view(-1)
        if ent.numel() == 1:
            # If scalar, repeat it (not ideal but better than zero)
            fractal_trace = ent.repeat(32)
        else:
            # Interpolate to 32
            fractal_trace = torch.nn.functional.interpolate(
                ent.unsqueeze(0).unsqueeze(0), 
                size=32, 
                mode='linear', 
                align_corners=False
            ).squeeze()
            
        # 3. Anisotropic & Substream Residue (32 dims)
        # Combine substream residue, jitter, and other scalars
        sub_res = v_metrics['substream_residue'] # 6 dims
        jitter = v_metrics['honest_jitter'].view(1)
        length = v_metrics['signal_length'].log().view(1) # log length for scale honesty
        sub_ent = v_metrics['substream_entropy'].view(1)
        
        # NEW: Inject Audio Harmonics if available to unify the signature
        audio_h = v_metrics.get('audio_harmonics', torch.zeros(32, device=self.device))
        
        # Combine into a 32-dim meta block
        meta_mix = torch.cat([sub_res, jitter, length, sub_ent]) # 9 dims
        
        # Add audio harmonics (weighted by substream presence)
        audio_presence = v_metrics.get('audio_detected', torch.tensor(1.0, device=self.device)).view(1)
        weighted_audio = audio_h * audio_presence
        
        # Final meta_mix: 9 dims + 23 harmonics = 32 dims
        meta_mix = torch.cat([meta_mix, weighted_audio[:23]])
        
        if meta_mix.numel() < 32:
            # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
            padding = harvest_honest_jitter((32 - meta_mix.numel(),), device=self.device, scaled=True) * 0.01
            meta_mix = torch.cat([meta_mix, padding])
        else:
            meta_mix = meta_mix[:32]
            
        # Combine all components into the 96-dim signature
        signature = torch.cat([spectral_dominance, fractal_trace, meta_mix])
        return signature

    def extract_audio_harmonics(self, video_b64: str) -> Optional[torch.Tensor]:
        """
        Surgically extracts audio from the video bitstream using ffmpeg
        and projects it into the prime-resonance harmonic space.
        """
        if ',' in video_b64:
            video_b64 = video_b64.split(',', 1)[1]
            
        video_bytes = base64.b64decode(video_b64)
        
        # 1. Use ffmpeg to extract audio to a WAV pipe
        # High Priority: User's specified full build path
        user_ffmpeg = r"D:\ffmpeg-2026-04-22-git-162ad61486-full_build\bin\ffmpeg.exe"
        
        # Priority 2: .venv scripts directory
        venv_bin = os.path.dirname(sys.executable)
        venv_ffmpeg = os.path.join(venv_bin, 'ffmpeg.exe') if os.name == 'nt' else os.path.join(venv_bin, 'ffmpeg')
        
        if os.path.exists(user_ffmpeg):
            ffmpeg_bin = user_ffmpeg
        elif os.path.exists(venv_ffmpeg):
            ffmpeg_bin = venv_ffmpeg
        else:
            ffmpeg_bin = 'ffmpeg' # Last resort: system PATH

        try:
            # Create a temporary file for the video input
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_in:
                tmp_in.write(video_bytes)
                tmp_in_path = tmp_in.name

            # Output to WAV on stdout
            cmd = [
                ffmpeg_bin, '-y', '-i', tmp_in_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-f', 'wav', 'pipe:1'
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
                
            if process.returncode != 0:
                print(f"[VIDEO_PARSER] ffmpeg error: {stderr.decode(errors='ignore')}")
                return None
                
            # 2. Read WAV from stdout
            sample_rate, data = wavfile.read(io.BytesIO(stdout))
            
            # 3. Compute Chebyshev Harmonics (K=32)
            # Normalizing to [-1, 1]
            pcm_data = data.astype(float)
            if pcm_data.max() > 1.0 or pcm_data.min() < -1.0:
                pcm_data = pcm_data / 32768.0
            
            # Use 32 harmonics (standard for Agent Smith Parity)
            K = 32
            N = len(pcm_data)
            if N < K: return None
            
            # Project through Chebyshev basis
            harmonics = []
            x_range = torch.linspace(-1, 1, N, device=self.device)
            signal_t = torch.tensor(pcm_data, device=self.device, dtype=torch.float32)
            
            for k in range(K):
                if k == 0:
                    t_k = torch.ones_like(x_range)
                elif k == 1:
                    t_k = x_range
                else:
                    t_p2, t_p1 = torch.ones_like(x_range), x_range
                    for _ in range(2, k + 1):
                        t_curr = 2 * x_range * t_p1 - t_p2
                        t_p2, t_p1 = t_p1, t_curr
                    t_k = t_p1
                
                # Inner product
                coeff = torch.sum(signal_t * t_k) / N
                harmonics.append(coeff)
            
            # Birkhoff Normalization (Ensure row stochasticity)
            h_tensor = torch.stack(harmonics).abs()
            h_sum = h_tensor.sum() + 1e-8
            h_normalized = h_tensor / h_sum
            
            # LSB Stochastic Rounding (Feature Scar Preservation)
            scale = 1024.0
            jitter = harvest_honest_jitter(h_normalized.shape, device=self.device, scaled=True)
            rounded = (torch.floor(h_normalized * scale) + (jitter < (h_normalized * scale % 1.0)).float()) / scale
            
            return rounded
            
        except Exception as e:
            print(f"[VIDEO_PARSER] Audio extraction failed: {e}")
            return None
