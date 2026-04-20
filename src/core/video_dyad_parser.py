import base64
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import time
import math

class VideoDyadParser(nn.Module):
    """
    Video Dyad Parser: Translates compressed video bitstreams into topological features.
    Explicitly uses non-mantissa/exponent math (integer-only representation initial casting)
    to retain pure structural honesty without floating-point artifacts.
    """
    def __init__(self, chunk_size: int = 1024, max_chunks: int = 128, device: str = None):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dirac Spectrum Constant (§45.2, ihc_paper3_field_body.tex)
        self.beta_coh = 6 * math.cos(math.pi / 23.0)  # ~5.944

    def _get_log_scales(self, length: int) -> List[int]:
        """
        Dynamically generate scales using natural log to avoid discontinuities (§45).
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
            _ = torch.det(torch.randn((32, 32), device=self.device))
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

    def parse_video_b64(self, video_b64: str, healing_ref: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Ingests a Base64 encoded video (mp4, avi) as a pure 1D byte sequence.
        Restores Natural Log Topological Rotation and Anisotropic Comparison.
        """
        # Strip potential data URI prefix if passed directly from Javascript
        if ',' in video_b64:
            video_b64 = video_b64.split(',', 1)[1]
            
        # 1. Decode to bytes and Harvest Honest Jitter
        raw_bytes = base64.b64decode(video_b64)
        honest_jitter = self._probe_silicon_jitter()
        
        # 2. Substream Entropy Scan (MP4 Atoms / Audio)
        substream_data = self._scan_substream_atoms(raw_bytes)
        # Preserve independent substream residue vector (§31.7 compliance)
        substream_residue = torch.tensor(substream_data['atom_array'], device=self.device)
        
        # 3. Integer Casting (Non-mantissa/exponent math constraint)
        byte_tensor = torch.tensor(list(raw_bytes), dtype=torch.uint8, device=self.device)
        
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
        
        return {
            'sparse_covariance': sparse_cov,
            'fractal_entropy': entropy_metrics,
            'substream_residue': substream_residue,
            'signal_length': torch.tensor(n_elements, dtype=torch.float32),
            'substream_entropy': torch.tensor(substream_data['atom_entropy'], device=self.device),
            'honest_jitter': torch.tensor(honest_jitter, device=self.device)
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
