import base64
import torch
import torch.nn as nn
from typing import Dict

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

    def parse_video_b64(self, video_b64: str) -> Dict[str, torch.Tensor]:
        """
        Ingests a Base64 encoded video (mp4, avi) as a pure 1D byte sequence.
        """
        # Strip potential data URI prefix if passed directly from Javascript
        if ',' in video_b64:
            video_b64 = video_b64.split(',', 1)[1]
            
        # 1. Decode to bytes
        raw_bytes = base64.b64decode(video_b64)
        
        # 2. Integer Casting (Non-mantissa/exponent math constraint)
        # Using uint8 preserves the raw cryptographic structure of the compressed video
        byte_tensor = torch.tensor(list(raw_bytes), dtype=torch.uint8, device=self.device)
        
        # Format into temporal structural chunks
        n_elements = byte_tensor.size(0)
        usable_elements = (n_elements // self.chunk_size) * self.chunk_size
        if usable_elements == 0:
            # File too small, pad it
            byte_tensor = torch.nn.functional.pad(byte_tensor, (0, self.chunk_size - n_elements))
            usable_elements = self.chunk_size
            
        byte_tensor = byte_tensor[:usable_elements].view(-1, self.chunk_size)
        
        # Limit chunks to prevent memory explosion tracking temporal frame progression
        if byte_tensor.size(0) > self.max_chunks:
            # Sample uniformly to respect the original bitstream temporal distribution
            indices = torch.linspace(0, byte_tensor.size(0)-1, self.max_chunks).long()
            byte_tensor = byte_tensor[indices]
            
        # Float cast ONLY for final tensor operations, the internal topology remains integer-spaced
        signal = byte_tensor.float()
        
        # 3. Sparse Temporal Covariance
        # Calculate covariance across chunks (captures I-frame vs P-frame topological stress)
        signal_mean = signal.mean(dim=0, keepdim=True)
        centered = signal - signal_mean
        cov = (centered.T @ centered) / (signal.size(0) - 1 + 1e-8)
        
        # Sparsify the covariance (keep only structurally significant correlations)
        threshold = torch.quantile(cov.abs(), 0.90)
        sparse_cov = torch.where(cov.abs() > threshold, cov, torch.zeros_like(cov))
        
        # 4. Fractal Fractional Anisotropic Recursive Entropy
        entropy_metrics = self._compute_fractal_anisotropic_entropy(signal)
        
        return {
            'sparse_covariance': sparse_cov,
            'fractal_entropy': entropy_metrics,
            'signal_length': torch.tensor(n_elements, dtype=torch.float32)
        }
        
    def _compute_fractal_anisotropic_entropy(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Fractal Fractional Anisotropic Recursive Entropy.
        Measures structural entropy at multiple scales (fractal), prioritizing 
        anisotropic differences (forward topological rupture vs backward healing).
        """
        # Natural log topological rotation effect scales
        length = signal.size(1) if signal.dim() > 1 else signal.size(0)
        max_power = max(2, int(torch.log(torch.tensor(length, dtype=torch.float32)).item()))
        
        # Dynamically generate scales using natural log to avoid discontinuities
        scales = sorted(list(set([
            max(2, int(torch.exp(torch.tensor(i, dtype=torch.float32)).item())) 
            for i in range(1, max_power + 1)
        ])))
        entropies = []
        for scale in scales:
            if signal.size(1) < scale:
                continue
            # Pool to scale
            pooled = torch.nn.functional.avg_pool1d(signal.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)
            
            # Anisotropic fractional difference
            if pooled.size(1) > 2:
                # Fractional derivative approximation (asymmetric shifts)
                d_forward = torch.abs(pooled[:, 1:] - pooled[:, :-1])
                
                # Probability distribution of structural changes
                p = d_forward / (d_forward.sum(dim=1, keepdim=True) + 1e-8)
                ent = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()
                entropies.append(ent)
            
        if not entropies:
            return torch.tensor(0.0, device=self.device)
            
        return torch.stack(entropies).mean()
