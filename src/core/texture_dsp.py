import torch
import numpy as np
from scipy import fftpack
import sys
import os
from typing import Optional, Any, Tuple, List

try:
    from image_extension import ImageProcessor, StructuralEntanglementNet, ConjugateColorTransport
except ImportError:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from image_extension import ImageProcessor, StructuralEntanglementNet, ConjugateColorTransport

class TextureDSPEngine:
    """
    Applies Digital Signal Processing (DSP) algorithms (like FFT and filters)
    to image textures, routing them through the existing StructuralEntanglementNet.
    This fulfills the "music filtering" requirement using the existing image architecture.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.image_processor = ImageProcessor(device=self.device)
        self.color_transport = ConjugateColorTransport(device=self.device)

    def apply_conjugate_color_filter(
        self,
        image_tensor: torch.Tensor,
        kappa: float = 0.15,
        prime_ladder: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Applies OKLab whitening, Gaussian mixture valley potential, and Brenier push-forward
        transport to image / texture tensor [B, C, H, W] or [B, H, W, 3].
        """
        if image_tensor.dim() == 4 and image_tensor.shape[1] == 3:
            # [B, 3, H, W] -> [B, H, W, 3]
            perm = image_tensor.permute(0, 2, 3, 1)
            transported = self.color_transport.transport_colors(perm, kappa=kappa, prime_ladder=prime_ladder)
            return transported.permute(0, 3, 1, 2)
        elif image_tensor.dim() == 4 and image_tensor.shape[3] == 3:
            return self.color_transport.transport_colors(image_tensor, kappa=kappa, prime_ladder=prime_ladder)
        elif image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
            return self.color_transport.transport_colors(image_tensor, kappa=kappa, prime_ladder=prime_ladder)
        return image_tensor

    def apply_cyclotomic_taper_window(
        self,
        tensor_2d: torch.Tensor,
        window_type: str = "hann",
        prime_p: int = 17
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a 2D tapering window to suppress spectral leakage from incommensurate frequencies
        (e.g. prime logarithms f_p = 2pi ln(p)) across rectangular frames (ECE2026 L42).
        Supports 'hann', 'blackman', and 'cyclotomic' (snapped to cyclotomic roots of unity).
        """
        H, W = tensor_2d.shape[-2], tensor_2d.shape[-1]
        dev = tensor_2d.device
        
        if window_type == "cyclotomic":
            try:
                from src.topology.modular_homology_fft import CyclotomicTDACompressor
                compressor = CyclotomicTDACompressor(p=prime_p, ring_size=min(64, H * W))
                quantized = compressor.cyclotomic_quantization(tensor_2d)
                # Modulate standard Hann envelope with cyclotomic phase shield
                wy = torch.hann_window(H, periodic=False, device=dev)
                wx = torch.hann_window(W, periodic=False, device=dev)
                base_window = torch.outer(wy, wx)
                window = base_window * (0.8 + 0.2 * torch.sigmoid(quantized))
            except Exception:
                wy = torch.hann_window(H, periodic=False, device=dev)
                wx = torch.hann_window(W, periodic=False, device=dev)
                window = torch.outer(wy, wx)
        elif window_type == "blackman":
            wy = torch.blackman_window(H, periodic=False, device=dev)
            wx = torch.blackman_window(W, periodic=False, device=dev)
            window = torch.outer(wy, wx)
        elif window_type == "hann":
            wy = torch.hann_window(H, periodic=False, device=dev)
            wx = torch.hann_window(W, periodic=False, device=dev)
            window = torch.outer(wy, wx)
        else:
            window = torch.ones((H, W), device=dev)
            
        tapered = tensor_2d * window
        return tapered, window

    def apply_spectral_filter(
        self,
        image_tensor: torch.Tensor,
        cutoff_freq: float = 0.5,
        window_type: str = "hann",
        use_parseval_norm: bool = True
    ) -> torch.Tensor:
        """
        Applies a low-pass filter to the texture using 2D FFT, analogous to audio filtering.
        Incorporates Parseval energy-conserving invariance (norm='ortho') and 2D window tapering
        to eliminate Dirichlet sinc leakage across incommensurate frequencies (ECE2026 L42).
        Input: [B, C, H, W]
        """
        if image_tensor.dim() == 4:
            B, C, H, W = image_tensor.shape
            dev = image_tensor.device
            
            # 1. Native PyTorch implementation on device for Parseval Invariance & speed
            try:
                # Apply 2D window tapering to suppress boundary switching artifacts
                if window_type in ("hann", "blackman", "cyclotomic"):
                    sample_2d = image_tensor[0, 0]
                    _, window_2d = self.apply_cyclotomic_taper_window(sample_2d, window_type=window_type)
                    window_broadcast = window_2d.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
                    windowed_input = image_tensor * window_broadcast
                else:
                    window_broadcast = torch.ones_like(image_tensor)
                    windowed_input = image_tensor

                # Parseval unitary 2D FFT (1/sqrt(HW) scaling)
                fft_norm = "ortho" if use_parseval_norm else "backward"
                F_spec = torch.fft.fft2(windowed_input, dim=(-2, -1), norm=fft_norm)
                F_shifted = torch.fft.fftshift(F_spec, dim=(-2, -1))
                
                # Circular frequency mask
                Y, X = torch.meshgrid(
                    torch.arange(H, device=dev, dtype=torch.float32),
                    torch.arange(W, device=dev, dtype=torch.float32),
                    indexing='ij'
                )
                center_y, center_x = H / 2.0, W / 2.0
                dist_from_center = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
                mask = (dist_from_center <= (cutoff_freq * max_radius)).float()
                mask_broadcast = mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
                
                F_filtered = F_shifted * mask_broadcast
                
                # Inverse 2D FFT with Parseval Invariance
                F_unshifted = torch.fft.ifftshift(F_filtered, dim=(-2, -1))
                reconstructed = torch.fft.ifft2(F_unshifted, dim=(-2, -1), norm=fft_norm).real
                
                # Normalize by window envelope if non-zero to restore unattenuated baseline
                if window_type in ("hann", "blackman", "cyclotomic"):
                    safe_win = torch.clamp(window_broadcast, min=0.15)
                    return reconstructed / safe_win
                return reconstructed
            except Exception:
                pass
            
            # Fallback to SciPy processing
            processed = torch.zeros_like(image_tensor)
            cpu_tensor = image_tensor.detach().cpu().numpy()
            
            for b in range(B):
                for c in range(C):
                    plane = cpu_tensor[b, c, :, :]
                    if window_type == "hann":
                        wy = np.hanning(H)
                        wx = np.hanning(W)
                        win2d = np.outer(wy, wx)
                        plane = plane * win2d
                    else:
                        win2d = np.ones((H, W))
                        
                    # Perform 2D FFT with 1/sqrt(HW) scaling
                    F1 = fftpack.fft2(plane)
                    if use_parseval_norm:
                        F1 = F1 / np.sqrt(H * W)
                    F2 = fftpack.fftshift(F1)
                    
                    Y, X = np.ogrid[:H, :W]
                    center_y, center_x = H // 2, W // 2
                    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                    max_radius = np.sqrt(center_x**2 + center_y**2)
                    mask = dist_from_center <= (cutoff_freq * max_radius)
                    
                    F2 = F2 * mask
                    
                    # Inverse FFT
                    F1_filtered = fftpack.ifftshift(F2)
                    image_filtered = fftpack.ifft2(F1_filtered).real
                    if use_parseval_norm:
                        image_filtered = image_filtered * np.sqrt(H * W)
                        
                    if window_type == "hann":
                        image_filtered = image_filtered / np.clip(win2d, 0.15, None)
                    processed[b, c, :, :] = torch.tensor(image_filtered, device=self.device)
                    
            return processed
        return image_tensor

    def process_and_embed(self, image_input, cutoff_freq: float = 0.5) -> torch.Tensor:
        """
        Loads an image, applies the spectral texture filter, and embeds it
        using the StructuralEntanglementNet (RNN/CNN backbone).
        """
        # Preprocess into windowed patches
        patches = self.image_processor.preprocess_image(image_input)
        
        # Apply DSP filter
        filtered_patches = self.apply_spectral_filter(patches, cutoff_freq=cutoff_freq)
        
        # Extract features using the sovereign vision pipeline
        embeddings = self.image_processor.forward(filtered_patches)
        
        return embeddings


class AdaptiveGaborWindowController:
    """
    Adaptive Gabor-Frame Window Controller based on Spectral Entropy (H_spec).
    Addresses ECE2026 L42 Time-Frequency Windowing & Switching Artifacts:
    - High Spectral Entropy (H_spec > threshold): transient Dirac epiphany / abrupt scene change.
      Dynamically contracts the frame window to preserve temporal/spatial localization and avoid vertical smear.
    - Low Spectral Entropy (H_spec <= threshold): stationary prime harmonic regime.
      Expands the frame window to maximize frequency resolution without incommensurate leakage.
    """
    def __init__(
        self,
        min_window: int = 16,
        max_window: int = 128,
        entropy_threshold: float = 0.5,
        device: str = "cpu"
    ):
        self.min_window = min_window
        self.max_window = max_window
        self.entropy_threshold = entropy_threshold
        self.device = device

    def compute_spectral_entropy(self, signal_or_frame: torch.Tensor) -> float:
        """
        Computes normalized spectral entropy H_spec in [0, 1].
        """
        import math
        flat = signal_or_frame.detach().flatten().float()
        if flat.numel() == 0:
            return 0.0
            
        # 1D or 2D RFFT power spectrum with unitary normalization
        freq = torch.fft.rfft(flat, norm="ortho")
        power = torch.abs(freq) ** 2
        power_sum = power.sum()
        if power_sum <= 1e-12:
            return 0.0
            
        prob = power / power_sum
        h_spec = -torch.sum(prob * torch.log(prob + 1e-12)).item()
        max_ent = math.log(max(2, prob.shape[0]))
        return float(min(1.0, max(0.0, h_spec / max_ent)))

    def determine_adaptive_window_size(self, signal_or_frame: torch.Tensor) -> int:
        """
        Determines the optimal Gabor window length based on H_spec.
        High entropy -> small window; Low entropy -> large window.
        """
        h_spec = self.compute_spectral_entropy(signal_or_frame)
        target_size = int(round(self.max_window - h_spec * (self.max_window - self.min_window)))
        target_size = max(self.min_window, min(self.max_window, target_size))
        return target_size

    def partition_adaptive_gabor_tiles(
        self,
        image_or_patch: torch.Tensor,
        stride: Optional[int] = None
    ) -> List[Tuple[torch.Tensor, int, float]]:
        """
        Slices image [C, H, W] into adaptive Gabor tiles depending on local spectral entropy.
        Returns list of (tile_tensor, window_size, h_spec).
        """
        from typing import List, Tuple
        if image_or_patch.dim() == 2:
            img = image_or_patch.unsqueeze(0)
        else:
            img = image_or_patch
            
        C, H, W = img.shape
        h_spec_global = self.compute_spectral_entropy(img)
        w_size = self.determine_adaptive_window_size(img)
        step = stride or max(4, w_size // 2)
        
        tiles = []
        for y in range(0, max(1, H - w_size + 1), step):
            for x in range(0, max(1, W - w_size + 1), step):
                tile = img[:, y:y + w_size, x:x + w_size]
                h_tile = self.compute_spectral_entropy(tile)
                tiles.append((tile, w_size, h_tile))
                
        if not tiles:
            tiles.append((img, min(H, W), h_spec_global))
        return tiles

