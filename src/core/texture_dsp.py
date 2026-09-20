import torch
import numpy as np
from scipy import fftpack
from image_extension import ImageProcessor, StructuralEntanglementNet

class TextureDSPEngine:
    """
    Applies Digital Signal Processing (DSP) algorithms (like FFT and filters)
    to image textures, routing them through the existing StructuralEntanglementNet.
    This fulfills the "music filtering" requirement using the existing image architecture.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.image_processor = ImageProcessor(device=self.device)

    def apply_spectral_filter(self, image_tensor: torch.Tensor, cutoff_freq: float = 0.5) -> torch.Tensor:
        """
        Applies a low-pass filter to the texture using 2D FFT, analogous to audio filtering.
        Input: [B, C, H, W]
        """
        if image_tensor.dim() == 4:
            B, C, H, W = image_tensor.shape
            processed = torch.zeros_like(image_tensor)
            
            # Move to CPU for SciPy processing
            cpu_tensor = image_tensor.detach().cpu().numpy()
            
            for b in range(B):
                for c in range(C):
                    # Perform 2D FFT
                    F1 = fftpack.fft2(cpu_tensor[b, c, :, :])
                    F2 = fftpack.fftshift(F1)
                    
                    # Create a circular low-pass filter mask
                    Y, X = np.ogrid[:H, :W]
                    center_y, center_x = H // 2, W // 2
                    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                    
                    # Cutoff defined as a percentage of the max radius
                    max_radius = np.sqrt(center_x**2 + center_y**2)
                    mask = dist_from_center <= (cutoff_freq * max_radius)
                    
                    F2 = F2 * mask
                    
                    # Inverse FFT
                    F1_filtered = fftpack.ifftshift(F2)
                    image_filtered = fftpack.ifft2(F1_filtered).real
                    
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
