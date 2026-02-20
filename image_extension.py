"""
Gyroidic Image Extension: Structural Entanglement Processor.

This module provides the "ImageProcessor" satellite component for the Gyroidic
Sparse Covariance Flux Reasoner. It replaces the legacy gradient-based hack
with a proper Convolutional Neural Network (CNN) backbone that supports:
1.  Feature Extraction: 768-dim embeddings compatible with the reasoned global state.
2.  Aspect Ratio Agnosticism: Uses letterbox padding instead of squashing.
3.  Diegetic Windowing: Tiles high-res images into manageable patches ("windows")
    to preserve local detail without losing global context.

Aligns with "MANDELBULB_DATASET_AUGMENTATION.md" by providing high-dimensional
feature vectors suitable for fractal projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
try:
    import torchvision.transforms as T
except ImportError:
    print("Warning: torchvision not found. Image processing will be limited.")
    T = None

from typing import Optional, List, Tuple, Union
import os

class StructuralEntanglementNet(nn.Module):
    """
    CNN Backbone for extracting "Structural Entanglement" features from image patches.
    
    Architecture:
    - Input: [Batch, 3, 64, 64] (RGB Patches)
    - Conv1: 3->32, 3x3, stride 1, padding 1
    - Pool1: 2x2
    - Conv2: 32->64, 3x3, stride 1, padding 1
    - Pool2: 2x2
    - Conv3: 64->128, 3x3, stride 1, padding 1
    - Global Average Pool
    - Projection: 128 -> 768 (Embedding Dimension)
    
    This is a lightweight "Satellite" network designed to be trained alongside
    the main reasoning core or frozen as a feature extractor.
    """
    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Projection to compatible embedding space (768-dim)
        self.projection = nn.Linear(128, output_dim)
        
        # Orthogonal initialization for stability
        nn.init.orthogonal_(self.projection.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 64, 64]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) # -> [32, 32, 32]
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) # -> [64, 16, 16]
        
        x = F.relu(self.conv3(x))
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1)) # -> [128, 1, 1]
        x = x.view(x.size(0), -1) # -> [B, 128]
        
        x = self.projection(x) # -> [B, 768]
        return x

class ImageProcessor(nn.Module):
    """
    High-fidelity Image Processor with "Diegetic Windowing".
    
    Handles:
    - Loading & Preprocessing
    - Aspect Ratio Preservation (Letterboxing)
    - Tiling (Windowing) for high-res support
    - Feature Extraction via StructuralEntanglementNet
    """
    def __init__(self, device: str = None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.backbone = StructuralEntanglementNet(output_dim=768)
        self.backbone.to(self.device)
        
        if T is not None:
            # Standard normalization for RGB images
            self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        else:
            self.normalize = None
        
        # Window parameters
        self.window_size = 64
        self.target_size = 512 # Canonical resolution for letterboxing high-res inputs

    def preprocess_image(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        """
        Load and tile an image.
        
        Returns:
            start_tensor: [Num_Tiles, 3, 64, 64] batch of image windows.
        """
        if T is None:
            print("Error: torchvision not available.")
            return torch.zeros(1, 3, 64, 64, device=self.device)

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                 # Fail gracefully logic - perhaps try loading as relative path or error
                 print(f"[ImageProcessor] File not found: {image_input}")
                 return torch.zeros(1, 3, 64, 64, device=self.device)
            try:
                img = Image.open(image_input).convert('RGB')
            except Exception as e:
                print(f"[ImageProcessor] Error loading {image_input}: {e}")
                # Return dummy tensor on failure (black square)
                return torch.zeros(1, 3, 64, 64, device=self.device)
        else:
            img = image_input
            
        # 1. Aspect Ratio Preserving Resize (Letterbox)
        w, h = img.size
        if max(w, h) > 0:
            scale = self.target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = self.target_size, self.target_size

        # Use appropriate resampling filter based on Pillow version
        if hasattr(Image, 'Resampling'):
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.LANCZOS
            
        img = img.resize((new_w, new_h), resample)
        
        # Create padded canvas
        canvas = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        # Center the image
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        
        # 2. Diegetic Windowing (Tiling)
        # Extract 64x64 tiles with stride 64 (non-overlapping for efficiency)
        if self.normalize:
            img_tensor = T.ToTensor()(canvas).to(self.device) # [3, 512, 512]
            img_tensor = self.normalize(img_tensor)
        else:
             # Fallback if no normalize
            img_tensor = T.ToTensor()(canvas).to(self.device)

        # Unfold to Extract Patches
        # Unfold H then W
        # Input: [C, H, W] -> Unfold H -> [C, NumH, W, Window] -> Unfold W -> [C, NumH, NumW, Window, Window]
        patches = img_tensor.unfold(1, self.window_size, self.window_size).unfold(2, self.window_size, self.window_size)
        # Shape: [3, 8, 8, 64, 64] for 512 input / 64 window
        
        patches = patches.contiguous().view(3, -1, self.window_size, self.window_size) # [3, 64, 64, 64]
        patches = patches.permute(1, 0, 2, 3) # [64, 3, 64, 64] (Batch of tiles)
        
        return patches

    def forward(self, image_input: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        End-to-end processing: Image -> 768-dim Embedding.
        
        Returns:
            embedding: [1, 768] (Global representation of the image)
        """
        # If input is already tensor tiles [B, 3, 64, 64]
        if isinstance(image_input, torch.Tensor):
             tiles = image_input
        else:
             tiles = self.preprocess_image(image_input) # [Num_Tiles, 3, 64, 64]
        
        if tiles.dim() == 4: # [Num_Tiles, 3, 64, 64]
            pass
        elif tiles.dim() == 3: # [3, 64, 64] - single tile
            tiles = tiles.unsqueeze(0)
            
        # Feature Extraction
        tile_features = self.backbone(tiles) # [Num_Tiles, 768]
        
        # Aggregation (Diegetic Consensus)
        # Max pooling captures the most salient features across all windows
        # Use simple max pool across the tiles dimension (dim 0)
        global_features = torch.max(tile_features, dim=0, keepdim=True)[0] # [1, 768]
        
        return global_features

    def get_embedding_layer(self):
        """Allow optimizer to discover trainable parameters."""
        return self.backbone.parameters()
