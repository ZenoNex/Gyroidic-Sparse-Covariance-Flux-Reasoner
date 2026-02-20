import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Any, Tuple
from .mandelbulb_gyroidic_augmenter import MandelbulbGyroidicAugmenter, AugmentationConfig

class MandelbulbAugmentedDataset(Dataset):
    """
    A PyTorch Dataset wrapper that applies Mandelbulb-Gyroidic augmentation.
    
    This dataset wraps an existing dataset (where __getitem__ returns (x, y))
    and applies toplogically-aware augmentation to the features 'x'.
    """
    
    def __init__(self, 
                 base_dataset: Dataset,
                 augmentation_config: Optional[AugmentationConfig] = None,
                 augmentation_factor: int = 1,
                 apply_online: bool = True,
                 cache_size: int = 1000):
        """
        Args:
            base_dataset: The underlying dataset to wrap.
            augmentation_config: Configuration for the Mandelbulb-Gyroidic augmenter.
            augmentation_factor: How many augmented variations to generate (if pre-computing).
                                 For online, this controls the intensity mixing.
            apply_online: If True, augments on-the-fly (slower, infinite variety).
                          If False, pre-computes augmentations (faster, fixed memory).
            cache_size: Number of augmented samples to cache if apply_online is True.
        """
        self.base_dataset = base_dataset
        self.config = augmentation_config or AugmentationConfig()
        self.augmenter = MandelbulbGyroidicAugmenter(self.config)
        self.apply_online = apply_online
        self.augmentation_factor = augmentation_factor
        
        # Pre-computation storage
        self.precomputed_data = []
        
        if not self.apply_online:
            self._precompute_dataset()
            
    def _precompute_dataset(self):
        """Pre-generates augmented samples for the entire dataset."""
        print(f"🌀 Pre-computing Mandelbulb augmentations for {len(self.base_dataset)} samples...")
        loader = DataLoader(self.base_dataset, batch_size=32, shuffle=False)
        
        for batch_idx, (X, y) in enumerate(loader):
            # X shape: [batch, features]
            aug_X, aug_y = self.augmenter(X, y, augmentation_factor=self.augmentation_factor)
            
            # Unpack into individual samples
            for i in range(aug_X.shape[0]):
                self.precomputed_data.append((aug_X[i], aug_y[i]))
                
        print(f"✅ Pre-computation complete. Total samples: {len(self.precomputed_data)}")

    def __len__(self):
        if self.apply_online:
            return len(self.base_dataset)
        else:
            return len(self.precomputed_data)
            
    def __getitem__(self, idx):
        if self.apply_online:
            # Get original sample
            X, y = self.base_dataset[idx]
            
            # Ensure X is 2D for the augmenter [1, features]
            if isinstance(X, torch.Tensor):
                X_batch = X.unsqueeze(0)
            else:
                X_batch = torch.tensor(X).unsqueeze(0)
                
            y_batch = torch.tensor([y]) if not isinstance(y, torch.Tensor) else y.unsqueeze(0)
            
            # Augment
            # Note: For online, we just generate 1 variation per access
            aug_X, aug_y = self.augmenter(X_batch, y_batch, augmentation_factor=1)
            
            return aug_X.squeeze(0), aug_y.squeeze(0)
        else:
            return self.precomputed_data[idx]

def create_mandelbulb_pipeline(
    dataset: Dataset, 
    batch_size: int = 32,
    num_workers: int = 0,
    config: Optional[AugmentationConfig] = None
) -> DataLoader:
    """
    Helper function to create a ready-to-use DataLoader with Mandelbulb augmentation.
    """
    augmented_ds = MandelbulbAugmentedDataset(dataset, augmentation_config=config, apply_online=True)
    return DataLoader(augmented_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
