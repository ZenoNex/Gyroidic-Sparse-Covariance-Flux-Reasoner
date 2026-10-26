import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Any, Tuple, Dict, Union
from .mandelbulb_gyroidic_augmenter import MandelbulbGyroidicAugmenter, AugmentationConfig

class MandelbulbAugmentedDataset(Dataset):
    """
    A PyTorch Dataset wrapper that applies Mandelbulb-Gyroidic augmentation.
    
    This dataset wraps an existing dataset (where __getitem__ returns (X, y) or a
    feature dictionary such as ConstraintDataset) and applies topologically-aware
    augmentation to the latent feature manifolds.
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
        self.config = augmentation_config or AugmentationConfig(
            mandelbulb_power=8,
            max_iterations=20,
            pressure_adaptation=False
        )
        self.augmenter = MandelbulbGyroidicAugmenter(self.config)
        self.apply_online = apply_online
        self.augmentation_factor = augmentation_factor
        
        # Pre-computation storage
        self.precomputed_data = []
        
        if not self.apply_online:
            self._precompute_dataset()
            
    def _precompute_dataset(self):
        """Pre-generates augmented samples for the entire dataset."""
        print(f"[MGDAS] Pre-computing Mandelbulb augmentations for {len(self.base_dataset)} samples...")
        loader = DataLoader(self.base_dataset, batch_size=32, shuffle=False)
        
        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                X, y = batch
                aug_X, aug_y = self.augmenter(X, y, augmentation_factor=self.augmentation_factor, chromatic_mode='neutral')
                for i in range(aug_X.shape[0]):
                    self.precomputed_data.append((aug_X[i], aug_y[i]))
            else:
                self.precomputed_data.extend(batch)
                
        print(f"[MGDAS] Pre-computation complete. Total samples: {len(self.precomputed_data)}")

    def __len__(self):
        if self.apply_online:
            return len(self.base_dataset)
        else:
            return len(self.precomputed_data)
            
    def __getitem__(self, idx):
        if self.apply_online:
            item = self.base_dataset[idx]
            
            # Handle dictionary-based samples (e.g., ConstraintDataset)
            if isinstance(item, dict):
                aug_item = {}
                for k, v in item.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)) and getattr(v, 'ndim', 0) >= 1 and (
                        k.endswith(('_emb', '_features', 'features', 'tensor')) or k in ('x', 'features')
                    ):
                        is_numpy = isinstance(v, np.ndarray)
                        t = torch.from_numpy(v).float() if is_numpy else v.float()
                        if t.ndim == 1:
                            t_batch = t.unsqueeze(0)
                            aug_t, _ = self.augmenter(t_batch, augmentation_factor=1, chromatic_mode='neutral')
                            aug_vec = aug_t.squeeze(0)
                        else:
                            aug_t, _ = self.augmenter(t, augmentation_factor=1, chromatic_mode='neutral')
                            aug_vec = aug_t
                        aug_item[k] = aug_vec.cpu().numpy().astype(np.float32) if is_numpy else aug_vec
                    else:
                        aug_item[k] = v
                return aug_item
                
            # Handle tuple/list samples (X, y)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                X, y = item
                if isinstance(X, torch.Tensor):
                    X_batch = X.unsqueeze(0) if X.ndim == 1 else X
                else:
                    X_batch = torch.tensor(X).float()
                    if X_batch.ndim == 1:
                        X_batch = X_batch.unsqueeze(0)
                        
                y_batch = torch.tensor([y]) if not isinstance(y, torch.Tensor) else y
                aug_X, aug_y = self.augmenter(X_batch, y_batch, augmentation_factor=1, chromatic_mode='neutral')
                return aug_X.squeeze(0), aug_y.squeeze(0)
            else:
                return item
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

