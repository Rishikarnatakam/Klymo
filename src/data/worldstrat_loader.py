"""
WorldStrat dataset loader for validation.

Loads paired LR/HR satellite image patches from the WorldStrat dataset
for computing PSNR/SSIM metrics.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    DATASETS_DIR,
    WORLDSTRAT_CONFIG,
    REFLECTANCE_MAX,
    SR_SCALE,
)


class WorldStratDataset(Dataset):
    """
    WorldStrat dataset for super-resolution validation.
    
    The dataset contains paired Sentinel-2 (LR) and SPOT/Pleiades (HR) 
    satellite imagery patches.
    
    Requires downloading the real dataset from Kaggle - no synthetic fallbacks.
    """
    
    def __init__(
        self,
        root_dir: Optional[Path] = None,
        split: str = "validation",
        max_samples: Optional[int] = None,
        lr_size: int = 128,
        hr_size: Optional[int] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize WorldStrat dataset.
        
        Args:
            root_dir: Root directory containing WorldStrat data
            split: Dataset split ("train", "validation", "test")
            max_samples: Maximum number of samples to load (for memory efficiency)
            lr_size: Size of LR patches
            hr_size: Size of HR patches (default: lr_size * SR_SCALE)
            transform: Optional transform to apply to samples
        """
        self.root_dir = root_dir or DATASETS_DIR / "worldstrat"
        self.split = split
        self.max_samples = max_samples or WORLDSTRAT_CONFIG["validation_samples"]
        self.lr_size = lr_size
        self.hr_size = hr_size or (lr_size * SR_SCALE)
        self.transform = transform
        
        # Load sample paths
        self.samples = self._load_sample_paths()
    
    def _load_sample_paths(self) -> List[Dict[str, Path]]:
        """
        Load paths to all valid sample pairs.
        
        Returns:
            List of dictionaries with 'lr' and 'hr' paths
        
        Raises:
            FileNotFoundError: If dataset is not found
        """
        samples = []
        
        # Check if dataset exists
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"WorldStrat dataset not found at {self.root_dir}\n"
                f"Likely cause: Kaggle download failed or wasn't run.\n"
                f"FIX:\n"
                f"1. Go to: https://www.kaggle.com/datasets/jucor1/worldstrat\n"
                f"2. Click 'Download' (or 'Agree') to accept the rules.\n"
                f"3. Re-run the download cell in the notebook."
            )
        
        # Look for LR/HR pairs in standard WorldStrat structure
        lr_dir = self.root_dir / "sentinel2"
        hr_dir = self.root_dir / "spot"
        
        if not lr_dir.exists() or not hr_dir.exists():
            # Try alternative structure
            lr_dir = self.root_dir / "LR"
            hr_dir = self.root_dir / "HR"
        
        if not lr_dir.exists() or not hr_dir.exists():
            # Try another common structure
            lr_dir = self.root_dir / "lr"
            hr_dir = self.root_dir / "hr"
        
        if lr_dir.exists() and hr_dir.exists():
            lr_files = sorted(lr_dir.glob("*.png")) + sorted(lr_dir.glob("*.tif"))
            
            for lr_path in lr_files[:self.max_samples]:
                # Find matching HR file
                hr_path = hr_dir / lr_path.name
                if hr_path.exists():
                    samples.append({"lr": lr_path, "hr": hr_path})
        
        if len(samples) == 0:
            raise FileNotFoundError(
                f"No valid LR/HR pairs found in {self.root_dir}\n"
                f"Expected structure:\n"
                f"  {self.root_dir}/sentinel2/*.png (or LR/)\n"
                f"  {self.root_dir}/spot/*.png (or HR/)\n"
                f"Please download the complete WorldStrat dataset from Kaggle."
            )
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample pair.
        
        Returns:
            Dictionary with 'lr', 'hr' tensors and metadata
        """
        sample = self.samples[idx]
        
        # Load from files
        lr = self._load_image(sample["lr"])
        hr = self._load_image(sample["hr"])
        name = sample["lr"].stem
        
        # Ensure correct sizes
        lr = self._crop_or_pad(lr, self.lr_size)
        hr = self._crop_or_pad(hr, self.hr_size)
        
        # Convert to tensors (CHW format)
        lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr).permute(2, 0, 1).float()
        
        result = {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "name": name,
        }
        
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and normalize an image file."""
        img = Image.open(path)
        arr = np.array(img).astype(np.float32)
        
        # Normalize based on bit depth
        if arr.max() > 1:
            if arr.max() > 255:
                # 16-bit
                arr = arr / REFLECTANCE_MAX
            else:
                # 8-bit
                arr = arr / 255.0
        
        # Ensure 3 channels
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        
        return np.clip(arr, 0, 1)
    
    def _crop_or_pad(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Crop or pad image to target size."""
        h, w = image.shape[:2]
        
        if h < target_size or w < target_size:
            # Pad
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='reflect'
            )
        
        # Center crop
        h, w = image.shape[:2]
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        
        return image[start_h:start_h + target_size, start_w:start_w + target_size]


def get_validation_loader(
    batch_size: int = 1,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """
    Get a DataLoader for WorldStrat validation.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for WorldStratDataset
    
    Returns:
        DataLoader for validation
    """
    dataset = WorldStratDataset(split="validation", **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
