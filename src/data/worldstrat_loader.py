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
from tqdm import tqdm

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
                f"Please download from Kaggle:\n"
                f"  kaggle datasets download -d julienco/worldstrat -p {self.root_dir} --unzip\n"
                f"Or from: https://www.kaggle.com/datasets/julienco/worldstrat"
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
    
    def _create_synthetic_samples(self) -> List[Dict[str, Any]]:
        """
        Create synthetic samples when dataset is not available.
        
        Returns:
            List of synthetic sample dictionaries
        """
        samples = []
        
        # Create directory for synthetic samples
        synthetic_dir = self.root_dir / "synthetic"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(min(10, self.max_samples)):
            # Create synthetic "satellite" imagery with patterns
            hr = self._generate_synthetic_scene(self.hr_size)
            
            # Create LR by downsampling HR
            lr = self._downsample(hr, SR_SCALE)
            
            samples.append({
                "lr_data": lr,
                "hr_data": hr,
                "name": f"synthetic_{i:03d}",
            })
        
        return samples
    
    def _generate_synthetic_scene(self, size: int) -> np.ndarray:
        """
        Generate a synthetic satellite-like scene.
        
        Creates patterns resembling roads, buildings, vegetation.
        """
        scene = np.zeros((size, size, 3), dtype=np.float32)
        
        # Base vegetation (green-ish)
        scene[:, :, 0] = 0.2 + np.random.random((size, size)) * 0.1
        scene[:, :, 1] = 0.3 + np.random.random((size, size)) * 0.15
        scene[:, :, 2] = 0.15 + np.random.random((size, size)) * 0.08
        
        # Add some "roads" (gray lines)
        for _ in range(np.random.randint(2, 5)):
            if np.random.random() > 0.5:
                # Horizontal road
                y = np.random.randint(size // 4, 3 * size // 4)
                width = np.random.randint(2, 6)
                scene[y:y+width, :, :] = 0.4 + np.random.random() * 0.1
            else:
                # Vertical road
                x = np.random.randint(size // 4, 3 * size // 4)
                width = np.random.randint(2, 6)
                scene[:, x:x+width, :] = 0.4 + np.random.random() * 0.1
        
        # Add some "buildings" (bright rectangles)
        for _ in range(np.random.randint(3, 8)):
            x = np.random.randint(0, size - 20)
            y = np.random.randint(0, size - 20)
            w = np.random.randint(5, 20)
            h = np.random.randint(5, 20)
            brightness = 0.5 + np.random.random() * 0.3
            scene[y:y+h, x:x+w, :] = brightness
        
        return np.clip(scene, 0, 1)
    
    def _downsample(self, image: np.ndarray, scale: int) -> np.ndarray:
        """Downsample image by given scale factor."""
        from skimage.transform import resize
        
        h, w = image.shape[:2]
        new_h, new_w = h // scale, w // scale
        
        return resize(image, (new_h, new_w), anti_aliasing=True).astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample pair.
        
        Returns:
            Dictionary with 'lr', 'hr' tensors and metadata
        """
        sample = self.samples[idx]
        
        # Check if synthetic or file-based
        if "lr_data" in sample:
            lr = sample["lr_data"]
            hr = sample["hr_data"]
            name = sample["name"]
        else:
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
