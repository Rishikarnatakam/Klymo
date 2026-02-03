"""
Preprocessing utilities for Sentinel-2 imagery.

Handles the conversion from 16-bit reflectance values to normalized
tensors suitable for neural network inference.
"""

import numpy as np
import torch
from typing import Union, Tuple
import cv2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import REFLECTANCE_MIN, REFLECTANCE_MAX, PREPROCESSING


def clip_reflectance(
    image: np.ndarray,
    min_val: float = REFLECTANCE_MIN,
    max_val: float = REFLECTANCE_MAX,
) -> np.ndarray:
    """
    Clip Sentinel-2 reflectance values to valid range.
    
    Args:
        image: Input image with raw reflectance values
        min_val: Minimum valid reflectance (default: 0)
        max_val: Maximum valid reflectance (default: 3000)
    
    Returns:
        Clipped image array
    """
    return np.clip(image, min_val, max_val)


def normalize_reflectance(
    image: np.ndarray,
    max_val: float = REFLECTANCE_MAX,
) -> np.ndarray:
    """
    Normalize reflectance values to [0, 1] range.
    
    Args:
        image: Clipped reflectance image
        max_val: Maximum reflectance value for normalization
    
    Returns:
        Normalized image in [0, 1] range
    """
    return image.astype(np.float32) / max_val


def denormalize(
    image: np.ndarray,
    max_val: float = REFLECTANCE_MAX,
) -> np.ndarray:
    """
    Convert normalized image back to reflectance values.
    
    Args:
        image: Normalized image in [0, 1] range
        max_val: Maximum reflectance value
    
    Returns:
        Denormalized image
    """
    return (image * max_val).astype(np.float32)


def to_8bit_visualization(
    image: np.ndarray,
    percentile_clip: Tuple[float, float] = (2, 98),
) -> np.ndarray:
    """
    Convert normalized image to 8-bit for visualization.
    
    Uses percentile clipping for better visual contrast.
    
    Args:
        image: Normalized image in [0, 1] range
        percentile_clip: Lower and upper percentiles for contrast stretching
    
    Returns:
        8-bit image suitable for display/saving
    """
    # Handle both HWC and CHW formats
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Percentile-based contrast stretching
    p_low, p_high = np.percentile(image, percentile_clip)
    
    # Avoid division by zero
    if p_high - p_low < 1e-6:
        p_high = p_low + 1e-6
    
    # Stretch and clip
    stretched = (image - p_low) / (p_high - p_low)
    stretched = np.clip(stretched, 0, 1)
    
    # Convert to 8-bit
    return (stretched * 255).astype(np.uint8)


def preprocess_sentinel2(
    image: np.ndarray,
    return_tensor: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Full preprocessing pipeline for Sentinel-2 imagery.
    
    Steps:
    1. Clip reflectance to valid range [0, 3000]
    2. Normalize to [0, 1]
    3. Optionally convert to PyTorch tensor
    
    Args:
        image: Raw Sentinel-2 image (H, W, C) or (C, H, W)
        return_tensor: If True, return PyTorch tensor
    
    Returns:
        Preprocessed image as numpy array or tensor
    """
    # Ensure HWC format for processing
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Step 1: Clip to valid range
    clipped = clip_reflectance(image)
    
    # Step 2: Normalize to [0, 1]
    normalized = normalize_reflectance(clipped)
    
    if return_tensor:
        # Convert to CHW format for PyTorch
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        return tensor
    
    return normalized


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array for visualization.
    
    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W)
    
    Returns:
        Numpy array in (H, W, C) format
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Move to CPU if needed
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert CHW to HWC
    array = tensor.numpy().transpose(1, 2, 0)
    
    return array


def apply_gamma_correction(
    image: np.ndarray,
    gamma: float = 2.2,
) -> np.ndarray:
    """
    Apply gamma correction for better visual appearance.
    
    Args:
        image: Normalized image in [0, 1] range
        gamma: Gamma value (default: 2.2 for sRGB)
    
    Returns:
        Gamma-corrected image
    """
    return np.power(image, 1.0 / gamma)


def match_histogram(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Match histogram of source image to reference.
    
    Used for color consistency between LR and SR outputs.
    
    Args:
        source: Image to transform
        reference: Reference image for histogram matching
    
    Returns:
        Histogram-matched image
    """
    from skimage import exposure
    
    matched = exposure.match_histograms(source, reference, channel_axis=-1)
    return matched.astype(np.float32)
