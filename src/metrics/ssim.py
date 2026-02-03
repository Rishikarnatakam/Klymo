"""
Structural Similarity Index (SSIM) computation.

Measures perceptual quality by comparing luminance, contrast, and structure.
"""

import numpy as np
import torch
from typing import Union, Optional, Tuple
from scipy import ndimage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import METRICS_CONFIG


def gaussian_kernel(
    size: int = 11,
    sigma: float = 1.5,
) -> np.ndarray:
    """Create a Gaussian kernel for SSIM computation."""
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def compute_ssim(
    sr: Union[np.ndarray, torch.Tensor],
    hr: Union[np.ndarray, torch.Tensor],
    data_range: float = METRICS_CONFIG["ssim_data_range"],
    window_size: int = METRICS_CONFIG["ssim_window_size"],
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """
    Compute SSIM between super-resolved and high-resolution images.
    
    SSIM combines luminance, contrast, and structural comparisons.
    Values range from -1 to 1, with 1 indicating identical images.
    
    Args:
        sr: Super-resolved image (H, W) or (H, W, C) or (C, H, W)
        hr: High-resolution ground truth
        data_range: Dynamic range of the images
        window_size: Size of the Gaussian window
        K1: Constant for luminance comparison
        K2: Constant for contrast comparison
    
    Returns:
        SSIM value (0-1 for similar images)
    """
    # Convert to numpy if tensor
    if isinstance(sr, torch.Tensor):
        sr = sr.detach().cpu().numpy()
    if isinstance(hr, torch.Tensor):
        hr = hr.detach().cpu().numpy()
    
    # Ensure float type
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)
    
    # Handle multi-channel images
    if sr.ndim == 3:
        if sr.shape[0] <= 4:  # CHW format
            sr = sr.transpose(1, 2, 0)
            hr = hr.transpose(1, 2, 0)
        
        # Compute SSIM for each channel and average
        ssim_values = []
        for c in range(sr.shape[2]):
            ssim_c = _ssim_single_channel(
                sr[:, :, c], hr[:, :, c],
                data_range, window_size, K1, K2
            )
            ssim_values.append(ssim_c)
        return float(np.mean(ssim_values))
    else:
        return _ssim_single_channel(sr, hr, data_range, window_size, K1, K2)


def _ssim_single_channel(
    sr: np.ndarray,
    hr: np.ndarray,
    data_range: float,
    window_size: int,
    K1: float,
    K2: float,
) -> float:
    """Compute SSIM for a single channel."""
    # Constants
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # Create Gaussian kernel
    kernel = gaussian_kernel(window_size, 1.5)
    
    # Compute local means
    mu_sr = ndimage.convolve(sr, kernel, mode='reflect')
    mu_hr = ndimage.convolve(hr, kernel, mode='reflect')
    
    # Compute local variances and covariance
    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr
    
    sigma_sr_sq = ndimage.convolve(sr ** 2, kernel, mode='reflect') - mu_sr_sq
    sigma_hr_sq = ndimage.convolve(hr ** 2, kernel, mode='reflect') - mu_hr_sq
    sigma_sr_hr = ndimage.convolve(sr * hr, kernel, mode='reflect') - mu_sr_hr
    
    # Ensure non-negative variances
    sigma_sr_sq = np.maximum(sigma_sr_sq, 0)
    sigma_hr_sq = np.maximum(sigma_hr_sq, 0)
    
    # SSIM formula
    numerator = (2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)
    denominator = (mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2)
    
    ssim_map = numerator / denominator
    
    return float(np.mean(ssim_map))


def ssim_batch(
    sr_batch: Union[np.ndarray, torch.Tensor],
    hr_batch: Union[np.ndarray, torch.Tensor],
    data_range: float = METRICS_CONFIG["ssim_data_range"],
) -> dict:
    """
    Compute SSIM for a batch of images.
    
    Args:
        sr_batch: Batch of SR images (B, C, H, W) or list
        hr_batch: Batch of HR ground truth images
        data_range: Dynamic range of the images
    
    Returns:
        Dictionary with mean, std, min, max SSIM values
    """
    # Handle different input formats
    if isinstance(sr_batch, torch.Tensor):
        sr_batch = sr_batch.detach().cpu().numpy()
    if isinstance(hr_batch, torch.Tensor):
        hr_batch = hr_batch.detach().cpu().numpy()
    
    if isinstance(sr_batch, list):
        sr_batch = np.array(sr_batch)
        hr_batch = np.array(hr_batch)
    
    # Add batch dimension if needed
    if sr_batch.ndim == 3:
        sr_batch = sr_batch[np.newaxis, ...]
        hr_batch = hr_batch[np.newaxis, ...]
    
    # Compute SSIM for each image
    ssim_values = []
    for i in range(len(sr_batch)):
        ssim = compute_ssim(sr_batch[i], hr_batch[i], data_range)
        ssim_values.append(ssim)
    
    ssim_values = np.array(ssim_values)
    
    return {
        "mean": float(np.mean(ssim_values)),
        "std": float(np.std(ssim_values)),
        "min": float(np.min(ssim_values)),
        "max": float(np.max(ssim_values)),
        "all_values": ssim_values.tolist(),
    }


def compute_ssim_y(
    sr: np.ndarray,
    hr: np.ndarray,
    data_range: float = METRICS_CONFIG["ssim_data_range"],
) -> float:
    """
    Compute SSIM on Y channel only (luminance).
    
    Args:
        sr: Super-resolved RGB image (H, W, 3) or (3, H, W)
        hr: High-resolution RGB image
        data_range: Dynamic range
    
    Returns:
        SSIM value
    """
    # Ensure HWC format
    if sr.ndim == 3 and sr.shape[0] == 3:
        sr = sr.transpose(1, 2, 0)
    if hr.ndim == 3 and hr.shape[0] == 3:
        hr = hr.transpose(1, 2, 0)
    
    # Convert RGB to Y (luminance)
    sr_y = 0.299 * sr[:, :, 0] + 0.587 * sr[:, :, 1] + 0.114 * sr[:, :, 2]
    hr_y = 0.299 * hr[:, :, 0] + 0.587 * hr[:, :, 1] + 0.114 * hr[:, :, 2]
    
    return compute_ssim(sr_y, hr_y, data_range)


# Optional: Use pytorch-msssim if available for GPU acceleration
try:
    from pytorch_msssim import ssim as pt_ssim, ms_ssim
    
    def compute_ssim_torch(
        sr: torch.Tensor,
        hr: torch.Tensor,
        data_range: float = 1.0,
    ) -> float:
        """
        Compute SSIM using PyTorch (GPU-accelerated).
        
        Args:
            sr: Super-resolved tensor (B, C, H, W) or (C, H, W)
            hr: High-resolution tensor
            data_range: Dynamic range
        
        Returns:
            SSIM value
        """
        if sr.dim() == 3:
            sr = sr.unsqueeze(0)
            hr = hr.unsqueeze(0)
        
        return float(pt_ssim(sr, hr, data_range=data_range))
    
    TORCH_SSIM_AVAILABLE = True
except ImportError:
    TORCH_SSIM_AVAILABLE = False
    compute_ssim_torch = None
