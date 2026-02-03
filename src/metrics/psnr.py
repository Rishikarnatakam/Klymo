"""
Peak Signal-to-Noise Ratio (PSNR) computation.

Standard metric for measuring image reconstruction quality.
"""

import numpy as np
import torch
from typing import Union, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import METRICS_CONFIG


def compute_psnr(
    sr: Union[np.ndarray, torch.Tensor],
    hr: Union[np.ndarray, torch.Tensor],
    max_val: float = METRICS_CONFIG["psnr_max_val"],
) -> float:
    """
    Compute PSNR between super-resolved and high-resolution images.
    
    PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    
    Higher values indicate better quality (typically 20-40 dB for SR).
    
    Args:
        sr: Super-resolved image
        hr: High-resolution ground truth
        max_val: Maximum pixel value (1.0 for normalized, 255 for 8-bit)
    
    Returns:
        PSNR value in dB
    """
    # Convert to numpy if tensor
    if isinstance(sr, torch.Tensor):
        sr = sr.detach().cpu().numpy()
    if isinstance(hr, torch.Tensor):
        hr = hr.detach().cpu().numpy()
    
    # Ensure same dtype
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)
    
    # Compute MSE
    mse = np.mean((sr - hr) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    
    return float(psnr)


def psnr_batch(
    sr_batch: Union[np.ndarray, torch.Tensor],
    hr_batch: Union[np.ndarray, torch.Tensor],
    max_val: float = METRICS_CONFIG["psnr_max_val"],
) -> dict:
    """
    Compute PSNR for a batch of images.
    
    Args:
        sr_batch: Batch of SR images (B, C, H, W) or list of images
        hr_batch: Batch of HR ground truth images
        max_val: Maximum pixel value
    
    Returns:
        Dictionary with mean, std, min, max PSNR values
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
    
    # Compute PSNR for each image
    psnr_values = []
    for i in range(len(sr_batch)):
        psnr = compute_psnr(sr_batch[i], hr_batch[i], max_val)
        psnr_values.append(psnr)
    
    psnr_values = np.array(psnr_values)
    
    # Filter out infinity values for statistics
    finite_psnr = psnr_values[np.isfinite(psnr_values)]
    
    return {
        "mean": float(np.mean(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        "std": float(np.std(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        "min": float(np.min(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        "max": float(np.max(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        "all_values": psnr_values.tolist(),
    }


def compute_psnr_y(
    sr: np.ndarray,
    hr: np.ndarray,
    max_val: float = METRICS_CONFIG["psnr_max_val"],
) -> float:
    """
    Compute PSNR on Y channel only (luminance).
    
    This is commonly used in SR benchmarks as it focuses on
    structural detail rather than color accuracy.
    
    Args:
        sr: Super-resolved RGB image (H, W, 3) or (3, H, W)
        hr: High-resolution RGB image
        max_val: Maximum pixel value
    
    Returns:
        PSNR value in dB
    """
    # Ensure HWC format
    if sr.ndim == 3 and sr.shape[0] == 3:
        sr = sr.transpose(1, 2, 0)
    if hr.ndim == 3 and hr.shape[0] == 3:
        hr = hr.transpose(1, 2, 0)
    
    # Convert RGB to Y (luminance)
    # Y = 0.299*R + 0.587*G + 0.114*B
    sr_y = 0.299 * sr[:, :, 0] + 0.587 * sr[:, :, 1] + 0.114 * sr[:, :, 2]
    hr_y = 0.299 * hr[:, :, 0] + 0.587 * hr[:, :, 1] + 0.114 * hr[:, :, 2]
    
    return compute_psnr(sr_y, hr_y, max_val)
