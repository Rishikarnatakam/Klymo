"""
Post-processing utilities for super-resolution outputs.

Includes hallucination guardrails and color consistency checks.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional
import cv2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import GUARDRAILS, SR_SCALE


def apply_color_consistency(
    sr_image: np.ndarray,
    lr_image: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Apply color consistency between SR output and LR input.
    
    Performs soft histogram matching to prevent color drift
    while preserving SR detail.
    
    Args:
        sr_image: Super-resolved image (H, W, C) in [0, 1]
        lr_image: Low-resolution input (h, w, C) in [0, 1]
        strength: Blending strength for histogram matching (0-1)
    
    Returns:
        Color-corrected SR image
    """
    from skimage import exposure
    
    # Ensure HWC format
    if sr_image.ndim == 3 and sr_image.shape[0] == 3:
        sr_image = sr_image.transpose(1, 2, 0)
    if lr_image.ndim == 3 and lr_image.shape[0] == 3:
        lr_image = lr_image.transpose(1, 2, 0)
    
    # Upscale LR to match SR size for histogram matching
    from skimage.transform import resize
    lr_upscaled = resize(lr_image, sr_image.shape[:2], order=3, anti_aliasing=True)
    
    # Match histogram
    matched = exposure.match_histograms(sr_image, lr_upscaled, channel_axis=-1)
    
    # Blend original SR with matched version
    result = sr_image * (1 - strength) + matched * strength
    
    return np.clip(result, 0, 1).astype(np.float32)


def compute_edge_map(
    image: np.ndarray,
    threshold1: int = 50,
    threshold2: int = 150,
) -> np.ndarray:
    """
    Compute Canny edge map for an image.
    
    Args:
        image: Input image (H, W, C) in [0, 1]
        threshold1: Canny lower threshold
        threshold2: Canny upper threshold
    
    Returns:
        Binary edge map
    """
    # Convert to 8-bit grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Compute Canny edges
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    return edges.astype(np.float32) / 255.0


def check_edge_consistency(
    sr_image: np.ndarray,
    lr_image: np.ndarray,
    threshold: float = GUARDRAILS["edge_consistency_threshold"],
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Check if SR output maintains edge consistency with LR input.
    
    New strong edges in SR that don't exist in bicubic-upscaled LR
    may indicate hallucinated features.
    
    Args:
        sr_image: Super-resolved image (H, W, C)
        lr_image: Low-resolution input (h, w, C)
        threshold: Maximum allowed edge difference score
    
    Returns:
        Tuple of (passed, score, details)
    """
    from skimage.transform import resize
    
    # Ensure HWC format
    if sr_image.ndim == 3 and sr_image.shape[0] == 3:
        sr_image = sr_image.transpose(1, 2, 0)
    if lr_image.ndim == 3 and lr_image.shape[0] == 3:
        lr_image = lr_image.transpose(1, 2, 0)
    
    # Upscale LR to match SR size (bicubic)
    lr_upscaled = resize(lr_image, sr_image.shape[:2], order=3, anti_aliasing=True)
    
    # Compute edge maps
    sr_edges = compute_edge_map(sr_image)
    lr_edges = compute_edge_map(lr_upscaled)
    
    # Dilate LR edges slightly to allow for minor shifts
    kernel = np.ones((3, 3), np.uint8)
    lr_edges_dilated = cv2.dilate(lr_edges, kernel, iterations=1)
    
    # Find new edges in SR that don't exist in LR
    new_edges = sr_edges * (1 - lr_edges_dilated)
    
    # Compute score (fraction of new edge pixels)
    sr_edge_count = np.sum(sr_edges > 0)
    new_edge_count = np.sum(new_edges > 0)
    
    if sr_edge_count > 0:
        new_edge_ratio = new_edge_count / sr_edge_count
    else:
        new_edge_ratio = 0.0
    
    passed = new_edge_ratio < threshold
    
    details = {
        "sr_edge_pixels": int(sr_edge_count),
        "new_edge_pixels": int(new_edge_count),
        "new_edge_ratio": float(new_edge_ratio),
        "threshold": threshold,
        "sr_edges": sr_edges,
        "new_edges": new_edges,
    }
    
    return passed, new_edge_ratio, details


def compute_ndvi(image: np.ndarray, nir_channel: int = 0, red_channel: int = 0) -> np.ndarray:
    """
    Compute approximate NDVI from RGB image.
    
    Note: This is a simplified approximation since true NDVI requires NIR band.
    We use the ratio of green to red as a vegetation indicator.
    
    Args:
        image: RGB image (H, W, 3) in [0, 1]
        nir_channel: NIR channel index (not used in RGB approximation)
        red_channel: Red channel index
    
    Returns:
        Vegetation index map
    """
    # Ensure HWC format
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # Extract channels (RGB order)
    red = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    
    # Compute Green-Red Vegetation Index (GRVI) as approximation
    # Higher values indicate more vegetation
    epsilon = 1e-6
    grvi = (green - red) / (green + red + epsilon)
    
    return grvi


def check_ndvi_stability(
    sr_image: np.ndarray,
    lr_image: np.ndarray,
    tolerance: float = GUARDRAILS["ndvi_tolerance"],
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Check if vegetation indices are stable between LR and SR.
    
    Large NDVI shifts may indicate unrealistic color changes
    or hallucinated vegetation/water features.
    
    Args:
        sr_image: Super-resolved image (H, W, C)
        lr_image: Low-resolution input (h, w, C)
        tolerance: Maximum allowed NDVI difference
    
    Returns:
        Tuple of (passed, score, details)
    """
    from skimage.transform import resize
    
    # Ensure HWC format
    if sr_image.ndim == 3 and sr_image.shape[0] == 3:
        sr_image = sr_image.transpose(1, 2, 0)
    if lr_image.ndim == 3 and lr_image.shape[0] == 3:
        lr_image = lr_image.transpose(1, 2, 0)
    
    # Downsample SR to match LR for fair comparison
    sr_downsampled = resize(sr_image, lr_image.shape[:2], order=3, anti_aliasing=True)
    
    # Compute NDVI for both
    lr_ndvi = compute_ndvi(lr_image)
    sr_ndvi = compute_ndvi(sr_downsampled)
    
    # Compute average absolute difference
    ndvi_diff = np.abs(sr_ndvi - lr_ndvi)
    mean_diff = float(np.mean(ndvi_diff))
    max_diff = float(np.max(ndvi_diff))
    
    passed = mean_diff < tolerance
    
    details = {
        "mean_ndvi_diff": mean_diff,
        "max_ndvi_diff": max_diff,
        "tolerance": tolerance,
        "lr_ndvi_mean": float(np.mean(lr_ndvi)),
        "sr_ndvi_mean": float(np.mean(sr_ndvi)),
    }
    
    return passed, mean_diff, details


def check_color_distribution(
    sr_image: np.ndarray,
    lr_image: np.ndarray,
    threshold: float = GUARDRAILS["color_histogram_threshold"],
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Check if color distribution is preserved between LR and SR.
    
    Uses histogram comparison to detect unrealistic color shifts.
    
    Args:
        sr_image: Super-resolved image (H, W, C)
        lr_image: Low-resolution input (h, w, C)
        threshold: Maximum allowed histogram difference
    
    Returns:
        Tuple of (passed, score, details)
    """
    # Ensure HWC format
    if sr_image.ndim == 3 and sr_image.shape[0] == 3:
        sr_image = sr_image.transpose(1, 2, 0)
    if lr_image.ndim == 3 and lr_image.shape[0] == 3:
        lr_image = lr_image.transpose(1, 2, 0)
    
    # Convert to 8-bit for histogram computation
    sr_8bit = (sr_image * 255).astype(np.uint8)
    lr_8bit = (lr_image * 255).astype(np.uint8)
    
    # Compute histograms for each channel
    channel_diffs = []
    for c in range(3):
        sr_hist = cv2.calcHist([sr_8bit], [c], None, [256], [0, 256])
        lr_hist = cv2.calcHist([lr_8bit], [c], None, [256], [0, 256])
        
        # Normalize histograms
        sr_hist = sr_hist / sr_hist.sum()
        lr_hist = lr_hist / lr_hist.sum()
        
        # Compute correlation (1 = identical, 0 = different)
        correlation = cv2.compareHist(sr_hist, lr_hist, cv2.HISTCMP_CORREL)
        channel_diffs.append(1 - correlation)  # Convert to distance
    
    mean_diff = float(np.mean(channel_diffs))
    
    passed = mean_diff < threshold
    
    details = {
        "channel_differences": channel_diffs,
        "mean_difference": mean_diff,
        "threshold": threshold,
    }
    
    return passed, mean_diff, details


def run_hallucination_checks(
    sr_image: np.ndarray,
    lr_image: np.ndarray,
) -> Dict[str, Any]:
    """
    Run all hallucination guardrail checks.
    
    Args:
        sr_image: Super-resolved image
        lr_image: Low-resolution input
    
    Returns:
        Dictionary with all check results
    """
    results = {
        "passed": True,
        "checks": {},
    }
    
    # Edge consistency check
    edge_passed, edge_score, edge_details = check_edge_consistency(sr_image, lr_image)
    results["checks"]["edge_consistency"] = {
        "passed": edge_passed,
        "score": edge_score,
        "details": {k: v for k, v in edge_details.items() if not isinstance(v, np.ndarray)},
    }
    if not edge_passed:
        results["passed"] = False
    
    # NDVI stability check
    ndvi_passed, ndvi_score, ndvi_details = check_ndvi_stability(sr_image, lr_image)
    results["checks"]["ndvi_stability"] = {
        "passed": ndvi_passed,
        "score": ndvi_score,
        "details": ndvi_details,
    }
    if not ndvi_passed:
        results["passed"] = False
    
    # Color distribution check
    color_passed, color_score, color_details = check_color_distribution(sr_image, lr_image)
    results["checks"]["color_distribution"] = {
        "passed": color_passed,
        "score": color_score,
        "details": color_details,
    }
    if not color_passed:
        results["passed"] = False
    
    return results
