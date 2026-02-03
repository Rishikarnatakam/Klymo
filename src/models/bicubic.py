"""
Bicubic upscaling baseline.

Provides a simple bicubic interpolation baseline for comparison
with neural network super-resolution methods.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import SR_SCALE


def bicubic_upscale(
    image: Union[np.ndarray, torch.Tensor],
    scale: int = SR_SCALE,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Upscale image using bicubic interpolation.
    
    Args:
        image: Input image as numpy array (H, W, C) or (C, H, W)
               or torch tensor (B, C, H, W) or (C, H, W)
        scale: Upscaling factor (default: 4)
    
    Returns:
        Upscaled image in same format as input
    """
    is_numpy = isinstance(image, np.ndarray)
    
    if is_numpy:
        # Convert numpy to tensor
        if image.ndim == 3:
            if image.shape[0] <= 4:  # CHW
                tensor = torch.from_numpy(image).unsqueeze(0).float()
            else:  # HWC
                tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        else:  # HW (grayscale)
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    else:
        tensor = image
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
    
    # Apply bicubic upscaling
    upscaled = F.interpolate(
        tensor,
        scale_factor=scale,
        mode='bicubic',
        align_corners=False,
        antialias=True,
    )
    
    # Clamp to valid range
    upscaled = torch.clamp(upscaled, 0, 1)
    
    if is_numpy:
        # Convert back to numpy
        result = upscaled.squeeze(0).numpy()
        if image.ndim == 3 and image.shape[0] > 4:  # Was HWC
            result = result.transpose(1, 2, 0)
        return result
    else:
        if image.dim() == 3:
            return upscaled.squeeze(0)
        return upscaled


class BicubicUpscaler:
    """
    Bicubic upscaler class with same interface as neural SR models.
    
    Used as a baseline for comparison.
    """
    
    def __init__(self, scale: int = SR_SCALE):
        """
        Initialize bicubic upscaler.
        
        Args:
            scale: Upscaling factor
        """
        self.scale = scale
        self.name = f"Bicubic {scale}×"
    
    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Upscale image.
        
        Args:
            image: Input image
        
        Returns:
            Upscaled image
        """
        return bicubic_upscale(image, self.scale)
    
    def inference(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run inference on a batch of images.
        
        Args:
            tensor: Input tensor (B, C, H, W)
        
        Returns:
            Upscaled tensor
        """
        return bicubic_upscale(tensor, self.scale)
    
    def to(self, device: str) -> "BicubicUpscaler":
        """No-op for compatibility with neural network models."""
        return self
    
    def eval(self) -> "BicubicUpscaler":
        """No-op for compatibility with neural network models."""
        return self
