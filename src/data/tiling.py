"""
Tiling utilities for processing large satellite images.

Implements tile extraction with overlap and seamless stitching
to avoid boundary artifacts in super-resolved outputs.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import TILE_SIZE, TILE_OVERLAP, SR_SCALE


@dataclass
class TileInfo:
    """Information about a single tile."""
    data: np.ndarray
    row_start: int
    col_start: int
    row_end: int
    col_end: int


class TileExtractor:
    """
    Extract overlapping tiles from large images.
    
    Handles padding for edge tiles and tracks tile positions
    for later stitching.
    """
    
    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        overlap: int = TILE_OVERLAP,
    ):
        """
        Initialize tile extractor.
        
        Args:
            tile_size: Size of each tile (square)
            overlap: Overlap between adjacent tiles
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
    
    def extract(
        self,
        image: np.ndarray,
    ) -> Generator[TileInfo, None, None]:
        """
        Extract tiles from image with overlap.
        
        Args:
            image: Input image (H, W, C) or (C, H, W)
        
        Yields:
            TileInfo objects containing tile data and position
        """
        # Ensure HWC format
        if image.ndim == 3 and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
        
        h, w = image.shape[:2]
        
        # Calculate padded size to cover full image
        pad_h = (self.stride - (h - self.tile_size) % self.stride) % self.stride
        pad_w = (self.stride - (w - self.tile_size) % self.stride) % self.stride
        
        # Pad image if needed
        if pad_h > 0 or pad_w > 0:
            if image.ndim == 3:
                image = np.pad(
                    image,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode='reflect'
                )
            else:
                image = np.pad(
                    image,
                    ((0, pad_h), (0, pad_w)),
                    mode='reflect'
                )
        
        padded_h, padded_w = image.shape[:2]
        
        # Extract tiles
        for row in range(0, padded_h - self.tile_size + 1, self.stride):
            for col in range(0, padded_w - self.tile_size + 1, self.stride):
                tile = image[
                    row:row + self.tile_size,
                    col:col + self.tile_size
                ]
                
                yield TileInfo(
                    data=tile.copy(),
                    row_start=row,
                    col_start=col,
                    row_end=min(row + self.tile_size, h),
                    col_end=min(col + self.tile_size, w),
                )
    
    def get_tile_count(self, height: int, width: int) -> Tuple[int, int]:
        """
        Calculate number of tiles in each dimension.
        
        Args:
            height: Image height
            width: Image width
        
        Returns:
            Tuple of (rows, cols) tile counts
        """
        rows = max(1, (height - self.overlap) // self.stride)
        cols = max(1, (width - self.overlap) // self.stride)
        return rows, cols


class TileStitcher:
    """
    Stitch super-resolved tiles back together.
    
    Uses linear blending in overlap regions to avoid
    visible seams in the final output.
    """
    
    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        overlap: int = TILE_OVERLAP,
        scale: int = SR_SCALE,
    ):
        """
        Initialize tile stitcher.
        
        Args:
            tile_size: Original LR tile size
            overlap: Overlap used during extraction
            scale: Super-resolution scale factor
        """
        self.lr_tile_size = tile_size
        self.lr_overlap = overlap
        self.lr_stride = tile_size - overlap
        
        # SR output dimensions
        self.sr_tile_size = tile_size * scale
        self.sr_overlap = overlap * scale
        self.sr_stride = self.lr_stride * scale
        self.scale = scale
        
        # Pre-compute blending weights
        self._compute_blend_weights()
    
    def _compute_blend_weights(self):
        """Compute linear blending weights for overlap regions."""
        # Create weight matrix for a single tile
        weights = np.ones((self.sr_tile_size, self.sr_tile_size), dtype=np.float32)
        
        # Linear ramp for overlap regions
        ramp = np.linspace(0, 1, self.sr_overlap)
        
        # Left edge
        weights[:, :self.sr_overlap] *= ramp[np.newaxis, :]
        
        # Right edge
        weights[:, -self.sr_overlap:] *= ramp[np.newaxis, ::-1]
        
        # Top edge
        weights[:self.sr_overlap, :] *= ramp[:, np.newaxis]
        
        # Bottom edge
        weights[-self.sr_overlap:, :] *= ramp[::-1, np.newaxis]
        
        self.blend_weights = weights
    
    def stitch(
        self,
        tiles: List[Tuple[np.ndarray, int, int]],
        output_height: int,
        output_width: int,
    ) -> np.ndarray:
        """
        Stitch tiles into a single image.
        
        Args:
            tiles: List of (tile_data, row_start, col_start) tuples
                   Row/col are in LR coordinates
            output_height: Final output height (in LR pixels, will be scaled)
            output_width: Final output width (in LR pixels, will be scaled)
        
        Returns:
            Stitched image
        """
        # Output size in SR coordinates
        sr_height = output_height * self.scale
        sr_width = output_width * self.scale
        
        # Determine number of channels
        sample_tile = tiles[0][0]
        if sample_tile.ndim == 3:
            if sample_tile.shape[0] <= 4:  # CHW format
                channels = sample_tile.shape[0]
                output = np.zeros((channels, sr_height, sr_width), dtype=np.float32)
                weight_sum = np.zeros((sr_height, sr_width), dtype=np.float32)
            else:  # HWC format
                channels = sample_tile.shape[2]
                output = np.zeros((sr_height, sr_width, channels), dtype=np.float32)
                weight_sum = np.zeros((sr_height, sr_width), dtype=np.float32)
        else:
            output = np.zeros((sr_height, sr_width), dtype=np.float32)
            weight_sum = np.zeros((sr_height, sr_width), dtype=np.float32)
        
        is_chw = sample_tile.ndim == 3 and sample_tile.shape[0] <= 4
        
        for tile_data, lr_row, lr_col in tiles:
            # Convert to SR coordinates
            sr_row = lr_row * self.scale
            sr_col = lr_col * self.scale
            
            # Get actual tile size (may be smaller at edges)
            if is_chw:
                _, th, tw = tile_data.shape
            elif tile_data.ndim == 3:
                th, tw, _ = tile_data.shape
            else:
                th, tw = tile_data.shape
            
            # Crop weights to match tile size
            weights = self.blend_weights[:th, :tw]
            
            # Calculate output region
            row_end = min(sr_row + th, sr_height)
            col_end = min(sr_col + tw, sr_width)
            
            # Actual region to write
            write_h = row_end - sr_row
            write_w = col_end - sr_col
            
            # Add weighted tile to output
            if is_chw:
                for c in range(channels):
                    output[c, sr_row:row_end, sr_col:col_end] += (
                        tile_data[c, :write_h, :write_w] * weights[:write_h, :write_w]
                    )
            elif tile_data.ndim == 3:
                for c in range(channels):
                    output[sr_row:row_end, sr_col:col_end, c] += (
                        tile_data[:write_h, :write_w, c] * weights[:write_h, :write_w]
                    )
            else:
                output[sr_row:row_end, sr_col:col_end] += (
                    tile_data[:write_h, :write_w] * weights[:write_h, :write_w]
                )
            
            weight_sum[sr_row:row_end, sr_col:col_end] += weights[:write_h, :write_w]
        
        # Normalize by weight sum
        weight_sum = np.maximum(weight_sum, 1e-10)  # Avoid division by zero
        
        if is_chw:
            for c in range(channels):
                output[c] /= weight_sum
        elif output.ndim == 3:
            for c in range(channels):
                output[:, :, c] /= weight_sum
        else:
            output /= weight_sum
        
        return output


def extract_tiles_tensor(
    tensor: torch.Tensor,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Extract tiles from a tensor.
    
    Args:
        tensor: Input tensor (B, C, H, W) or (C, H, W)
        tile_size: Size of each tile
        overlap: Overlap between tiles
    
    Returns:
        Tuple of (tiles tensor, original size)
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    b, c, h, w = tensor.shape
    stride = tile_size - overlap
    
    # Use unfold for efficient tile extraction
    tiles = tensor.unfold(2, tile_size, stride).unfold(3, tile_size, stride)
    
    # Reshape to (num_tiles, C, tile_size, tile_size)
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
    tiles = tiles.view(-1, c, tile_size, tile_size)
    
    return tiles, (h, w)
