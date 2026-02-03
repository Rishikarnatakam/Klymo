"""
Basic tests for the Sentinel-2 SR pipeline.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPreprocessing:
    """Tests for data preprocessing functions."""
    
    def test_clip_reflectance(self):
        """Test reflectance clipping to valid range."""
        from src.data.preprocessing import clip_reflectance
        
        # Create test image with values outside range
        image = np.array([[-100, 500], [2000, 5000]], dtype=np.float32)
        
        clipped = clip_reflectance(image, min_val=0, max_val=3000)
        
        assert clipped.min() >= 0
        assert clipped.max() <= 3000
        assert clipped[0, 0] == 0
        assert clipped[0, 1] == 500
        assert clipped[1, 0] == 2000
        assert clipped[1, 1] == 3000
    
    def test_normalize_reflectance(self):
        """Test normalization to [0, 1] range."""
        from src.data.preprocessing import normalize_reflectance
        
        image = np.array([[0, 1500], [3000, 1500]], dtype=np.float32)
        
        normalized = normalize_reflectance(image, max_val=3000)
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert np.isclose(normalized[0, 1], 0.5)
    
    def test_preprocess_sentinel2(self):
        """Test full preprocessing pipeline."""
        from src.data.preprocessing import preprocess_sentinel2
        
        # Simulate 16-bit Sentinel-2 data
        image = np.random.randint(0, 4000, (128, 128, 3)).astype(np.float32)
        
        tensor = preprocess_sentinel2(image, return_tensor=True)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)
        assert tensor.min() >= 0
        assert tensor.max() <= 1


class TestTiling:
    """Tests for tile extraction and stitching."""
    
    def test_tile_extractor(self):
        """Test tile extraction with overlap."""
        from src.data.tiling import TileExtractor
        
        extractor = TileExtractor(tile_size=64, overlap=16)
        
        # Create test image
        image = np.random.rand(128, 128, 3).astype(np.float32)
        
        tiles = list(extractor.extract(image))
        
        # Should get 4 tiles for 128x128 with stride 48
        assert len(tiles) > 0
        
        for tile in tiles:
            assert tile.data.shape[:2] == (64, 64)
    
    def test_tile_stitcher(self):
        """Test tile stitching with blending."""
        from src.data.tiling import TileExtractor, TileStitcher
        
        tile_size = 64
        overlap = 16
        scale = 4
        
        extractor = TileExtractor(tile_size=tile_size, overlap=overlap)
        stitcher = TileStitcher(tile_size=tile_size, overlap=overlap, scale=scale)
        
        # Create test image
        h, w = 128, 128
        image = np.random.rand(h, w, 3).astype(np.float32)
        
        # Extract and "super-resolve" tiles (just upscale for test)
        sr_tiles = []
        for tile_info in extractor.extract(image):
            # Simulate 4x upscaling
            sr_tile = np.repeat(np.repeat(tile_info.data, scale, axis=0), scale, axis=1)
            sr_tiles.append((sr_tile, tile_info.row_start, tile_info.col_start))
        
        # Stitch
        stitched = stitcher.stitch(sr_tiles, h, w)
        
        assert stitched.shape == (h * scale, w * scale, 3)


class TestModels:
    """Tests for SR models."""
    
    def test_bicubic_upscale(self):
        """Test bicubic upscaling baseline."""
        from src.models.bicubic import bicubic_upscale
        
        # Test with numpy array
        image = np.random.rand(64, 64, 3).astype(np.float32)
        upscaled = bicubic_upscale(image, scale=4)
        
        assert upscaled.shape == (256, 256, 3)
        
        # Test with tensor
        tensor = torch.rand(1, 3, 64, 64)
        upscaled_tensor = bicubic_upscale(tensor, scale=4)
        
        assert upscaled_tensor.shape == (1, 3, 256, 256)


class TestMetrics:
    """Tests for PSNR and SSIM computation."""
    
    def test_psnr_identical(self):
        """Test PSNR of identical images is infinite."""
        from src.metrics.psnr import compute_psnr
        
        image = np.random.rand(64, 64, 3).astype(np.float32)
        psnr = compute_psnr(image, image)
        
        assert psnr == float('inf')
    
    def test_psnr_different(self):
        """Test PSNR of different images is finite."""
        from src.metrics.psnr import compute_psnr
        
        image1 = np.random.rand(64, 64, 3).astype(np.float32)
        image2 = np.random.rand(64, 64, 3).astype(np.float32)
        
        psnr = compute_psnr(image1, image2)
        
        assert np.isfinite(psnr)
        assert psnr > 0
    
    def test_ssim_identical(self):
        """Test SSIM of identical images is 1.0."""
        from src.metrics.ssim import compute_ssim
        
        image = np.random.rand(64, 64, 3).astype(np.float32)
        ssim = compute_ssim(image, image)
        
        assert np.isclose(ssim, 1.0, atol=1e-6)
    
    def test_ssim_range(self):
        """Test SSIM is in valid range."""
        from src.metrics.ssim import compute_ssim
        
        image1 = np.random.rand(64, 64, 3).astype(np.float32)
        image2 = np.random.rand(64, 64, 3).astype(np.float32)
        
        ssim = compute_ssim(image1, image2)
        
        assert -1 <= ssim <= 1


class TestPostprocessing:
    """Tests for post-processing and hallucination checks."""
    
    def test_color_consistency(self):
        """Test color consistency adjustment."""
        from src.inference.postprocess import apply_color_consistency
        
        sr = np.random.rand(256, 256, 3).astype(np.float32)
        lr = np.random.rand(64, 64, 3).astype(np.float32)
        
        result = apply_color_consistency(sr, lr, strength=0.5)
        
        assert result.shape == sr.shape
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_edge_consistency_check(self):
        """Test edge consistency check."""
        from src.inference.postprocess import check_edge_consistency
        
        # Create similar images (should pass)
        lr = np.random.rand(64, 64, 3).astype(np.float32)
        from skimage.transform import resize
        sr = resize(lr, (256, 256), order=3)
        
        passed, score, details = check_edge_consistency(sr, lr)
        
        # Should mostly pass for interpolated images
        assert isinstance(passed, bool)
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
