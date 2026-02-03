"""
End-to-end super-resolution inference pipeline.

Handles the complete workflow from input image to super-resolved output,
including preprocessing, tiling, inference, and post-processing.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    SR_SCALE,
    TILE_SIZE,
    TILE_OVERLAP,
    DEMO_DIR,
    VISUALIZATIONS_DIR,
)
from src.data.preprocessing import (
    preprocess_sentinel2,
    to_8bit_visualization,
    tensor_to_numpy,
)
from src.data.tiling import TileExtractor, TileStitcher
from src.models.swinir import SwinIRWrapper
from src.models.bicubic import BicubicUpscaler
from src.inference.postprocess import (
    apply_color_consistency,
    run_hallucination_checks,
)


class SuperResolutionPipeline:
    """
    Complete super-resolution pipeline for Sentinel-2 imagery.
    
    Pipeline stages:
    1. Preprocessing (clip, normalize)
    2. Tiling (for memory efficiency)
    3. SR inference (per tile)
    4. Tile stitching
    5. Post-processing (color consistency)
    6. Hallucination checks
    """
    
    def __init__(
        self,
        scale: int = SR_SCALE,
        tile_size: int = TILE_SIZE,
        tile_overlap: int = TILE_OVERLAP,
        device: Optional[str] = None,
        use_tiling: bool = True,
        apply_postprocess: bool = True,
        run_checks: bool = True,
    ):
        """
        Initialize the super-resolution pipeline.
        
        Args:
            scale: Super-resolution scale factor
            tile_size: Size of tiles for processing
            tile_overlap: Overlap between tiles
            device: Compute device ('cuda' or 'cpu')
            use_tiling: Whether to use tiling for large images
            apply_postprocess: Whether to apply color consistency
            run_checks: Whether to run hallucination checks
        """
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tiling = use_tiling
        self.apply_postprocess = apply_postprocess
        self.run_checks = run_checks
        
        # Initialize components
        print(f"Initializing SR pipeline (scale={scale}×, device={self.device})")
        
        self.sr_model = SwinIRWrapper(scale=scale, device=self.device)
        self.bicubic = BicubicUpscaler(scale=scale)
        self.tile_extractor = TileExtractor(tile_size=tile_size, overlap=tile_overlap)
        self.tile_stitcher = TileStitcher(tile_size=tile_size, overlap=tile_overlap, scale=scale)
        
        print("Pipeline ready")
    
    def run(
        self,
        image: Union[np.ndarray, str, Path],
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete super-resolution pipeline.
        
        Args:
            image: Input image as numpy array (H, W, C), or path to image file
            return_intermediate: Whether to return intermediate results
        
        Returns:
            Dictionary with:
            - 'sr': Super-resolved image
            - 'bicubic': Bicubic upscaled baseline
            - 'lr': Original low-resolution input
            - 'checks': Hallucination check results (if enabled)
            - 'intermediate': Intermediate results (if requested)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        # Ensure correct format (H, W, C) and range [0, 1]
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        if image.max() > 1:
            if image.max() > 255:
                # 16-bit
                image = np.clip(image, 0, 3000) / 3000.0
            else:
                # 8-bit
                image = image / 255.0
        
        image = image.astype(np.float32)
        
        results = {
            'lr': image,
            'intermediate': {} if return_intermediate else None,
        }
        
        h, w = image.shape[:2]
        
        # Step 1: Bicubic baseline
        print("Computing bicubic baseline...")
        lr_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        bicubic_tensor = self.bicubic.inference(lr_tensor)
        bicubic_np = bicubic_tensor.squeeze(0).numpy().transpose(1, 2, 0)
        results['bicubic'] = bicubic_np
        
        # Step 2: Super-resolution
        if self.use_tiling and (h > self.tile_size or w > self.tile_size):
            print(f"Processing with tiling ({h}×{w} image)...")
            sr_np = self._run_tiled(image, return_intermediate, results)
        else:
            print("Processing full image...")
            sr_np = self._run_single(image)
        
        # Step 3: Post-processing
        if self.apply_postprocess:
            print("Applying color consistency...")
            sr_np = apply_color_consistency(sr_np, image, strength=0.3)
        
        results['sr'] = sr_np
        
        # Step 4: Hallucination checks
        if self.run_checks:
            print("Running hallucination checks...")
            results['checks'] = run_hallucination_checks(sr_np, image)
            
            if results['checks']['passed']:
                print("✓ All hallucination checks passed")
            else:
                print("⚠ Some hallucination checks failed:")
                for check_name, check_result in results['checks']['checks'].items():
                    status = "✓" if check_result['passed'] else "✗"
                    print(f"  {status} {check_name}: {check_result['score']:.4f}")
        
        return results
    
    def _run_single(self, image: np.ndarray) -> np.ndarray:
        """Run SR on a single image without tiling."""
        # Convert to tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Run inference
        sr_tensor = self.sr_model.inference(tensor)
        
        # Convert back to numpy
        if sr_tensor.is_cuda:
            sr_tensor = sr_tensor.cpu()
        
        return sr_tensor.numpy().transpose(1, 2, 0)
    
    def _run_tiled(
        self,
        image: np.ndarray,
        return_intermediate: bool,
        results: Dict,
    ) -> np.ndarray:
        """Run SR on a large image using tiling."""
        h, w = image.shape[:2]
        
        # Extract tiles
        tiles = list(self.tile_extractor.extract(image))
        print(f"  Extracted {len(tiles)} tiles")
        
        # Process each tile
        sr_tiles = []
        for tile_info in tqdm(tiles, desc="Processing tiles"):
            # Convert to tensor
            tile_tensor = torch.from_numpy(tile_info.data).permute(2, 0, 1).float()
            
            # Run SR
            sr_tensor = self.sr_model.inference(tile_tensor)
            
            # Convert back
            if sr_tensor.is_cuda:
                sr_tensor = sr_tensor.cpu()
            sr_tile = sr_tensor.numpy().transpose(1, 2, 0)
            
            # Store with position (in LR coordinates)
            sr_tiles.append((sr_tile, tile_info.row_start, tile_info.col_start))
        
        if return_intermediate:
            results['intermediate']['tiles'] = sr_tiles
        
        # Stitch tiles together
        print("  Stitching tiles...")
        sr_image = self.tile_stitcher.stitch(sr_tiles, h, w)
        
        return sr_image
    
    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load an image from file."""
        path = Path(path)
        
        if path.suffix.lower() in ['.tif', '.tiff']:
            # Try rasterio for GeoTIFF
            try:
                import rasterio
                with rasterio.open(path) as src:
                    image = src.read()  # (C, H, W)
                    image = image.transpose(1, 2, 0)  # (H, W, C)
            except ImportError:
                # Fall back to PIL
                img = Image.open(path)
                image = np.array(img)
        else:
            img = Image.open(path)
            image = np.array(img)
        
        return image.astype(np.float32)
    
    def compare(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run pipeline and create comparison visualization.
        
        Args:
            image: Input image
            save_path: Optional path to save comparison image
        
        Returns:
            Dictionary with comparison images
        """
        results = self.run(image)
        
        # Create side-by-side comparison
        lr = results['lr']
        bicubic = results['bicubic']
        sr = results['sr']
        
        # Upscale LR for visual comparison
        from skimage.transform import resize
        lr_upscaled = resize(lr, sr.shape[:2], order=3)
        
        # Convert to 8-bit
        lr_8bit = to_8bit_visualization(lr_upscaled)
        bicubic_8bit = to_8bit_visualization(bicubic)
        sr_8bit = to_8bit_visualization(sr)
        
        # Create comparison image
        comparison = np.concatenate([lr_8bit, bicubic_8bit, sr_8bit], axis=1)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(comparison).save(save_path)
            print(f"Saved comparison to {save_path}")
        
        return {
            'lr': lr_8bit,
            'bicubic': bicubic_8bit,
            'sr': sr_8bit,
            'comparison': comparison,
            'checks': results.get('checks'),
        }
    
    def save_results(
        self,
        results: Dict[str, Any],
        name: str,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Save pipeline results to disk.
        
        Args:
            results: Results from run() or compare()
            name: Base name for output files
            output_dir: Output directory (default: VISUALIZATIONS_DIR)
        
        Returns:
            Dictionary mapping result names to file paths
        """
        output_dir = output_dir or VISUALIZATIONS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for key in ['lr', 'bicubic', 'sr']:
            if key in results:
                img = results[key]
                if img.max() <= 1:
                    img = to_8bit_visualization(img)
                
                path = output_dir / f"{name}_{key}.png"
                Image.fromarray(img).save(path)
                saved_paths[key] = path
        
        return saved_paths


def run_demo(
    location: str = "delhi",
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete demo using GEE tiles.
    
    Args:
        location: Demo location name
        save_outputs: Whether to save output images
    
    Returns:
        Pipeline results
    """
    from src.data.gee_fetcher import GEEFetcher
    
    print(f"\n{'='*60}")
    print(f"Running SR demo for {location}")
    print(f"{'='*60}\n")
    
    # Fetch tile
    fetcher = GEEFetcher(authenticate=True)
    tile = fetcher.fetch_tile(location)
    
    if tile is None:
        raise RuntimeError(f"Failed to fetch tile for {location}")
    
    # Save LR tile
    fetcher.save_tile(tile, f"lr_{location}")
    
    # Run pipeline
    pipeline = SuperResolutionPipeline()
    results = pipeline.run(tile)
    
    if save_outputs:
        # Save outputs
        pipeline.save_results(results, location, DEMO_DIR)
        
        # Save SR tile
        sr_8bit = to_8bit_visualization(results['sr'])
        Image.fromarray(sr_8bit).save(DEMO_DIR / f"sr_{location}.png")
        
        # Save bicubic
        bicubic_8bit = to_8bit_visualization(results['bicubic'])
        Image.fromarray(bicubic_8bit).save(DEMO_DIR / f"bicubic_{location}.png")
        
        print(f"\nOutputs saved to {DEMO_DIR}")
    
    return results


if __name__ == "__main__":
    # Run demo
    run_demo("delhi")
