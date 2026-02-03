"""
Google Earth Engine fetcher for Sentinel-2 imagery.

Provides utilities to fetch Sentinel-2 L2A tiles from GEE
for real-world inference.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timedelta
import requests
from io import BytesIO
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    LOCATIONS,
    DEFAULT_LOCATION,
    GEE_CONFIG,
    SENTINEL2_BANDS,
    RGB_BANDS,
    TILE_SIZE,
    REFLECTANCE_MAX,
    VISUALIZATIONS_DIR,
)


class GEEFetcher:
    """
    Fetch Sentinel-2 imagery from Google Earth Engine.
    
    Requires GEE authentication - no fallbacks.
    """
    
    def __init__(
        self,
        authenticate: bool = True,
        project: Optional[str] = None,
    ):
        """
        Initialize GEE fetcher.
        
        Args:
            authenticate: Whether to authenticate with GEE
            project: GEE project ID (optional, uses default if None)
        """
        self.ee = None
        self.authenticated = False
        self.project = project
        
        if authenticate:
            self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Earth Engine."""
        try:
            import ee
            self.ee = ee
            
            # Try to initialize
            try:
                if self.project:
                    ee.Initialize(project=self.project)
                else:
                    ee.Initialize()
                self.authenticated = True
                print("✓ Successfully authenticated with Google Earth Engine")
            except Exception as e:
                raise RuntimeError(
                    f"GEE authentication failed: {e}\n"
                    "Please run: earthengine authenticate"
                )
                
        except ImportError:
            raise ImportError(
                "earthengine-api not installed.\n"
                "Install with: pip install earthengine-api"
            )
    
    def fetch_tile(
        self,
        location: str = DEFAULT_LOCATION,
        tile_size: int = TILE_SIZE,
        date_range: Optional[Tuple[str, str]] = None,
        cloud_threshold: int = GEE_CONFIG["cloud_threshold"],
    ) -> np.ndarray:
        """
        Fetch a Sentinel-2 tile for the specified location.
        
        Args:
            location: Location key from LOCATIONS config
            tile_size: Size of tile to fetch (in pixels)
            date_range: Optional (start_date, end_date) tuple, format YYYY-MM-DD
            cloud_threshold: Maximum cloud coverage percentage
        
        Returns:
            RGB image as numpy array (H, W, 3) normalized to [0, 1]
        
        Raises:
            RuntimeError: If not authenticated or fetch fails
        """
        if not self.authenticated:
            raise RuntimeError(
                "Not authenticated with Google Earth Engine.\n"
                "Please run 'earthengine authenticate' first."
            )
        
        # Get location coordinates
        if location in LOCATIONS:
            lat = LOCATIONS[location]["lat"]
            lon = LOCATIONS[location]["lon"]
        else:
            raise ValueError(f"Unknown location: {location}. Use one of {list(LOCATIONS.keys())}")
        
        # Set date range
        if date_range is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=GEE_CONFIG["date_range_days"])
            date_range = (
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
        
        ee = self.ee
        
        # Define point of interest
        point = ee.Geometry.Point([lon, lat])
        
        # Define region (small buffer around point)
        buffer_size = (tile_size * 10) / 2
        region = point.buffer(buffer_size).bounds()
        
        # Get Sentinel-2 collection
        collection = (
            ee.ImageCollection(GEE_CONFIG["collection"])
            .filterBounds(point)
            .filterDate(date_range[0], date_range[1])
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        
        # Get the clearest image
        image = collection.first()
        
        if image is None:
            raise RuntimeError(f"No clear images found for {location} in date range {date_range}")
        
        # Select RGB bands
        rgb_image = image.select(RGB_BANDS)
        
        # Get the image as a numpy array
        url = rgb_image.getThumbURL({
            "region": region,
            "dimensions": f"{tile_size}x{tile_size}",
            "format": "png",
            "min": 0,
            "max": REFLECTANCE_MAX,
        })
        
        # Download the image
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            arr = np.array(img).astype(np.float32) / 255.0
            
            # Ensure 3 channels
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            
            return arr
        else:
            raise RuntimeError(f"Failed to download image: HTTP {response.status_code}")
    
    def save_tile(
        self,
        tile: np.ndarray,
        name: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save a tile to disk.
        
        Args:
            tile: Image array (H, W, 3) in [0, 1] range
            name: Output filename (without extension)
            output_dir: Output directory (default: VISUALIZATIONS_DIR)
        
        Returns:
            Path to saved file
        """
        output_dir = output_dir or VISUALIZATIONS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to 8-bit
        img_8bit = (tile * 255).astype(np.uint8)
        
        # Save as PNG
        output_path = output_dir / f"{name}.png"
        Image.fromarray(img_8bit).save(output_path)
        
        print(f"Saved tile to {output_path}")
        return output_path
