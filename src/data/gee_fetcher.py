"""
Google Earth Engine fetcher for Sentinel-2 imagery.

Provides utilities to fetch Sentinel-2 L2A tiles from GEE
for real-world inference demonstrations.
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
    DEMO_LOCATIONS,
    DEFAULT_DEMO_LOCATION,
    GEE_CONFIG,
    SENTINEL2_BANDS,
    RGB_BANDS,
    TILE_SIZE,
    REFLECTANCE_MAX,
    DEMO_DIR,
)


class GEEFetcher:
    """
    Fetch Sentinel-2 imagery from Google Earth Engine.
    
    Provides methods to retrieve small tiles for demo purposes
    without downloading entire scenes.
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
                print(f"⚠️ GEE initialization failed: {e}")
                print("\nTo use real Sentinel-2 data, please authenticate:")
                print("  1. Run: earthengine authenticate")
                print("  2. Follow the browser prompts")
                print("  3. Re-run this script")
                self.authenticated = False
                
        except ImportError:
            print("⚠️ earthengine-api not installed.")
            print("Install with: pip install earthengine-api")
            self.authenticated = False
    
    def fetch_tile(
        self,
        location: str = DEFAULT_DEMO_LOCATION,
        tile_size: int = TILE_SIZE,
        date_range: Optional[Tuple[str, str]] = None,
        cloud_threshold: int = GEE_CONFIG["cloud_threshold"],
    ) -> Optional[np.ndarray]:
        """
        Fetch a Sentinel-2 tile for the specified location.
        
        Args:
            location: Location key from DEMO_LOCATIONS or custom coords
            tile_size: Size of tile to fetch (in pixels)
            date_range: Optional (start_date, end_date) tuple, format YYYY-MM-DD
            cloud_threshold: Maximum cloud coverage percentage
        
        Returns:
            RGB image as numpy array (H, W, 3) normalized to [0, 1]
        
        Raises:
            RuntimeError: If not authenticated with GEE
        """
        if not self.authenticated:
            raise RuntimeError(
                "Not authenticated with Google Earth Engine.\n"
                "Please run 'earthengine authenticate' in terminal first,\n"
                "or use authenticate=True when creating GEEFetcher.\n"
                "Real Sentinel-2 data is required - no synthetic fallback."
            )
        
        # Get location coordinates
        if location in DEMO_LOCATIONS:
            lat = DEMO_LOCATIONS[location]["lat"]
            lon = DEMO_LOCATIONS[location]["lon"]
        else:
            raise ValueError(f"Unknown location: {location}. Use one of {list(DEMO_LOCATIONS.keys())}")
        
        # Set date range
        if date_range is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=GEE_CONFIG["date_range_days"])
            date_range = (
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
        
        try:
            ee = self.ee
            
            # Define point of interest
            point = ee.Geometry.Point([lon, lat])
            
            # Define region (small buffer around point)
            # 10m/pixel * tile_size pixels = region size in meters
            buffer_size = (tile_size * 10) / 2  # Half the tile size
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
                print(f"No clear images found for {location} in date range")
                return self._get_fallback_tile(location, tile_size)
            
            # Select RGB bands
            rgb_image = image.select(RGB_BANDS)
            
            # Get the image as a numpy array
            # Use getThumbURL for small tiles (avoids export complexity)
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
                print(f"Failed to download image: {response.status_code}")
                return self._get_fallback_tile(location, tile_size)
                
        except Exception as e:
            print(f"Error fetching from GEE: {e}")
            return self._get_fallback_tile(location, tile_size)
    
    def _get_fallback_tile(
        self,
        location: str,
        tile_size: int,
    ) -> np.ndarray:
        """
        Generate a realistic fallback tile when GEE is unavailable.
        
        Creates a synthetic urban/rural scene based on location.
        """
        print(f"Using synthetic fallback tile for {location}")
        
        np.random.seed(hash(location) % 2**32)
        
        # Create base image
        if location == "delhi":
            # Dense urban - more buildings, roads
            tile = self._generate_urban_tile(tile_size)
        else:
            # Mixed urban/rural
            tile = self._generate_mixed_tile(tile_size)
        
        return tile
    
    def _generate_urban_tile(self, size: int) -> np.ndarray:
        """Generate a synthetic urban satellite tile."""
        tile = np.zeros((size, size, 3), dtype=np.float32)
        
        # Urban base (gray-beige)
        tile[:, :, 0] = 0.35 + np.random.random((size, size)) * 0.1
        tile[:, :, 1] = 0.32 + np.random.random((size, size)) * 0.08
        tile[:, :, 2] = 0.28 + np.random.random((size, size)) * 0.08
        
        # Add road grid
        road_spacing = size // 8
        road_width = max(2, size // 50)
        
        for i in range(0, size, road_spacing):
            # Horizontal roads
            if i + road_width < size:
                tile[i:i+road_width, :, :] = 0.25 + np.random.random() * 0.05
            # Vertical roads
            if i + road_width < size:
                tile[:, i:i+road_width, :] = 0.25 + np.random.random() * 0.05
        
        # Add buildings (bright rectangles)
        num_buildings = (size * size) // 400
        for _ in range(num_buildings):
            x = np.random.randint(0, size - 15)
            y = np.random.randint(0, size - 15)
            w = np.random.randint(5, 15)
            h = np.random.randint(5, 15)
            
            # Building color (various shades)
            brightness = 0.4 + np.random.random() * 0.35
            color = np.array([brightness, brightness * 0.95, brightness * 0.9])
            tile[y:y+h, x:x+w, :] = color
        
        # Add some vegetation patches
        num_parks = size // 50
        for _ in range(num_parks):
            x = np.random.randint(0, size - 20)
            y = np.random.randint(0, size - 20)
            w = np.random.randint(10, 25)
            h = np.random.randint(10, 25)
            
            # Green color
            tile[y:y+h, x:x+w, 0] = 0.15 + np.random.random() * 0.1
            tile[y:y+h, x:x+w, 1] = 0.25 + np.random.random() * 0.15
            tile[y:y+h, x:x+w, 2] = 0.1 + np.random.random() * 0.08
        
        return np.clip(tile, 0, 1)
    
    def _generate_mixed_tile(self, size: int) -> np.ndarray:
        """Generate a synthetic mixed urban/rural tile."""
        tile = np.zeros((size, size, 3), dtype=np.float32)
        
        # Agricultural/vegetation base
        tile[:, :, 0] = 0.2 + np.random.random((size, size)) * 0.1
        tile[:, :, 1] = 0.3 + np.random.random((size, size)) * 0.15
        tile[:, :, 2] = 0.15 + np.random.random((size, size)) * 0.08
        
        # Add field patterns
        field_size = size // 4
        for i in range(0, size, field_size):
            for j in range(0, size, field_size):
                # Random field color variation
                base_green = 0.25 + np.random.random() * 0.2
                tile[i:i+field_size, j:j+field_size, 0] = 0.15 + np.random.random() * 0.1
                tile[i:i+field_size, j:j+field_size, 1] = base_green
                tile[i:i+field_size, j:j+field_size, 2] = 0.1 + np.random.random() * 0.08
        
        # Add a river or water body
        if np.random.random() > 0.5:
            river_y = size // 2 + np.random.randint(-size//4, size//4)
            river_width = np.random.randint(5, 15)
            tile[river_y:river_y+river_width, :, 0] = 0.1
            tile[river_y:river_y+river_width, :, 1] = 0.15
            tile[river_y:river_y+river_width, :, 2] = 0.25
        
        # Add some roads
        for _ in range(np.random.randint(2, 4)):
            if np.random.random() > 0.5:
                y = np.random.randint(0, size)
                width = np.random.randint(2, 4)
                tile[y:y+width, :, :] = 0.35
            else:
                x = np.random.randint(0, size)
                width = np.random.randint(2, 4)
                tile[:, x:x+width, :] = 0.35
        
        # Add small settlement
        settlement_x = np.random.randint(size//4, 3*size//4)
        settlement_y = np.random.randint(size//4, 3*size//4)
        settlement_size = size // 5
        
        for _ in range(np.random.randint(5, 15)):
            x = settlement_x + np.random.randint(-settlement_size//2, settlement_size//2)
            y = settlement_y + np.random.randint(-settlement_size//2, settlement_size//2)
            w = np.random.randint(3, 8)
            h = np.random.randint(3, 8)
            
            x = max(0, min(x, size - w))
            y = max(0, min(y, size - h))
            
            brightness = 0.45 + np.random.random() * 0.25
            tile[y:y+h, x:x+w, :] = brightness
        
        return np.clip(tile, 0, 1)
    
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
            output_dir: Output directory (default: DEMO_DIR)
        
        Returns:
            Path to saved file
        """
        output_dir = output_dir or DEMO_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to 8-bit
        img_8bit = (tile * 255).astype(np.uint8)
        
        # Save as PNG
        output_path = output_dir / f"{name}.png"
        Image.fromarray(img_8bit).save(output_path)
        
        print(f"Saved tile to {output_path}")
        return output_path
    
    def fetch_demo_tiles(
        self,
        locations: Optional[List[str]] = None,
        tile_size: int = TILE_SIZE,
    ) -> Dict[str, np.ndarray]:
        """
        Fetch demo tiles for multiple locations.
        
        Args:
            locations: List of location keys (default: all DEMO_LOCATIONS)
            tile_size: Size of tiles to fetch
        
        Returns:
            Dictionary mapping location names to image arrays
        """
        locations = locations or list(DEMO_LOCATIONS.keys())
        
        tiles = {}
        for loc in locations:
            print(f"Fetching tile for {loc}...")
            tile = self.fetch_tile(loc, tile_size)
            if tile is not None:
                tiles[loc] = tile
                self.save_tile(tile, f"lr_{loc}", DEMO_DIR)
        
        return tiles
