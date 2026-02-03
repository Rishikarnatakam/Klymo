"""
Configuration constants for Sentinel-2 Super-Resolution Pipeline.
"""

import os
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "model_zoo"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Create directories if they don't exist
for dir_path in [OUTPUTS_DIR, MODEL_DIR, DATASETS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Output subdirectories
METRICS_DIR = OUTPUTS_DIR / "metrics"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

for dir_path in [METRICS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Sentinel-2 Configuration
# ============================================================================

# Sentinel-2 band mappings for RGB
SENTINEL2_BANDS = {
    "red": "B4",
    "green": "B3",
    "blue": "B2",
    "nir": "B8",  # Near infrared for NDVI
}

# RGB band order for image composition
RGB_BANDS = ["B4", "B3", "B2"]

# Sentinel-2 reflectance range (16-bit)
REFLECTANCE_MIN = 0
REFLECTANCE_MAX = 3000  # Typical surface reflectance upper bound

# Resolution
SENTINEL2_RESOLUTION = 10  # meters per pixel
TARGET_RESOLUTION = 2.5    # meters per pixel after 4× SR

# ============================================================================
# Super-Resolution Configuration
# ============================================================================

SR_SCALE = 4  # Fixed 4× super-resolution

# Tile configuration
TILE_SIZE = 256  # Input tile size in pixels
TILE_OVERLAP = 32  # Overlap for seamless stitching

# Output tile size after SR
OUTPUT_TILE_SIZE = TILE_SIZE * SR_SCALE  # 1024 pixels

# ============================================================================
# SwinIR Model Configuration
# ============================================================================

SWINIR_CONFIG = {
    # Model architecture parameters
    "img_size": 64,
    "patch_size": 1,
    "in_chans": 3,
    "embed_dim": 180,
    "depths": [6, 6, 6, 6, 6, 6],
    "num_heads": [6, 6, 6, 6, 6, 6],
    "window_size": 8,
    "mlp_ratio": 2.0,
    "upscale": SR_SCALE,
    "img_range": 1.0,
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
}

# Pretrained model URL (classical SR, 4×)
SWINIR_MODEL_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
SWINIR_MODEL_NAME = "swinir_classical_x4.pth"

# ============================================================================
# Predefined Locations for GEE Fetching
# ============================================================================

# Real geographic coordinates for inference testing
LOCATIONS = {
    "delhi": {
        "name": "Delhi, India",
        "lat": 28.6139,
        "lon": 77.2090,
        "description": "Dense urban area with mixed land use",
    },
    "kanpur": {
        "name": "Kanpur, India",
        "lat": 26.4499,
        "lon": 80.3319,
        "description": "Industrial city along Ganges River",
    },
}

DEFAULT_LOCATION = "delhi"

# ============================================================================
# Google Earth Engine Configuration
# ============================================================================

GEE_CONFIG = {
    "collection": "COPERNICUS/S2_SR_HARMONIZED",  # Sentinel-2 L2A
    "cloud_threshold": 10,  # Maximum cloud coverage percentage
    "date_range_days": 90,  # Look back N days for clear imagery
}

# ============================================================================
# Preprocessing Configuration
# ============================================================================

PREPROCESSING = {
    "clip_min": REFLECTANCE_MIN,
    "clip_max": REFLECTANCE_MAX,
    "normalize_range": (0.0, 1.0),
}

# ============================================================================
# Metrics Configuration
# ============================================================================

METRICS_CONFIG = {
    "psnr_max_val": 1.0,  # For normalized images
    "ssim_window_size": 11,
    "ssim_data_range": 1.0,
}

# ============================================================================
# Hallucination Guardrails
# ============================================================================

GUARDRAILS = {
    # Edge consistency threshold (Canny edge difference)
    "edge_consistency_threshold": 0.15,
    
    # NDVI stability tolerance
    "ndvi_tolerance": 0.1,
    
    # Color histogram difference threshold
    "color_histogram_threshold": 0.2,
}

# ============================================================================
# Validation Configuration
# ============================================================================

WORLDSTRAT_CONFIG = {
    "kaggle_dataset": "julienco/worldstrat",
    "validation_samples": 100,  # Maximum samples for validation
    "patch_size": 128,  # Validation patch size
}
