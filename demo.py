"""
Quick demo script to test the super-resolution pipeline.
Run with: python demo.py
"""

import numpy as np
from src.inference.pipeline import SuperResolutionPipeline

print("Initializing pipeline on CPU...")
pipeline = SuperResolutionPipeline(device='cpu')

print("Creating test image (128x128)...")
test_image = np.random.rand(128, 128, 3).astype('float32')

print("Running super-resolution (this may take 30-60 seconds on CPU)...")
results = pipeline.run(test_image)

print(f"\nResults:")
print(f"  Input shape:  {results['lr'].shape}")
print(f"  Bicubic shape: {results['bicubic'].shape}")
print(f"  SR Output shape: {results['sr'].shape}")
print("\nDone!")
