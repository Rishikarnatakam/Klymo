# Sentinel-2 Super-Resolution Pipeline

A geospatially faithful 4× super-resolution pipeline for Sentinel-2 satellite imagery using SwinIR.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This project enhances Sentinel-2 imagery from 10m/pixel to 2.5m/pixel resolution using a pretrained SwinIR transformer model. **We prioritize geospatial correctness over visual sharpness** — a believable image beats an impressive fake.

### Sample Results

| Low Resolution (10m) | Bicubic 4× | SwinIR 4× (2.5m) |
|---------------------|------------|------------------|
| ![LR](outputs/demo/lr_sample.png) | ![Bicubic](outputs/demo/bicubic_sample.png) | ![SR](outputs/demo/sr_sample.png) |

## 🔬 Technical Decisions

### Why SwinIR?

1. **Transformer-based architecture** - Captures global context better than CNNs
2. **Proven performance** - State-of-the-art PSNR/SSIM on standard benchmarks
3. **No adversarial training** - Avoids hallucination-prone GAN dynamics
4. **Stable inference** - Deterministic outputs, no sampling artifacts
5. **Pretrained availability** - High-quality weights trained on natural images

### Why 4× Super-Resolution?

- **Balanced enhancement**: 10m → 2.5m is aggressive but achievable
- **Hallucination control**: Higher scales (8×) create more artifacts
- **Practical utility**: 2.5m resolution enables meaningful analysis improvements
- **Model availability**: SwinIR 4× pretrained models are well-tested

### Why Hallucination Avoidance?

Satellite imagery is used for:
- Urban planning
- Agricultural monitoring
- Disaster response
- Environmental analysis

**Hallucinated features (fake buildings, roads, vegetation) can lead to dangerous decisions.** Our pipeline enforces:

- No diffusion models (high hallucination risk)
- No aggressive GAN loss
- Conservative SR objectives
- Edge consistency validation
- NDVI stability checks

### Limitations

- **Not for feature detection**: Don't use SR output for counting buildings or detecting new roads
- **Texture synthesis**: Complex textures (dense urban, forest canopy) may show repetitive patterns
- **Cloud/shadow handling**: Works best on clear imagery; artifacts possible near cloud edges
- **Spectral fidelity**: Optimized for RGB; other bands may need separate treatment

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/sentinel2-sr.git
cd sentinel2-sr
pip install -r requirements.txt
```

### One-Click Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/sentinel2_sr_colab.ipynb)

### Run Inference

```python
from src.inference.pipeline import SuperResolutionPipeline

# Initialize pipeline
pipeline = SuperResolutionPipeline(scale=4, device='cuda')

# Run on Sentinel-2 image
sr_image = pipeline.run("path/to/sentinel2_tile.tif")
```

### Interactive Demo

```bash
streamlit run app/streamlit_app.py
```

## 📊 Metrics

Evaluated on WorldStrat validation set:

| Method | PSNR (dB) ↑ | SSIM ↑ |
|--------|-------------|--------|
| Bicubic 4× | 24.32 | 0.6821 |
| **SwinIR 4×** | **28.47** | **0.8156** |

## 🏗️ Pipeline Architecture

```
Sentinel-2 LR (10m/pixel)
    │
    ▼
┌─────────────────┐
│  Preprocessing  │  ◄── Clip [0, 3000], Normalize [0, 1]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Tiling      │  ◄── 256×256 patches with overlap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SwinIR 4×     │  ◄── Pretrained transformer SR
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │  ◄── Color consistency, edge checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tile Stitching │  ◄── Overlap blending
└────────┬────────┘
         │
         ▼
    Output (2.5m/pixel)
```

## 📁 Project Structure

```
sentinel2-sr/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py              # Configuration constants
│   ├── data/
│   │   ├── worldstrat_loader.py
│   │   ├── gee_fetcher.py
│   │   ├── preprocessing.py
│   │   └── tiling.py
│   ├── models/
│   │   ├── swinir.py
│   │   └── bicubic.py
│   ├── inference/
│   │   ├── pipeline.py
│   │   └── postprocess.py
│   └── metrics/
│       ├── psnr.py
│       ├── ssim.py
│       └── hallucination.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── sentinel2_sr_colab.ipynb
└── outputs/
    ├── metrics/
    ├── visualizations/
    └── demo/
```

## 🛰️ Data Sources

### WorldStrat (Validation)
- **Source**: [Kaggle WorldStrat Dataset](https://www.kaggle.com/datasets/julienco/worldstrat)
- **Purpose**: PSNR/SSIM computation
- **Usage**: Small paired LR/HR patches only

### Google Earth Engine (Inference)
- **Source**: Sentinel-2 L2A
- **Bands**: B4 (Red), B3 (Green), B2 (Blue)
- **Resolution**: 10m native → 2.5m super-resolved

## ⚠️ Hallucination Guardrails

The system implements multiple checks:

1. **No Diffusion Models**: Eliminates stochastic hallucinations
2. **Edge Consistency**: SR edges must align with original LR edges
3. **NDVI Stability**: Vegetation indices preserved within tolerance
4. **Color Distribution**: Histogram matching prevents color drift

**Failure Conditions** (treated as errors):
- Buildings appearing in forested areas
- Roads appearing in water bodies
- New structures not present in original

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 References

- [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)
- [WorldStrat Dataset](https://github.com/worldstrat/worldstrat)
- [Google Earth Engine](https://earthengine.google.com/)
