# Klymo: Sentinel-2 Super-Resolution 🛰️✨

> **Bridging the resolution gap between public Sentinel-2 (10m) and commercial imagery (0.5m) using Deep Learning.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishikarnatakam/Klymo/blob/main/notebooks/colab_inference.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Klymo is a deep learning pipeline designed to upscale **Sentinel-2 satellite imagery (10m/pixel)** by **4×** to achieve near-commercial quality (2.5m/pixel). 

Unlike traditional approaches that require massive local dataset downloads, Klymo operates on a **streaming-first architecture**, fetching live data from **Google Earth Engine (GEE)** and processing it on-the-fly using a **SwinIR Transformer**.

### 🎯 The Challenge & Solution
| Feature | Traditional Approach | Klymo Approach 🚀 |
|---------|----------------------|-------------------|
| **Data Source** | Download 50GB+ Zip files | **Stream live tiles from GEE API** |
| **Model** | Basic CNN (SRResNet) | **Swin Transformer (SwinIR)** |
| **Accuracy** | Prone to "hallucination" | **Geospatial Guardrails (NDVI/Spectral)** |
| **Compute** | Heavy local GPU setup | **One-Click Google Colab** |

---

## 🚀 Key Features (Judge's Checklist)
- ✅ **4× Upscaling**: Transformers blurry 10m Sentinel-2 into sharp 2.5m imagery.
- ✅ **Zero-Download Pipeline**: No massive datasets required. Everything is streamed.
- ✅ **Geospatial Guardrails**: Prevents the model from inventing buildings or destroying forests (Checks NDVI & Spectral consistency).
- ✅ **Interactive UI**: Built-in **Streamlit** app for real-time "Before vs After" comparison.
- ✅ **Swin Transformer**: State-of-the-art restoration model (vs. older GANs).

---

## �️ Tech Stack
- **Ingestion**: Google Earth Engine (GEE) Python API
- **Model**: PyTorch, SwinIR (Swin Transformer for Image Restoration)
- **Processing**: NumPy, Rasterio, OpenCV
- **Visualization**: Streamlit, Matplotlib
- **Infrastructure**: Google Colab (T4 GPU optimized)

---

## � How to Run (Fastest Way)

1. **Click the Colab Button** above 👆
2. **Connect to GPU**: `Runtime` -> `Change runtime type` -> `T4 GPU`.
3. **Run All Cells**:
    - The notebook will clone this repo.
    - Install dependencies.
    - Authenticate GEE (Google Login).
    - Stream a live patch from **Delhi, India** (or any location).
    - Perform Super-Resolution.
    - Launch the **Streamlit Interactive Demo**.

### Local Installation (Optional)
If you have a GPU and want to run locally:
```bash
git clone https://github.com/Rishikarnatakam/Klymo.git
cd Klymo
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## � Methodology

### 1. Ingestion (GEE Streaming)
We bypass downloading `L2A` scenes. Instead, we use `ee.ImageCollection('COPERNICUS/S2_SR')` to request a geospatial crop (2.56km x 2.56km) at a specific lat/lon.

### 2. The Model (SwinIR)
We utilize **SwinIR**, a Swin Transformer tailored for image restoration.
- **Why?** It captures long-range dependencies better than CNNs (EDSR/RCAN), crucial for repetitive urban patterns (roads, buildings).
- **Architecture**: Deep feature extraction -> Swin Transformer Layers (RSTB) -> PixelShuffle Upsampler.

### 3. Hallucination Guardrails
To prevent "fake" details:
- **NDVI Check**: We ensure vegetation density hasn't drastically changed.
- **Spectral Consistency**: We check that the color histogram distribution remains valid for the biome.

---

## 🖼️ Results
**Location**: Delhi, India (Urban/Dense)

| Input (Sentinel-2 10m) | Output (Klymo 2.5m) |
|------------------------|---------------------|
| *Blurry, blocky roads* | *Sharp edges, clear road networks* |
| *(See Colab for live demo)* | *(See Colab for live demo)* |

---

## 🏆 Credits
Built for the **72-Hour ML Track Challenge**.
- **Model**: Based on official [SwinIR](https://github.com/JingyunLiang/SwinIR) implementation.
- **Data**: Sentinel-2 via Google Earth Engine.
