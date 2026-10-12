<div align="center">

# 🌾 ID Crop Yield Sentinel
### Spatio-Temporal Deep Learning & Multi-Spectral Remote Sensing for Rice Yield Estimation and Crop Failure (*Puso*) Early Warning in Indonesia
**Applied Remote Sensing & Deep Learning · Master's Thesis Research Track**

[![Python CI](https://github.com/KevinNaserwan/id-crop-yield-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinNaserwan/id-crop-yield-sentinel/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Google Earth Engine](https://img.shields.io/badge/GEE-API-34A853.svg)](https://earthengine.google.com/)

</div>

---

## 📌 Research Motivation & National Urgency

In Indonesia, staple food security is deeply vulnerable to severe tropical weather volatility—specifically intense **El Niño droughts** and **La Niña floodings**. Furthermore, tropical agricultural monitoring in Southeast Asia suffers from persistent **heavy cloud cover (>70%)** during monsoon planting seasons, rendering conventional optical satellites ineffective.

### Key Contributions:
1. **Multi-Modal Satellite Fusion**: Integrates **Sentinel-1 C-Band Synthetic Aperture Radar (SAR)** (cloud-penetrating microwave backscatter VV & VH) with cloud-filtered **Sentinel-2 Level-2A multi-spectral optical data**.
2. **Phenological Trajectory Modeling**: Employs a **Bi-directional LSTM with Self-Attention** (`SpatioTemporalYieldNet`) to model spectral reflectance across the ~120-day paddy growth cycle.
3. **Dual-Task Prediction**: Simultaneously forecasts harvest yield ($\text{Tons/Hectare}$) and evaluates the probability of catastrophic harvest failure (*puso*).

---

## 🏛️ End-to-End Processing Pipeline

```
+-------------------------------------------------------------------------+
|                  GEOSPATIAL SENTINEL DATA PIPELINE                     |
+-------------------------------------------------------------------------+

 [ Indonesian Paddy Parcel Boundary Polygons (GeoJSON / Shapefile) ]
                                    |
                                    v
                 [ Google Earth Engine (GEE) Ingestion API ]
                                    |
                 +------------------+------------------+
                 |                                     |
                 v (Optical L2A)                       v (SAR C-Band)
 +-------------------------------+     +-------------------------------+
 | Sentinel-2 Surface Reflectance|     | Sentinel-1 GRD Backscatter    |
 +---------------+---------------+     +---------------+---------------+
                 |                                     |
                 v                                     v
 | QA60 Bitmask Cloud Filter     |     | Radiometric Calibration       |
 | Savitzky-Golay Interpolation  |     | Lee Speckle Noise Filter      |
 | Indices: NDVI, NDRE, NDWI, EVI|     | Cross-Ratio VH / VV           |
 +---------------+---------------+     +---------------+---------------+
                 |                                     |
                 +------------------+------------------+
                                    |
                                    v
 +---------------------------------------------------------------------+
 |                 TEMPORAL FEATURE FUSION MATRIX                      |
 +----------------------------------+----------------------------------+
                                    |
                                    v
 +---------------------------------------------------------------------+
 |              SpatioTemporalYieldNet (Bi-LSTM + Attention)           |
 +----------------------------------+----------------------------------+
                                    |
                 +------------------+------------------+
                 |                                     |
                 v                                     v
 [ Grain Yield Output: Metric Tons / Ha ]   [ Risk Index: Puso Alert ]
```

---

## 📐 Spectral Index Formulations

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{B8 - B4}{B8 + B4}$$

$$\text{NDRE} = \frac{\text{NIR} - \text{RedEdge}}{\text{NIR} + \text{RedEdge}} = \frac{B8 - B5}{B8 + B5} \quad \text{(Chlorophyll sensitivity)}$$

$$\text{EVI} = 2.5 \times \frac{\text{NIR} - \text{Red}}{\text{NIR} + 6.0 \cdot \text{Red} - 7.5 \cdot \text{Blue} + 1.0} \quad \text{(High-biomass saturation resilience)}$$

$$\text{SAR Cross-Ratio (dB)} = \sigma^0_{\text{VH}} - \sigma^0_{\text{VV}} \quad \text{(Surface water & paddy tillering proxy)}$$

---

## 📂 Project Structure

```
id-crop-yield-sentinel/
├── data/
│   ├── raw/                 # GEE geojson boundary polygons & raw spectral timeseries
│   └── processed/           # Cloud-filtered, temporally-aligned feature tensors
├── docs/                    # Thesis methodology drafts & scientific manuscripts
├── notebooks/               # Case studies on Central/East Java & South Sumatra paddies
│   └── 01_spectral_eda.ipynb
├── src/
│   ├── gee_extractor.py     # Earth Engine ingestion & QA60 cloud bitmasking
│   ├── preprocessor.py      # Savitzky-Golay smoothing & temporal interpolation
│   └── models/
│       ├── yield_net.py     # Bi-LSTM with Self-Attention neural architecture
│       └── attention_pooling.py # Dynamic spatial attention mechanisms
├── tests/                   # Automated pytest suites
└── requirements.txt         # Dependencies
```

---

## 🚀 Getting Started

### Prerequisites & Installation
```bash
# Clone the repository
git clone git@github.com:KevinNaserwan/id-crop-yield-sentinel.git
cd id-crop-yield-sentinel

# Install dependencies
pip install -r requirements.txt
```

### Verify Deep Learning Model Architecture
```bash
python -c "import torch; from src.models.yield_net import SpatioTemporalYieldNet; m = SpatioTemporalYieldNet(); x = torch.randn(4, 16, 5); y, f = m(x); print('Model initialized. Output shape:', y.shape)"
```

---

## 🔬 Scientific Roadmap & Publication Targets
- [x] Multi-spectral vegetation index extraction pipeline
- [x] Spatio-temporal Deep Learning network definition (Bi-LSTM + Attention)
- [ ] Integration with BPS Kerangka Sampel Area (KSA) ground truth statistics
- [ ] Full evaluation across monsoon planting seasons using Sentinel-1 SAR fusion
- [ ] Manuscript preparation for submission to *IEEE Geoscience and Remote Sensing Letters (GRSL)* or *Remote Sensing of Environment*.

---

## 👤 Author & Research Citation
- **Principal Investigator**: Kevin Putrayudha Naserwan ([@KevinNaserwan](https://github.com/KevinNaserwan))
- **Affiliation**: Universitas Sriwijaya (Alumnus) · Master's Thesis Research Track
