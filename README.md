# ID Crop Yield Sentinel

[![Python CI](https://github.com/KevinNaserwan/id-crop-yield-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinNaserwan/id-crop-yield-sentinel/actions)

Multispectral remote sensing and deep learning pipeline for **Indonesian rice crop yield estimation and early crop failure (*puso*) warning**. Designed as an applied research repository for Master's thesis work and national food security intelligence.

## Motivation & National Urgency
Indonesia faces extreme seasonal climate shifts (El Niño / La Niña events), erratic rainfall, and cloud cover challenges in tropical agricultural monitoring. This system leverages:
1. **Cloud-Penetrating Synthetic Aperture Radar (SAR)**: Sentinel-1 C-band backscatter (VV/VH polarizations).
2. **Optical Multi-Spectral Imagery**: Sentinel-2 Level-2A surface reflectance (NDVI, NDRE, NDWI).
3. **Spatio-Temporal Deep Learning**: Bi-directional LSTM with self-attention mechanism to capture phenological trajectory across the ~120-day paddy growth cycle.

## Architecture
```
├── data/
│   ├── raw/             # GEE geojson boundaries & raw spectral timeseries
│   └── processed/       # Interpolated, cloud-filtered features
├── src/
│   ├── gee_extractor.py # Earth Engine extraction & cloud masking (QA60)
│   ├── preprocessor.py  # Savitzky-Golay filtering & temporal alignment
│   └── models/
│       └── yield_net.py # Bi-LSTM + Attention regression & classification
├── notebooks/           # Case studies on Central/East Java & South Sumatra paddies
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt
python -c "import torch; from src.models.yield_net import SpatioTemporalYieldNet; m = SpatioTemporalYieldNet(); print(m)"
```

## Research Roadmap
- [x] Multi-spectral vegetation index extraction pipeline (NDVI, NDRE, NDWI)
- [x] Spatio-temporal Deep Learning network definition (Bi-LSTM + Attention)
- [ ] Integration with BPS Kerangka Sampel Area (KSA) ground truth statistics
- [ ] Evaluation across cloud-heavy monsoon seasons using Sentinel-1 SAR fusion
- [ ] Publication manuscript drafting for remote sensing conference / Q2 journal

## License
MIT
