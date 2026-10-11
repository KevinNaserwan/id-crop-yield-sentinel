"""
Google Earth Engine (GEE) Extractor for Sentinel-2 Optical and Sentinel-1 SAR imagery
over agricultural paddy parcels in Indonesia.
"""
from typing import Dict, Any, List
import numpy as np

class SentinelIndexCalculator:
    """
    Computes key vegetation and moisture indices essential for rice crop phenology:
    - NDVI: Normalized Difference Vegetation Index (Canopy greenness)
    - NDRE: Normalized Difference Red Edge (Chlorophyll content)
    - NDWI: Normalized Difference Water Index (Paddy surface flooding status)
    """

    @staticmethod
    def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        denominator = nir + red
        return np.where(denominator != 0, (nir - red) / denominator, 0.0)

    @staticmethod
    def calculate_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        denominator = nir + red_edge
        return np.where(denominator != 0, (nir - red_edge) / denominator, 0.0)

    @staticmethod
    def calculate_ndwi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        denominator = nir + swir
        return np.where(denominator != 0, (nir - swir) / denominator, 0.0)

class CloudMaskFilter:
    """
    Applies QA60 bitmask filter for Sentinel-2 Level-2A surface reflectance.
    """
    @staticmethod
    def mask_s2_clouds(qa_band: int) -> bool:
        # Bit 10: Opaque clouds, Bit 11: Cirrus clouds
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        is_cloudy = ((qa_band & cloud_bit_mask) != 0) or ((qa_band & cirrus_bit_mask) != 0)
        return not is_cloudy
