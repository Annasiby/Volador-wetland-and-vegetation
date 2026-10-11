import numpy as np
from rasterio import band
from scipy.ndimage import gaussian_filter
from typing import Dict

class VegetationIndexProcessor:
    """
    Computes standard vegetation indices (NDVI, GNDVI, NDRE) under two configurable
    modes: 'training' (with validity masking and Gaussian smoothing) and 'inference'
    (standard pixel-wise computation).
    """
    def __init__(
        self,
        mode: str = "training",
        epsilon: float = 1e-6,
        min_denominator: float = 0.05,
        min_band_value: float = 0.01,
        gaussian_sigma: float = 1.0
    ):
        self.mode = mode.lower()
        self.epsilon = epsilon
        self.min_denominator = min_denominator
        self.min_band_value = min_band_value
        self.gaussian_sigma = gaussian_sigma

    def _compute_index_inference(
        self,
        nir: np.ndarray,
        band: np.ndarray
    ) -> np.ndarray:
        """
        Computes a spectral index using the exact validity rules used
        by the validated standalone model preprocessing.
        """

        denominator = nir + band

        result = np.zeros_like(nir, dtype=np.float32)

        valid = (
            (denominator > self.min_denominator) &
            (nir > self.min_band_value) &
            (band > self.min_band_value)
        )

        result[valid] = (
            (nir[valid] - band[valid]) /
            (denominator[valid] + self.epsilon)
        )

        result[~np.isfinite(result)] = 0.0

        return np.clip(
            result,
            -1.0,
            1.0
        ).astype(np.float32)

    def _compute_index_training(self, nir: np.ndarray, band: np.ndarray) -> np.ndarray:
        denominator = nir + band
        result = np.zeros_like(nir, dtype=np.float32)
        
        valid = (
            (denominator > self.min_denominator) &
            (nir > self.min_band_value) &
            (band > self.min_band_value)
        )
        
        result[valid] = (nir[valid] - band[valid]) / (denominator[valid] + self.epsilon)
        result[~np.isfinite(result)] = 0.0
        result = np.clip(result, -1.0, 1.0)
        
        if self.gaussian_sigma > 0:
            result = gaussian_filter(result, sigma=self.gaussian_sigma)
            result = np.clip(result, -1.0, 1.0)
            
        return result

    def compute_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Computes NDVI, GNDVI, and NDRE indices from preprocessed bands.
        """
        green = bands["green"]
        red = bands["red"]
        rededge = bands["rededge"]
        nir = bands["nir"]

        indices = {}

        if self.mode == "training":
            indices["ndvi"] = self._compute_index_training(nir, red)
            indices["gndvi"] = self._compute_index_training(nir, green)
            indices["ndre"] = self._compute_index_training(nir, rededge)
        else:
            indices["ndvi"] = self._compute_index_inference(nir, red)
            indices["gndvi"] = self._compute_index_inference(nir, green)
            indices["ndre"] = self._compute_index_inference(nir, rededge)

        return indices
