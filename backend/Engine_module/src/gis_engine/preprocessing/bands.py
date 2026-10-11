import numpy as np
from typing import Dict


class BandProcessor:
    """
    Standardizes raw multispectral bands using the same 2%-98% percentile
    preprocessing used by the validated model inference pipeline.

    Processing:
        1. Convert to float32
        2. Replace NaN/Inf with 0
        3. Compute 2nd and 98th percentiles
        4. Scale values into [0, 1]
        5. Clip outliers to [0, 1]
    """

    def __init__(
        self,
        scaling_factor: float = 1.0,
        clip_range: tuple = (0.0, 1.0)
    ):
        # Kept for configuration/API compatibility.
        # Percentile standardization does not use a fixed scaling factor.
        self.scaling_factor = scaling_factor
        self.clip_range = clip_range

    @staticmethod
    def _percentile_normalize(array: np.ndarray) -> np.ndarray:
        """
        Apply the validated 2%-98% percentile standardization.

        This matches the standalone model preprocessing:
            p2 = percentile(band, 2)
            p98 = percentile(band, 98)
            normalized = (band - p2) / (p98 - p2 + 1e-6)
            clip to [0, 1]
        """
        arr = array.astype(np.float32)

        # Remove invalid values before percentile calculation.
        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        p2 = np.percentile(arr, 2)
        p98 = np.percentile(arr, 98)

        denominator = p98 - p2 + 1e-6

        normalized = (arr - p2) / denominator

        normalized = np.clip(
            normalized,
            0.0,
            1.0
        )

        return normalized.astype(np.float32)

    def process_bands(
        self,
        bands: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Standardize each multispectral band independently.

        Returns:
            Dict[str, np.ndarray]:
                Each band as float32 values in [0, 1].
        """
        processed = {}

        for name, array in bands.items():
            processed[name] = self._percentile_normalize(array)

        return processed