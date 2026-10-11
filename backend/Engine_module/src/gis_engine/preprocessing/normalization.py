import numpy as np

class Normalizer:
    """
    Handles min-max normalization of individual channel bands to the [0, 1] range.
    """
    @staticmethod
    def normalize_channel(channel: np.ndarray) -> np.ndarray:
        """
        Applies per-channel min/max normalization, converting invalid values to 0.0.
        """
        channel = channel.astype(np.float32)
        channel = np.nan_to_num(
            channel,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        
        cmin = np.min(channel)
        cmax = np.max(channel)
        
        if (cmax - cmin) < 1e-8:
            return np.zeros_like(channel, dtype=np.float32)
            
        return ((channel - cmin) / (cmax - cmin)).astype(np.float32)
