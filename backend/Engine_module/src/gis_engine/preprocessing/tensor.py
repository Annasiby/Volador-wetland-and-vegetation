import numpy as np
from typing import Dict, List
from .normalization import Normalizer

class TensorBuilder:
    """
    Compiles normalized indices and bands into the structured 5-channel model input tensor,
    verifying spatial and numeric contracts before execution.
    """
    def __init__(self, channel_order: List[str] = None):
        if channel_order is None:
            self.channel_order = ["ndvi", "gndvi", "ndre", "nir", "rededge"]
        else:
            self.channel_order = channel_order

    def build_tensor(self, indices: Dict[str, np.ndarray], preprocessed_bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combines and normalizes input data into the expected (1, 5, H, W) float32 tensor.
        """
        # Pool all available channels
        data_pool = {**indices, **preprocessed_bands}
        
        normalized_channels = []
        for channel_name in self.channel_order:
            if channel_name not in data_pool:
                raise KeyError(
                    f"Required channel '{channel_name}' was not found in computed "
                    f"indices or preprocessed bands."
                )
            
            raw_channel = data_pool[channel_name]
            norm_channel = Normalizer.normalize_channel(raw_channel)
            normalized_channels.append(norm_channel)

        # Stack into (5, H, W)
        tensor = np.stack(normalized_channels, axis=0)
        
        # Add batch dimension to get (1, 5, H, W)
        tensor_batch = np.expand_dims(tensor, axis=0).astype(np.float32)

        # Validate final tensor properties
        self._validate_tensor(tensor_batch)

        return tensor_batch

    def _validate_tensor(self, tensor: np.ndarray):
        if tensor.dtype != np.float32:
            raise TypeError(f"Model input tensor must be float32, got {tensor.dtype}")
            
        if tensor.ndim != 4:
            raise ValueError(f"Model input tensor must be 4D (1, C, H, W), got shape {tensor.shape}")
            
        if tensor.shape[0] != 1:
            raise ValueError(f"Batch dimension of model input tensor must be 1, got {tensor.shape[0]}")
            
        if tensor.shape[1] != len(self.channel_order):
            raise ValueError(
                f"Model input channel count must be {len(self.channel_order)}, "
                f"got {tensor.shape[1]}"
            )
            
        if np.any(np.isnan(tensor)):
            raise ValueError("Model input tensor contains NaN values.")
            
        if np.any(np.isinf(tensor)):
            raise ValueError("Model input tensor contains infinite values.")
            
        if tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
            raise ValueError(f"Invalid spatial dimensions: {tensor.shape[2]}x{tensor.shape[3]}")
