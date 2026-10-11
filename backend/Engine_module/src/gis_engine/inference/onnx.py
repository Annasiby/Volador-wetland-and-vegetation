import os
import logging
import time
import numpy as np
import onnxruntime as ort
from typing import Tuple

logger = logging.getLogger(__name__)

class ONNXModelWrapper:
    """
    ONNX Runtime model wrapper that loads the production wetland classification model,
    manages CPU/CUDA execution providers, and runs inference.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.execution_provider = None
        self.loading_time_ms = 0.0
        self.load_model()

    def load_model(self):
        """
        Loads the inference session and detects active provider.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")
            
        available_providers = ort.get_available_providers()
        
        # CUDA preference fallback
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            logger.info(
                "[onnx] GPU provider (CUDA) unavailable. Using CPUExecutionProvider."
            )
            providers = ["CPUExecutionProvider"]

        start_time = time.perf_counter()
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as exc:
            # Hard fallback to standard CPU provider
            logger.warning(
                "[onnx] Initial provider list %s failed (%s). "
                "Falling back to CPUExecutionProvider.",
                providers, exc
            )
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            
        self.loading_time_ms = (time.perf_counter() - start_time) * 1000.0
        active_providers = self.session.get_providers()
        self.execution_provider = active_providers[0] if active_providers else "CPUExecutionProvider"
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs model inference on the preprocessed 5-channel input tensor.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - raw logits: shape (1, 4, H, W)
                - class map: shape (H, W) containing integer IDs 0-3
        """
        if self.session is None:
            raise RuntimeError("Model session is not loaded.")
            
        if tensor.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (1, 5, H, W), got {tensor.shape}")
            
        if tensor.shape[1] != 5:
            raise ValueError(f"Input tensor must have 5 channels, got {tensor.shape[1]}")

        # Execute session forward pass
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        raw_logits = outputs[0]  # shape (1, 4, H, W)
        
        # Argmax along the classes dimension (axis 1)
        class_map = np.argmax(raw_logits, axis=1)  # shape (1, H, W)
        class_map = np.squeeze(class_map, axis=0)  # shape (H, W)
        
        return raw_logits.astype(np.float32), class_map.astype(np.int32)
