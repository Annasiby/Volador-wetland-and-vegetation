from pathlib import Path
from typing import Dict, Any, Tuple
import time
import numpy as np
from ..ingestion.loader import RasterLoader
from ..ingestion.validator import InputValidator
from ..ingestion.metadata import RasterMetadata
from ..preprocessing.bands import BandProcessor
from ..preprocessing.indices import VegetationIndexProcessor
from ..preprocessing.tensor import TensorBuilder
from .onnx import ONNXModelWrapper

class SingleDateInferenceEngine:
    """
    Coordinator class that manages execution of the Phase 1 GIS vegetation-health pipeline
    for a single date scene directory.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Ingestion
        self.loader = RasterLoader(config["ingestion"]["band_mapping"])
        self.validator = InputValidator(require_crs=config["preprocessing"]["require_crs"])
        
        # Preprocessing
        self.band_processor = BandProcessor(
            scaling_factor=config["preprocessing"]["band_scaling_factor"]
        )
        self.index_processor = VegetationIndexProcessor(
            mode=config["preprocessing"]["mode"],
            epsilon=config["preprocessing"]["epsilon"],
            min_denominator=config["preprocessing"]["min_denominator"],
            min_band_value=config["preprocessing"]["min_band_value"],
            gaussian_sigma=config["preprocessing"]["gaussian_sigma"]
        )
        self.tensor_builder = TensorBuilder(config["model"]["channel_order"])
        
        # ONNX model setup
        self.model = ONNXModelWrapper(config["model"]["path"])

    def run_pipeline(self, scene_dir: Path) -> Tuple[np.ndarray, np.ndarray, RasterMetadata, Dict[str, Any]]:
        """
        Runs the full Phase 1 processing pipeline:
        Ingestion Loader & Validator -> Band Processor -> Index Processor ->
        Tensor Builder -> ONNX model prediction.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, RasterMetadata, Dict[str, Any]]:
                - raw logits: shape (1, 4, H, W)
                - class map: shape (H, W)
                - metadata: RasterMetadata details
                - run_stats: timing and provider metrics dictionary
        """
        scene_dir = Path(scene_dir)
        
        # 1. Ingestion loader
        bands_data, metadata = self.loader.load_bands(scene_dir)
        
        # 2. Validation
        bands_paths = {}
        for name in ["green", "red", "rededge", "nir"]:
            bands_paths[name] = self.loader._find_band_file(scene_dir, name)
            
        self.validator.validate_files(bands_paths)
        self.validator.validate_arrays(bands_data)
        
        # 3. Preprocessing
        preprocessed_bands = self.band_processor.process_bands(bands_data)
        indices = self.index_processor.compute_indices(preprocessed_bands)
        
        # 4. Build Tensor
        start_tensor = time.perf_counter()
        tensor = self.tensor_builder.build_tensor(indices, preprocessed_bands)
        tensor_creation_time = (time.perf_counter() - start_tensor) * 1000.0
        
        # 5. Model execution
        start_inference = time.perf_counter()
        logits, class_map = self.model.predict(tensor)
        inference_time_ms = (time.perf_counter() - start_inference) * 1000.0
        
        # 6. Gather metrics
        run_stats = {
            "inference_time_ms": inference_time_ms,
            "tensor_creation_time_ms": tensor_creation_time,
            "model_loading_time_ms": self.model.loading_time_ms,
            "execution_provider": self.model.execution_provider
        }
        
        return logits, class_map, metadata, run_stats
