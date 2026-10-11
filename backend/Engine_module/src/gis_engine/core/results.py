from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
from ..alignment.spatial import ReferenceGrid


@dataclass
class TwoDateInferenceResult:
    """
    Structured container for the output of a two-date model inference run.
    """
    t1_health_map: np.ndarray  # Shape: (H, W), dtype: uint8, values: 0-3, 255 for NoData
    t2_health_map: np.ndarray  # Shape: (H, W), dtype: uint8, values: 0-3, 255 for NoData
    t1_logits: np.ndarray      # Shape: (1, 4, H, W), dtype: float32
    t2_logits: np.ndarray      # Shape: (1, 4, H, W), dtype: float32
    t1_probabilities: np.ndarray  # Shape: (4, H, W), dtype: float32
    t2_probabilities: np.ndarray  # Shape: (4, H, W), dtype: float32
    t1_confidence: np.ndarray     # Shape: (H, W), dtype: float32
    t2_confidence: np.ndarray     # Shape: (H, W), dtype: float32
    t1_valid_mask: np.ndarray     # Shape: (H, W), dtype: bool
    t2_valid_mask: np.ndarray     # Shape: (H, W), dtype: bool
    overlap_mask: np.ndarray      # Shape: (H, W), dtype: bool
    reference_grid: ReferenceGrid
    t1_metadata: Dict[str, Any]
    t2_metadata: Dict[str, Any]
    class_statistics_t1: Dict[str, Any]
    class_statistics_t2: Dict[str, Any]
    performance: Dict[str, Any]


@dataclass
class AnalysisResult:
    """
    Structured production result returned by the public API and pipeline.

    This object contains references/paths to all outputs rather than
    full raster arrays, to keep the public API lightweight and serialisable.

    Fields:
        run_id:           Unique identifier for this analysis run.
        status:           "SUCCESS", "WARNING", or "FAILED".
        output_directory: Absolute path to the run output directory.
        outputs:          Mapping of output key to relative path within
                          output_directory (e.g., "vegetation/t1_health.tif").
        metrics:          All computed scalar and area metrics.
        validation:       QA status flags and summary.
        warnings:         Non-blocking warning messages accumulated during the run.
        errors:           Error messages (populated if status != "SUCCESS").
        metadata:         Provenance and execution metadata (run_id, timestamps,
                          model hash, input fingerprints, CRS, etc.).
    """
    run_id: str
    status: str                           # "SUCCESS" | "WARNING" | "FAILED"
    output_directory: str
    outputs: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
