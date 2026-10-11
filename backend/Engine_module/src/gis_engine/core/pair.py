from pathlib import Path
from typing import Dict, Any
import time
import hashlib
import numpy as np

from .results import TwoDateInferenceResult
from ..inference.model import SingleDateInferenceEngine
from ..inference.probabilities import calculate_softmax, calculate_confidence
from ..alignment.temporal import TemporalAligner
from ..alignment.spatial import ReferenceGrid
from ..output.geotiff import GeoTIFFWriter
from ..output.json import write_pair_metadata_json

CLASS_NAMES = {
    0: "Background / Bare Soil",
    1: "Severe Stress",
    2: "Moderate Stress",
    3: "Healthy Vegetation"
}

def calculate_class_stats(class_map: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Calculates class distribution statistics over the specified mask.
    Only pixels where mask is True are considered.
    """
    total_valid = int(np.sum(mask))
    
    stats = {
        "total_valid_pixels": total_valid,
        "classes": {}
    }
    
    class_label_map = {
        0: "Background",
        1: "Severe",
        2: "Moderate",
        3: "Healthy"
    }
    
    for class_id, name in class_label_map.items():
        if total_valid > 0:
            count = int(np.sum((class_map == class_id) & mask))
            pct = (count / total_valid) * 100.0
        else:
            count = 0
            pct = 0.0
        
        stats["classes"][str(class_id)] = {
            "name": name,
            "count": count,
            "percentage": float(pct)
        }
        
    return stats

def analyze_pair(
    t1_scene: Path,
    t2_scene: Path,
    model_path: Path,
    output_dir: Path,
    config: Dict[str, Any]
) -> TwoDateInferenceResult:
    """
    Runs Phase 3 Two-Date Model Inference pipeline.
    
    1. Validates T1/T2 spatial compatibility (must be already aligned).
    2. Runs Phase 1 single-date inference on T1.
    3. Runs Phase 1 single-date inference on T2.
    4. Calculates softmax probabilities and maximum confidence map for both dates.
    5. Propagates valid data masks to output maps.
    6. Saves outputs, statistics, and metadata to disk.
    7. Returns TwoDateInferenceResult object.
    """
    t1_scene = Path(t1_scene)
    t2_scene = Path(t2_scene)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    # 1. Initialize TemporalAligner to load & validate bands
    aligner = TemporalAligner(config)
    t1_bands, t1_meta = aligner.load_and_validate_date(t1_scene)
    t2_bands, t2_meta = aligner.load_and_validate_date(t2_scene)
    
    t1_grid = ReferenceGrid.from_metadata(t1_meta)
    t2_grid = ReferenceGrid.from_metadata(t2_meta)
    
    # Verify that they share the exact same spatial grid (transform, resolution, etc.)
    if not aligner.spatial_aligner.check_grids_equal(t1_grid, t2_grid):
        raise ValueError(
            "T1 and T2 spatial grids are not aligned. Please perform Phase 2 temporal alignment "
            "before executing Phase 3 Two-Date Inference."
        )
        
    # 2. Setup configuration overrides and initialize model
    config = dict(config)
    config["model"]["path"] = str(model_path)
    engine = SingleDateInferenceEngine(config)
    
    # Measure performance times
    pair_start_time = time.perf_counter()
    
    # T1 run
    t1_run_start = time.perf_counter()
    t1_logits, t1_class_map, _, t1_stats = engine.run_pipeline(t1_scene)
    t1_run_time = (time.perf_counter() - t1_run_start) * 1000.0
    
    # T2 run
    t2_run_start = time.perf_counter()
    t2_logits, t2_class_map, _, t2_stats = engine.run_pipeline(t2_scene)
    t2_run_time = (time.perf_counter() - t2_run_start) * 1000.0
    
    # Performance calculations
    t1_inference_time = t1_stats["inference_time_ms"]
    t1_preprocessing_time = t1_run_time - t1_inference_time
    
    t2_inference_time = t2_stats["inference_time_ms"]
    t2_preprocessing_time = t2_run_time - t2_inference_time
    
    total_pair_time = (time.perf_counter() - pair_start_time) * 1000.0
    
    # 3. Build validity masks and overlap mask
    t1_valid_mask = np.ones((t1_grid.height, t1_grid.width), dtype=bool)
    for name, arr in t1_bands.items():
        t1_valid_mask &= aligner.spatial_aligner.get_validity_mask(arr, t1_meta.nodata)
        
    t2_valid_mask = np.ones((t2_grid.height, t2_grid.width), dtype=bool)
    for name, arr in t2_bands.items():
        t2_valid_mask &= aligner.spatial_aligner.get_validity_mask(arr, t2_meta.nodata)
        
    overlap_mask = t1_valid_mask & t2_valid_mask
    
    # 4. Argmax and mask health maps with date-valid mask (NoData = 255)
    t1_health_map = np.where(t1_valid_mask, t1_class_map, 255).astype(np.uint8)
    t2_health_map = np.where(t2_valid_mask, t2_class_map, 255).astype(np.uint8)
    
    # 5. Softmax and confidence calculation
    t1_probs = calculate_softmax(t1_logits)
    t2_probs = calculate_softmax(t2_logits)
    
    t1_conf = calculate_confidence(t1_probs)
    t2_conf = calculate_confidence(t2_probs)
    
    # 6. Statistics
    t1_stats_valid = calculate_class_stats(t1_class_map, t1_valid_mask)
    t1_stats_overlap = calculate_class_stats(t1_class_map, overlap_mask)
    
    t2_stats_valid = calculate_class_stats(t2_class_map, t2_valid_mask)
    t2_stats_overlap = calculate_class_stats(t2_class_map, overlap_mask)
    
    # 7. Write outputs to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    t1_out_dir = output_dir / "T1"
    t2_out_dir = output_dir / "T2"
    
    t1_out_dir.mkdir(parents=True, exist_ok=True)
    t2_out_dir.mkdir(parents=True, exist_ok=True)
    
    writer = GeoTIFFWriter(config["output"]["colormap"])
    
    # Save T1 outputs
    writer.write_health_map(str(t1_out_dir / "health.tif"), t1_health_map, t1_meta)
    writer.write_logits(str(t1_out_dir / "logits"), t1_logits)  # GeoTIFFWriter.write_logits adds .npy
    np.save(str(t1_out_dir / "probabilities.npy"), t1_probs)
    writer.write_confidence_map(str(t1_out_dir / "confidence.tif"), t1_conf, t1_meta)
    
    # Save T2 outputs
    writer.write_health_map(str(t2_out_dir / "health.tif"), t2_health_map, t2_meta)
    writer.write_logits(str(t2_out_dir / "logits"), t2_logits)  # GeoTIFFWriter.write_logits adds .npy
    np.save(str(t2_out_dir / "probabilities.npy"), t2_probs)
    writer.write_confidence_map(str(t2_out_dir / "confidence.tif"), t2_conf, t2_meta)
    
    # Save overlap mask
    writer.write_mask_map(str(output_dir / "overlap_mask.tif"), overlap_mask, t1_meta)
    
    # Save single-date metadata JSON files
    writer.write_metadata_json(
        output_path=str(t1_out_dir / "metadata.json"),
        input_scene_dir=str(t1_scene),
        model_path=config["model"]["path"],
        input_shape=(1, 5, t1_grid.height, t1_grid.width),
        output_shape=(1, 4, t1_grid.height, t1_grid.width),
        class_mapping=CLASS_NAMES,
        channel_order=config["model"]["channel_order"],
        preproc_config=config["preprocessing"],
        metadata=t1_meta,
        inference_provider=t1_stats["execution_provider"],
        inference_time_ms=t1_inference_time
    )
    
    writer.write_metadata_json(
        output_path=str(t2_out_dir / "metadata.json"),
        input_scene_dir=str(t2_scene),
        model_path=config["model"]["path"],
        input_shape=(1, 5, t2_grid.height, t2_grid.width),
        output_shape=(1, 4, t2_grid.height, t2_grid.width),
        class_mapping=CLASS_NAMES,
        channel_order=config["model"]["channel_order"],
        preproc_config=config["preprocessing"],
        metadata=t2_meta,
        inference_provider=t2_stats["execution_provider"],
        inference_time_ms=t2_inference_time
    )
    
    # Compute SHA-256 model hash
    model_hash = ""
    if model_path.is_file():
        sha = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        model_hash = sha.hexdigest()
        
    overlap_pct = (np.sum(overlap_mask) / overlap_mask.size) * 100.0
    
    # Generate root pair metadata
    pair_metadata = {
        "pair_name": output_dir.name,
        "t1_date_dir": str(t1_scene),
        "t2_date_dir": str(t2_scene),
        "model_path": str(model_path),
        "model_hash": model_hash,
        "model_input_shape": [1, 5, t1_grid.height, t1_grid.width],
        "model_output_shape": [1, 4, t1_grid.height, t1_grid.width],
        "channel_order": config["model"]["channel_order"],
        "class_mapping": CLASS_NAMES,
        "preprocessing_configuration": config["preprocessing"],
        "t1_spatial_metadata": {
            "crs": str(t1_meta.crs),
            "transform": [t1_meta.transform.a, t1_meta.transform.b, t1_meta.transform.c,
                          t1_meta.transform.d, t1_meta.transform.e, t1_meta.transform.f],
            "dimensions": [t1_meta.height, t1_meta.width],
            "resolution": list(t1_meta.resolution),
            "extent": list(t1_meta.extent)
        },
        "t2_spatial_metadata": {
            "crs": str(t2_meta.crs),
            "transform": [t2_meta.transform.a, t2_meta.transform.b, t2_meta.transform.c,
                          t2_meta.transform.d, t2_meta.transform.e, t2_meta.transform.f],
            "dimensions": [t2_meta.height, t2_meta.width],
            "resolution": list(t2_meta.resolution),
            "extent": list(t2_meta.extent)
        },
        "reference_grid": t1_grid.to_dict(),
        "overlap_information": {
            "overlap_pixel_count": int(np.sum(overlap_mask)),
            "overlap_percentage": float(overlap_pct)
        },
        "performance": {
            "t1": {
                "preprocessing_time_ms": float(t1_preprocessing_time),
                "inference_time_ms": float(t1_inference_time),
                "inference_provider": t1_stats["execution_provider"]
            },
            "t2": {
                "preprocessing_time_ms": float(t2_preprocessing_time),
                "inference_time_ms": float(t2_inference_time),
                "inference_provider": t2_stats["execution_provider"]
            },
            "total_pair_processing_time_ms": float(total_pair_time)
        },
        "probabilities_generated": True,
        "confidence_generated": True,
        "class_statistics": {
            "t1": {
                "date_valid": t1_stats_valid,
                "common_overlap": t1_stats_overlap
            },
            "t2": {
                "date_valid": t2_stats_valid,
                "common_overlap": t2_stats_overlap
            }
        },
        "software_version": "1.0.0"
    }
    
    write_pair_metadata_json(str(output_dir / "pair_metadata.json"), pair_metadata)
    
    return TwoDateInferenceResult(
        t1_health_map=t1_health_map,
        t2_health_map=t2_health_map,
        t1_logits=t1_logits,
        t2_logits=t2_logits,
        t1_probabilities=t1_probs,
        t2_probabilities=t2_probs,
        t1_confidence=t1_conf,
        t2_confidence=t2_conf,
        t1_valid_mask=t1_valid_mask,
        t2_valid_mask=t2_valid_mask,
        overlap_mask=overlap_mask,
        reference_grid=t1_grid,
        t1_metadata=pair_metadata["t1_spatial_metadata"],
        t2_metadata=pair_metadata["t2_spatial_metadata"],
        class_statistics_t1=pair_metadata["class_statistics"]["t1"],
        class_statistics_t2=pair_metadata["class_statistics"]["t2"],
        performance=pair_metadata["performance"]
    )

