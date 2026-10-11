import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


from ..alignment.spatial import ReferenceGrid
from .transition import TRANSITIONS

# Locked transitions and categories for Phase 6
DEGRADATION_IDS = {9, 13, 14}
RECOVERY_IDS = {6, 7, 11}
CANOPY_LOSS_IDS = {4, 8, 12}
BACKGROUND_TO_VEGETATION_IDS = {1, 2, 3}
STABLE_VEGETATION_IDS = {5, 10, 15}
STABLE_BACKGROUND_IDS = {0}

@dataclass
class DegradationResult:
    """
    Container for the outputs of Phase 6 Degradation & Recovery Engine.
    """
    degradation_map: np.ndarray        # (H, W) uint8: 0=False, 1=True, 255=NoData
    recovery_map: np.ndarray           # (H, W) uint8: 0=False, 1=True, 255=NoData
    residual_stress: np.ndarray        # (H, W) uint8: 0=False, 1=True, 255=NoData
    canopy_loss: np.ndarray            # (H, W) uint8: 0=False, 1=True, 255=NoData
    statistics: Dict[str, Any]
    thresholds: Dict[str, Any]
    transition_counts: Dict[str, int]
    metadata: Dict[str, Any]


def _load_input_raster(val: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Helper to load a raster file path or handle an existing numpy array and extract metadata."""
    if isinstance(val, (str, Path)):
        path = Path(val)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        with rasterio.open(path) as src:
            arr = src.read(1)
            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "resolution": src.res,
                "bounds": src.bounds,
                "nodata": src.nodata
            }
            return arr, meta
    elif isinstance(val, np.ndarray):
        return val, None
    else:
        raise TypeError(f"Expected str, Path, or numpy array, got {type(val)}")


def analyze_degradation(
    transition_map: Union[str, Path, np.ndarray],
    cva_magnitude: Union[str, Path, np.ndarray],
    delta_ndvi: Union[str, Path, np.ndarray],
    delta_gndvi: Union[str, Path, np.ndarray],
    delta_ndre: Union[str, Path, np.ndarray],
    valid_mask: Union[str, Path, np.ndarray],
    config: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    reference_grid: Optional[ReferenceGrid] = None
) -> DegradationResult:
    """
    Performs Phase 6 Degradation & Recovery analysis.
    Combines Phase 4 categorical transitions with Phase 5 continuous CVA.
    """
    # 1. Load inputs and check types
    trans_arr, trans_meta = _load_input_raster(transition_map)
    cva_arr, cva_meta = _load_input_raster(cva_magnitude)
    ndvi_arr, ndvi_meta = _load_input_raster(delta_ndvi)
    gndvi_arr, gndvi_meta = _load_input_raster(delta_gndvi)
    ndre_arr, ndre_meta = _load_input_raster(delta_ndre)
    mask_arr, mask_meta = _load_input_raster(valid_mask)

    # 2. Extract spatial metadata
    crs = None
    transform = None
    width = None
    height = None
    resolution = None
    bounds = None

    # Get from metadata if available
    metas = [m for m in [trans_meta, cva_meta, ndvi_meta, gndvi_meta, ndre_meta, mask_meta] if m is not None]
    if metas:
        ref_m = metas[0]
        crs = ref_m["crs"]
        transform = ref_m["transform"]
        width = ref_m["width"]
        height = ref_m["height"]
        resolution = ref_m["resolution"]
        bounds = ref_m["bounds"]
    elif reference_grid is not None:
        crs = reference_grid.crs
        transform = reference_grid.transform
        width = reference_grid.width
        height = reference_grid.height
        resolution = reference_grid.resolution
        bounds = reference_grid.bounds
    else:
        # Construct fallback grid parameters based on shape
        height, width = trans_arr.shape
        crs = CRS.from_epsg(4326)
        transform = Affine.identity()
        resolution = (1.0, 1.0)
        bounds = (0.0, 0.0, float(width), float(height))

    # Strict spatial metadata check
    for m in metas:
        if m["width"] != width or m["height"] != height:
            raise ValueError(f"Geometry mismatch: Dimensions do not match. Expected {width}x{height}, got {m['width']}x{m['height']}.")
        if crs is not None and m["crs"] is not None and crs != m["crs"]:
            raise ValueError(f"CRS mismatch: Expected {crs}, got {m['crs']}.")
        if transform is not None and m["transform"] is not None and not np.allclose(
            [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
            [m["transform"].a, m["transform"].b, m["transform"].c, m["transform"].d, m["transform"].e, m["transform"].f],
            atol=1e-5
        ):
            raise ValueError(f"Transform mismatch: Expected {transform}, got {m['transform']}.")
        if resolution is not None and m["resolution"] is not None and not np.allclose(resolution, m["resolution"], atol=1e-5):
            raise ValueError(f"Resolution mismatch: Expected {resolution}, got {m['resolution']}.")
        if bounds is not None and m["bounds"] is not None and not np.allclose(
            [bounds[0], bounds[1], bounds[2], bounds[3]],
            [m["bounds"][0], m["bounds"][1], m["bounds"][2], m["bounds"][3]],
            atol=1e-5
        ):
            raise ValueError(f"Bounds mismatch: Expected {bounds}, got {m['bounds']}.")

    # Direct shape check
    shape = trans_arr.shape
    if (cva_arr.shape != shape or 
        ndvi_arr.shape != shape or 
        gndvi_arr.shape != shape or 
        ndre_arr.shape != shape or 
        mask_arr.shape != shape):
        raise ValueError("Geometry mismatch: Array shapes are not identical.")

    # 3. Handle NoData mapping
    # Validate transition IDs - raise error if there are unrecognized IDs (other than valid 0-15 and NoData 255)
    valid_ids = set(TRANSITIONS.keys())
    unique_trans_ids = np.unique(trans_arr)
    for tid in unique_trans_ids:
        if tid != 255 and tid not in valid_ids:
            raise ValueError(f"Invalid transition ID encountered: {tid}. Authoritative range is 0-15.")

    # Define invalid pixels
    # transition_map uses 255 for NoData
    # cva uses -9999.0 for NoData
    # mask is boolean, False means invalid
    # Also handle NaNs in CVA and delta bands
    cva_nodata_val = -9999.0
    if cva_meta is not None and cva_meta["nodata"] is not None:
        cva_nodata_val = cva_meta["nodata"]

    invalid_mask = (
        (mask_arr == 0) | 
        (trans_arr == 255) | 
        (cva_arr == cva_nodata_val) | 
        np.isnan(cva_arr) | 
        np.isnan(ndvi_arr) | 
        np.isnan(gndvi_arr) | 
        np.isnan(ndre_arr)
    )
    valid_pixels = ~invalid_mask
    valid_pixel_count = int(np.sum(valid_pixels))

    # 4. Process configurations
    change_config = config.get("change", {})
    res_config = change_config.get("residual_stress", {})
    res_enabled = res_config.get("enabled", True)
    thresh_config = res_config.get("threshold", {"method": "percentile", "value": 95.0})
    thresh_method = thresh_config.get("method", "percentile")
    thresh_val = thresh_config.get("value", 95.0)
    directional_rule = res_config.get("directional_rule", None)

    # 5. Compute CVA residual stress threshold
    # Population for thresholding: valid stable vegetation pixels (5, 10, 15) ONLY
    stable_veg_mask = np.isin(trans_arr, list(STABLE_VEGETATION_IDS)) & valid_pixels
    stable_veg_count = int(np.sum(stable_veg_mask))

    cva_threshold = 0.0
    population_count = stable_veg_count

    if stable_veg_count > 0:
        cva_stable_population = cva_arr[stable_veg_mask]
        if thresh_method == "percentile":
            cva_threshold = float(np.percentile(cva_stable_population, thresh_val))
        elif thresh_method in ("absolute", "externally supplied"):
            cva_threshold = float(thresh_val)
        elif thresh_method == "robust-statistical":
            median = np.median(cva_stable_population)
            mad = np.median(np.abs(cva_stable_population - median))
            cva_threshold = float(median + thresh_val * mad)
        else:
            raise ValueError(f"Unknown threshold method: {thresh_method}")
    else:
        # Fallback if stable veg population is empty
        if thresh_method in ("absolute", "externally supplied"):
            cva_threshold = float(thresh_val)
        else:
            cva_threshold = 0.0

    # 6. Initialize output arrays (uint8, default 0, NoData = 255)
    degradation_map = np.zeros(shape, dtype=np.uint8)
    recovery_map = np.zeros(shape, dtype=np.uint8)
    residual_stress = np.zeros(shape, dtype=np.uint8)
    canopy_loss_map = np.zeros(shape, dtype=np.uint8)

    # Apply invalid mask to outputs
    degradation_map[invalid_mask] = 255
    recovery_map[invalid_mask] = 255
    residual_stress[invalid_mask] = 255
    canopy_loss_map[invalid_mask] = 255

    # 7. Apply vectorized rules on valid pixels
    if valid_pixel_count > 0:
        # Categorical maps:
        # Degradation
        degradation_mask = np.isin(trans_arr, list(DEGRADATION_IDS)) & valid_pixels
        degradation_map[degradation_mask] = 1

        # Recovery
        recovery_mask = np.isin(trans_arr, list(RECOVERY_IDS)) & valid_pixels
        recovery_map[recovery_mask] = 1

        # Canopy Loss
        canopy_loss_mask = np.isin(trans_arr, list(CANOPY_LOSS_IDS)) & valid_pixels
        canopy_loss_map[canopy_loss_mask] = 1

        # Residual Stress
        if res_enabled:
            # Must be stable vegetation transition
            cand_residual_mask = stable_veg_mask & (cva_arr >= cva_threshold)
            
            # Apply directional check if configured
            if directional_rule == "all_negative":
                cand_residual_mask &= (ndvi_arr < 0) & (gndvi_arr < 0) & (ndre_arr < 0)
            elif directional_rule == "any_negative":
                cand_residual_mask &= (ndvi_arr < 0) | (gndvi_arr < 0) | (ndre_arr < 0)
            elif directional_rule == "ndvi_negative":
                cand_residual_mask &= (ndvi_arr < 0)
                
            residual_stress[cand_residual_mask] = 1

    # 8. Compute transition specific counts on valid pixels
    transition_counts = {}
    for tid, info in TRANSITIONS.items():
        name = info["name"] # B_to_B etc. Replaced special char to match requested format
        count = int(np.sum((trans_arr == tid) & valid_pixels))
        transition_counts[name] = count

    # 9. Compute summary statistics
    deg_count = int(np.sum(degradation_map == 1))
    rec_count = int(np.sum(recovery_map == 1))
    canopy_count = int(np.sum(canopy_loss_map == 1))
    bg_to_veg_count = int(np.sum(np.isin(trans_arr, list(BACKGROUND_TO_VEGETATION_IDS)) & valid_pixels))
    stable_veg_count = int(np.sum(stable_veg_mask))
    stable_bg_count = int(np.sum((trans_arr == 0) & valid_pixels))
    stable_total_count = stable_veg_count + stable_bg_count
    residual_stress_count = int(np.sum(residual_stress == 1))

    def get_pct(cnt):
        return (cnt / valid_pixel_count * 100.0) if valid_pixel_count > 0 else 0.0

    # Helper to stratify CVA statistics by category
    def calculate_cva_stats(mask_sel):
        if np.sum(mask_sel) == 0:
            return {"mean": None, "median": None, "p95": None, "min": None, "max": None, "std": None}
        cva_vals = cva_arr[mask_sel]
        return {
            "mean": float(np.mean(cva_vals)),
            "median": float(np.median(cva_vals)),
            "p95": float(np.percentile(cva_vals, 95.0)),
            "min": float(np.min(cva_vals)),
            "max": float(np.max(cva_vals)),
            "std": float(np.std(cva_vals))
        }

    cva_stats = {
        "degradation": calculate_cva_stats(degradation_mask & valid_pixels if valid_pixel_count > 0 else np.zeros_like(valid_pixels)),
        "recovery": calculate_cva_stats(recovery_mask & valid_pixels if valid_pixel_count > 0 else np.zeros_like(valid_pixels)),
        "canopy_loss": calculate_cva_stats(canopy_loss_mask & valid_pixels if valid_pixel_count > 0 else np.zeros_like(valid_pixels)),
        "stable_vegetation": calculate_cva_stats(stable_veg_mask),
        "background_to_vegetation": calculate_cva_stats(np.isin(trans_arr, list(BACKGROUND_TO_VEGETATION_IDS)) & valid_pixels)
    }

    stats = {
        "valid_pixel_count": valid_pixel_count,
        "nodata_pixel_count": int(np.sum(invalid_mask)),
        "degradation_pixel_count": deg_count,
        "degradation_percentage": get_pct(deg_count),
        "recovery_pixel_count": rec_count,
        "recovery_percentage": get_pct(rec_count),
        "residual_stress_pixel_count": residual_stress_count,
        "residual_stress_percentage": get_pct(residual_stress_count),
        "canopy_loss_pixel_count": canopy_count,
        "canopy_loss_percentage": get_pct(canopy_count),
        "background_to_vegetation_count": bg_to_veg_count,
        "background_to_vegetation_percentage": get_pct(bg_to_veg_count),
        "stable_count": stable_total_count,
        "stable_vegetation_count": stable_veg_count,
        "stable_background_count": stable_bg_count,
        "transition_counts": transition_counts,
        "cva_statistics_by_category": cva_stats
    }

    thresholds = {
        "residual_cva_threshold": {
            "method": thresh_method,
            "configured_value": thresh_val,
            "threshold_value": cva_threshold,
            "population_count": population_count
        }
    }

    # 10. Generate Metadata
    metadata = {
        "transition_definitions": {str(k): v for k, v in TRANSITIONS.items()},
        "degradation_transition_ids": list(DEGRADATION_IDS),
        "recovery_transition_ids": list(RECOVERY_IDS),
        "canopy_loss_transition_ids": list(CANOPY_LOSS_IDS),
        "residual_stress_rule": {
            "enabled": res_enabled,
            "threshold_method": thresh_method,
            "threshold_configured_value": thresh_val,
            "threshold_value": cva_threshold,
            "directional_rule": directional_rule
        },
        "spatial_grid": {
            "crs": str(crs) if crs is not None else None,
            "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f] if transform is not None else None,
            "width": width,
            "height": height,
            "resolution": list(resolution) if resolution is not None else None,
            "bounds": [bounds[0], bounds[1], bounds[2], bounds[3]] if bounds is not None else None
        },
        "nodata_policy": {
            "input_transition_nodata": 255,
            "input_cva_nodata": cva_nodata_val,
            "output_nodata": 255
        },
        "disclaimer": "Vegetation-to-background transitions are not interpreted as physical wetland depletion in Phase 6.",
        "software_version": "1.0.0"
    }

    # 11. Save outputs to disk
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Build raster profile
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.uint8,
            'crs': crs,
            'transform': transform,
            'nodata': 255,
            'compress': 'lzw'
        }

        with rasterio.open(out_path / "degradation_map.tif", 'w', **profile) as dst:
            dst.write(degradation_map, 1)

        with rasterio.open(out_path / "recovery_map.tif", 'w', **profile) as dst:
            dst.write(recovery_map, 1)

        with rasterio.open(out_path / "residual_stress.tif", 'w', **profile) as dst:
            dst.write(residual_stress, 1)

        with rasterio.open(out_path / "canopy_loss_map.tif", 'w', **profile) as dst:
            dst.write(canopy_loss_map, 1)

        # Save metadata and stats
        with open(out_path / "degradation_statistics.json", "w") as f:
            json.dump(stats, f, indent=4)

        # Write degradation_metadata.json
        # Include inputs if strings/paths
        metadata_to_save = metadata.copy()
        metadata_to_save["inputs"] = {
            "transition_map": str(transition_map) if isinstance(transition_map, (str, Path)) else "ndarray",
            "cva_magnitude": str(cva_magnitude) if isinstance(cva_magnitude, (str, Path)) else "ndarray",
            "delta_ndvi": str(delta_ndvi) if isinstance(delta_ndvi, (str, Path)) else "ndarray",
            "delta_gndvi": str(delta_gndvi) if isinstance(delta_gndvi, (str, Path)) else "ndarray",
            "delta_ndre": str(delta_ndre) if isinstance(delta_ndre, (str, Path)) else "ndarray",
            "valid_mask": str(valid_mask) if isinstance(valid_mask, (str, Path)) else "ndarray"
        }
        with open(out_path / "degradation_metadata.json", "w") as f:
            json.dump(metadata_to_save, f, indent=4)

    return DegradationResult(
        degradation_map=degradation_map,
        recovery_map=recovery_map,
        residual_stress=residual_stress,
        canopy_loss=canopy_loss_map,
        statistics=stats,
        thresholds=thresholds,
        transition_counts=transition_counts,
        metadata=metadata
    )
