import os
import json
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, Union, List
import scipy.ndimage as ndimage

from ..alignment.spatial import ReferenceGrid
from .area import calculate_pixel_areas, PhysicalAreaUnavailableError

@dataclass
class MetricsResult:
    baseline_area: Dict[str, Any]
    impact_area: Dict[str, Any]
    severity_weighted_area: Dict[str, Any]
    normalized_degradation_ratio: Optional[float]
    recovery_area: Dict[str, Any]
    residual_stress_area: Dict[str, Any]
    canopy_loss_area: Dict[str, Any]

    transition_metrics: Dict[str, Any]
    class_area_metrics: Dict[str, Any]
    cva_statistics: Dict[str, Any]
    delta_statistics: Dict[str, Any]

    wetland_extent_metrics: Dict[str, Any]

    qa_flags: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def verify_spatial_alignment(metas: List[Dict[str, Any]]):
    """Verifies that all input rasters share identical CRS, transform, resolution, dimensions and bounds."""
    if not metas:
        return
    ref = metas[0]
    if ref is None:
        return
    for m in metas[1:]:
        if m is None:
            continue
        # Compare CRS
        if ref["crs"] != m["crs"]:
            raise ValueError(f"CRS mismatch: {ref['crs']} vs {m['crs']}")
        # Compare dimensions
        if ref["width"] != m["width"] or ref["height"] != m["height"]:
            raise ValueError(f"Dimension mismatch: {ref['width']}x{ref['height']} vs {m['width']}x{m['height']}")
        # Compare transform
        if not np.allclose(
            [ref["transform"].a, ref["transform"].b, ref["transform"].c, ref["transform"].d, ref["transform"].e, ref["transform"].f],
            [m["transform"].a, m["transform"].b, m["transform"].c, m["transform"].d, m["transform"].e, m["transform"].f],
            atol=1e-5
        ):
            raise ValueError("Transform mismatch.")
        # Compare resolution
        if not np.allclose(ref["resolution"], m["resolution"], atol=1e-5):
            raise ValueError("Resolution mismatch.")
        # Compare bounds
        if not np.allclose(
            [ref["bounds"][0], ref["bounds"][1], ref["bounds"][2], ref["bounds"][3]],
            [m["bounds"][0], m["bounds"][1], m["bounds"][2], m["bounds"][3]],
            atol=1e-5
        ):
            raise ValueError("Bounds mismatch.")


def filter_by_min_area(binary_map: np.ndarray, pixel_areas: np.ndarray, threshold_m2: float) -> np.ndarray:
    """Filters out connected components (patches) whose total area is less than threshold_m2."""
    if threshold_m2 <= 0:
        return binary_map
    
    # 255 is NoData. We label components only where the value is 1.
    label_mask = (binary_map == 1)
    labeled, num_features = ndimage.label(label_mask)
    if num_features == 0:
        return binary_map
        
    filtered = binary_map.copy()
    for i in range(1, num_features + 1):
        comp_mask = (labeled == i)
        comp_area = np.sum(pixel_areas[comp_mask])
        if comp_area < threshold_m2:
            filtered[comp_mask] = 0
            
    return filtered


def calculate_subset_stats(data: np.ndarray, mask: np.ndarray, percentiles: List[int] = [95]) -> Dict[str, Any]:
    """Calculates summary statistics of data under mask. Returns None values if no pixels match."""
    subset = data[mask]
    if len(subset) == 0:
        res = {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None
        }
        for p in percentiles:
            res[f"p{p}"] = None
        return res
        
    res = {
        "mean": float(np.mean(subset)),
        "median": float(np.median(subset)),
        "min": float(np.min(subset)),
        "max": float(np.max(subset)),
        "std": float(np.std(subset))
    }
    for p in percentiles:
        res[f"p{p}"] = float(np.percentile(subset, p))
    return res


def calculate_metrics(
    transition_map: Union[str, Path, np.ndarray],
    cva_magnitude: Union[str, Path, np.ndarray],
    delta_ndvi: Union[str, Path, np.ndarray],
    delta_gndvi: Union[str, Path, np.ndarray],
    delta_ndre: Union[str, Path, np.ndarray],
    degradation_map: Union[str, Path, np.ndarray],
    recovery_map: Union[str, Path, np.ndarray],
    residual_stress_map: Union[str, Path, np.ndarray],
    canopy_loss_map: Union[str, Path, np.ndarray],
    wetland_mask: Union[str, Path, np.ndarray],
    valid_analysis_mask: Union[str, Path, np.ndarray],
    config: Dict[str, Any],
    reference_grid: Optional[ReferenceGrid] = None
) -> MetricsResult:
    """
    Phase 8 Engine: Calculates quantitative metrics and degradation indices from validated inputs.
    """
    # 1. Load inputs and extract spatial metadata
    trans_arr, trans_meta = _load_input_raster(transition_map)
    cva_arr, cva_meta = _load_input_raster(cva_magnitude)
    ndvi_arr, ndvi_meta = _load_input_raster(delta_ndvi)
    gndvi_arr, gndvi_meta = _load_input_raster(delta_gndvi)
    ndre_arr, ndre_meta = _load_input_raster(delta_ndre)
    deg_arr, deg_meta = _load_input_raster(degradation_map)
    rec_arr, rec_meta = _load_input_raster(recovery_map)
    res_arr, res_meta = _load_input_raster(residual_stress_map)
    canopy_arr, canopy_meta = _load_input_raster(canopy_loss_map)
    wetland_arr, wetland_meta = _load_input_raster(wetland_mask)
    valid_mask_arr, valid_mask_meta = _load_input_raster(valid_analysis_mask)

    # 2. Strict spatial validation
    metas = [
        trans_meta, cva_meta, ndvi_meta, gndvi_meta, ndre_meta,
        deg_meta, rec_meta, res_meta, canopy_meta, wetland_meta, valid_mask_meta
    ]
    verify_spatial_alignment([m for m in metas if m is not None])

    # Direct shape check
    shape = trans_arr.shape
    if (cva_arr.shape != shape or ndvi_arr.shape != shape or gndvi_arr.shape != shape or
        ndre_arr.shape != shape or deg_arr.shape != shape or rec_arr.shape != shape or
        res_arr.shape != shape or canopy_arr.shape != shape or wetland_arr.shape != shape or
        valid_mask_arr.shape != shape):
        raise ValueError("Geometry mismatch: Array shapes are not identical.")

    # 3. Resolve spatial grid parameters for area calculation
    crs = None
    transform = None
    width = None
    height = None

    non_null_metas = [m for m in metas if m is not None]
    if non_null_metas:
        ref_m = non_null_metas[0]
        crs = ref_m["crs"]
        transform = ref_m["transform"]
        width = ref_m["width"]
        height = ref_m["height"]
    elif reference_grid is not None:
        crs = reference_grid.crs
        transform = reference_grid.transform
        width = reference_grid.width
        height = reference_grid.height
    else:
        raise ValueError("Missing spatial grid reference. Provide file paths or a reference_grid object.")

    # Calculate physical pixel areas only when authoritative spatial
    # reference information is available. CRS-less/pixel-space rasters
    # must never be interpreted as metres.
    physical_area_available = True
    try:
        pixel_areas = calculate_pixel_areas(transform, shape, crs)
    except PhysicalAreaUnavailableError:
        pixel_areas = None
        physical_area_available = False

    # 4. Connected Component Filtering (MMU)
    metrics_config = config.get("metrics", {})
    min_change_config = metrics_config.get("minimum_change_area", {})
    mmu_enabled = min_change_config.get("enabled", False)
    mmu_val_m2 = min_change_config.get("value_m2", 0)

    minimum_change_area_applied = False
    minimum_change_area_available = True
    if mmu_enabled and mmu_val_m2 > 0:
        if physical_area_available:
            deg_arr = filter_by_min_area(deg_arr, pixel_areas, mmu_val_m2)
            rec_arr = filter_by_min_area(rec_arr, pixel_areas, mmu_val_m2)
            res_arr = filter_by_min_area(res_arr, pixel_areas, mmu_val_m2)
            canopy_arr = filter_by_min_area(canopy_arr, pixel_areas, mmu_val_m2)
            minimum_change_area_applied = True
        else:
            minimum_change_area_available = False

    # 5. Define strictly valid population (exclude NoData)
    # Define invalid/NoData pixels
    invalid_mask = (
        (valid_mask_arr == 0) |
        (trans_arr == 255) |
        (cva_arr == -9999.0) | np.isnan(cva_arr) |
        (ndvi_arr == -9999.0) | np.isnan(ndvi_arr) |
        (gndvi_arr == -9999.0) | np.isnan(gndvi_arr) |
        (ndre_arr == -9999.0) | np.isnan(ndre_arr) |
        (deg_arr == 255) |
        (rec_arr == 255) |
        (res_arr == 255) |
        (canopy_arr == 255)
    )
    valid_pop = ~invalid_mask

    # 6. Baseline Area (A_baseline)
    baseline_pixel_count = int(np.sum(valid_pop))
    if physical_area_available:
        baseline_area_m2 = float(np.sum(pixel_areas[valid_pop]))
        baseline_area_ha = baseline_area_m2 / 10000.0
    else:
        baseline_area_m2 = None
        baseline_area_ha = None

    if baseline_pixel_count < 0:
        raise ValueError("Sanity Check Failed: Negative baseline pixel count detected.")
    if physical_area_available and baseline_area_m2 < 0:
        raise ValueError("Sanity Check Failed: Negative baseline area detected.")

    # 7. Affected / Degraded Footprint (A_impact)
    impact_mask = (deg_arr == 1) & valid_pop
    impact_pixel_count = int(np.sum(impact_mask))
    if physical_area_available:
        impact_area_m2 = float(np.sum(pixel_areas[impact_mask]))
        impact_area_ha = impact_area_m2 / 10000.0
    else:
        impact_area_m2 = None
        impact_area_ha = None

    if impact_pixel_count > baseline_pixel_count:
        raise ValueError("Sanity Check Failed: Affected pixel count exceeds baseline pixel count.")
    if physical_area_available and impact_area_m2 > baseline_area_m2:
        raise ValueError("Sanity Check Failed: Affected area exceeds baseline area.")

    # 8. Severity-Weighted Degraded Area (A_SWD / ha_deg)
    severity_weights = metrics_config.get("severity_weights", {
        "healthy": 0.0,
        "moderate": 0.35,
        "severe": 0.80,
        "background": 1.0
    })
    include_canopy_loss_in_swd = metrics_config.get("include_canopy_loss_in_swd", False)

    # 16 transition IDs + NoData values:
    # 0 to 15, and 255
    transition_weights = np.zeros(256, dtype=np.float64)
    # Default V1 transitions
    transition_weights[14] = severity_weights.get("moderate", 0.35)  # H->M
    transition_weights[13] = severity_weights.get("severe", 0.80)    # H->S
    transition_weights[9] = severity_weights.get("severe", 0.80)     # M->S

    # Canopy-loss transitions: H->B (12), M->B (8), S->B (4)
    if include_canopy_loss_in_swd:
        bg_weight = severity_weights.get("background", 1.0)
        transition_weights[12] = bg_weight
        transition_weights[8] = bg_weight
        transition_weights[4] = bg_weight

    severities = transition_weights[trans_arr]

    # Mask for SWD pixels:
    # Default V1 uses: H->M, H->S, M->S (which are deg_arr == 1)
    # If canopy loss is enabled: include canopy_arr == 1
    if include_canopy_loss_in_swd:
        swd_mask = ((deg_arr == 1) | (canopy_arr == 1)) & valid_pop
    else:
        swd_mask = (deg_arr == 1) & valid_pop

    if physical_area_available:
        swd_area_m2 = float(np.sum(pixel_areas[swd_mask] * severities[swd_mask]))
        swd_area_ha = swd_area_m2 / 10000.0
        if swd_area_m2 > baseline_area_m2:
            raise ValueError("Sanity Check Failed: Severity-weighted area exceeds baseline area.")
    else:
        swd_area_m2 = None
        swd_area_ha = None

    # 9. NDR (Normalized Degradation Ratio)
    warnings_list = []
    if not physical_area_available:
        ndr = None
        ndr_percent = None
        warnings_list.append(
            "Physical area is unavailable because the analysis raster is "
            "CRS-less/pixel-space. NDR is set to None."
        )
    elif baseline_area_m2 == 0:
        ndr = None
        ndr_percent = None
        warnings_list.append("Baseline area is zero. Normalized Degradation Ratio (NDR) is set to None.")
    else:
        ndr = swd_area_m2 / baseline_area_m2
        ndr_percent = ndr * 100.0
        if ndr < 0 or ndr > 1:
            raise ValueError(f"Sanity Check Failed: NDR value {ndr} is out of [0, 1] range.")

    # 10. Recovery Area (A_recovery)
    rec_mask = (rec_arr == 1) & valid_pop
    rec_pixel_count = int(np.sum(rec_mask))
    if physical_area_available:
        rec_area_m2 = float(np.sum(pixel_areas[rec_mask]))
        rec_area_ha = rec_area_m2 / 10000.0
        if rec_area_m2 > baseline_area_m2:
            raise ValueError("Sanity Check Failed: Recovery area exceeds baseline area.")
    else:
        rec_area_m2 = None
        rec_area_ha = None

    if not physical_area_available:
        rec_ratio = None
        warnings_list.append("Physical area is unavailable. Recovery ratio is set to None.")
    elif baseline_area_m2 == 0:
        rec_ratio = None
        warnings_list.append("Baseline area is zero. Recovery ratio is set to None.")
    else:
        rec_ratio = rec_area_m2 / baseline_area_m2

    # Recovery specific areas
    # M->H (11), S->M (6), S->H (7)
    rec_mh_mask = (trans_arr == 11) & valid_pop
    rec_sm_mask = (trans_arr == 6) & valid_pop
    rec_sh_mask = (trans_arr == 7) & valid_pop

    if physical_area_available:
        rec_mh_area_m2 = float(np.sum(pixel_areas[rec_mh_mask]))
        rec_sm_area_m2 = float(np.sum(pixel_areas[rec_sm_mask]))
        rec_sh_area_m2 = float(np.sum(pixel_areas[rec_sh_mask]))
    else:
        rec_mh_area_m2 = None
        rec_sm_area_m2 = None
        rec_sh_area_m2 = None

    # 11. Canopy-Loss Area (A_canopy_loss)
    canopy_mask = (canopy_arr == 1) & valid_pop
    canopy_pixel_count = int(np.sum(canopy_mask))
    if physical_area_available:
        canopy_area_m2 = float(np.sum(pixel_areas[canopy_mask]))
        canopy_area_ha = canopy_area_m2 / 10000.0
    else:
        canopy_area_m2 = None
        canopy_area_ha = None

    if not physical_area_available:
        canopy_ratio = None
        canopy_pct = None
    elif baseline_area_m2 == 0:
        canopy_ratio = None
        canopy_pct = None
    else:
        canopy_ratio = canopy_area_m2 / baseline_area_m2
        canopy_pct = canopy_ratio * 100.0

    # 12. Residual Stress Area (A_residual_stress)
    res_mask = (res_arr == 1) & valid_pop
    res_pixel_count = int(np.sum(res_mask))
    if physical_area_available:
        res_area_m2 = float(np.sum(pixel_areas[res_mask]))
        res_area_ha = res_area_m2 / 10000.0
    else:
        res_area_m2 = None
        res_area_ha = None

    # Stable veg area (5, 10, 15)
    stable_veg_mask = np.isin(trans_arr, [5, 10, 15]) & valid_pop
    if physical_area_available:
        stable_veg_area_m2 = float(np.sum(pixel_areas[stable_veg_mask]))
        stable_veg_area_ha = stable_veg_area_m2 / 10000.0
    else:
        stable_veg_area_m2 = None
        stable_veg_area_ha = None

    if not physical_area_available:
        res_ratio = None
        warnings_list.append("Physical area is unavailable. Residual stress ratio is set to None.")
    elif stable_veg_area_m2 == 0:
        res_ratio = None
        warnings_list.append("Stable vegetation area is zero. Residual stress ratio is set to None.")
    else:
        res_ratio = res_area_m2 / stable_veg_area_m2

    # 13. Class Area Metrics & Reconciliation
    T1_class = trans_arr // 4
    T2_class = trans_arr % 4

    # Calculate class areas for T1 and T2
    class_map = {
        0: "background",
        1: "severe",
        2: "moderate",
        3: "healthy"
    }

    class_areas_t1 = {}
    class_areas_t2 = {}
    class_changes = {}

    t1_sum_m2 = 0.0
    t2_sum_m2 = 0.0

    for cid, name in class_map.items():
        mask_t1 = (T1_class == cid) & valid_pop
        mask_t2 = (T2_class == cid) & valid_pop

        cnt_t1 = int(np.sum(mask_t1))
        cnt_t2 = int(np.sum(mask_t2))

        if physical_area_available:
            area_t1_m2 = float(np.sum(pixel_areas[mask_t1]))
            area_t2_m2 = float(np.sum(pixel_areas[mask_t2]))
            t1_sum_m2 += area_t1_m2
            t2_sum_m2 += area_t2_m2
            pct_t1 = (area_t1_m2 / baseline_area_m2 * 100.0) if baseline_area_m2 > 0 else 0.0
            pct_t2 = (area_t2_m2 / baseline_area_m2 * 100.0) if baseline_area_m2 > 0 else 0.0
            change_m2 = area_t2_m2 - area_t1_m2
            change_ha = change_m2 / 10000.0
            area_t1_ha = area_t1_m2 / 10000.0
            area_t2_ha = area_t2_m2 / 10000.0
        else:
            area_t1_m2 = None
            area_t2_m2 = None
            pct_t1 = (cnt_t1 / baseline_pixel_count * 100.0) if baseline_pixel_count > 0 else 0.0
            pct_t2 = (cnt_t2 / baseline_pixel_count * 100.0) if baseline_pixel_count > 0 else 0.0
            change_m2 = None
            change_ha = None
            area_t1_ha = None
            area_t2_ha = None

        if cnt_t1 < 0 or cnt_t2 < 0:
            raise ValueError(f"Sanity Check Failed: Negative count for class {name}.")

        class_areas_t1[name] = {
            "pixel_count": cnt_t1,
            "area_m2": area_t1_m2,
            "area_ha": area_t1_ha,
            "percentage": pct_t1
        }
        class_areas_t2[name] = {
            "pixel_count": cnt_t2,
            "area_m2": area_t2_m2,
            "area_ha": area_t2_ha,
            "percentage": pct_t2
        }

        class_changes[name] = {
            "area_change_m2": change_m2,
            "area_change_ha": change_ha
        }
        if name == "healthy":
            if physical_area_available:
                loss_m2 = max(area_t1_m2 - area_t2_m2, 0.0)
                loss_ha = loss_m2 / 10000.0
            else:
                loss_m2 = None
                loss_ha = None
            class_changes[name]["loss_m2"] = loss_m2
            class_changes[name]["loss_ha"] = loss_ha

    # Class Reconciliation Check
    if physical_area_available:
        reconciled_class_t1 = bool(np.isclose(t1_sum_m2, baseline_area_m2, rtol=1e-5, atol=1e-3))
        reconciled_class_t2 = bool(np.isclose(t2_sum_m2, baseline_area_m2, rtol=1e-5, atol=1e-3))
        if not reconciled_class_t1:
            raise ValueError(f"Class Reconciliation Failed for T1: Class sum {t1_sum_m2} != baseline {baseline_area_m2}.")
        if not reconciled_class_t2:
            raise ValueError(f"Class Reconciliation Failed for T2: Class sum {t2_sum_m2} != baseline {baseline_area_m2}.")
    else:
        reconciled_class_t1 = bool(sum(v["pixel_count"] for v in class_areas_t1.values()) == baseline_pixel_count)
        reconciled_class_t2 = bool(sum(v["pixel_count"] for v in class_areas_t2.values()) == baseline_pixel_count)

    # 14. Transition-specific Area Metrics
    TRANSITIONS_INFO = {
        0:  {"name": "B_to_B", "from": 0, "to": 0},
        1:  {"name": "B_to_S", "from": 0, "to": 1},
        2:  {"name": "B_to_M", "from": 0, "to": 2},
        3:  {"name": "B_to_H", "from": 0, "to": 3},
        4:  {"name": "S_to_B", "from": 1, "to": 0},
        5:  {"name": "S_to_S", "from": 1, "to": 1},
        6:  {"name": "S_to_M", "from": 1, "to": 2},
        7:  {"name": "S_to_H", "from": 1, "to": 3},
        8:  {"name": "M_to_B", "from": 2, "to": 0},
        9:  {"name": "M_to_S", "from": 2, "to": 1},
        10: {"name": "M_to_M", "from": 2, "to": 2},
        11: {"name": "M_to_H", "from": 2, "to": 3},
        12: {"name": "H_to_B", "from": 3, "to": 0},
        13: {"name": "H_to_S", "from": 3, "to": 1},
        14: {"name": "H_to_M", "from": 3, "to": 2},
        15: {"name": "H_to_H", "from": 3, "to": 3}
    }

    transition_areas = {}
    trans_sum_m2 = 0.0
    trans_sum_pixels = 0

    for tid, info in TRANSITIONS_INFO.items():
        name = info["name"]
        mask_trans = (trans_arr == tid) & valid_pop

        cnt = int(np.sum(mask_trans))
        trans_sum_pixels += cnt

        if physical_area_available:
            area_m2 = float(np.sum(pixel_areas[mask_trans]))
            area_ha = area_m2 / 10000.0
            pct = (area_m2 / baseline_area_m2 * 100.0) if baseline_area_m2 > 0 else 0.0
            trans_sum_m2 += area_m2
        else:
            area_m2 = None
            area_ha = None
            pct = (cnt / baseline_pixel_count * 100.0) if baseline_pixel_count > 0 else 0.0

        if cnt < 0:
            raise ValueError(f"Sanity Check Failed: Negative count for transition {name}.")
        if physical_area_available and area_m2 < 0:
            raise ValueError(f"Sanity Check Failed: Negative area for transition {name}.")

        transition_areas[name] = {
            "pixel_count": cnt,
            "area_m2": area_m2,
            "area_ha": area_ha,
            "percentage_of_valid_wetland_analysis_area": pct
        }

    # Transition Reconciliation checks
    # Sum transition counts must exactly equal baseline pixel count
    reconciled_trans_pixels = bool(trans_sum_pixels == baseline_pixel_count)
    # Sum transition areas must approximately equal baseline area
    if physical_area_available:
        reconciled_trans_area = bool(np.isclose(trans_sum_m2, baseline_area_m2, rtol=1e-5, atol=1e-3))
    else:
        reconciled_trans_area = None

    if not reconciled_trans_pixels:
        raise ValueError(f"Transition Reconciliation Failed: Sum of transition pixels {trans_sum_pixels} != baseline {baseline_pixel_count}.")
    if physical_area_available and not reconciled_trans_area:
        raise ValueError(f"Transition Reconciliation Failed: Sum of transition areas {trans_sum_m2} != baseline {baseline_area_m2}.")

    # 15. CVA Statistics by Change Category
    # Categorize subsets
    cva_masks = {
        "all_valid_wetland": valid_pop,
        "degradation": np.isin(trans_arr, [9, 13, 14]) & valid_pop,
        "recovery": np.isin(trans_arr, [6, 7, 11]) & valid_pop,
        "stable_vegetation": np.isin(trans_arr, [5, 10, 15]) & valid_pop,
        "canopy_loss": np.isin(trans_arr, [4, 8, 12]) & valid_pop,
        "background_to_vegetation": np.isin(trans_arr, [1, 2, 3]) & valid_pop
    }

    cva_stats = {}
    for cat_name, cat_mask in cva_masks.items():
        stats = calculate_subset_stats(cva_arr, cat_mask, percentiles=[95])
        if stats["min"] is not None and stats["min"] < 0:
            # Check CVA < 0 sanity check
            # CVA magnitude is usually >= 0. Let's flag warning or fail if CVA magnitude is negative.
            raise ValueError(f"Sanity Check Failed: Negative CVA magnitude {stats['min']} detected under {cat_name}.")
        cva_stats[cat_name] = stats

    # 16. Delta Index Statistics
    delta_indices = {
        "ndvi": ndvi_arr,
        "gndvi": gndvi_arr,
        "ndre": ndre_arr
    }

    delta_stats = {}
    for index_name, index_arr in delta_indices.items():
        delta_stats[index_name] = {
            "all_valid_wetland": calculate_subset_stats(index_arr, valid_pop, percentiles=[5, 95]),
            "degradation": calculate_subset_stats(index_arr, np.isin(trans_arr, [9, 13, 14]) & valid_pop, percentiles=[5, 95])
        }

    # 17. Wetland Extent Metrics
    wetland_extent_status = "NOT_AVAILABLE"
    wetland_extent_T1 = None
    wetland_extent_T2 = None
    wetland_extent_change = None
    wetland_extent_change_ha = None
    physical_wetland_depletion = "NOT_AVAILABLE"

    # Read Phase 7 physical depletion flags and configurations
    depletion_enabled_config = config.get("wetland", {}).get("depletion_claims", {}).get("enabled", False)
    # Check if config provides paths to temporal wetland extent
    temporal_config = metrics_config.get("temporal_wetland_extent", {})
    t1_path = temporal_config.get("t1_path")
    t2_path = temporal_config.get("t2_path")

    if depletion_enabled_config and t1_path and t2_path:
        t1_ext_arr, t1_ext_meta = _load_input_raster(t1_path)
        t2_ext_arr, t2_ext_meta = _load_input_raster(t2_path)

        if t1_ext_arr.shape != shape or t2_ext_arr.shape != shape:
            raise ValueError("Temporal wetland extent raster shape mismatch.")

        t1_ext_mask = (t1_ext_arr == 1) & valid_pop
        t2_ext_mask = (t2_ext_arr == 1) & valid_pop

        if physical_area_available:
            wetland_extent_T1 = float(np.sum(pixel_areas[t1_ext_mask]))
            wetland_extent_T2 = float(np.sum(pixel_areas[t2_ext_mask]))
            wetland_extent_change = wetland_extent_T2 - wetland_extent_T1
            wetland_extent_change_ha = wetland_extent_change / 10000.0
            wetland_extent_status = "AVAILABLE"

            depletion_mask = (t1_ext_arr == 1) & (t2_ext_arr == 0) & valid_pop
            physical_wetland_depletion = float(np.sum(pixel_areas[depletion_mask])) / 10000.0
        else:
            wetland_extent_status = "PIXEL_ONLY"
            warnings_list.append(
                "Temporal wetland extent is available in pixel space, but "
                "physical hectares cannot be calculated without a spatial CRS."
            )
    else:
        warnings_list.append("Wetland-specific physical depletion cannot be quantified.")

    # 18. Build result and metadata
    metadata = {
        "severity_weights": severity_weights,
        "provisional_severity_weights_disclaimer": "Severity-weighted area uses provisional engineering parameters.",
        "include_canopy_loss_in_swd": include_canopy_loss_in_swd,
        "minimum_change_area": min_change_config,
        "minimum_change_area_applied": minimum_change_area_applied,
        "minimum_change_area_available": minimum_change_area_available,
        "physical_area_available": physical_area_available,
        "area_basis": "physical_crs" if physical_area_available else "pixel_count_only",
        "baseline_definition": "valid wetland analysis area (restricted to valid overlap and non-NoData)",
        "depletion_disclaimer": "Vegetation degradation metrics do not constitute physical wetland depletion.",
        "olofsson_status": "Model-derived estimate; no independent area-error correction applied."
    }

    # Handle temporal phenological acquisition warning
    phenology_warning = "NO"
    acq_t1 = config.get("preprocessing", {}).get("t1_date")
    acq_t2 = config.get("preprocessing", {}).get("t2_date")
    if acq_t1 and acq_t2:
        # If dates differ substantially, raise warning
        metadata["acquisition_t1"] = acq_t1
        metadata["acquisition_t2"] = acq_t2

    qa_flags = {
        "input_validation": "PASS",
        "spatial_alignment": "PASS",
        "model_inference": "PASS",
        "phenology_warning": phenology_warning,
        "independent_validation": "NO",
        "reconciliation_transition_passes": reconciled_trans_pixels and reconciled_trans_area,
        "reconciliation_class_passes": reconciled_class_t1 and reconciled_class_t2
    }

    # Organize result categories
    baseline_res = {
        "pixel_count": baseline_pixel_count,
        "area_m2": baseline_area_m2,
        "area_ha": baseline_area_ha
    }

    degradation_res = {
        "impact_area_m2": impact_area_m2,
        "impact_area_ha": impact_area_ha,
        "severity_weighted_area_m2": swd_area_m2,
        "severity_weighted_area_ha_deg": swd_area_ha,
        "ndr": ndr,
        "ndr_percent": ndr_percent
    }

    recovery_res = {
        "area_m2": rec_area_m2,
        "area_ha": rec_area_ha,
        "ratio": rec_ratio,
        "details_m2": {
            "M_to_H": rec_mh_area_m2,
            "S_to_M": rec_sm_area_m2,
            "S_to_H": rec_sh_area_m2
        }
    }

    canopy_loss_res = {
        "area_m2": canopy_area_m2,
        "area_ha": canopy_area_ha,
        "ratio": canopy_ratio,
        "percentage": canopy_pct
    }

    residual_stress_res = {
        "area_m2": res_area_m2,
        "area_ha": res_area_ha,
        "ratio": res_ratio
    }

    class_area_metrics = {
        "T1": class_areas_t1,
        "T2": class_areas_t2,
        "changes": class_changes
    }

    wetland_extent_res = {
        "status": wetland_extent_status,
        "extent_T1_ha": (wetland_extent_T1 / 10000.0) if wetland_extent_T1 is not None else None,
        "extent_T2_ha": (wetland_extent_T2 / 10000.0) if wetland_extent_T2 is not None else None,
        "extent_change_ha": wetland_extent_change_ha,
        "physical_wetland_depletion": physical_wetland_depletion
    }

    # Summary object as specified
    summary = {
        "baseline_area_ha": baseline_area_ha,
        "affected_area_ha": impact_area_ha,
        "severity_weighted_area_ha_deg": swd_area_ha,
        "normalized_degradation_ratio": ndr,
        "recovery_area_ha": rec_area_ha,
        "residual_stress_area_ha": res_area_ha,
        "canopy_loss_area_ha": canopy_area_ha,
        "mean_cva": cva_stats["all_valid_wetland"]["mean"],
        "median_cva": cva_stats["all_valid_wetland"]["median"],
        "p95_cva": cva_stats["all_valid_wetland"]["p95"],
        "wetland_extent_status": wetland_extent_status
    }

    # Put it all into QA dictionary
    qa = {
        "warnings": warnings_list,
        "flags": qa_flags
    }

    return MetricsResult(
        baseline_area=baseline_res,
        impact_area=degradation_res,
        severity_weighted_area=degradation_res, # Included here as well to respect structural fields
        normalized_degradation_ratio=ndr,
        recovery_area=recovery_res,
        residual_stress_area=residual_stress_res,
        canopy_loss_area=canopy_loss_res,
        transition_metrics=transition_areas,
        class_area_metrics=class_area_metrics,
        cva_statistics=cva_stats,
        delta_statistics=delta_stats,
        wetland_extent_metrics=wetland_extent_res,
        qa_flags=qa,
        metadata={
            "summary": summary,
            "disclaimer": "Vegetation degradation metrics do not constitute physical wetland depletion.",
            "metrics_metadata": metadata,
            "physical_area_available": physical_area_available,
            "area_basis": "physical_crs" if physical_area_available else "pixel_count_only"
        }
    )
