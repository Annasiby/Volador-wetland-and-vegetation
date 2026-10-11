import os
import csv
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.crs import CRS

from ..alignment.spatial import ReferenceGrid
from ..metrics.area import calculate_pixel_areas

# Authoritative class contract mapping
CLASS_DESCRIPTIONS = {
    0: "Background",
    1: "Severe",
    2: "Moderate",
    3: "Healthy"
}

# Authoritative transition dictionary
TRANSITIONS = {
    0:  {"from": 0, "to": 0, "name": "B_to_B", "category": "STABLE"},
    1:  {"from": 0, "to": 1, "name": "B_to_S", "category": "BACKGROUND_TO_VEGETATION"},
    2:  {"from": 0, "to": 2, "name": "B_to_M", "category": "BACKGROUND_TO_VEGETATION"},
    3:  {"from": 0, "to": 3, "name": "B_to_H", "category": "BACKGROUND_TO_VEGETATION"},

    4:  {"from": 1, "to": 0, "name": "S_to_B", "category": "VEGETATION_TO_BACKGROUND"},
    5:  {"from": 1, "to": 1, "name": "S_to_S", "category": "STABLE"},
    6:  {"from": 1, "to": 2, "name": "S_to_M", "category": "RECOVERY"},
    7:  {"from": 1, "to": 3, "name": "S_to_H", "category": "RECOVERY"},

    8:  {"from": 2, "to": 0, "name": "M_to_B", "category": "VEGETATION_TO_BACKGROUND"},
    9:  {"from": 2, "to": 1, "name": "M_to_S", "category": "DEGRADATION"},
    10: {"from": 2, "to": 2, "name": "M_to_M", "category": "STABLE"},
    11: {"from": 2, "to": 3, "name": "M_to_H", "category": "RECOVERY"},

    12: {"from": 3, "to": 0, "name": "H_to_B", "category": "VEGETATION_TO_BACKGROUND"},
    13: {"from": 3, "to": 1, "name": "H_to_S", "category": "DEGRADATION"},
    14: {"from": 3, "to": 2, "name": "H_to_M", "category": "DEGRADATION"},
    15: {"from": 3, "to": 3, "name": "H_to_H", "category": "STABLE"}
}

@dataclass
class TransitionResult:
    """
    Container for the outputs of the temporal transition analysis.
    """
    transition_map: np.ndarray          # (H, W) uint8 map containing 0-15 or 255 (NoData)
    transition_counts: Dict[int, int]   # transition_id -> pixel count
    transition_areas_m2: Dict[int, Optional[float]]
    transition_areas_ha: Dict[int, Optional[float]]
    transition_percentages: Dict[int, float]
    
    category_counts: Dict[str, int]     # category -> pixel count
    category_areas_m2: Dict[str, Optional[float]]
    category_areas_ha: Dict[str, Optional[float]]
    category_percentages: Dict[str, float]
    
    valid_mask: np.ndarray              # Common valid-overlap mask (H, W) bool
    nodata_count: int
    reference_grid: ReferenceGrid
    metadata: Dict[str, Any]

def _load_input(val: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Optional[Any]]:
    """Helper to load a raster file path or handle an existing numpy array."""
    if isinstance(val, (str, Path)):
        path = Path(val)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        src = rasterio.open(path)
        return src.read(1), src
    elif isinstance(val, np.ndarray):
        return val, None
    else:
        raise TypeError(f"Expected str, Path, or numpy array, got {type(val)}")

def calculate_transitions(
    t1_health_map: Union[str, Path, np.ndarray],
    t2_health_map: Union[str, Path, np.ndarray],
    valid_overlap_mask: Union[str, Path, np.ndarray],
    reference_grid: Optional[ReferenceGrid] = None,
    output_dir: Optional[Union[str, Path]] = None,
    pair_name: Optional[str] = None
) -> TransitionResult:
    """
    Performs temporal post-classification transition analysis.
    
    Checks spatial alignment, validates class IDs, calculates transition IDs, 
    computes area and percentages, and saves maps, CSVs, and JSON logs.
    """
    # 1. Load data and dataset readers (if paths are provided)
    t1_arr, t1_src = _load_input(t1_health_map)
    t2_arr, t2_src = _load_input(t2_health_map)
    mask_arr, mask_src = _load_input(valid_overlap_mask)
    
    # 2. Extract and validate spatial parameters
    crs = None
    transform = None
    width = None
    height = None
    resolution = None
    bounds = None
    
    if t1_src is not None:
        crs = t1_src.crs
        transform = t1_src.transform
        width = t1_src.width
        height = t1_src.height
        resolution = t1_src.res
        bounds = t1_src.bounds
    elif reference_grid is not None:
        crs = reference_grid.crs
        transform = reference_grid.transform
        width = reference_grid.width
        height = reference_grid.height
        resolution = reference_grid.resolution
        bounds = reference_grid.bounds
    else:
        raise ValueError("Missing spatial grid reference. Provide file paths or a reference_grid object.")
        
    # Validate geometry matching
    if t1_arr.shape != t2_arr.shape:
        raise ValueError(f"Shape mismatch: T1 {t1_arr.shape} vs T2 {t2_arr.shape}")
    if mask_arr.shape != t1_arr.shape:
        raise ValueError(f"Mask shape {mask_arr.shape} does not match maps {t1_arr.shape}")
        
    if t2_src is not None:
        if t1_src.crs != t2_src.crs:
            raise ValueError(f"CRS mismatch: T1 ({t1_src.crs}) vs T2 ({t2_src.crs})")
        if not np.allclose(t1_src.transform, t2_src.transform, atol=1e-5):
            raise ValueError("Transform mismatch between T1 and T2.")
        if t1_src.width != t2_src.width or t1_src.height != t2_src.height:
            raise ValueError("Dimensions mismatch between T1 and T2.")
        if not np.allclose(t1_src.res, t2_src.res, atol=1e-5):
            raise ValueError("Resolution mismatch between T1 and T2.")
        if not np.allclose(t1_src.bounds, t2_src.bounds, atol=1e-5):
            raise ValueError("Bounds mismatch between T1 and T2.")
            
    if mask_src is not None and t1_src is not None:
        if t1_src.crs != mask_src.crs:
            raise ValueError(f"CRS mismatch: T1 ({t1_src.crs}) vs Mask ({mask_src.crs})")
        if not np.allclose(t1_src.transform, mask_src.transform, atol=1e-5):
            raise ValueError("Transform mismatch between T1 and Mask.")
            
    if reference_grid is not None and t1_src is not None:
        if reference_grid.crs != t1_src.crs:
            raise ValueError("Reference grid CRS does not match T1 CRS.")
        if not np.allclose(reference_grid.transform, t1_src.transform, atol=1e-5):
            raise ValueError("Reference grid transform does not match T1 transform.")
        if reference_grid.width != t1_src.width or reference_grid.height != t1_src.height:
            raise ValueError("Reference grid dimensions do not match T1 dimensions.")
            
    # 3. Validate class IDs
    # Expected classes: 0, 1, 2, 3 or NoData (255)
    valid_classes = {0, 1, 2, 3, 255}
    t1_unique = np.unique(t1_arr)
    t2_unique = np.unique(t2_arr)
    
    if not set(t1_unique).issubset(valid_classes):
        invalid_vals = set(t1_unique) - valid_classes
        raise ValueError(f"T1 health map contains invalid class IDs: {invalid_vals}")
    if not set(t2_unique).issubset(valid_classes):
        invalid_vals = set(t2_unique) - valid_classes
        raise ValueError(f"T2 health map contains invalid class IDs: {invalid_vals}")
        
    # Build final reference grid if not provided
    if reference_grid is None:
        reference_grid = ReferenceGrid(
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            resolution=resolution,
            bounds=bounds
        )
        
    # 4. Determine common valid overlap mask
    # A pixel is valid only if T1 is valid (not 255), T2 is valid (not 255), and overlap mask is not 0
    valid_overlap = (t1_arr != 255) & (t2_arr != 255) & (mask_arr != 0)
    
    if not np.any(valid_overlap):
        raise ValueError("The raster pair contains no valid overlapping pixels (entirely invalid/NoData).")
        
    # 5. Calculate transition map
    # Vectorized computation of transition_id = T1_class * 4 + T2_class
    transition_map = np.full(t1_arr.shape, 255, dtype=np.uint8)
    transition_map[valid_overlap] = t1_arr[valid_overlap].astype(np.uint8) * 4 + t2_arr[valid_overlap].astype(np.uint8)
    
    # 6. Area Calculation
    # Physical area is available only when the raster has an authoritative
    # spatial reference. CRS-less pixel-space data must never be treated
    # as metres or m².
    physical_area_available = True

    try:
        pixel_areas = calculate_pixel_areas(
            transform,
            t1_arr.shape,
            crs
        )
        total_valid_area_m2 = float(
            np.sum(pixel_areas[valid_overlap])
        )
    except Exception as exc:
        from ..metrics.area import PhysicalAreaUnavailableError

        if not isinstance(exc, PhysicalAreaUnavailableError):
            raise

        pixel_areas = None
        total_valid_area_m2 = None
        physical_area_available = False

    total_valid_pixels = int(np.sum(valid_overlap))
    nodata_count = int(np.sum(~valid_overlap))    
    
    # 7. Collect Statistics
    transition_counts = {}
    transition_areas_m2 = {}
    transition_areas_ha = {}
    transition_percentages = {}
    
    category_counts = {"DEGRADATION": 0, "RECOVERY": 0, "VEGETATION_TO_BACKGROUND": 0, "BACKGROUND_TO_VEGETATION": 0, "STABLE": 0}
    category_areas_m2 = {k: 0.0 for k in category_counts.keys()}
    category_areas_ha = {k: 0.0 for k in category_counts.keys()}
    category_percentages = {k: 0.0 for k in category_counts.keys()}
    
    for tid in range(16):
        match_mask = (transition_map == tid) & valid_overlap
        count = int(np.sum(match_mask))

        if physical_area_available:
            area_m2 = float(np.sum(pixel_areas[match_mask]))
            area_ha = area_m2 / 10000.0
            pct = (
                area_m2 / total_valid_area_m2 * 100.0
                if total_valid_area_m2 and total_valid_area_m2 > 0
                else 0.0
            )
        else:
            area_m2 = None
            area_ha = None
            pct = (
                count / total_valid_pixels * 100.0
                if total_valid_pixels > 0
                else 0.0
            )

        transition_counts[tid] = count
        transition_areas_m2[tid] = area_m2
        transition_areas_ha[tid] = area_ha
        transition_percentages[tid] = pct

        cat = TRANSITIONS[tid]["category"]
        category_counts[cat] += count

        if physical_area_available:
            category_areas_m2[cat] += area_m2
        
    for cat in category_counts.keys():
        if physical_area_available:
            category_areas_ha[cat] = category_areas_m2[cat] / 10000.0
            category_percentages[cat] = (
                category_areas_m2[cat] / total_valid_area_m2 * 100.0
                if total_valid_area_m2 and total_valid_area_m2 > 0
                else 0.0
            )
        else:
            category_areas_m2[cat] = None
            category_areas_ha[cat] = None
            category_percentages[cat] = (
                category_counts[cat] / total_valid_pixels * 100.0
                if total_valid_pixels > 0
                else 0.0
            )

    # Close resources if opened
    if t1_src is not None:
        t1_src.close()
    if t2_src is not None:
        t2_src.close()
    if mask_src is not None:
        mask_src.close()
        
    # 8. Write outputs if output_dir is specified
    metadata = {}
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        p_name = pair_name or out_path.name
        
        # Write GeoTIFF Transition Map
        map_path = out_path / "transition_map.tif"
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
        with rasterio.open(map_path, 'w', **profile) as dst:
            dst.write(transition_map, 1)
            # Add a color table/colormap for transition IDs 0-15 if wanted
            # Simple gray color table or categorical colormap
            colormap = {}
            # Base color assignments for 5 categories (QGIS style)
            # STABLE = Green/Gray, DEGRADATION = Red, RECOVERY = Blue, 
            # VEGETATION_TO_BACKGROUND = Brown, BACKGROUND_TO_VEGETATION = Light Green
            cat_colors = {
                "STABLE": (180, 180, 180, 255),
                "DEGRADATION": (230, 0, 0, 255),
                "RECOVERY": (0, 120, 230, 255),
                "VEGETATION_TO_BACKGROUND": (150, 100, 50, 255),
                "BACKGROUND_TO_VEGETATION": (160, 220, 160, 255)
            }
            for tid in range(16):
                colormap[tid] = cat_colors[TRANSITIONS[tid]["category"]]
            # Fill empty slots up to 256
            for idx in range(16, 256):
                colormap[idx] = (0, 0, 0, 0)
            dst.write_colormap(1, colormap)
            
        # Write Transition Matrix CSV
        csv_path = out_path / "transition_matrix.csv"
        with open(csv_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "from_class", "to_class", "transition_id", "transition_name",
                "category", "count", "area_m2", "area_ha", "percentage"
            ])
            for tid in range(16):
                t_def = TRANSITIONS[tid]
                csv_writer.writerow([
                    CLASS_DESCRIPTIONS[t_def["from"]],
                    CLASS_DESCRIPTIONS[t_def["to"]],
                    tid,
                    t_def["name"],
                    t_def["category"],
                    transition_counts[tid],
                    transition_areas_m2[tid],
                    transition_areas_ha[tid],
                    transition_percentages[tid]
                ])
                
        # Write Statistics JSON
        stats_path = out_path / "transition_statistics.json"
        stats_dict = {
            "transitions": {
                str(tid): {
                    "name": TRANSITIONS[tid]["name"],
                    "from_class": CLASS_DESCRIPTIONS[TRANSITIONS[tid]["from"]],
                    "to_class": CLASS_DESCRIPTIONS[TRANSITIONS[tid]["to"]],
                    "category": TRANSITIONS[tid]["category"],
                    "count": transition_counts[tid],
                    "area_m2": transition_areas_m2[tid],
                    "area_ha": transition_areas_ha[tid],
                    "percentage": transition_percentages[tid]
                } for tid in range(16)
            },
            "categories": {
                cat: {
                    "count": category_counts[cat],
                    "area_m2": category_areas_m2[cat],
                    "area_ha": category_areas_ha[cat],
                    "percentage": category_percentages[cat]
                } for cat in category_counts.keys()
            },
            "summary": {
                "total_valid_pixels": total_valid_pixels,
                "total_valid_area_m2": total_valid_area_m2,
                "total_valid_area_ha": (
                    total_valid_area_m2 / 10000.0
                    if physical_area_available and total_valid_area_m2 is not None
                    else None
                ),
                "nodata_pixels": nodata_count
            }
        }
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=4)
            
        # Write Metadata JSON
        meta_path = out_path / "transition_metadata.json"
        
        # Format transform list for serialization
        transform_list = [
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f
        ] if transform is not None else None
        
        metadata = {
            "pair_name": p_name,
            "t1_health_map": str(t1_health_map) if isinstance(t1_health_map, (str, Path)) else "ndarray",
            "t2_health_map": str(t2_health_map) if isinstance(t2_health_map, (str, Path)) else "ndarray",
            "reference_grid": {
                "crs": str(crs),
                "transform": transform_list,
                "width": width,
                "height": height,
                "resolution": list(resolution) if resolution is not None else None,
                "bounds": list(bounds) if bounds is not None else None
            },
            "overlap_information": {
                "valid_pixel_count": total_valid_pixels,
                "nodata_pixel_count": nodata_count,
                "overlap_area_m2": total_valid_area_m2,
                "overlap_area_ha": (
                    total_valid_area_m2 / 10000.0
                    if physical_area_available and total_valid_area_m2 is not None
                    else None
                )
            },
            "class_mapping": CLASS_DESCRIPTIONS,
            "transition_mapping": {
                str(tid): {
                    "name": TRANSITIONS[tid]["name"],
                    "from": CLASS_DESCRIPTIONS[TRANSITIONS[tid]["from"]],
                    "to": CLASS_DESCRIPTIONS[TRANSITIONS[tid]["to"]]
                } for tid in range(16)
            },
            "transition_category_definitions": {
                "DEGRADATION": ["H_to_M", "H_to_S", "M_to_S"],
                "RECOVERY": ["M_to_H", "S_to_M", "S_to_H"],
                "VEGETATION_TO_BACKGROUND": ["H_to_B", "M_to_B", "S_to_B"],
                "BACKGROUND_TO_VEGETATION": ["B_to_H", "B_to_M", "B_to_S"],
                "STABLE": ["B_to_B", "S_to_S", "M_to_M", "H_to_H"]
            },
            "physical_area_available": physical_area_available,
            "area_basis": (
                "physical_crs"
                if physical_area_available
                else "pixel_count_only"
            ),
            "total_valid_pixels": total_valid_pixels,
            "area_calculation_method": (
                "ellipsoidal quadrangle area on WGS84 ellipsoid"
                if getattr(crs, "is_geographic", False)
                else (
                    "projected grid cell area"
                    if physical_area_available
                    else "physical area unavailable; pixel counts only"
                )
            ),            "software_version": "1.0.0",
            "configuration_version": "1.0.0"
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
    return TransitionResult(
        transition_map=transition_map,
        transition_counts=transition_counts,
        transition_areas_m2=transition_areas_m2,
        transition_areas_ha=transition_areas_ha,
        transition_percentages=transition_percentages,
        category_counts=category_counts,
        category_areas_m2=category_areas_m2,
        category_areas_ha=category_areas_ha,
        category_percentages=category_percentages,
        valid_mask=valid_overlap,
        nodata_count=nodata_count,
        reference_grid=reference_grid,
        metadata=metadata
    )
