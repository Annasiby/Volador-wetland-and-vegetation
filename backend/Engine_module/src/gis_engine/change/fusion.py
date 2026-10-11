from cv2 import transform
import numpy as np
import rasterio
import json
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from rasterio.crs import CRS
from rasterio.transform import Affine

from ..alignment.spatial import ReferenceGrid
from ..metrics.area import calculate_pixel_areas

# Fusion Category Definitions
FUSION_CATEGORIES = {
    0: "nodata",
    1: "stable",
    2: "vegetation_degradation",
    3: "vegetation_recovery",
    4: "surface_water_gain",
    5: "surface_water_loss",
    6: "hydro_ecological_degradation",
    7: "hydro_ecological_recovery",
    8: "vegetation_to_background",
    9: "ambiguous_mixed_evidence"
}

LEVEL_MAPPING = {
    0: "NODATA",
    1: "VEGETATION_ONLY",
    2: "SURFACE_WATER_ONLY",
    3: "HYDRO_ECOLOGICAL",
    4: "INDEPENDENT_WETLAND_SUPPORTED"
}

def _load_input_raster(val: Union[str, Path, np.ndarray], reference_grid: Optional[ReferenceGrid] = None) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
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
        meta = None
        if reference_grid is not None:
            meta = {
                "crs": reference_grid.crs,
                "transform": reference_grid.transform,
                "width": reference_grid.width,
                "height": reference_grid.height,
                "resolution": reference_grid.resolution,
                "bounds": reference_grid.bounds,
                "nodata": None
            }
        return val, meta
    else:
        raise TypeError(f"Expected str, Path, or numpy array, got {type(val)}")

def _validate_grids(metas: list) -> bool:
    """Validates that all input rasters share the exact same spatial grid geometry."""
    if not metas or all(m is None for m in metas):
        return True
    
    # Filter out None metas (e.g. from raw array inputs without grid)
    valid_metas = [m for m in metas if m is not None]
    if not valid_metas:
        return True
        
    ref = valid_metas[0]
    for m in valid_metas[1:]:
        # Exact CRS check
        if ref["crs"] != m["crs"]:
            raise ValueError(f"CRS mismatch: {ref['crs']} vs {m['crs']}")
            
        # Dimension checks
        if ref["width"] != m["width"] or ref["height"] != m["height"]:
            raise ValueError(
                f"Dimension mismatch: {ref['height']}x{ref['width']} vs {m['height']}x{m['width']}"
            )
            
        # Tolerance checks for float parameters
        if not np.allclose(ref["transform"], m["transform"], atol=1e-5):
            raise ValueError("Transform mismatch between input rasters.")
            
        if not np.allclose(ref["resolution"], m["resolution"], atol=1e-5):
            raise ValueError("Resolution mismatch between input rasters.")
            
        if not np.allclose(ref["bounds"], m["bounds"], atol=1e-5):
            raise ValueError("Extent/Bounds mismatch between input rasters.")
            
    return True

def get_fusion_lut() -> np.ndarray:
    """
    Constructs a 2D lookup table (LUT) mapping (v_trans, w_change) -> fusion_class_id.
    v_trans index range: 0-15
    w_change index range: 0-3
    """
    lut = np.zeros((256, 256), dtype=np.uint8)
    
    for v in range(16):
        for w in range(4):
            is_veg_deg = v in {9, 13, 14}
            is_veg_rec = v in {6, 7, 11}
            is_veg_bg = v in {4, 8, 12}
            
            is_w_loss = (w == 3)
            is_w_gain = (w == 2)
            is_w_stable = (w in {0, 1})
            
            if is_veg_deg and is_w_loss:
                val = 6 # Hydro-Ecological Degradation Evidence
            elif is_veg_rec and is_w_gain:
                val = 7 # Hydro-Ecological Recovery Evidence
            elif is_veg_deg and is_w_gain:
                val = 9 # Ambiguous / Mixed Evidence
            elif is_veg_rec and is_w_loss:
                val = 9 # Ambiguous / Mixed Evidence
            elif is_veg_deg and is_w_stable:
                val = 2 # Vegetation Degradation
            elif is_veg_rec and is_w_stable:
                val = 3 # Vegetation Recovery
            elif is_veg_bg:
                val = 8 # Vegetation-to-Background Change
            elif is_w_loss:
                val = 5 # Surface-Water Loss
            elif is_w_gain:
                val = 4 # Surface-Water Gain
            else:
                val = 1 # Stable
                
            lut[v, w] = val
            
    return lut

def calculate_metrics_dict(
    fusion_map: np.ndarray,
    v_trans: np.ndarray,
    w_change: np.ndarray,
    pixel_areas: Optional[np.ndarray],
    valid_mask: np.ndarray,
    wetland_mask_subset: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Calculates all areas, ratios, and support statistics for the valid subset."""
    physical_area_available = pixel_areas is not None

    def area_m2(mask: np.ndarray) -> Optional[float]:
        if not mask.any():
            return 0.0 if physical_area_available else None
        if physical_area_available:
            return float(np.sum(pixel_areas[mask]))
        return None

    def pixel_count(mask: np.ndarray) -> int:
        return int(np.sum(mask))
    if wetland_mask_subset is not None:
        sub_valid = valid_mask & wetland_mask_subset
    else:
        sub_valid = valid_mask
        
    # Total valid area
    total_pixel_count = pixel_count(sub_valid)

    if physical_area_available:
        total_area_m2 = float(np.sum(pixel_areas[sub_valid]))
        total_area_ha = total_area_m2 / 10000.0
    else:
        total_area_m2 = None
        total_area_ha = None
    
    # Vegetation Degradation (9, 13, 14)
    veg_deg_mask = sub_valid & np.isin(v_trans, [9, 13, 14])
    if physical_area_available:
        veg_deg_m2 = float(np.sum(pixel_areas[veg_deg_mask]))
        veg_deg_ha = veg_deg_m2 / 10000.0
    else:
        veg_deg_m2 = None
        veg_deg_ha = None
    
    # Vegetation Recovery (6, 7, 11)
    veg_rec_mask = sub_valid & np.isin(v_trans, [6, 7, 11])
    if physical_area_available:
        veg_rec_m2 = float(np.sum(pixel_areas[veg_rec_mask]))
        veg_rec_ha = veg_rec_m2 / 10000.0
    else:
        veg_rec_m2 = None
        veg_rec_ha = None
    
    # Water Loss (3)
    w_loss_mask = sub_valid & (w_change == 3)
    if physical_area_available:
        w_loss_m2 = float(np.sum(pixel_areas[w_loss_mask]))
        w_loss_ha = w_loss_m2 / 10000.0
    else:
        w_loss_m2 = None
        w_loss_ha = None

    # Water Gain (2)
    w_gain_mask = sub_valid & (w_change == 2)
    if physical_area_available:
        w_gain_m2 = float(np.sum(pixel_areas[w_gain_mask]))
        w_gain_ha = w_gain_m2 / 10000.0
    else:
        w_gain_m2 = None
        w_gain_ha = None
    
    # Hydro-Ecological Degradation Evidence (fusion_map == 6)
    hydro_deg_mask = sub_valid & (fusion_map == 6)
    if physical_area_available:
        hydro_deg_m2 = float(np.sum(pixel_areas[hydro_deg_mask]))
        hydro_deg_ha = hydro_deg_m2 / 10000.0
    else:
        hydro_deg_m2 = None
        hydro_deg_ha = None

    # Hydro-Ecological Recovery Evidence (fusion_map == 7)
    hydro_rec_mask = sub_valid & (fusion_map == 7)
    if physical_area_available:
        hydro_rec_m2 = float(np.sum(pixel_areas[hydro_rec_mask]))
        hydro_rec_ha = hydro_rec_m2 / 10000.0
    else:
        hydro_rec_m2 = None
        hydro_rec_ha = None
    
    # Overlap Ratios
    hydro_degradation_overlap_ratio = hydro_deg_m2 / veg_deg_m2 if veg_deg_m2 > 0 else None
    water_loss_supported_fraction = hydro_deg_m2 / w_loss_m2 if w_loss_m2 > 0 else None
    hydro_recovery_overlap_ratio = hydro_rec_m2 / veg_rec_m2 if veg_rec_m2 > 0 else None
    
    # Vegetation-to-Background Change (fusion_map == 8)
    veg_bg_mask = sub_valid & (fusion_map == 8)
    if physical_area_available:
        veg_bg_m2 = float(np.sum(pixel_areas[veg_bg_mask]))
        veg_bg_ha = veg_bg_m2 / 10000.0
    else:
        veg_bg_m2 = None
        veg_bg_ha = None
    
    # Vegetation-only vs Water-supported change
    veg_deg_with_w_loss_ha = hydro_deg_ha
    veg_deg_without_w_loss_ha = veg_deg_ha - veg_deg_with_w_loss_ha
    
    veg_rec_with_w_gain_ha = hydro_rec_ha
    veg_rec_without_w_gain_ha = veg_rec_ha - veg_rec_with_w_gain_ha
    
    return {
        "total_area_m2": total_area_m2,
        "total_area_ha": total_area_ha,
        "vegetation_degradation_area_ha": veg_deg_ha,
        "vegetation_recovery_area_ha": veg_rec_ha,
        "water_loss_area_ha": w_loss_ha,
        "water_gain_area_ha": w_gain_ha,
        "hydro_ecological_degradation_area_ha": hydro_deg_ha,
        "hydro_ecological_recovery_area_ha": hydro_rec_ha,
        "vegetation_degradation_with_water_loss_ha": veg_deg_with_w_loss_ha,
        "vegetation_degradation_without_water_loss_ha": veg_deg_without_w_loss_ha,
        "vegetation_recovery_with_water_gain_ha": veg_rec_with_w_gain_ha,
        "vegetation_recovery_without_water_gain_ha": veg_rec_without_w_gain_ha,
        "hydro_degradation_overlap_ratio": hydro_degradation_overlap_ratio,
        "hydro_recovery_overlap_ratio": hydro_recovery_overlap_ratio,
        "water_loss_supported_fraction": water_loss_supported_fraction,
        "vegetation_to_background_area_ha": veg_bg_ha
    }

def fuse_evidence(
    transition_map: Union[str, Path, np.ndarray],
    water_change: Union[str, Path, np.ndarray],
    valid_analysis_mask: Union[str, Path, np.ndarray],
    config: Dict[str, Any],
    delta_ndvi: Optional[Union[str, Path, np.ndarray]] = None,
    delta_gndvi: Optional[Union[str, Path, np.ndarray]] = None,
    delta_ndre: Optional[Union[str, Path, np.ndarray]] = None,
    cva_magnitude: Optional[Union[str, Path, np.ndarray]] = None,
    wetland_mask: Optional[Union[str, Path, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    reference_grid: Optional[ReferenceGrid] = None,
    independent_hydrological_source_accepted: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Executes the Phase 11 Additive Multi-Evidence Fusion layer.
    """
    # 1. Load data and metadata
    v_arr, v_meta = _load_input_raster(transition_map, reference_grid)
    w_arr, w_meta = _load_input_raster(water_change, reference_grid)
    mask_arr, mask_meta = _load_input_raster(valid_analysis_mask, reference_grid)
    
    d_ndvi_arr, d_ndvi_meta = (None, None) if delta_ndvi is None else _load_input_raster(delta_ndvi, reference_grid)
    wetland_arr, wetland_meta = (None, None) if wetland_mask is None else _load_input_raster(wetland_mask, reference_grid)
    
    # 2. Validate grid geometry
    _validate_grids([v_meta, w_meta, mask_meta, d_ndvi_meta, wetland_meta])
    
    # Resolve metadata to use for writing
    active_meta = next(m for m in [v_meta, w_meta, mask_meta] if m is not None)
    crs = active_meta["crs"]
    transform = active_meta["transform"]
    width = active_meta["width"]
    height = active_meta["height"]
    
    # 3. Create combined valid mask
    veg_valid = (v_arr != 255)
    hydro_valid = (w_arr != 255)
    analysis_valid = mask_arr.astype(bool)
    
    fusion_valid = veg_valid & hydro_valid & analysis_valid
    
    # 4. Perform vectorized lookup
    lut = get_fusion_lut()
    
    # Fast vectorized table lookup
    fusion_map = np.zeros((height, width), dtype=np.uint8)
    fusion_map[fusion_valid] = lut[v_arr[fusion_valid], w_arr[fusion_valid]]
    
    # 5. Physical Depletion Assessment
    fusion_config = config.get("fusion", {})
    if independent_hydrological_source_accepted is None:
        independent_hydrological_source_accepted = fusion_config.get(
            "independent_hydrological_source_accepted", False
        )
        
    depletion_status = "NOT_ASSESSED"
    if independent_hydrological_source_accepted and wetland_arr is not None:
        # Check if independent validation is supported
        # If there is a spatial wetland boundary, and water loss occurs inside it
        depletion_status = "SUPPORTED_BY_INDEPENDENT_EVIDENCE"
    elif independent_hydrological_source_accepted:
        depletion_status = "NOT_SUPPORTED" # Accepted but no independent dataset available to validate
        
    # 6. Evidence Level Assignment
    evidence_level_map = np.zeros((height, width), dtype=np.uint8)
    if fusion_valid.any():
        level_lut = np.zeros(10, dtype=np.uint8)
        # Class 2, 3, 8 -> VEGETATION_ONLY (1)
        level_lut[2] = 1
        level_lut[3] = 1
        level_lut[8] = 1
        # Class 4, 5 -> SURFACE_WATER_ONLY (2)
        level_lut[4] = 2
        level_lut[5] = 2
        # Class 6, 7 -> HYDRO_ECOLOGICAL (3) or INDEPENDENT_WETLAND_SUPPORTED (4)
        level_lut[6] = 3
        level_lut[7] = 3
        # Class 1 (stable) -> 1
        level_lut[1] = 1
        # Class 9 (ambiguous) -> 3
        level_lut[9] = 3
        
        evidence_level_map[fusion_valid] = level_lut[fusion_map[fusion_valid]]
        
        # If independent wetland is supported, promote class 6/7 inside wetland mask
        if depletion_status == "SUPPORTED_BY_INDEPENDENT_EVIDENCE" and wetland_arr is not None:
            in_wetland = (wetland_arr == 1)
            promo_mask = fusion_valid & in_wetland & np.isin(fusion_map, [6, 7])
            evidence_level_map[promo_mask] = 4
            
    # 7. CVA Consistency Check
    cva_inconsistent_map = np.zeros((height, width), dtype=np.uint8)
    spectral_transition_inconsistency = False
    
    check_cva = fusion_config.get("cva_consistency_check", True)
    if check_cva and d_ndvi_arr is not None:
        deg_mask = fusion_valid & np.isin(v_arr, [9, 13, 14])
        rec_mask = fusion_valid & np.isin(v_arr, [6, 7, 11])
        
        # Degradation transition but NDVI increased
        deg_incon = deg_mask & (d_ndvi_arr > 0.0)
        # Recovery transition but NDVI decreased
        rec_incon = rec_mask & (d_ndvi_arr < 0.0)
        
        cva_inconsistent_map[deg_incon | rec_incon] = 1
        spectral_transition_inconsistency = bool(np.any(cva_inconsistent_map == 1))
        
    # 8. Calculate physical areas
    physical_area_available = True

    try:
        pixel_areas = calculate_pixel_areas(
            transform,
            (height, width),
            crs
        )
    except Exception as exc:
        from ..metrics.area import PhysicalAreaUnavailableError

        if not isinstance(exc, PhysicalAreaUnavailableError):
            raise

        pixel_areas = None
        physical_area_available = False
    
    # Baseline analysis metrics
    metrics = calculate_metrics_dict(fusion_map, v_arr, w_arr, pixel_areas, fusion_valid)
    
    # Calculate wetland-restricted metrics
    wetland_specific_metrics_available = False
    wetland_metrics = {}
    
    # Propagate NoData wetland mask: pixels where wetland_arr is NoData (e.g. 255) are invalid in wetland subset
    wetland_valid_mask = None
    if wetland_arr is not None:
        wetland_valid_mask = (wetland_arr != 255)
        # We only calculate wetland restricted metrics if wetland mask contains valid data
        if np.any(wetland_arr == 1):
            wetland_specific_metrics_available = True
            wetland_sub = (wetland_arr == 1) & wetland_valid_mask
            wetland_metrics = calculate_metrics_dict(fusion_map, v_arr, w_arr, pixel_areas, fusion_valid, wetland_sub)
            
    # Invariant Verification (within float tolerance)
    valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    if physical_area_available:
        class_areas = [
            float(np.sum(pixel_areas[fusion_valid & (fusion_map == c)]))
            for c in valid_classes
        ]

        sum_class_areas_m2 = sum(class_areas)

        assert np.allclose(
            sum_class_areas_m2,
            metrics["total_area_m2"],
            atol=1e-2
    )
    sum_class_areas_m2 = sum(class_areas)
    
    # Asserting invariants:
    assert np.allclose(sum_class_areas_m2, metrics["total_area_m2"], atol=1e-2), "Invariant failed: Sum of class areas does not equal total valid area."
    if metrics["total_area_ha"] is not None:
        assert (
            metrics["hydro_ecological_degradation_area_ha"]
            <= metrics["vegetation_degradation_area_ha"] + 1e-5
        )

        assert (
            metrics["hydro_ecological_degradation_area_ha"]
            <= metrics["water_loss_area_ha"] + 1e-5
        )

        assert (
            metrics["hydro_ecological_recovery_area_ha"]
            <= metrics["vegetation_recovery_area_ha"] + 1e-5
        )

        assert (
            metrics["hydro_ecological_recovery_area_ha"]
            <= metrics["water_gain_area_ha"] + 1e-5
        )
    
    # 9. Format final outputs and metrics
    output_metrics = {
        "total_area_ha": metrics["total_area_ha"],
        "vegetation_degradation_area_ha": metrics["vegetation_degradation_area_ha"],
        "vegetation_recovery_area_ha": metrics["vegetation_recovery_area_ha"],
        "water_loss_area_ha": metrics["water_loss_area_ha"],
        "water_gain_area_ha": metrics["water_gain_area_ha"],
        "hydro_ecological_degradation_area_ha": metrics["hydro_ecological_degradation_area_ha"],
        "hydro_ecological_recovery_area_ha": metrics["hydro_ecological_recovery_area_ha"],
        "vegetation_degradation_with_water_loss_ha": metrics["vegetation_degradation_with_water_loss_ha"],
        "vegetation_degradation_without_corresponding_surface_water_loss_ha": metrics["vegetation_degradation_without_water_loss_ha"],
        "vegetation_recovery_with_water_gain_ha": metrics["vegetation_recovery_with_water_gain_ha"],
        "hydro_degradation_overlap_ratio": metrics["hydro_degradation_overlap_ratio"],
        "hydro_recovery_overlap_ratio": metrics["hydro_recovery_overlap_ratio"],
        "water_loss_supported_fraction": metrics["water_loss_supported_fraction"],
        "vegetation_to_background_area_ha": metrics["vegetation_to_background_area_ha"],
        "spectral_transition_inconsistency_detected": spectral_transition_inconsistency,
        "physical_depletion_assessment": depletion_status,
        "wetland_specific_metrics_available": wetland_specific_metrics_available
    }
    
    if wetland_specific_metrics_available:
        output_metrics["hydro_ecological_degradation_area_inside_wetland_ha"] = wetland_metrics["hydro_ecological_degradation_area_ha"]
        output_metrics["hydro_ecological_recovery_area_inside_wetland_ha"] = wetland_metrics["hydro_ecological_recovery_area_ha"]
        output_metrics["wetland_restricted_metrics"] = wetland_metrics
        
    # Generate CSV Evidence Matrix
    matrix_rows = []
    
    # Standard translation mapping helper
    v_trans_labels = {
        0: "B_to_B", 1: "B_to_S", 2: "B_to_M", 3: "B_to_H",
        4: "S_to_B", 5: "S_to_S", 6: "S_to_M", 7: "S_to_H",
        8: "M_to_B", 9: "M_to_S", 10: "M_to_M", 11: "M_to_H",
        12: "H_to_B", 13: "H_to_S", 14: "H_to_M", 15: "H_to_H"
    }
    w_change_labels = {
        0: "stable_non_water", 1: "stable_water", 2: "water_gain", 3: "water_loss"
    }
    
    # Calculate unique combinations in valid analysis area
    for v in range(16):
        for w in range(4):
            combo_mask = fusion_valid & (v_arr == v) & (w_arr == w)
            count = int(np.sum(combo_mask))
            if count > 0:
                area_m2 = float(np.sum(pixel_areas[combo_mask]))
                area_ha = area_m2 / 10000.0
                val_id = int(lut[v, w])
                matrix_rows.append({
                    "vegetation_transition": v_trans_labels[v],
                    "water_change": w_change_labels[w],
                    "pixel_count": count,
                    "area_m2": area_m2,
                    "area_ha": area_ha,
                    "interpretation": FUSION_CATEGORIES[val_id]
                })
                
    # 10. Save rasters and data files to disk
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Profile for writing categorical uint8 GeoTIFF maps
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.uint8,
            'crs': crs,
            'transform': transform,
            'nodata': 0,
            'compress': 'lzw'
        }
        
        # Write fusion raster
        with rasterio.open(output_dir / "evidence_fusion.tif", 'w', **profile) as dst:
            dst.write(fusion_map, 1)
            
        # Write evidence level raster
        with rasterio.open(output_dir / "evidence_level.tif", 'w', **profile) as dst:
            dst.write(evidence_level_map, 1)
            
        # Write spectral inconsistency map
        with rasterio.open(output_dir / "cva_inconsistency.tif", 'w', **profile) as dst:
            dst.write(cva_inconsistent_map, 1)
            
        # Write metrics JSON
        with open(output_dir / "fusion_metrics.json", "w", encoding="utf-8") as f:
            json.dump(output_metrics, f, indent=4)
            
        # Write Evidence Matrix CSV
        with open(output_dir / "evidence_matrix.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "vegetation_transition", "water_change", "pixel_count", "area_m2", "area_ha", "interpretation"
            ])
            writer.writeheader()
            writer.writerows(matrix_rows)
            
    return output_metrics
