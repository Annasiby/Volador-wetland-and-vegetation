import numpy as np
import rasterio
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from gis_engine.metrics.area import PhysicalAreaUnavailableError

def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    nodata_mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Normalized Difference Water Index (NDWI).
    NDWI = (Green - NIR) / (Green + NIR + epsilon)
    
    Parameters:
        green: Green band numpy array.
        nir: NIR band numpy array.
        nodata_mask: Optional boolean mask where True indicates NoData.
        epsilon: Configurable offset to prevent divide-by-zero or instability.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - ndwi: Float32 array where invalid/NoData pixels are set to -9999.0.
            - valid_mask: Boolean mask indicating valid pixels (True = valid).
    """
    # 1. Start with initial finiteness validity
    g_valid = np.isfinite(green)
    n_valid = np.isfinite(nir)
    
    # 2. Propagate NoData from input bands
    if nodata_mask is not None:
        g_valid &= ~nodata_mask
        n_valid &= ~nodata_mask
        
    valid_mask = g_valid & n_valid
    
    # 3. Prevent invalid denominator (denominator must be positive and > 1e-8)
    denom = green + nir + epsilon
    valid_mask &= (denom > 1e-8)
    
    # Initialize output array with the standard NoData value
    ndwi = np.full(green.shape, -9999.0, dtype=np.float32)
    
    # 4. Perform division only on valid pixels
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi[valid_mask] = (green[valid_mask] - nir[valid_mask]) / denom[valid_mask]
        
    # 5. Protect against non-finite outputs in calculations
    non_finite = ~np.isfinite(ndwi) & valid_mask
    if np.any(non_finite):
        ndwi[non_finite] = -9999.0
        valid_mask[non_finite] = False
        
    # 6. Clip valid outputs to standard index bounds [-1.0, 1.0]
    ndwi[valid_mask] = np.clip(ndwi[valid_mask], -1.0, 1.0)
    
    return ndwi, valid_mask

def classify_optical_water(
    green_path: Path,
    nir_path: Path,
    threshold: float = 0.1,
    nodata: Optional[float] = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classifies water using NDWI on aligned Green and NIR band rasters.
    Preserves backward compatibility for Phase 7 fallback.
    
    Returns:
        Tuple of (water_mask, valid_mask)
    """
    with rasterio.open(green_path) as src_g, rasterio.open(nir_path) as src_n:
        green = src_g.read(1).astype(np.float32)
        nir = src_n.read(1).astype(np.float32)
        
        # Build nodata mask based on nodata value
        nodata_mask = None
        if nodata is not None:
            if np.isnan(nodata):
                nodata_mask = np.isnan(green) | np.isnan(nir)
            else:
                nodata_mask = (green == nodata) | (nir == nodata)
                
        ndwi, valid_mask = calculate_ndwi(green, nir, nodata_mask=nodata_mask)
        water_mask = (ndwi >= threshold) & valid_mask
        
    return water_mask, valid_mask

def analyze_hydrology(
    t1_aligned_dir: Path,
    t2_aligned_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    wetland_mask_path: Optional[Path] = None,
    overlap_mask_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes the Phase 10 Hydrological/Surface-Water Evidence pipeline.
    
    1. Loads aligned Green/NIR bands for T1 and T2.
    2. Calculates T1 and T2 NDWI maps using safety bounds and propagates NoData.
    3. Builds the common temporal valid mask.
    4. Classifies water pixels using the configurable NDWI threshold.
    5. Calculates transition states and maps them to the public schema:
         0 = stable non-water
         1 = stable water
         2 = water gain
         3 = water loss
         255 = invalid/NoData
    6. Calculates geospatial areas inside the analysis grid and the optional wetland mask.
    7. Exports GeoTIFF rasters and JSON metrics.
    """
    t1_aligned_dir = Path(t1_aligned_dir)
    t2_aligned_dir = Path(t2_aligned_dir)
    output_dir = Path(output_dir)
    
    # 1. Read configurations
    water_config = config.get("water", {})
    threshold = water_config.get("ndwi_threshold", 0.0)
    epsilon = water_config.get("epsilon", 1e-6)
    
    t1_green_path = t1_aligned_dir / "green.tif"
    t1_nir_path = t1_aligned_dir / "nir.tif"
    t2_green_path = t2_aligned_dir / "green.tif"
    t2_nir_path = t2_aligned_dir / "nir.tif"
    
    for path in [t1_green_path, t1_nir_path, t2_green_path, t2_nir_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required band file for Phase 10: {path}")
            
    # 2. Ingest T1 and T2 data
    with rasterio.open(t1_green_path) as src:
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
        resolution = src.res
        t1_nodata = src.nodata
        t1_green = src.read(1).astype(np.float32)
        
    with rasterio.open(t1_nir_path) as src:
        t1_nir = src.read(1).astype(np.float32)
        
    with rasterio.open(t2_green_path) as src:
        t2_nodata = src.nodata
        t2_green = src.read(1).astype(np.float32)
        
    with rasterio.open(t2_nir_path) as src:
        t2_nir = src.read(1).astype(np.float32)
        
    # 3. Create input NoData masks
    t1_nodata_mask = (t1_green == t1_nodata) | (t1_nir == t1_nodata) if t1_nodata is not None else None
    t2_nodata_mask = (t2_green == t2_nodata) | (t2_nir == t2_nodata) if t2_nodata is not None else None
    
    # 4. Calculate NDWI & valid masks
    ndwi_t1, valid_t1 = calculate_ndwi(t1_green, t1_nir, nodata_mask=t1_nodata_mask, epsilon=epsilon)
    ndwi_t2, valid_t2 = calculate_ndwi(t2_green, t2_nir, nodata_mask=t2_nodata_mask, epsilon=epsilon)
    
    # 5. Build common valid mask
    final_valid = valid_t1 & valid_t2
    
    if overlap_mask_path is not None and Path(overlap_mask_path).is_file():
        with rasterio.open(overlap_mask_path) as src:
            overlap_mask = src.read(1).astype(bool)
            final_valid &= overlap_mask
            
    # 6. Water Mask Classification (0 = non-water, 1 = water, 255 = NoData)
    water_mask_t1 = np.full((height, width), 255, dtype=np.uint8)
    water_mask_t1[final_valid] = np.where(ndwi_t1[final_valid] >= threshold, 1, 0)
    
    water_mask_t2 = np.full((height, width), 255, dtype=np.uint8)
    water_mask_t2[final_valid] = np.where(ndwi_t2[final_valid] >= threshold, 1, 0)
    
    # 7. Calculate temporal delta_ndwi
    delta_ndwi = np.full((height, width), -9999.0, dtype=np.float32)
    delta_ndwi[final_valid] = ndwi_t2[final_valid] - ndwi_t1[final_valid]
    
    # 8. Water transition classification
    water_change = np.full((height, width), 255, dtype=np.uint8)
    if np.any(final_valid):
        # Compact ID: water_t1 * 2 + water_t2
        compact_ids = water_mask_t1[final_valid].astype(np.uint8) * 2 + water_mask_t2[final_valid].astype(np.uint8)
        
        # Public mappings:
        # Internal 0 (0->0) -> Public 0 (stable non-water)
        # Internal 1 (0->1) -> Public 2 (water gain)
        # Internal 2 (1->0) -> Public 3 (water loss)
        # Internal 3 (1->1) -> Public 1 (stable water)
        public_ids = np.select(
            condlist=[
                compact_ids == 0,
                compact_ids == 1,
                compact_ids == 2,
                compact_ids == 3
            ],
            choicelist=[0, 2, 3, 1],
            default=255
        )
        water_change[final_valid] = public_ids
        
    # 9. Spatial area metrics calculations
    # Physical area is authoritative only when the raster is georeferenced.
    # CRS-less identity/pixel-space imagery must not be converted into fake m²/ha.
    from gis_engine.metrics.area import calculate_pixel_areas

    physical_area_available = True
    area_basis = "physical_area"
    pixel_areas = None

    try:
        pixel_areas = calculate_pixel_areas(
            transform, (height, width), crs
        )
    except PhysicalAreaUnavailableError:
        physical_area_available = False
        area_basis = "pixel_count"

    valid_pixel_count = int(np.sum(final_valid))
    nodata_pixel_count = int(np.sum(~final_valid))

    def get_metrics_dict(subset_mask: np.ndarray) -> Dict[str, Any]:
        sub_valid = final_valid & subset_mask

        w_t1 = sub_valid & (water_mask_t1 == 1)
        w_t2 = sub_valid & (water_mask_t2 == 1)
        w_gain = sub_valid & (water_change == 2)
        w_loss = sub_valid & (water_change == 3)

        w_t1_pixels = int(np.sum(w_t1))
        w_t2_pixels = int(np.sum(w_t2))
        w_gain_pixels = int(np.sum(w_gain))
        w_loss_pixels = int(np.sum(w_loss))

        result = {
            "water_pixel_count_t1": w_t1_pixels,
            "water_pixel_count_t2": w_t2_pixels,
            "water_gain_pixel_count": w_gain_pixels,
            "water_loss_pixel_count": w_loss_pixels,
            "area_basis": area_basis,
            "physical_area_available": physical_area_available,
            "water_area_t1_m2": None,
            "water_area_t2_m2": None,
            "water_area_t1_ha": None,
            "water_area_t2_ha": None,
            "water_gain_m2": None,
            "water_loss_m2": None,
            "water_gain_ha": None,
            "water_loss_ha": None,
            "net_water_change_m2": None,
            "net_water_change_ha": None,
        }

        if physical_area_available and pixel_areas is not None:
            w_t1_m2 = float(np.sum(pixel_areas[w_t1]))
            w_t2_m2 = float(np.sum(pixel_areas[w_t2]))
            w_gain_m2 = float(np.sum(pixel_areas[w_gain]))
            w_loss_m2 = float(np.sum(pixel_areas[w_loss]))
            net_change_m2 = w_t2_m2 - w_t1_m2

            result.update({
                "water_area_t1_m2": w_t1_m2,
                "water_area_t2_m2": w_t2_m2,
                "water_area_t1_ha": w_t1_m2 / 10000.0,
                "water_area_t2_ha": w_t2_m2 / 10000.0,
                "water_gain_m2": w_gain_m2,
                "water_loss_m2": w_loss_m2,
                "water_gain_ha": w_gain_m2 / 10000.0,
                "water_loss_ha": w_loss_m2 / 10000.0,
                "net_water_change_m2": net_change_m2,
                "net_water_change_ha": net_change_m2 / 10000.0,
            })

        return result

    # Analysis area metrics (entire valid overlap area)
    analysis_metrics = get_metrics_dict(np.ones((height, width), dtype=bool))
    
    # Optional wetland restricted metrics
    wetland_boundary_used = False
    wetland_metrics = None
    
    if wetland_mask_path is not None and Path(wetland_mask_path).is_file():
        with rasterio.open(wetland_mask_path) as src:
            wetland_mask = src.read(1)
            # 1 represents wetland
            wetland_sub = (wetland_mask == 1)
            if np.any(wetland_sub):
                wetland_boundary_used = True
                wetland_metrics = get_metrics_dict(wetland_sub)
                
    # Build complete metadata and metrics content
    metrics_content = {
        "index": "NDWI",
        "t1_date": t1_aligned_dir.parent.name.split("_")[0] if "_" in t1_aligned_dir.parent.name else "T1",
        "t2_date": t1_aligned_dir.parent.name.split("_")[1] if "_" in t1_aligned_dir.parent.name else "T2",
        "threshold": threshold,
        "epsilon": epsilon,
        "valid_pixel_count": valid_pixel_count,
        "nodata_pixel_count": nodata_pixel_count,
        "physical_area_available": physical_area_available,
        "area_basis": area_basis,
        "wetland_boundary_used": wetland_boundary_used,
        **analysis_metrics
    }
    
    if wetland_boundary_used and wetland_metrics is not None:
        metrics_content["wetland_restricted_metrics"] = wetland_metrics
        
    # 10. Write GeoTIFF and JSON outputs
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        profile_float = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.float32,
            'crs': crs,
            'transform': transform,
            'nodata': -9999.0,
            'compress': 'lzw'
        }
        
        profile_uint8 = {
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
        
        with rasterio.open(output_dir / "ndwi_t1.tif", 'w', **profile_float) as dst:
            dst.write(ndwi_t1, 1)
        with rasterio.open(output_dir / "ndwi_t2.tif", 'w', **profile_float) as dst:
            dst.write(ndwi_t2, 1)
        with rasterio.open(output_dir / "delta_ndwi.tif", 'w', **profile_float) as dst:
            dst.write(delta_ndwi, 1)
            
        with rasterio.open(output_dir / "water_mask_t1.tif", 'w', **profile_uint8) as dst:
            dst.write(water_mask_t1, 1)
        with rasterio.open(output_dir / "water_mask_t2.tif", 'w', **profile_uint8) as dst:
            dst.write(water_mask_t2, 1)
        with rasterio.open(output_dir / "water_change.tif", 'w', **profile_uint8) as dst:
            dst.write(water_change, 1)
            
        with open(output_dir / "water_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_content, f, indent=4)
            
    return metrics_content
