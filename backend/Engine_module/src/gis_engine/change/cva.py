import os
import json
import math
import numpy as np
import rasterio
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union, List

from ..alignment.spatial import ReferenceGrid, SpatialAligner
from ..ingestion.metadata import RasterMetadata

@dataclass
class CVAResult:
    """
    Container for the outputs of Change Vector Analysis.
    """
    delta_ndvi: np.ndarray
    delta_gndvi: np.ndarray
    delta_ndre: np.ndarray
    cva_magnitude: np.ndarray
    statistics: Dict[str, Any]
    valid_mask: np.ndarray
    metadata: Dict[str, Any]
    cva_standardized: Optional[np.ndarray] = None

def _load_input_indices(
    indices_input: Union[str, Path, Dict[str, Union[str, Path, np.ndarray]]],
    config: Optional[Dict[str, Any]] = None,
    default_grid: Optional[ReferenceGrid] = None
) -> Tuple[Dict[str, np.ndarray], RasterMetadata]:
    """
    Loads index rasters from a dictionary, a directory of precomputed indices,
    or a directory of raw bands (by computing them dynamically).
    Returns a dictionary of loaded index arrays and spatial metadata.
    """
    indices = {}
    representative_meta = None

    # 1. Resolve dict input
    if isinstance(indices_input, dict):
        # Standardize keys to lowercase
        input_dict = {k.lower(): v for k, v in indices_input.items()}
        for key in ["ndvi", "gndvi", "ndre"]:
            if key not in input_dict:
                raise KeyError(f"Missing required CVA index key: '{key}' in input dictionary.")
            
            val = input_dict[key]
            if isinstance(val, (str, Path)):
                path = Path(val)
                if not path.is_file():
                    raise FileNotFoundError(f"Index file not found: {path}")
                with rasterio.open(path) as src:
                    indices[key] = src.read(1).astype(np.float32)
                    if representative_meta is None:
                        representative_meta = RasterMetadata(
                            crs=src.crs,
                            transform=src.transform,
                            width=src.width,
                            height=src.height,
                            resolution=src.res,
                            extent=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                            nodata=src.nodata
                        )
            elif isinstance(val, np.ndarray):
                indices[key] = val.astype(np.float32)
            else:
                raise TypeError(f"Expected str, Path, or numpy array for key '{key}', got {type(val)}")
                
        # If all arrays were numpy arrays, build metadata from default_grid
        if representative_meta is None:
            if default_grid is not None:
                representative_meta = RasterMetadata(
                    crs=default_grid.crs,
                    transform=default_grid.transform,
                    width=default_grid.width,
                    height=default_grid.height,
                    resolution=default_grid.resolution,
                    extent=default_grid.bounds,
                    nodata=None
                )
            else:
                raise ValueError("Missing spatial grid reference. Provide file paths or a reference_grid object.")
                
        return indices, representative_meta

    # 2. Resolve Path or string input
    elif isinstance(indices_input, (str, Path)):
        path = Path(indices_input)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
            
        if path.is_dir():
            # Check if it contains pre-calculated indices (e.g., *ndvi.tif etc.)
            index_files = {}
            for name in ["ndvi", "gndvi", "ndre"]:
                matches = (
                    list(path.glob(f"*{name}*.tif")) + 
                    list(path.glob(f"*{name.upper()}*.tif")) + 
                    list(path.glob(f"*{name}*.TIF")) + 
                    list(path.glob(f"*{name.upper()}*.TIF"))
                )
                if matches:
                    index_files[name] = matches[0]
            
            if len(index_files) == 3:
                # Load pre-calculated indices
                for key in ["ndvi", "gndvi", "ndre"]:
                    filepath = index_files[key]
                    with rasterio.open(filepath) as src:
                        indices[key] = src.read(1).astype(np.float32)
                        if representative_meta is None:
                            representative_meta = RasterMetadata(
                                crs=src.crs,
                                transform=src.transform,
                                width=src.width,
                                height=src.height,
                                resolution=src.res,
                                extent=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                                nodata=src.nodata
                            )
                return indices, representative_meta
            else:
                # Load raw bands and compute indices dynamically
                if config is None:
                    # Load default config
                    default_config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
                    if default_config_path.is_file():
                        with open(default_config_path, "r") as f:
                            config = json.load(f)
                    else:
                        raise ValueError(
                            f"Configuration dictionary or config.json is required to process raw bands in directory {path}."
                        )
                
                from ..ingestion.loader import RasterLoader
                from ..preprocessing.bands import BandProcessor
                from ..preprocessing.indices import VegetationIndexProcessor
                
                band_mapping = config["ingestion"]["band_mapping"]
                loader = RasterLoader(band_mapping)
                bands_data, representative_meta = loader.load_bands(path)
                
                band_processor = BandProcessor(
                    scaling_factor=config["preprocessing"]["band_scaling_factor"]
                )
                index_processor = VegetationIndexProcessor(
                    mode=config["preprocessing"]["mode"],
                    epsilon=config["preprocessing"]["epsilon"],
                    min_denominator=config["preprocessing"]["min_denominator"],
                    min_band_value=config["preprocessing"]["min_band_value"],
                    gaussian_sigma=config["preprocessing"]["gaussian_sigma"]
                )
                
                preprocessed_bands = band_processor.process_bands(bands_data)
                computed = index_processor.compute_indices(preprocessed_bands)
                
                indices = {k.lower(): v for k, v in computed.items()}
                return indices, representative_meta
        else:
            raise ValueError(f"Input path must be a directory containing index/band files, got a file: {path}")
            
    else:
        raise TypeError(f"Expected str, Path, or dict, got {type(indices_input)}")

def _load_mask(mask_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Loads valid overlap mask as a numpy array."""
    if isinstance(mask_input, (str, Path)):
        path = Path(mask_input)
        if not path.is_file():
            raise FileNotFoundError(f"Mask file not found: {path}")
        with rasterio.open(path) as src:
            return src.read(1)
    elif isinstance(mask_input, np.ndarray):
        return mask_input
    else:
        raise TypeError(f"Expected str, Path, or numpy array for mask, got {type(mask_input)}")

def _calculate_array_stats(arr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Calculates statistics over valid masked pixels."""
    valid_data = arr[mask]
    if len(valid_data) == 0:
        return {
            "count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0
        }
    return {
        "count": int(len(valid_data)),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "median": float(np.median(valid_data)),
        "std": float(np.std(valid_data)),
        "p95": float(np.percentile(valid_data, 95))
    }

def _calculate_delta_stats(arr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Calculates directional and magnitude stats for a delta index over valid pixels."""
    valid_data = arr[mask]
    if len(valid_data) == 0:
        return {
            "count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0,
            "pct_positive": 0.0, "pct_negative": 0.0, "pct_zero": 0.0
        }
    count = len(valid_data)
    pos_count = np.sum(valid_data > 1e-8)
    neg_count = np.sum(valid_data < -1e-8)
    zero_count = np.sum(np.abs(valid_data) <= 1e-8)
    
    return {
        "count": int(count),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "median": float(np.median(valid_data)),
        "std": float(np.std(valid_data)),
        "p95": float(np.percentile(valid_data, 95)),
        "pct_positive": float((pos_count / count) * 100.0),
        "pct_negative": float((neg_count / count) * 100.0),
        "pct_zero": float((zero_count / count) * 100.0)
    }

def calculate_cva(
    t1_indices: Union[str, Path, Dict[str, Union[str, Path, np.ndarray]]],
    t2_indices: Union[str, Path, Dict[str, Union[str, Path, np.ndarray]]],
    valid_overlap_mask: Union[str, Path, np.ndarray],
    reference_grid: Optional[ReferenceGrid] = None,
    output_dir: Optional[Union[str, Path]] = None,
    pair_name: Optional[str] = None,
    transition_map: Optional[Union[str, Path, np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None
) -> CVAResult:
    """
    Performs Change Vector Analysis (CVA) between T1 and T2 vegetation indices.
    
    Validates geometries, masks invalid pixels, computes raw and standardized CVA 
    magnitudes + individual deltas, calculates descriptive stats, and saves GeoTIFF + JSON results.
    """
    # 1. Load indices and masks
    t1_data, t1_meta = _load_input_indices(t1_indices, config, reference_grid)
    t2_data, t2_meta = _load_input_indices(t2_indices, config, reference_grid)
    mask_arr = _load_mask(valid_overlap_mask)

    # 2. Extract and validate spatial parameters
    crs = None
    transform = None
    width = None
    height = None
    resolution = None
    bounds = None

    if t1_meta is not None:
        crs = t1_meta.crs
        transform = t1_meta.transform
        width = t1_meta.width
        height = t1_meta.height
        resolution = t1_meta.resolution
        bounds = t1_meta.extent
    elif reference_grid is not None:
        crs = reference_grid.crs
        transform = reference_grid.transform
        width = reference_grid.width
        height = reference_grid.height
        resolution = reference_grid.resolution
        bounds = reference_grid.bounds
    else:
        raise ValueError("Missing spatial grid reference. Provide file paths or a reference_grid object.")

    # Spatial contract check: T1 vs T2
    if t2_meta is not None:
        if str(t1_meta.crs) != str(t2_meta.crs):
            raise ValueError(f"CRS mismatch: T1 ({t1_meta.crs}) vs T2 ({t2_meta.crs})")
        if not np.allclose(t1_meta.transform, t2_meta.transform, atol=1e-5):
            raise ValueError("Transform mismatch between T1 and T2.")
        if t1_meta.width != t2_meta.width or t1_meta.height != t2_meta.height:
            raise ValueError("Dimensions mismatch between T1 and T2.")
        if not np.allclose(t1_meta.resolution, t2_meta.resolution, atol=1e-5):
            raise ValueError("Resolution mismatch between T1 and T2.")
        if not np.allclose(t1_meta.extent, t2_meta.extent, atol=1e-5):
            raise ValueError("Bounds mismatch between T1 and T2.")

    # Validate shapes of all index arrays and mismatch between T1 and T2
    t1_shape = t1_data["ndvi"].shape
    t2_shape = t2_data["ndvi"].shape
    if t1_shape != t2_shape:
        raise ValueError(f"Dimensions mismatch between T1 shape {t1_shape} and T2 shape {t2_shape}")
        
    for key in ["ndvi", "gndvi", "ndre"]:
        if t1_data[key].shape != t1_shape:
            raise ValueError(f"T1 index '{key}' shape {t1_data[key].shape} does not match ndvi shape {t1_shape}")
        if t2_data[key].shape != t2_shape:
            raise ValueError(f"T2 index '{key}' shape {t2_data[key].shape} does not match ndvi shape {t2_shape}")

    # Validate mask matches T1 dimensions
    ndvi_shape = t1_shape
    if mask_arr.shape != ndvi_shape:
        raise ValueError(f"Mask shape {mask_arr.shape} does not match indices shape {ndvi_shape}")

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

    # 3. Establish pixel-wise validity mask
    # Pixel is valid if: mask is not 0 AND index values are not NoData/NaN
    combined_valid = (mask_arr != 0)

    # Filter out NaNs or NoData
    for key in ["ndvi", "gndvi", "ndre"]:
        # T1 indices checks
        combined_valid &= ~np.isnan(t1_data[key])
        if t1_meta.nodata is not None:
            if not np.isnan(t1_meta.nodata):
                combined_valid &= (t1_data[key] != t1_meta.nodata)
        
        # T2 indices checks
        combined_valid &= ~np.isnan(t2_data[key])
        if t2_meta.nodata is not None:
            if not np.isnan(t2_meta.nodata):
                combined_valid &= (t2_data[key] != t2_meta.nodata)

    if not np.any(combined_valid):
        raise ValueError("The raster pair contains no valid overlapping pixels (entirely invalid/NoData).")

    # 4. Calculate raw deltas: positive is T2 > T1
    delta_ndvi = np.full(ndvi_shape, -9999.0, dtype=np.float32)
    delta_gndvi = np.full(ndvi_shape, -9999.0, dtype=np.float32)
    delta_ndre = np.full(ndvi_shape, -9999.0, dtype=np.float32)

    delta_ndvi[combined_valid] = t2_data["ndvi"][combined_valid] - t1_data["ndvi"][combined_valid]
    delta_gndvi[combined_valid] = t2_data["gndvi"][combined_valid] - t1_data["gndvi"][combined_valid]
    delta_ndre[combined_valid] = t2_data["ndre"][combined_valid] - t1_data["ndre"][combined_valid]

    # 5. Compute Raw CVA (Euclidean magnitude)
    cva_magnitude = np.full(ndvi_shape, -9999.0, dtype=np.float32)
    cva_magnitude[combined_valid] = np.sqrt(
        delta_ndvi[combined_valid]**2 +
        delta_gndvi[combined_valid]**2 +
        delta_ndre[combined_valid]**2
    )

    # 6. Compute Standardized CVA
    # Calculate scale/std over valid pixels
    std_ndvi = float(np.std(delta_ndvi[combined_valid]))
    std_gndvi = float(np.std(delta_gndvi[combined_valid]))
    std_ndre = float(np.std(delta_ndre[combined_valid]))

    z_ndvi = np.zeros(ndvi_shape, dtype=np.float32)
    z_gndvi = np.zeros(ndvi_shape, dtype=np.float32)
    z_ndre = np.zeros(ndvi_shape, dtype=np.float32)

    if std_ndvi > 1e-8:
        z_ndvi[combined_valid] = delta_ndvi[combined_valid] / std_ndvi
    if std_gndvi > 1e-8:
        z_gndvi[combined_valid] = delta_gndvi[combined_valid] / std_gndvi
    if std_ndre > 1e-8:
        z_ndre[combined_valid] = delta_ndre[combined_valid] / std_ndre

    cva_standardized = np.full(ndvi_shape, -9999.0, dtype=np.float32)
    cva_standardized[combined_valid] = np.sqrt(
        z_ndvi[combined_valid]**2 +
        z_gndvi[combined_valid]**2 +
        z_ndre[combined_valid]**2
    )

    # 7. Collect Statistics over valid pixels
    stats = {
        "delta_ndvi": _calculate_delta_stats(delta_ndvi, combined_valid),
        "delta_gndvi": _calculate_delta_stats(delta_gndvi, combined_valid),
        "delta_ndre": _calculate_delta_stats(delta_ndre, combined_valid),
        "cva_magnitude": _calculate_array_stats(cva_magnitude, combined_valid),
        "cva_standardized": _calculate_array_stats(cva_standardized, combined_valid)
    }

    # 8. Phase 4 integration (Transition Map statistics)
    transition_stats = {}
    if transition_map is not None:
        trans_arr = _load_mask(transition_map)
        if trans_arr.shape == ndvi_shape:
            # Map transition ID ranges to categories
            TRANSITION_CATEGORIES = {
                "DEGRADATION": [9, 13, 14],
                "RECOVERY": [6, 7, 11],
                "VEGETATION_TO_BACKGROUND": [4, 8, 12],
                "BACKGROUND_TO_VEGETATION": [1, 2, 3],
                "STABLE": [0, 5, 10, 15]
            }
            for cat_name, t_ids in TRANSITION_CATEGORIES.items():
                cat_mask = np.isin(trans_arr, t_ids) & combined_valid
                if np.any(cat_mask):
                    cat_cva = cva_magnitude[cat_mask]
                    transition_stats[cat_name] = {
                        "pixel_count": int(np.sum(cat_mask)),
                        "min": float(np.min(cat_cva)),
                        "max": float(np.max(cat_cva)),
                        "mean": float(np.mean(cat_cva)),
                        "median": float(np.median(cat_cva)),
                        "std": float(np.std(cat_cva)),
                        "p95": float(np.percentile(cat_cva, 95))
                    }
                else:
                    transition_stats[cat_name] = {
                        "pixel_count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0
                    }

    # 9. Mathematical Reconciliation check
    # CVA^2 = delta_ndvi^2 + delta_gndvi^2 + delta_ndre^2
    valid_cva = cva_magnitude[combined_valid]
    expected_cva_sq = (
        delta_ndvi[combined_valid]**2 +
        delta_gndvi[combined_valid]**2 +
        delta_ndre[combined_valid]**2
    )
    recon_diff = np.abs(valid_cva**2 - expected_cva_sq)
    max_recon_err = float(np.max(recon_diff))
    mean_recon_err = float(np.mean(recon_diff))

    # Affine transform serialization
    transform_list = [
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f
    ] if transform is not None else None

    # Build metadata dictionary (always populated)
    metadata = {
        "pair_name": pair_name or (Path(output_dir).name if output_dir is not None else "cva_pair"),
        "t1_source": str(t1_indices) if isinstance(t1_indices, (str, Path)) else "dict/ndarray",
        "t2_source": str(t2_indices) if isinstance(t2_indices, (str, Path)) else "dict/ndarray",
        "feature_vector": ["NDVI", "GNDVI", "NDRE"],
        "delta_definition": "T2 - T1 (increase is positive, decrease is negative)",
        "cva_formula": "sqrt(delta_ndvi^2 + delta_gndvi^2 + delta_ndre^2)",
        "standardization": {
            "method": "z-score divide by index standard deviation",
            "scales": {
                "std_ndvi": std_ndvi,
                "std_gndvi": std_gndvi,
                "std_ndre": std_ndre
            }
        },
        "mathematical_reconciliation": {
            "max_absolute_error_cva_squared": max_recon_err,
            "mean_absolute_error_cva_squared": mean_recon_err
        },
        "spatial_metadata": {
            "crs": str(crs) if crs is not None else None,
            "transform": transform_list,
            "width": width,
            "height": height,
            "resolution": list(resolution) if resolution is not None else None,
            "bounds": list(bounds) if bounds is not None else None
        },
        "overlap_information": {
            "valid_pixel_count": int(np.sum(combined_valid)),
            "nodata_pixel_count": int(np.sum(~combined_valid)),
            "nodata_value": -9999.0
        },
        "statistics": stats,
        "software_version": "1.0.0"
    }

    if transition_stats:
        metadata["transition_map_statistics"] = transition_stats

    # 10. Write outputs if output_dir is specified
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        profile = {
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

        # Save rasters
        with rasterio.open(out_path / "cva_magnitude.tif", 'w', **profile) as dst:
            dst.write(cva_magnitude, 1)
        with rasterio.open(out_path / "delta_ndvi.tif", 'w', **profile) as dst:
            dst.write(delta_ndvi, 1)
        with rasterio.open(out_path / "delta_gndvi.tif", 'w', **profile) as dst:
            dst.write(delta_gndvi, 1)
        with rasterio.open(out_path / "delta_ndre.tif", 'w', **profile) as dst:
            dst.write(delta_ndre, 1)
        with rasterio.open(out_path / "cva_standardized.tif", 'w', **profile) as dst:
            dst.write(cva_standardized, 1)

        with open(out_path / "cva_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    return CVAResult(
        delta_ndvi=delta_ndvi,
        delta_gndvi=delta_gndvi,
        delta_ndre=delta_ndre,
        cva_magnitude=cva_magnitude,
        cva_standardized=cva_standardized,
        statistics=stats,
        valid_mask=combined_valid,
        metadata=metadata
    )
