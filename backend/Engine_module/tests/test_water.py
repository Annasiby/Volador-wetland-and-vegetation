import pytest
import numpy as np
import tempfile
import json
import os
import rasterio
from pathlib import Path
from rasterio.transform import from_origin
from rasterio.crs import CRS

from src.gis_engine.wetland.water import calculate_ndwi, classify_optical_water, analyze_hydrology

def test_ndwi_formula_exact():
    # Green = 0.6, NIR = 0.2 -> (0.6 - 0.2) / (0.6 + 0.2 + 0.0) = 0.5 (with epsilon = 0.0)
    green = np.array([0.6], dtype=np.float32)
    nir = np.array([0.2], dtype=np.float32)
    ndwi, valid = calculate_ndwi(green, nir, epsilon=0.0)
    assert np.allclose(ndwi[valid], 0.5)

def test_ndwi_equal_bands():
    green = np.array([0.4], dtype=np.float32)
    nir = np.array([0.4], dtype=np.float32)
    ndwi, valid = calculate_ndwi(green, nir, epsilon=0.0)
    assert np.allclose(ndwi[valid], 0.0)

def test_ndwi_green_greater_than_nir():
    green = np.array([0.8], dtype=np.float32)
    nir = np.array([0.3], dtype=np.float32)
    ndwi, valid = calculate_ndwi(green, nir)
    assert np.all(ndwi[valid] > 0.0)

def test_ndwi_green_less_than_nir():
    green = np.array([0.2], dtype=np.float32)
    nir = np.array([0.7], dtype=np.float32)
    ndwi, valid = calculate_ndwi(green, nir)
    assert np.all(ndwi[valid] < 0.0)

def test_ndwi_nodata_propagation():
    green = np.array([0.5, 0.6], dtype=np.float32)
    nir = np.array([0.3, 0.4], dtype=np.float32)
    nodata_mask = np.array([True, False]) # first pixel is nodata
    
    ndwi, valid = calculate_ndwi(green, nir, nodata_mask=nodata_mask)
    assert not valid[0]
    assert valid[1]
    assert ndwi[0] == -9999.0

def test_ndwi_nan_inf_handling():
    green = np.array([np.nan, 0.5, np.inf, 0.6], dtype=np.float32)
    nir = np.array([0.3, np.nan, 0.4, np.inf], dtype=np.float32)
    
    ndwi, valid = calculate_ndwi(green, nir)
    assert not valid[0]
    assert not valid[1]
    assert not valid[2]
    assert not valid[3]
    assert np.all(ndwi[~valid] == -9999.0)

def test_ndwi_divide_by_zero_protection():
    green = np.array([0.0, 0.000000001], dtype=np.float32)
    nir = np.array([0.0, 0.0], dtype=np.float32)
    
    ndwi, valid = calculate_ndwi(green, nir, epsilon=0.0)
    assert not valid[0]
    assert not valid[1]
    assert np.all(ndwi[~valid] == -9999.0)

def test_water_threshold_classification():
    green = np.array([0.6, 0.2], dtype=np.float32)
    nir = np.array([0.2, 0.6], dtype=np.float32)
    
    # Threshold 0.0
    # NDWI 1st pixel: 0.5 >= 0.0 -> water (1)
    # NDWI 2nd pixel: -0.5 < 0.0 -> non-water (0)
    ndwi, valid = calculate_ndwi(green, nir, epsilon=0.0)
    
    water_mask = np.where(ndwi >= 0.0, 1, 0)
    assert water_mask[0] == 1
    assert water_mask[1] == 0

def test_synthetic_water_transition_case():
    # T1 water state mask (0 = non-water, 1 = water)
    # [
    #     [0, 0, 1, 1],
    #     [0, 1, 1, 0]
    # ]
    # T2 water state mask (0 = non-water, 1 = water)
    # [
    #     [0, 1, 1, 0],
    #     [0, 1, 0, 0]
    # ]
    # Transitions should map:
    # 0 -> 0 : stable non-water (0)
    # 0 -> 1 : water gain (2)
    # 1 -> 0 : water loss (3)
    # 1 -> 1 : stable water (1)
    
    t1_water = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 0]
    ], dtype=np.uint8)
    
    t2_water = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 0]
    ], dtype=np.uint8)
    
    final_valid = np.ones((2, 4), dtype=bool)
    final_valid[1, 3] = False # introduce NoData element
    
    # Calculate transitions
    compact_ids = t1_water * 2 + t2_water
    public_change = np.select(
        condlist=[
            compact_ids == 0, # 0->0 stable non-water
            compact_ids == 1, # 0->1 water gain
            compact_ids == 2, # 1->0 water loss
            compact_ids == 3  # 1->1 stable water
        ],
        choicelist=[0, 2, 3, 1],
        default=255
    )
    
    # Apply valid mask
    public_change[~final_valid] = 255
    
    # Expected:
    # Row 0:
    # 0->0 -> 0
    # 0->1 -> 2
    # 1->1 -> 1
    # 1->0 -> 3
    # Row 1:
    # 0->0 -> 0
    # 1->1 -> 1
    # 1->0 -> 3
    # 0->0 -> 255 (invalid)
    
    expected = np.array([
        [0, 2, 1, 3],
        [0, 1, 3, 255]
    ], dtype=np.uint8)
    
    assert np.array_equal(public_change, expected)

def create_dummy_raster(path: Path, val: float, crs: str = "EPSG:32632", nodata: float = -9999.0):
    transform = from_origin(500000, 5000000, 10.0, 10.0)
    profile = {
        'driver': 'GTiff',
        'height': 4,
        'width': 4,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': crs,
        'transform': transform,
        'nodata': nodata
    }
    arr = np.full((4, 4), val, dtype=np.float32)
    # Add NoData value to bottom right
    arr[3, 3] = nodata
    
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)
        
    return transform, crs

def test_analyze_hydrology_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "output"
        
        t1_dir.mkdir()
        t2_dir.mkdir()
        
        # Write dummy bands
        # T1: Green = 0.6, NIR = 0.2 -> NDWI = 0.5
        # T2: Green = 0.4, NIR = 0.4 -> NDWI = 0.0
        # Valid overlap shape (4,4) with bottom right as NoData
        trans, crs = create_dummy_raster(t1_dir / "green.tif", 0.6)
        create_dummy_raster(t1_dir / "nir.tif", 0.2)
        create_dummy_raster(t2_dir / "green.tif", 0.4)
        create_dummy_raster(t2_dir / "nir.tif", 0.4)
        
        config = {
            "water": {
                "ndwi_threshold": 0.1,
                "epsilon": 0.0
            }
        }
        
        metrics = analyze_hydrology(
            t1_aligned_dir=t1_dir,
            t2_aligned_dir=t2_dir,
            output_dir=output_dir,
            config=config
        )
        
        # Verify metrics contents
        assert metrics["index"] == "NDWI"
        assert metrics["threshold"] == 0.1
        assert metrics["valid_pixel_count"] == 15
        assert metrics["nodata_pixel_count"] == 1
        
        # T1 NDWI: all valid are 0.5. Since 0.5 >= 0.1, all are water (1)
        # T2 NDWI: all valid are 0.0. Since 0.0 < 0.1, all are non-water (0)
        # This is water loss (3) for all 15 valid pixels!
        assert metrics["water_gain_m2"] == 0.0
        assert metrics["water_loss_m2"] > 0.0
        assert metrics["water_area_t1_m2"] > 0.0
        assert metrics["water_area_t2_m2"] == 0.0
        
        # Verify files were generated
        assert (output_dir / "ndwi_t1.tif").is_file()
        assert (output_dir / "ndwi_t2.tif").is_file()
        assert (output_dir / "delta_ndwi.tif").is_file()
        assert (output_dir / "water_mask_t1.tif").is_file()
        assert (output_dir / "water_mask_t2.tif").is_file()
        assert (output_dir / "water_change.tif").is_file()
        assert (output_dir / "water_metrics.json").is_file()
        
        # Validate mathematical delta calculation
        with rasterio.open(output_dir / "ndwi_t1.tif") as s_t1, \
             rasterio.open(output_dir / "ndwi_t2.tif") as s_t2, \
             rasterio.open(output_dir / "delta_ndwi.tif") as s_d:
            
            t1_ndwi_arr = s_t1.read(1)
            t2_ndwi_arr = s_t2.read(1)
            delta_ndwi_arr = s_d.read(1)
            
            valid = (t1_ndwi_arr != -9999.0) & (t2_ndwi_arr != -9999.0)
            assert np.allclose(delta_ndwi_arr[valid], t2_ndwi_arr[valid] - t1_ndwi_arr[valid])
            assert np.all(delta_ndwi_arr[~valid] == -9999.0)
            
        # Validate spatial preservation
        with rasterio.open(output_dir / "water_change.tif") as s_c:
            assert s_c.crs == CRS.from_string(crs)
            assert s_c.transform == trans
            assert s_c.width == 4
            assert s_c.height == 4
            assert s_c.nodata == 255

def test_analyze_hydrology_reproducibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir1 = tmp_path / "out1"
        output_dir2 = tmp_path / "out2"
        
        t1_dir.mkdir()
        t2_dir.mkdir()
        
        create_dummy_raster(t1_dir / "green.tif", 0.7)
        create_dummy_raster(t1_dir / "nir.tif", 0.3)
        create_dummy_raster(t2_dir / "green.tif", 0.5)
        create_dummy_raster(t2_dir / "nir.tif", 0.4)
        
        config = {
            "water": {
                "ndwi_threshold": 0.0,
                "epsilon": 1e-6
            }
        }
        
        metrics1 = analyze_hydrology(t1_dir, t2_dir, output_dir1, config)
        metrics2 = analyze_hydrology(t1_dir, t2_dir, output_dir2, config)
        
        # Verify metrics equivalence
        assert metrics1 == metrics2
        
        # Compare raster contents
        files = ["ndwi_t1.tif", "ndwi_t2.tif", "delta_ndwi.tif", "water_mask_t1.tif", "water_mask_t2.tif", "water_change.tif"]
        for f in files:
            with rasterio.open(output_dir1 / f) as s1, rasterio.open(output_dir2 / f) as s2:
                arr1 = s1.read(1)
                arr2 = s2.read(1)
                assert np.array_equal(arr1, arr2)
