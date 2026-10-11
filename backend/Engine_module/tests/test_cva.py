import pytest
import numpy as np
import tempfile
import json
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from pathlib import Path

from src.gis_engine.change.cva import calculate_cva
from src.gis_engine.alignment.spatial import ReferenceGrid

def get_dummy_grid(shape=(4, 4), is_geographic=False):
    """Helper to construct a dummy ReferenceGrid."""
    height, width = shape
    if is_geographic:
        crs = CRS.from_epsg(4326)
        transform = from_origin(10.0, 50.0, 0.0001, 0.0001)
        resolution = (0.0001, 0.0001)
        bounds = (10.0, 50.0 - height * 0.0001, 10.0 + width * 0.0001, 50.0)
    else:
        crs = CRS.from_epsg(32632)  # UTM 32N
        transform = from_origin(500000, 5000000, 10.0, 10.0)
        resolution = (10.0, 10.0)
        bounds = (500000, 5000000 - height * 10.0, 500000 + width * 10.0, 5000000)
        
    return ReferenceGrid(
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        resolution=resolution,
        bounds=bounds
    )

def test_cva_001_zero_change():
    """TEST-CVA-001: Zero change. T1 == T2 -> CVA = 0."""
    t1 = {"ndvi": np.full((3, 3), 0.5), "gndvi": np.full((3, 3), 0.6), "ndre": np.full((3, 3), 0.4)}
    t2 = {"ndvi": np.full((3, 3), 0.5), "gndvi": np.full((3, 3), 0.6), "ndre": np.full((3, 3), 0.4)}
    mask = np.ones((3, 3), dtype=bool)
    grid = get_dummy_grid(shape=(3, 3))
    
    result = calculate_cva(t1, t2, mask, grid)
    assert np.allclose(result.delta_ndvi, 0.0)
    assert np.allclose(result.delta_gndvi, 0.0)
    assert np.allclose(result.delta_ndre, 0.0)
    assert np.allclose(result.cva_magnitude, 0.0)

def test_cva_002_only_ndvi_changes():
    """TEST-CVA-002: Only NDVI changes. Delta = [1, 0, 0] -> CVA = 1."""
    t1 = {"ndvi": np.full((2, 2), 0.0), "gndvi": np.full((2, 2), 0.5), "ndre": np.full((2, 2), 0.5)}
    t2 = {"ndvi": np.full((2, 2), 1.0), "gndvi": np.full((2, 2), 0.5), "ndre": np.full((2, 2), 0.5)}
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    assert np.allclose(result.delta_ndvi, 1.0)
    assert np.allclose(result.delta_gndvi, 0.0)
    assert np.allclose(result.delta_ndre, 0.0)
    assert np.allclose(result.cva_magnitude, 1.0)

def test_cva_003_known_vector():
    """TEST-CVA-003: Known vector. Delta = [1, 2, 2] -> CVA = 3."""
    t1 = {"ndvi": np.full((2, 2), 0.0), "gndvi": np.full((2, 2), 0.0), "ndre": np.full((2, 2), 0.0)}
    t2 = {"ndvi": np.full((2, 2), 1.0), "gndvi": np.full((2, 2), 2.0), "ndre": np.full((2, 2), 2.0)}
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    assert np.allclose(result.cva_magnitude, 3.0)

def test_cva_004_negative_vector():
    """TEST-CVA-004: Negative vector. Delta = [-1, -2, -2] -> CVA = 3."""
    t1 = {"ndvi": np.full((2, 2), 1.0), "gndvi": np.full((2, 2), 2.0), "ndre": np.full((2, 2), 2.0)}
    t2 = {"ndvi": np.full((2, 2), 0.0), "gndvi": np.full((2, 2), 0.0), "ndre": np.full((2, 2), 0.0)}
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    assert np.allclose(result.delta_ndvi, -1.0)
    assert np.allclose(result.delta_gndvi, -2.0)
    assert np.allclose(result.delta_ndre, -2.0)
    assert np.allclose(result.cva_magnitude, 3.0)

def test_cva_005_mixed_direction():
    """TEST-CVA-005: Mixed direction. Delta = [1, -2, 2] -> CVA = 3."""
    t1 = {"ndvi": np.full((2, 2), 0.0), "gndvi": np.full((2, 2), 2.0), "ndre": np.full((2, 2), 0.0)}
    t2 = {"ndvi": np.full((2, 2), 1.0), "gndvi": np.full((2, 2), 0.0), "ndre": np.full((2, 2), 2.0)}
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    assert np.allclose(result.cva_magnitude, 3.0)

def test_cva_006_nodata():
    """TEST-CVA-006: NoData. Invalid pixel must remain NoData (-9999.0). It must NOT become zero."""
    t1 = {"ndvi": np.array([[0.5, 0.5], [0.5, 0.5]]), "gndvi": np.array([[0.6, 0.6], [0.6, 0.6]]), "ndre": np.array([[0.4, 0.4], [0.4, 0.4]])}
    t2 = {"ndvi": np.array([[0.8, 0.8], [0.8, 0.8]]), "gndvi": np.array([[0.9, 0.9], [0.9, 0.9]]), "ndre": np.array([[0.7, 0.7], [0.7, 0.7]])}
    
    # Bottom-right pixel is invalid (0) in overlap mask
    mask = np.array([[1, 1], [1, 0]], dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    assert result.delta_ndvi[1, 1] == -9999.0
    assert result.cva_magnitude[1, 1] == -9999.0
    assert result.delta_ndvi[0, 0] != -9999.0
    assert result.cva_magnitude[0, 0] != -9999.0

def test_cva_007_valid_zero():
    """TEST-CVA-007: Valid zero. A valid unchanged pixel must have CVA = 0 and remain valid."""
    t1 = {"ndvi": np.array([[0.5, 0.5], [0.5, 0.5]]), "gndvi": np.array([[0.6, 0.6], [0.6, 0.6]]), "ndre": np.array([[0.4, 0.4], [0.4, 0.4]])}
    t2 = {"ndvi": np.array([[0.5, 0.8], [0.5, 0.8]]), "gndvi": np.array([[0.6, 0.9], [0.6, 0.9]]), "ndre": np.array([[0.4, 0.7], [0.4, 0.7]])}
    # Left column is unchanged (T1 == T2), right column changed. All valid.
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    assert result.cva_magnitude[0, 0] == 0.0
    assert result.cva_magnitude[1, 0] == 0.0
    assert result.cva_magnitude[0, 1] > 0.0

def test_cva_008_standardized_cva():
    """TEST-CVA-008: Standardized CVA. Verify scale calculation and standardized magnitude on synthetic dataset."""
    # Standard deviation calculation:
    # Let delta_ndvi = [1.0, 3.0] -> mean = 2.0, variance = 1.0, std = 1.0
    # Let delta_gndvi = [2.0, 6.0] -> mean = 4.0, variance = 4.0, std = 2.0
    # Let delta_ndre = [3.0, 9.0] -> mean = 6.0, variance = 9.0, std = 3.0
    # z_ndvi = delta_ndvi / std_ndvi = [1.0, 3.0] / 1.0 = [1.0, 3.0]
    # z_gndvi = delta_gndvi / std_gndvi = [2.0, 6.0] / 2.0 = [1.0, 3.0]
    # z_ndre = delta_ndre / std_ndre = [3.0, 9.0] / 3.0 = [1.0, 3.0]
    # cva_standardized = sqrt(z_ndvi^2 + z_gndvi^2 + z_ndre^2)
    # At pixel 0: sqrt(1^2 + 1^2 + 1^2) = sqrt(3) = 1.73205
    # At pixel 1: sqrt(3^2 + 3^2 + 3^2) = sqrt(27) = 5.19615
    t1 = {"ndvi": np.array([0.0, 0.0]), "gndvi": np.array([0.0, 0.0]), "ndre": np.array([0.0, 0.0])}
    t2 = {"ndvi": np.array([1.0, 3.0]), "gndvi": np.array([2.0, 6.0]), "ndre": np.array([3.0, 9.0])}
    mask = np.ones((2,), dtype=bool)
    grid = get_dummy_grid(shape=(1, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    assert np.allclose(result.cva_standardized, [np.sqrt(3), np.sqrt(27)])
    assert np.allclose(result.metadata["standardization"]["scales"]["std_ndvi"], 1.0)
    assert np.allclose(result.metadata["standardization"]["scales"]["std_gndvi"], 2.0)
    assert np.allclose(result.metadata["standardization"]["scales"]["std_ndre"], 3.0)

def test_cva_009_zero_standard_deviation():
    """TEST-CVA-009: Zero standard deviation. Verify no NaN or Inf is produced."""
    t1 = {"ndvi": np.full((2, 2), 0.0), "gndvi": np.full((2, 2), 0.0), "ndre": np.full((2, 2), 0.0)}
    t2 = {"ndvi": np.full((2, 2), 1.0), "gndvi": np.full((2, 2), 1.0), "ndre": np.full((2, 2), 1.0)}
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    # std of all deltas is 0 because all pixels have delta = 1.0
    # Thus z-scores must be set to 0.0, resulting in standardized CVA = 0.0.
    assert not np.any(np.isnan(result.cva_standardized))
    assert not np.any(np.isinf(result.cva_standardized))
    assert np.allclose(result.cva_standardized[result.valid_mask], 0.0)

def test_cva_010_geometry_mismatch():
    """TEST-CVA-010: Geometry mismatch (shape, CRS, bounds). Expected: clear failure."""
    t1 = {"ndvi": np.ones((2, 2)), "gndvi": np.ones((2, 2)), "ndre": np.ones((2, 2))}
    t2 = {"ndvi": np.ones((3, 3)), "gndvi": np.ones((3, 3)), "ndre": np.ones((3, 3))}
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    # Shape mismatch between T1 and T2
    with pytest.raises(ValueError, match="Dimensions mismatch"):
        calculate_cva(t1, t2, mask, grid)

def test_cva_011_mask_mismatch():
    """TEST-CVA-011: Mask mismatch. Expected: clear failure."""
    t1 = {"ndvi": np.ones((2, 2)), "gndvi": np.ones((2, 2)), "ndre": np.ones((2, 2))}
    t2 = {"ndvi": np.ones((2, 2)), "gndvi": np.ones((2, 2)), "ndre": np.ones((2, 2))}
    mask = np.ones((3, 3), dtype=bool)  # Mismatch mask shape
    grid = get_dummy_grid(shape=(2, 2))
    
    with pytest.raises(ValueError, match="Mask shape"):
        calculate_cva(t1, t2, mask, grid)

def test_cva_012_statistics():
    """TEST-CVA-012: Statistics. Verify min, max, mean, median, std, P95."""
    t1 = {"ndvi": np.array([0.0, 0.0]), "gndvi": np.array([0.0, 0.0]), "ndre": np.array([0.0, 0.0])}
    t2 = {"ndvi": np.array([1.0, 2.0]), "gndvi": np.array([0.0, 0.0]), "ndre": np.array([0.0, 0.0])}
    mask = np.ones((2,), dtype=bool)
    grid = get_dummy_grid(shape=(1, 2))
    
    result = calculate_cva(t1, t2, mask, grid)
    # For delta_ndvi: values are 1.0, 2.0.
    # min=1.0, max=2.0, mean=1.5, median=1.5, std=0.5, p95=1.95
    stats = result.statistics["delta_ndvi"]
    assert stats["min"] == 1.0
    assert stats["max"] == 2.0
    assert stats["mean"] == 1.5
    assert stats["median"] == 1.5
    assert stats["std"] == 0.5
    assert np.allclose(stats["p95"], 1.95)

def test_cva_013_raster_metadata():
    """TEST-CVA-013: Raster metadata. Verify CRS, transform, width, height, resolution, bounds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        grid = get_dummy_grid(shape=(4, 4))
        
        # Save dummy index GeoTIFFs
        t1_paths = {}
        t2_paths = {}
        profile = {
            'driver': 'GTiff', 'height': 4, 'width': 4, 'count': 1, 'dtype': rasterio.float32,
            'crs': grid.crs, 'transform': grid.transform, 'nodata': -9999.0
        }
        
        for name in ["ndvi", "gndvi", "ndre"]:
            t1_p = tmp_path / f"t1_{name}.tif"
            t2_p = tmp_path / f"t2_{name}.tif"
            
            with rasterio.open(t1_p, 'w', **profile) as dst:
                dst.write(np.full((4, 4), 0.2, dtype=np.float32), 1)
            with rasterio.open(t2_p, 'w', **profile) as dst:
                dst.write(np.full((4, 4), 0.5, dtype=np.float32), 1)
                
            t1_paths[name] = t1_p
            t2_paths[name] = t2_p
            
        mask = np.ones((4, 4), dtype=bool)
        out_dir = tmp_path / "outputs"
        
        result = calculate_cva(t1_paths, t2_paths, mask, grid, output_dir=out_dir)
        
        # Verify the saved files have exact same metadata
        for out_name in ["cva_magnitude.tif", "delta_ndvi.tif", "delta_gndvi.tif", "delta_ndre.tif"]:
            filepath = out_dir / out_name
            assert filepath.is_file()
            with rasterio.open(filepath) as src:
                assert src.crs == grid.crs
                assert np.allclose(src.transform, grid.transform)
                assert src.width == grid.width
                assert src.height == grid.height
                assert np.allclose(src.res, grid.resolution)
                assert np.allclose(src.bounds, grid.bounds)
                assert src.nodata == -9999.0

def test_cva_014_reproducibility():
    """TEST-CVA-014: Reproducibility. Run twice, expected identical output."""
    t1 = {"ndvi": np.random.rand(5, 5), "gndvi": np.random.rand(5, 5), "ndre": np.random.rand(5, 5)}
    t2 = {"ndvi": np.random.rand(5, 5), "gndvi": np.random.rand(5, 5), "ndre": np.random.rand(5, 5)}
    mask = (np.random.rand(5, 5) > 0.2)
    grid = get_dummy_grid(shape=(5, 5))
    
    res1 = calculate_cva(t1, t2, mask, grid)
    res2 = calculate_cva(t1, t2, mask, grid)
    
    assert np.allclose(res1.cva_magnitude, res2.cva_magnitude)
    assert np.allclose(res1.delta_ndvi, res2.delta_ndvi)
    assert np.allclose(res1.delta_gndvi, res2.delta_gndvi)
    assert np.allclose(res1.delta_ndre, res2.delta_ndre)
    if res1.cva_standardized is not None:
        assert np.allclose(res1.cva_standardized, res2.cva_standardized)
