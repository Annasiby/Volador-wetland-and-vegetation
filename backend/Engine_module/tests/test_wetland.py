import pytest
import numpy as np
import tempfile
import json
import os
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from pathlib import Path

from src.gis_engine.alignment.spatial import ReferenceGrid
from src.gis_engine.wetland.mask import build_wetland_mask, WetlandMaskResult
from src.gis_engine.change.degradation import analyze_degradation

def get_test_grid(shape=(10, 10), crs_code=32632):
    height, width = shape
    crs = CRS.from_epsg(crs_code)
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

def create_geojson_file(coords, properties=None, geom_type="Polygon"):
    if properties is None:
        properties = {}
    
    # Auto-detect CRS based on coordinate values
    def find_first_coord(nested_list):
        if not isinstance(nested_list, list):
            return None
        if len(nested_list) == 0:
            return None
        if not isinstance(nested_list[0], list):
            return nested_list
        return find_first_coord(nested_list[0])
        
    first_pt = find_first_coord(coords)
    crs_code = 4326
    if first_pt is not None and len(first_pt) >= 2:
        x, y = first_pt[0], first_pt[1]
        # UTM coordinate values are typically > 180 or < -180
        if abs(x) > 180 or abs(y) > 90:
            crs_code = 32632
            
    geom = {
        "type": geom_type,
        "coordinates": coords
    }
    
    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": properties
    }
    
    feature_collection = {
        "type": "FeatureCollection",
        "features": [feature],
        "crs": {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{crs_code}"
            }
        }
    }
    
    fd, path = tempfile.mkstemp(suffix=".geojson")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(feature_collection, f)
    return path

# TEST-W7-001: Simple polygon rasterization
def test_w7_001_simple_rasterization():
    grid = get_test_grid(shape=(10, 10))
    # Define polygon that covers center pixels (row 4-6, col 4-6)
    # Coordinate range for row 4-6: Y from 5000000 - 70 = 4999930 to 5000000 - 30 = 4999970
    # Coordinate range for col 4-6: X from 500000 + 40 = 500040 to 500000 + 70 = 500070
    poly_coords = [[
        [500035, 4999925], # bottom-left
        [500035, 4999975], # top-left
        [500075, 4999975], # top-right
        [500075, 4999925], # bottom-right
        [500035, 4999925]  # close ring
    ]]
    
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        
        # Verify that center pixels are rasterized as 1 and outer as 0
        mask = res.wetland_mask
        assert mask[4, 4] == 1
        assert mask[5, 5] == 1
        assert mask[0, 0] == 0
        assert res.status == "WETLAND_BOUNDARY_AVAILABLE"
    finally:
        os.remove(path)

# TEST-W7-002: Polygon outside raster
def test_w7_002_polygon_outside():
    grid = get_test_grid(shape=(10, 10))
    # Polygon far away from grid
    poly_coords = [[
        [600000, 6000000],
        [600000, 6000100],
        [600100, 600100],
        [600100, 600000],
        [600000, 6000000]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        assert np.sum(res.wetland_mask == 1) == 0
    finally:
        os.remove(path)

# TEST-W7-003: Polygon partially overlapping raster
def test_w7_003_polygon_partial_overlap():
    grid = get_test_grid(shape=(10, 10))
    # Polygon partially inside (X: 500040 to 500150, Y: 4999930 to 4999970)
    poly_coords = [[
        [500035, 4999925],
        [500035, 4999975],
        [500150, 4999975],
        [500150, 4999925],
        [500035, 4999925]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        # Inside grid pixels should be 1
        assert res.wetland_mask[4, 4] == 1
        # It shouldn't crash or overflow
        assert res.wetland_mask.shape == (10, 10)
    finally:
        os.remove(path)

# TEST-W7-004: Multi-polygon
def test_w7_004_multi_polygon():
    grid = get_test_grid(shape=(10, 10))
    # Two polygons
    coords = [
        [[[500015, 4999975], [500015, 4999985], [500025, 4999985], [500025, 4999975], [500015, 4999975]]],
        [[[500075, 4999915], [500075, 4999925], [500085, 4999925], [500085, 4999915], [500075, 4999915]]]
    ]
    path = create_geojson_file(coords, geom_type="MultiPolygon")
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        # Both areas should be represented in the union mask
        assert res.wetland_mask[2, 2] == 1  # near top-left area
        assert res.wetland_mask[8, 8] == 1  # near bottom-right area
    finally:
        os.remove(path)

# TEST-W7-005: CRS transformation
def test_w7_005_crs_transformation():
    grid = get_test_grid(shape=(10, 10), crs_code=32632) # EPSG:32632 (meter unit)
    
    # We will create boundary in WGS84 (EPSG:4326, degrees unit)
    # Target EPSG:32632 center (~500000, ~5000000) corresponds roughly to lat/lon 45.15, 9.0
    # Let's verify reprojection works by passing EPSG:4326 coords
    poly_coords = [[
        [9.0, 45.15],
        [9.0, 45.16],
        [9.01, 45.16],
        [9.01, 45.15],
        [9.0, 45.15]
    ]]
    # GeoJSON loader defaults to EPSG:4326 if not specified
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        assert res.wetland_mask.shape == (10, 10)
        assert res.status == "WETLAND_BOUNDARY_AVAILABLE"
    finally:
        os.remove(path)

# TEST-W7-006: Reference-grid preservation
def test_w7_006_reference_grid_preservation():
    grid = get_test_grid(shape=(15, 20), crs_code=4326)
    config = {"wetland": {"source": "auto"}}
    res = build_wetland_mask(source="auto", reference_grid=grid, config=config)
    assert res.wetland_mask.shape == (15, 20)
    assert res.metadata["target_crs"] == str(grid.crs)
    assert np.allclose(res.metadata["reference_grid"]["transform"], grid.to_dict()["transform"])

# TEST-W7-007: all_touched behavior
def test_w7_007_all_touched_behavior():
    grid = get_test_grid(shape=(10, 10))
    # Let's create a tiny polygon that just barely skims a pixel boundary
    poly_coords = [[
        [500041, 4999941],
        [500041, 4999942],
        [500042, 4999942],
        [500042, 4999941],
        [500041, 4999941]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        # Default all_touched: False
        config_default = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path},
                "rasterization": {"all_touched": False}
            }
        }
        res_default = build_wetland_mask(source="boundary", reference_grid=grid, config=config_default)
        
        # all_touched: True
        config_touched = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path},
                "rasterization": {"all_touched": True}
            }
        }
        res_touched = build_wetland_mask(source="boundary", reference_grid=grid, config=config_touched)
        
        # When all_touched is True, it should have equal or more pixels than when it is False
        assert np.sum(res_touched.wetland_mask == 1) >= np.sum(res_default.wetland_mask == 1)
    finally:
        os.remove(path)

# TEST-W7-008: Class selection
def test_w7_008_class_selection():
    grid = get_test_grid(shape=(10, 10))
    # Create feature collection with properties: class_name
    poly_coords = [[
        [500045, 4999935], [500045, 4999965], [500065, 4999965], [500065, 4999935], [500045, 4999935]
    ]]
    
    # Feature 1: class_name = "marsh"
    # Feature 2: class_name = "forest"
    # We will write both into GeoJSON
    geom = {"type": "Polygon", "coordinates": poly_coords}
    features = [
        {"type": "Feature", "geometry": geom, "properties": {"class_name": "marsh"}},
        {"type": "Feature", "geometry": geom, "properties": {"class_name": "forest"}}
    ]
    fc = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::32632"
            }
        }
    }
    
    fd, path = tempfile.mkstemp(suffix=".geojson")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(fc, f)
        
    try:
        # Filter: select "marsh" only
        config = {
            "wetland": {
                "source": "official",
                "official_dataset": {
                    "path": path,
                    "classes": ["marsh"],
                    "class_field": "class_name"
                }
            }
        }
        res = build_wetland_mask(source="official", reference_grid=grid, config=config)
        assert res.source_type == "OFFICIAL_DATASET"
        assert res.metadata["feature_selection_rule"] == ["marsh"]
        
        # If classes don't match, should fail with class semantic configuration error
        config_err = {
            "wetland": {
                "source": "official",
                "official_dataset": {
                    "path": path,
                    "classes": ["desert"],
                    "class_field": "class_name"
                }
            }
        }
        with pytest.raises(ValueError, match="Dataset class semantics require configuration"):
            build_wetland_mask(source="official", reference_grid=grid, config=config_err)
    finally:
        os.remove(path)

# TEST-W7-009: NoData distinction
def test_w7_009_nodata_distinction():
    grid = get_test_grid(shape=(10, 10))
    poly_coords = [[
        [500045, 4999935], [500045, 4999965], [500065, 4999965], [500065, 4999935], [500045, 4999935]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        # Pass overlap mask where pixel (0,0) is NoData (0) and other is 1
        overlap = np.ones((10, 10), dtype=np.uint8)
        overlap[0, 0] = 0
        
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config, overlap_mask=overlap)
        
        # Valid analysis mask should be 0 at (0, 0) because it's invalid overlap
        # Raw mask represents spatial boundary so it is still 0 (since it is outside the polygon)
        assert res.wetland_mask[0, 0] == 0
        assert res.valid_analysis_mask[0, 0] == 0
    finally:
        os.remove(path)

# TEST-W7-010: Valid overlap masking
def test_w7_010_valid_overlap_masking():
    grid = get_test_grid(shape=(10, 10))
    poly_coords = [[
        [500045, 4999935], [500045, 4999965], [500065, 4999965], [500065, 4999935], [500045, 4999935]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        # Overlap mask is invalid at row 5, col 5
        overlap = np.ones((10, 10), dtype=np.uint8)
        overlap[5, 5] = 0
        
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config, overlap_mask=overlap)
        
        # At row 5, col 5, the raw wetland mask is 1 (wetland exists geometrically)
        # But valid analysis mask is 0 (excluded because temporal overlap is invalid)
        assert res.wetland_mask[5, 5] == 1
        assert res.valid_analysis_mask[5, 5] == 0
    finally:
        os.remove(path)

# TEST-W7-011: Area calculation
def test_w7_011_area_calculation():
    grid = get_test_grid(shape=(10, 10), crs_code=32632) # EPSG:32632 is projected in meters
    # Let's create a polygon covering exactly 4 pixels. Each pixel is 10m x 10m = 100 m^2.
    # Total area should be 400 m^2.
    # Cover rows 3,4 and cols 3,4
    # X range: 500030 to 500050. Y range: 4999950 to 4999970
    poly_coords = [[
        [500030, 4999950],
        [500030, 4999970],
        [500050, 4999970],
        [500050, 4999950],
        [500030, 4999950]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        assert abs(res.wetland_area_m2 - 400.0) < 1e-3
    finally:
        os.remove(path)

# TEST-W7-012: Hectare conversion
def test_w7_012_hectare_conversion():
    grid = get_test_grid(shape=(10, 10), crs_code=32632)
    poly_coords = [[
        [500000, 4999900],
        [500000, 5000000],
        [501000, 5000000], # 1000m x 100m = 100,000 m^2 = 10 ha
        [501000, 4999900],
        [500000, 4999900]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        # 10,000 m^2 = 1 ha. Let's verify scaling is correct.
        assert abs(res.wetland_area_ha - (res.wetland_area_m2 / 10000.0)) < 1e-6
    finally:
        os.remove(path)

# TEST-W7-013: No boundary available
def test_w7_013_no_boundary_available():
    grid = get_test_grid(shape=(10, 10))
    config = {"wetland": {"source": "auto"}}
    res = build_wetland_mask(source="auto", reference_grid=grid, config=config)
    assert res.status == "WETLAND_BOUNDARY_UNAVAILABLE"
    assert res.physical_wetland_depletion_enabled is False

# TEST-W7-014: Project boundary priority
def test_w7_014_project_boundary_priority():
    grid = get_test_grid(shape=(10, 10))
    poly_coords = [[
        [500045, 4999935], [500045, 4999965], [500065, 4999965], [500065, 4999935], [500045, 4999935]
    ]]
    path_boundary = create_geojson_file(poly_coords)
    path_official = create_geojson_file(poly_coords)
    try:
        # Both exist, project boundary should take precedence
        config = {
            "wetland": {
                "source": "auto",
                "boundary": {"path": path_boundary},
                "official_dataset": {"path": path_official}
            }
        }
        res = build_wetland_mask(source="auto", reference_grid=grid, config=config)
        assert res.source_type == "PROJECT_BOUNDARY"
        assert res.status == "WETLAND_BOUNDARY_AVAILABLE"
    finally:
        os.remove(path_boundary)
        os.remove(path_official)

# TEST-W7-015: Official dataset priority
def test_w7_015_official_dataset_priority():
    grid = get_test_grid(shape=(10, 10))
    poly_coords = [[
        [500045, 4999935], [500045, 4999965], [500065, 4999965], [500065, 4999935], [500045, 4999935]
    ]]
    path_official = create_geojson_file(poly_coords)
    try:
        # Project boundary is missing, official dataset should take precedence over optical/JRC
        config = {
            "wetland": {
                "source": "auto",
                "boundary": {"path": None},
                "official_dataset": {"path": path_official},
                "jrc": {"enabled": True},
                "optical": {"enabled": True}
            }
        }
        res = build_wetland_mask(source="auto", reference_grid=grid, config=config)
        assert res.source_type == "OFFICIAL_DATASET"
        assert res.status == "WETLAND_BOUNDARY_AVAILABLE"
    finally:
        os.remove(path_official)

# TEST-W7-016: JRC distinction
def test_w7_016_jrc_distinction():
    grid = get_test_grid(shape=(10, 10))
    # We will write a small JRC raster to disk to load
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 255,
        "width": 10,
        "height": 10,
        "count": 1,
        "crs": grid.crs,
        "transform": grid.transform
    }
    fd, path_jrc = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    
    with rasterio.open(path_jrc, "w", **profile) as dst:
        data = np.zeros((10, 10), dtype=np.uint8)
        data[4:6, 4:6] = 80 # persistence value
        dst.write(data, 1)
        
    try:
        config = {
            "wetland": {
                "source": "auto",
                "jrc": {
                    "enabled": True,
                    "path": path_jrc
                }
            }
        }
        res = build_wetland_mask(source="auto", reference_grid=grid, config=config)
        
        # Verify status is WETLAND_EXTENT_INFERRED and source is JRC
        assert res.source_type == "JRC_WATER_CONTEXT"
        assert res.status == "WETLAND_EXTENT_INFERRED"
        assert res.metadata["source_metadata"]["processing_method"] == "hydrological/water-history context"
        assert res.physical_wetland_depletion_enabled is False
    finally:
        os.remove(path_jrc)

# TEST-W7-017: Reproducibility
def test_w7_017_reproducibility():
    grid = get_test_grid(shape=(10, 10))
    poly_coords = [[
        [500045, 4999935], [500045, 4999965], [500065, 4999965], [500065, 4999935], [500045, 4999935]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res1 = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        res2 = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        
        assert np.array_equal(res1.wetland_mask, res2.wetland_mask)
        assert np.array_equal(res1.valid_analysis_mask, res2.valid_analysis_mask)
    finally:
        os.remove(path)

# Phase 6 Integration Test
def test_phase6_integration():
    grid = get_test_grid(shape=(10, 10))
    
    # We will simulate Phase 6 output rasters:
    # degradation_map (1 where degraded, 0 otherwise)
    # recovery_map (1 where recovered, 0 otherwise)
    degradation_map = np.zeros((10, 10), dtype=np.uint8)
    degradation_map[3:7, 3:7] = 1 # center degraded
    
    # Create wetland boundary that covers columns 0-5
    # X from 500000 to 500050. Y from 4999900 to 5000000
    poly_coords = [[
        [500000, 4999900],
        [500000, 5000000],
        [500050, 5000000],
        [500050, 4999900],
        [500000, 4999900]
    ]]
    path = create_geojson_file(poly_coords)
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(source="boundary", reference_grid=grid, config=config)
        
        # Apply wetland mask
        # Conceptually: wetland_degradation = degradation_map AND wetland_mask
        wetland_degradation = degradation_map & res.wetland_mask
        
        # Column 0-4 are inside wetland (so mask is 1). Column 5-9 are outside (mask is 0).
        # Degradation is inside rows 3-6 and cols 3-6.
        # Intersect is rows 3-6 and cols 3-4.
        assert np.sum(wetland_degradation == 1) == 8  # 4 rows * 2 columns
        assert np.all(wetland_degradation[:, 5:] == 0) # outside-wetland pixels are excluded
    finally:
        os.remove(path)

# End-to-end synthetic test
def test_e2e_synthetic():
    grid = get_test_grid(shape=(10, 10))
    
    # Create synthetic boundary, overlap mask, and Phase 6 maps
    poly_coords = [[
        [500025, 4999925], [500025, 4999975], [500075, 4999975], [500075, 4999925], [500025, 4999925]
    ]]
    path = create_geojson_file(poly_coords)
    
    overlap = np.ones((10, 10), dtype=np.uint8)
    overlap[8:, :] = 0 # bottom 2 rows are invalid overlap
    
    degradation = np.ones((10, 10), dtype=np.uint8) # everywhere degraded
    
    try:
        config = {
            "wetland": {
                "source": "boundary",
                "boundary": {"path": path}
            }
        }
        res = build_wetland_mask(
            source="boundary",
            reference_grid=grid,
            config=config,
            overlap_mask=overlap
        )
        
        # Expected: wetland-masked output contains only pixels where wetland = 1 and overlap is valid (1)
        wetland_deg = degradation & res.valid_analysis_mask
        
        # Check rows that are valid overlap (0 to 7) and inside polygon (col 3-7, row 3-7)
        # Row 8-9 are invalid, so mask is 0
        assert np.all(wetland_deg[8:, :] == 0)
        assert np.all(wetland_deg[:8, :3] == 0)
        assert np.all(wetland_deg[3:8, 3:8] == 1)
    finally:
        os.remove(path)
