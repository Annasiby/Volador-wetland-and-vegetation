import pytest
import numpy as np
import tempfile
import json
import csv
import rasterio
from pathlib import Path
from rasterio.transform import from_origin
from rasterio.crs import CRS

from src.gis_engine.alignment.spatial import ReferenceGrid
from src.gis_engine.change.fusion import fuse_evidence, FUSION_CATEGORIES

@pytest.fixture
def base_grid():
    transform = from_origin(500000, 5000000, 10.0, 10.0)
    crs = CRS.from_string("EPSG:32632")
    return ReferenceGrid(
        crs=crs,
        transform=transform,
        width=4,
        height=4,
        resolution=(10.0, 10.0),
        bounds=(500000.0, 4999960.0, 500040.0, 5000000.0)
    )

@pytest.fixture
def base_config():
    return {
        "fusion": {
            "vegetation_degradation_transitions": [9, 13, 14],
            "vegetation_recovery_transitions": [6, 7, 11],
            "water_gain_class": 2,
            "water_loss_class": 3,
            "fusion_category_ids": {
                "nodata": 0,
                "stable": 1,
                "vegetation_degradation": 2,
                "vegetation_recovery": 3,
                "surface_water_gain": 4,
                "surface_water_loss": 5,
                "hydro_ecological_degradation": 6,
                "hydro_ecological_recovery": 7,
                "vegetation_to_background": 8,
                "ambiguous_mixed_evidence": 9
            },
            "independent_hydrological_source_accepted": False,
            "cva_consistency_check": True
        }
    }

def test_fusion_categories_and_invariants(base_grid, base_config):
    # Test cases:
    # 1. stable vegetation (10) + stable water (0) -> Stable (1)
    # 2. vegetation degradation (13) + stable water (0) -> Vegetation Degradation (2)
    # 3. vegetation recovery (11) + stable water (0) -> Vegetation Recovery (3)
    # 4. stable vegetation (10) + water loss (3) -> Surface-Water Loss (5)
    # 5. stable vegetation (10) + water gain (2) -> Surface-Water Gain (4)
    # 6. vegetation degradation (13) + water loss (3) -> Hydro-Ecological Degradation Evidence (6)
    # 7. vegetation recovery (11) + water gain (2) -> Hydro-Ecological Recovery Evidence (7)
    # 8. vegetation-to-background (8) + water gain (2) -> Vegetation-to-Background Change (8)
    # 9. vegetation-to-background (8) + water loss (3) -> Vegetation-to-Background Change (8)
    # 10. NoData vegetation (255) + stable water (0) -> NoData (0)
    # 11. stable vegetation (10) + NoData water (255) -> NoData (0)
    # 12. NoData vegetation (255) + NoData water (255) -> NoData (0)
    # 13. Ambiguous: veg degradation (13) + water gain (2) -> Ambiguous (9)
    # 14. Ambiguous: veg recovery (11) + water loss (3) -> Ambiguous (9)
    # 15. stable background (0) + stable water (0) -> Stable (1)
    # 16. background-to-vegetation (2) + water loss (3) -> Surface-Water Loss (5)
    
    # We construct a 4x4 grid corresponding to these 16 cases.
    v_trans = np.array([
        [10, 13, 11, 10],
        [10, 13, 11,  8],
        [ 8, 255, 10, 255],
        [13, 11,  0,  2]
    ], dtype=np.uint8)
    
    w_change = np.array([
        [ 0,  0,  0,  3],
        [ 2,  3,  2,  2],
        [ 3,  0, 255, 255],
        [ 2,  3,  0,  3]
    ], dtype=np.uint8)
    
    # Analysis mask - all True
    analysis_mask = np.ones((4, 4), dtype=bool)
    # Let's set the last pixel as invalid in analysis mask
    analysis_mask[3, 3] = False
    
    # NDVI delta - let's make it consistent with transitions
    # (NDVI decreases for deg, increases for rec)
    # 13 is deg -> negative delta
    # 11 is rec -> positive delta
    # 0, 10 is stable -> 0.0
    delta_ndvi = np.array([
        [0.0, -0.2, 0.2, 0.0],
        [0.0, -0.3, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.4, 0.3, 0.0, 0.0]
    ], dtype=np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        metrics = fuse_evidence(
            transition_map=v_trans,
            water_change=w_change,
            valid_analysis_mask=analysis_mask,
            config=base_config,
            delta_ndvi=delta_ndvi,
            output_dir=tmp_path,
            reference_grid=base_grid
        )
        
        # Verify outputs were generated
        assert (tmp_path / "evidence_fusion.tif").is_file()
        assert (tmp_path / "evidence_level.tif").is_file()
        assert (tmp_path / "cva_inconsistency.tif").is_file()
        assert (tmp_path / "fusion_metrics.json").is_file()
        assert (tmp_path / "evidence_matrix.csv").is_file()
        
        # Read fusion raster back
        with rasterio.open(tmp_path / "evidence_fusion.tif") as src:
            fusion_arr = src.read(1)
            assert src.crs == base_grid.crs
            assert src.transform == base_grid.transform
            assert src.width == 4
            assert src.height == 4
            assert src.nodata == 0
            
        # Expected outputs for each pixel:
        # Pixel (0,0): 10 (stable veg) + 0 (stable water) -> 1 (stable)
        # Pixel (0,1): 13 (deg) + 0 (stable water) -> 2 (veg degradation)
        # Pixel (0,2): 11 (rec) + 0 (stable water) -> 3 (veg recovery)
        # Pixel (0,3): 10 (stable veg) + 3 (water loss) -> 5 (surface-water loss)
        # Pixel (1,0): 10 (stable veg) + 2 (water gain) -> 4 (surface-water gain)
        # Pixel (1,1): 13 (deg) + 3 (water loss) -> 6 (hydro-ecological degradation)
        # Pixel (1,2): 11 (rec) + 2 (water gain) -> 7 (hydro-ecological recovery)
        # Pixel (1,3): 8 (veg_to_bg) + 2 (water gain) -> 8 (vegetation-to-background)
        # Pixel (2,0): 8 (veg_to_bg) + 3 (water loss) -> 8 (vegetation-to-background)
        # Pixel (2,1): 255 (NoData) + 0 -> 0 (NoData)
        # Pixel (2,2): 10 + 255 (NoData) -> 0 (NoData)
        # Pixel (2,3): 255 (NoData) + 255 (NoData) -> 0 (NoData)
        # Pixel (3,0): 13 (deg) + 2 (water gain) -> 9 (ambiguous)
        # Pixel (3,1): 11 (rec) + 3 (water loss) -> 9 (ambiguous)
        # Pixel (3,2): 0 (stable bg) + 0 (stable water) -> 1 (stable)
        # Pixel (3,3): mask is False -> 0 (NoData)
        
        expected = np.array([
            [1, 2, 3, 5],
            [4, 6, 7, 8],
            [8, 0, 0, 0],
            [9, 9, 1, 0]
        ], dtype=np.uint8)
        
        assert np.array_equal(fusion_arr, expected)
        
        # Verify evidence levels
        # Class 1, 2, 3, 8 -> VEGETATION_ONLY (1)
        # Class 4, 5 -> SURFACE_WATER_ONLY (2)
        # Class 6, 7, 9 -> HYDRO_ECOLOGICAL (3) (Wait, class 9 is Ambiguous, maps to 3)
        # Class 0 -> NODATA (0)
        with rasterio.open(tmp_path / "evidence_level.tif") as src:
            level_arr = src.read(1)
            
        expected_levels = np.array([
            [1, 1, 1, 2],
            [2, 3, 3, 1],
            [1, 0, 0, 0],
            [3, 3, 1, 0]
        ], dtype=np.uint8)
        
        assert np.array_equal(level_arr, expected_levels)
        
        # Mathematical Invariants from metrics
        # total area must equal sum of class areas
        total_area_ha = metrics["total_area_ha"]
        veg_deg_ha = metrics["vegetation_degradation_area_ha"]
        w_loss_ha = metrics["water_loss_area_ha"]
        hydro_deg_ha = metrics["hydro_ecological_degradation_area_ha"]
        
        assert hydro_deg_ha <= veg_deg_ha + 1e-6
        assert hydro_deg_ha <= w_loss_ha + 1e-6
        
        # Overlap ratio check
        # veg_deg occurs in (0,1) [class 2], (1,1) [class 6], (3,0) [class 9] -> 3 pixels
        # hydro_deg occurs in (1,1) [class 6] -> 1 pixel
        # ratio = 1/3 = 0.333333
        assert np.allclose(metrics["hydro_degradation_overlap_ratio"], 1.0 / 3.0)
        # water loss occurs in (0,3) [class 5], (1,1) [class 6], (2,0) [class 8], (3,1) [class 9] -> 4 pixels
        # supported fraction = 1/4 = 0.25
        assert np.allclose(metrics["water_loss_supported_fraction"], 0.25)

def test_zero_area_division_protection(base_grid, base_config):
    # No degradation, no recovery, no water change
    v_trans = np.full((4, 4), 10, dtype=np.uint8)
    w_change = np.full((4, 4), 0, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    
    metrics = fuse_evidence(
        transition_map=v_trans,
        water_change=w_change,
        valid_analysis_mask=mask,
        config=base_config,
        reference_grid=base_grid
    )
    
    assert metrics["vegetation_degradation_area_ha"] == 0.0
    assert metrics["vegetation_recovery_area_ha"] == 0.0
    assert metrics["hydro_degradation_overlap_ratio"] is None
    assert metrics["hydro_recovery_overlap_ratio"] is None
    assert metrics["water_loss_supported_fraction"] is None

def test_wetland_mask_restriction(base_grid, base_config):
    v_trans = np.array([
        [13, 13, 10, 10],
        [13, 13, 10, 10],
        [10, 10, 10, 10],
        [10, 10, 10, 10]
    ], dtype=np.uint8)
    
    w_change = np.array([
        [ 3,  3,  0,  0],
        [ 0,  0,  0,  0],
        [ 0,  0,  0,  0],
        [ 0,  0,  0,  0]
    ], dtype=np.uint8)
    
    # 4 degradation pixels total.
    # 2 are hydro-ecological degradation (at 0,0 and 0,1).
    
    # Wetland mask covers the left half (column 0 & 1)
    wetland_mask = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ], dtype=np.uint8)
    
    metrics = fuse_evidence(
        transition_map=v_trans,
        water_change=w_change,
        valid_analysis_mask=np.ones((4,4), dtype=bool),
        config=base_config,
        wetland_mask=wetland_mask,
        reference_grid=base_grid
    )
    
    assert metrics["wetland_specific_metrics_available"] is True
    # total veg degradation in wetland: 4 pixels
    # total hydro deg in wetland: 2 pixels
    # Let's verify values inside wetland restricted metrics
    assert metrics["hydro_ecological_degradation_area_inside_wetland_ha"] > 0.0
    
    wm = metrics["wetland_restricted_metrics"]
    assert np.allclose(wm["hydro_degradation_overlap_ratio"], 2.0 / 4.0)

def test_physical_depletion_assessment_logic(base_grid, base_config):
    v_trans = np.full((4,4), 13, dtype=np.uint8)
    w_change = np.full((4,4), 3, dtype=np.uint8)
    mask = np.ones((4,4), dtype=bool)
    wetland_mask = np.full((4,4), 1, dtype=np.uint8)
    
    # Default is False -> NOT_ASSESSED
    metrics_default = fuse_evidence(v_trans, w_change, mask, base_config, reference_grid=base_grid)
    assert metrics_default["physical_depletion_assessment"] == "NOT_ASSESSED"
    
    # Accepted but no wetland mask -> NOT_SUPPORTED
    metrics_no_mask = fuse_evidence(
        v_trans, w_change, mask, base_config,
        reference_grid=base_grid,
        independent_hydrological_source_accepted=True
    )
    assert metrics_no_mask["physical_depletion_assessment"] == "NOT_SUPPORTED"
    
    # Accepted with wetland mask -> SUPPORTED_BY_INDEPENDENT_EVIDENCE
    metrics_supported = fuse_evidence(
        v_trans, w_change, mask, base_config,
        wetland_mask=wetland_mask,
        reference_grid=base_grid,
        independent_hydrological_source_accepted=True
    )
    assert metrics_supported["physical_depletion_assessment"] == "SUPPORTED_BY_INDEPENDENT_EVIDENCE"

def test_cva_consistency_checks(base_grid, base_config):
    # (0,0) is degradation (13) with positive NDVI (+0.1) -> inconsistent!
    # (0,1) is degradation (13) with negative NDVI (-0.1) -> consistent!
    # (0,2) is recovery (11) with negative NDVI (-0.1) -> inconsistent!
    # (0,3) is recovery (11) with positive NDVI (+0.1) -> consistent!
    v_trans = np.array([[13, 13, 11, 11]], dtype=np.uint8)
    w_change = np.array([[0, 0, 0, 0]], dtype=np.uint8)
    mask = np.ones((1, 4), dtype=bool)
    delta_ndvi = np.array([[0.1, -0.1, -0.1, 0.1]], dtype=np.float32)
    
    grid = ReferenceGrid(
        crs=base_grid.crs,
        transform=base_grid.transform,
        width=4,
        height=1,
        resolution=(10.0, 10.0),
        bounds=base_grid.bounds
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = fuse_evidence(
            transition_map=v_trans,
            water_change=w_change,
            valid_analysis_mask=mask,
            config=base_config,
            delta_ndvi=delta_ndvi,
            reference_grid=grid,
            output_dir=Path(tmpdir)
        )
        
        assert metrics["spectral_transition_inconsistency_detected"] is True
        
        # Read inconsistency map
        with rasterio.open(Path(tmpdir) / "cva_inconsistency.tif") as src:
            incon_arr = src.read(1)
            
        assert np.array_equal(incon_arr, np.array([[1, 0, 1, 0]], dtype=np.uint8))
