import pytest
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.gis_engine.alignment.spatial import ReferenceGrid
from src.gis_engine.change.transition import calculate_transitions, TRANSITIONS, CLASS_DESCRIPTIONS

def get_dummy_grid(shape=(4, 4), is_geographic=False):
    """Helper to construct a dummy ReferenceGrid."""
    height, width = shape
    if is_geographic:
        crs = CRS.from_epsg(4326)
        transform = from_origin(10.0, 50.0, 0.0001, 0.0001)
        resolution = (0.0001, 0.0001)
        bounds = (10.0, 50.0 - height * 0.0001, 10.0 + width * 0.0001, 50.0)
    else:
        crs = CRS.from_epsg(32632) # Projected UTM zone 32N
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

def test_t4_001_all_16_transitions():
    """TEST-T4-001: Construct synthetic arrays with all combinations; verify all 0-15 appear."""
    t1 = np.repeat(np.arange(4), 4).reshape(4, 4)
    t2 = np.tile(np.arange(4), (4, 1))
    # This covers:
    # Row 0 (T1=0): T2=0,1,2,3 -> transition IDs 0,1,2,3
    # Row 1 (T1=1): T2=0,1,2,3 -> transition IDs 4,5,6,7
    # Row 2 (T1=2): T2=0,1,2,3 -> transition IDs 8,9,10,11
    # Row 3 (T1=3): T2=0,1,2,3 -> transition IDs 12,13,14,15
    mask = np.ones((4, 4), dtype=bool)
    grid = get_dummy_grid()
    
    result = calculate_transitions(t1, t2, mask, grid)
    assert np.all(np.isin(result.transition_map, np.arange(16)))
    # Check that all 0-15 IDs appear exactly once
    unique_ids = np.unique(result.transition_map)
    assert len(unique_ids) == 16
    assert np.array_equal(unique_ids, np.arange(16))

def test_t4_002_transition_formula():
    """TEST-T4-002: Verify transition_id = t1 * 4 + t2 formula."""
    grid = get_dummy_grid()
    for t1_val in range(4):
        for t2_val in range(4):
            t1 = np.full((1, 1), t1_val, dtype=np.uint8)
            t2 = np.full((1, 1), t2_val, dtype=np.uint8)
            mask = np.ones((1, 1), dtype=bool)
            
            result = calculate_transitions(t1, t2, mask, grid)
            expected_id = t1_val * 4 + t2_val
            assert result.transition_map[0, 0] == expected_id

def test_t4_003_class_mapping():
    """TEST-T4-003: Verify Class Mapping Contract."""
    assert CLASS_DESCRIPTIONS[0] == "Background"
    assert CLASS_DESCRIPTIONS[1] == "Severe"
    assert CLASS_DESCRIPTIONS[2] == "Moderate"
    assert CLASS_DESCRIPTIONS[3] == "Healthy"

def test_t4_004_to_016_individual_transitions():
    """
    TEST-T4-004 through TEST-T4-016: Verify transition mapping and categories.
    TEST-T4-004: H->M = 14 (DEGRADATION)
    TEST-T4-005: H->S = 13 (DEGRADATION)
    TEST-T4-006: M->S = 9 (DEGRADATION)
    TEST-T4-007: M->H = 11 (RECOVERY)
    TEST-T4-008: S->M = 6 (RECOVERY)
    TEST-T4-009: S->H = 7 (RECOVERY)
    TEST-T4-010: H->B = 12 (VEGETATION_TO_BACKGROUND)
    TEST-T4-011: M->B = 8 (VEGETATION_TO_BACKGROUND)
    TEST-T4-012: S->B = 4 (VEGETATION_TO_BACKGROUND)
    TEST-T4-013: B->H = 3 (BACKGROUND_TO_VEGETATION)
    TEST-T4-014: B->M = 2 (BACKGROUND_TO_VEGETATION)
    TEST-T4-015: B->S = 1 (BACKGROUND_TO_VEGETATION)
    TEST-T4-016: Stable transitions.
    """
    assert TRANSITIONS[14]["category"] == "DEGRADATION"     # H->M
    assert TRANSITIONS[13]["category"] == "DEGRADATION"     # H->S
    assert TRANSITIONS[9]["category"] == "DEGRADATION"      # M->S
    
    assert TRANSITIONS[11]["category"] == "RECOVERY"        # M->H
    assert TRANSITIONS[6]["category"] == "RECOVERY"         # S->M
    assert TRANSITIONS[7]["category"] == "RECOVERY"         # S->H
    
    assert TRANSITIONS[12]["category"] == "VEGETATION_TO_BACKGROUND" # H->B
    assert TRANSITIONS[8]["category"] == "VEGETATION_TO_BACKGROUND"  # M->B
    assert TRANSITIONS[4]["category"] == "VEGETATION_TO_BACKGROUND"  # S->B
    
    assert TRANSITIONS[3]["category"] == "BACKGROUND_TO_VEGETATION" # B->H
    assert TRANSITIONS[2]["category"] == "BACKGROUND_TO_VEGETATION" # B->M
    assert TRANSITIONS[1]["category"] == "BACKGROUND_TO_VEGETATION" # B->S
    
    # Stable transitions
    assert TRANSITIONS[0]["name"] == "B_to_B"
    assert TRANSITIONS[0]["category"] == "STABLE"
    assert TRANSITIONS[5]["name"] == "S_to_S"
    assert TRANSITIONS[5]["category"] == "STABLE"
    assert TRANSITIONS[10]["name"] == "M_to_M"
    assert TRANSITIONS[10]["category"] == "STABLE"
    assert TRANSITIONS[15]["name"] == "H_to_H"
    assert TRANSITIONS[15]["category"] == "STABLE"

def test_t4_017_nodata():
    """TEST-T4-017: NoData - Invalid pixels receive NoData (255) and do not become B->B (0)."""
    t1 = np.array([[0, 255], [3, 0]], dtype=np.uint8)
    t2 = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    mask = np.array([[1, 1], [1, 0]], dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_transitions(t1, t2, mask, grid)
    # Check pixels:
    # (0,0): T1=0, T2=0, mask=1 -> transition_id = 0
    # (0,1): T1=255, T2=0, mask=1 -> NoData (255)
    # (1,0): T1=3, T2=255, mask=1 -> NoData (255)
    # (1,1): T1=0, T2=255, mask=0 -> NoData (255)
    assert result.transition_map[0, 0] == 0
    assert result.transition_map[0, 1] == 255
    assert result.transition_map[1, 0] == 255
    assert result.transition_map[1, 1] == 255

def test_t4_018_background_validity():
    """TEST-T4-018: A valid B->B pixel must remain transition ID 0 and be counted."""
    t1 = np.full((2, 2), 0, dtype=np.uint8)
    t2 = np.full((2, 2), 0, dtype=np.uint8)
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_transitions(t1, t2, mask, grid)
    assert result.transition_map[0, 0] == 0
    assert result.transition_counts[0] == 4
    assert result.category_counts["STABLE"] == 4

def test_t4_019_transition_counts():
    """TEST-T4-019: Known synthetic distribution. Verify exact counts."""
    t1 = np.array([[3, 3], [1, 0]], dtype=np.uint8)
    t2 = np.array([[2, 1], [3, 0]], dtype=np.uint8)
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_transitions(t1, t2, mask, grid)
    # Transitions:
    # (0,0): 3->2 (H->M, tid=14)
    # (0,1): 3->1 (H->S, tid=13)
    # (1,0): 1->3 (S->H, tid=7)
    # (1,1): 0->0 (B->B, tid=0)
    assert result.transition_counts[14] == 1
    assert result.transition_counts[13] == 1
    assert result.transition_counts[7] == 1
    assert result.transition_counts[0] == 1
    assert sum(result.transition_counts.values()) == 4

def test_t4_020_percentages_sum():
    """TEST-T4-020: Verify transition percentages sum to approx 100%."""
    t1 = np.repeat(np.arange(4), 4).reshape(4, 4)
    t2 = np.tile(np.arange(4), (4, 1))
    mask = np.ones((4, 4), dtype=bool)
    grid = get_dummy_grid(shape=(4, 4))
    
    result = calculate_transitions(t1, t2, mask, grid)
    percentages_sum = sum(result.transition_percentages.values())
    assert pytest.approx(percentages_sum, abs=1e-5) == 100.0

def test_t4_021_area_calculation():
    """TEST-T4-021: Verify area = count * pixel_area for projected raster."""
    t1 = np.full((4, 4), 3, dtype=np.uint8)
    t2 = np.full((4, 4), 2, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    grid = get_dummy_grid(shape=(4, 4), is_geographic=False) # 10m x 10m pixels
    
    result = calculate_transitions(t1, t2, mask, grid)
    # Pixel area = 10 * 10 = 100 m^2
    # Total valid pixels = 16
    # Total area = 1600 m^2
    assert result.transition_areas_m2[14] == 1600.0
    assert result.transition_areas_ha[14] == 0.16

def test_t4_022_hectare_conversion():
    """TEST-T4-022: Verify 10,000 m^2 = 1 ha."""
    t1 = np.full((10, 10), 3, dtype=np.uint8)
    t2 = np.full((10, 10), 3, dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    grid = get_dummy_grid(shape=(10, 10), is_geographic=False)
    # Ensure pixel size is 10m x 10m
    grid.transform = from_origin(500000, 5000000, 10.0, 10.0)
    grid.resolution = (10.0, 10.0)
    
    result = calculate_transitions(t1, t2, mask, grid)
    # Total pixels = 100. Area = 100 * 100 = 10,000 m^2
    # This must equal exactly 1 ha
    assert result.transition_areas_m2[15] == 10000.0
    assert result.transition_areas_ha[15] == 1.0

def test_t4_023_zero_transitions():
    """TEST-T4-023: Row still exists when count = 0, with zeros for metrics."""
    t1 = np.full((2, 2), 3, dtype=np.uint8)
    t2 = np.full((2, 2), 3, dtype=np.uint8)
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_transitions(t1, t2, mask, grid)
    # Only H->H (tid=15) occurs.
    # Other 15 transitions must be 0
    for tid in range(15):
        assert result.transition_counts[tid] == 0
        assert result.transition_areas_m2[tid] == 0.0
        assert result.transition_percentages[tid] == 0.0

def test_t4_024_spatial_metadata():
    """TEST-T4-024: Verify output preserves reference grid parameters."""
    t1 = np.full((2, 2), 3, dtype=np.uint8)
    t2 = np.full((2, 2), 3, dtype=np.uint8)
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    result = calculate_transitions(t1, t2, mask, grid)
    assert result.reference_grid.crs == grid.crs
    assert result.reference_grid.transform == grid.transform
    assert result.reference_grid.width == grid.width
    assert result.reference_grid.height == grid.height
    assert result.reference_grid.resolution == grid.resolution
    assert result.reference_grid.bounds == grid.bounds

def test_t4_025_invalid_class():
    """TEST-T4-025: Insert class value 4; verify clear validation error."""
    t1 = np.array([[3, 4]], dtype=np.uint8)
    t2 = np.array([[3, 3]], dtype=np.uint8)
    mask = np.ones((1, 2), dtype=bool)
    grid = get_dummy_grid(shape=(1, 2))
    
    with pytest.raises(ValueError, match="invalid class ID"):
        calculate_transitions(t1, t2, mask, grid)

def test_t4_026_shape_mismatch():
    """TEST-T4-026: T1 and T2 different shapes; verify clear failure."""
    t1 = np.full((2, 2), 3, dtype=np.uint8)
    t2 = np.full((2, 3), 3, dtype=np.uint8)
    mask = np.ones((2, 2), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_transitions(t1, t2, mask, grid)

def test_t4_027_mask_mismatch():
    """TEST-T4-027: Mask dimensions differ from health map; verify clear failure."""
    t1 = np.full((2, 2), 3, dtype=np.uint8)
    t2 = np.full((2, 2), 3, dtype=np.uint8)
    mask = np.ones((2, 3), dtype=bool)
    grid = get_dummy_grid(shape=(2, 2))
    
    with pytest.raises(ValueError, match="Mask shape"):
        calculate_transitions(t1, t2, mask, grid)
