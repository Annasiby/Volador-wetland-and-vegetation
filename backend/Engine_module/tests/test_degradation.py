import pytest
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.gis_engine.change.degradation import analyze_degradation, DegradationResult
from src.gis_engine.alignment.spatial import ReferenceGrid

def get_dummy_grid(shape=(4, 4)):
    height, width = shape
    crs = CRS.from_epsg(32632)
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

def get_base_inputs(shape=(1, 1), transition_id=15, cva=0.1, delta=0.0):
    transition_map = np.full(shape, transition_id, dtype=np.uint8)
    cva_magnitude = np.full(shape, cva, dtype=np.float32)
    delta_ndvi = np.full(shape, delta, dtype=np.float32)
    delta_gndvi = np.full(shape, delta, dtype=np.float32)
    delta_ndre = np.full(shape, delta, dtype=np.float32)
    valid_mask = np.ones(shape, dtype=bool)
    return transition_map, cva_magnitude, delta_ndvi, delta_gndvi, delta_ndre, valid_mask

def get_default_config(method="percentile", value=95.0, directional_rule=None):
    return {
        "change": {
            "residual_stress": {
                "enabled": True,
                "threshold": {
                    "method": method,
                    "value": value
                },
                "directional_rule": directional_rule
            }
        }
    }

# TEST-D6-001: H→M (transition ID = 14)
def test_d6_001_h_to_m():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=14)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 1
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0

# TEST-D6-002: H→S (transition ID = 13)
def test_d6_002_h_to_s():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=13)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 1
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0

# TEST-D6-003: M→S (transition ID = 9)
def test_d6_003_m_to_s():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=9)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 1
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0

# TEST-D6-004: M→H (transition ID = 11)
def test_d6_004_m_to_h():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=11)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 1
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0

# TEST-D6-005: S→M (transition ID = 6)
def test_d6_005_s_to_m():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=6)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 1
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0

# TEST-D6-006: S→H (transition ID = 7)
def test_d6_006_s_to_h():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=7)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 1
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0

# TEST-D6-007: H→B (transition ID = 12)
def test_d6_007_h_to_b():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=12)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 1

# TEST-D6-008: M→B (transition ID = 8)
def test_d6_008_m_to_b():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=8)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 1

# TEST-D6-009: S→B (transition ID = 4)
def test_d6_009_s_to_b():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=4)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 1

# TEST-D6-010: B→H (transition ID = 3)
def test_d6_010_b_to_h():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=3)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0
    assert res.statistics["background_to_vegetation_count"] == 1

# TEST-D6-011: B→M (transition ID = 2)
def test_d6_011_b_to_m():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=2)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0
    assert res.statistics["background_to_vegetation_count"] == 1

# TEST-D6-012: B→S (transition ID = 1)
def test_d6_012_b_to_s():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=1)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 0
    assert res.recovery_map[0, 0] == 0
    assert res.residual_stress[0, 0] == 0
    assert res.canopy_loss[0, 0] == 0
    assert res.statistics["background_to_vegetation_count"] == 1

# TEST-D6-013: H→H with low CVA
def test_d6_013_h_to_h_low_cva():
    t = np.array([15, 15], dtype=np.uint8)
    c = np.array([0.1, 0.9], dtype=np.float32)
    # The stable veg population has values 0.1, 0.9.
    # 95th percentile will be 0.86.
    # Pixel 0 (CVA=0.1) is below threshold -> residual = 0
    m = np.ones(2, dtype=bool)
    d = np.zeros(2, dtype=np.float32)
    grid = get_dummy_grid(shape=(1, 2))
    
    res = analyze_degradation(t, c, d, d, d, m, get_default_config(method="percentile", value=95.0), reference_grid=grid)
    assert res.degradation_map[0] == 0
    assert res.recovery_map[0] == 0
    assert res.residual_stress[0] == 0 # below threshold (0.1 < 0.86)
    assert res.residual_stress[1] == 1 # above threshold (0.9 >= 0.86)

# TEST-D6-014: H→H with high CVA
def test_d6_014_h_to_h_high_cva():
    # Use absolute threshold 0.5. CVA is 0.6.
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=15, cva=0.6)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(method="absolute", value=0.5), reference_grid=grid)
    assert res.residual_stress[0, 0] == 1

# TEST-D6-015: M→M with high CVA
def test_d6_015_m_to_m_high_cva():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=10, cva=0.6)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(method="absolute", value=0.5), reference_grid=grid)
    assert res.residual_stress[0, 0] == 1

# TEST-D6-016: S→S with high CVA
def test_d6_016_s_to_s_high_cva():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=5, cva=0.6)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(method="absolute", value=0.5), reference_grid=grid)
    assert res.residual_stress[0, 0] == 1

# TEST-D6-017: B→B with high CVA
def test_d6_017_b_to_b_high_cva():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=0, cva=0.6)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(method="absolute", value=0.5), reference_grid=grid)
    assert res.residual_stress[0, 0] == 0 # B->B excluded from vegetation residual stress

# TEST-D6-018: Degradation with high CVA
def test_d6_018_degradation_precedence():
    # H->S (13) with high CVA (0.8) vs absolute threshold (0.5)
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=13, cva=0.8)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(method="absolute", value=0.5), reference_grid=grid)
    assert res.degradation_map[0, 0] == 1
    assert res.residual_stress[0, 0] == 0 # primary transition takes precedence

# TEST-D6-019: Recovery with high CVA
def test_d6_019_recovery_precedence():
    # S->H (7) with high CVA (0.8) vs absolute threshold (0.5)
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=7, cva=0.8)
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(method="absolute", value=0.5), reference_grid=grid)
    assert res.recovery_map[0, 0] == 1
    assert res.residual_stress[0, 0] == 0 # primary transition takes precedence

# TEST-D6-020: NoData propagation
def test_d6_020_nodata():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=15)
    m[0, 0] = False # Set invalid
    grid = get_dummy_grid(shape=(1, 1))
    res = analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert res.degradation_map[0, 0] == 255
    assert res.recovery_map[0, 0] == 255
    assert res.residual_stress[0, 0] == 255
    assert res.canopy_loss[0, 0] == 255

# TEST-D6-021: Invalid transition ID
def test_d6_021_invalid_transition_id():
    t, c, d_ndvi, d_gndvi, d_ndre, m = get_base_inputs(transition_id=16) # Invalid ID 16
    grid = get_dummy_grid(shape=(1, 1))
    with pytest.raises(ValueError) as excinfo:
        analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert "Invalid transition ID" in str(excinfo.value)

# TEST-D6-022: Geometry shape mismatch
def test_d6_022_geometry_shape_mismatch():
    t = np.full((1, 2), 15, dtype=np.uint8)
    c = np.full((1, 1), 0.1, dtype=np.float32)
    d_ndvi = np.full((1, 1), 0.0, dtype=np.float32)
    d_gndvi = np.full((1, 1), 0.0, dtype=np.float32)
    d_ndre = np.full((1, 1), 0.0, dtype=np.float32)
    m = np.ones((1, 1), dtype=bool)
    grid = get_dummy_grid(shape=(1, 1))
    with pytest.raises(ValueError) as excinfo:
        analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert "Geometry mismatch" in str(excinfo.value)

# TEST-D6-023: Mask shape mismatch
def test_d6_023_mask_shape_mismatch():
    t = np.full((1, 1), 15, dtype=np.uint8)
    c = np.full((1, 1), 0.1, dtype=np.float32)
    d_ndvi = np.full((1, 1), 0.0, dtype=np.float32)
    d_gndvi = np.full((1, 1), 0.0, dtype=np.float32)
    d_ndre = np.full((1, 1), 0.0, dtype=np.float32)
    m = np.ones((1, 2), dtype=bool)
    grid = get_dummy_grid(shape=(1, 1))
    with pytest.raises(ValueError) as excinfo:
        analyze_degradation(t, c, d_ndvi, d_gndvi, d_ndre, m, get_default_config(), reference_grid=grid)
    assert "Geometry mismatch" in str(excinfo.value)

# TEST-D6-024: Threshold reproducibility
def test_d6_024_reproducibility():
    # Build a stable vegetation array with multiple values
    t = np.array([15, 10, 5, 15, 0], dtype=np.uint8)
    c = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    m = np.ones(5, dtype=bool)
    d = np.zeros(5, dtype=np.float32)
    grid = get_dummy_grid(shape=(1, 5))
    
    cfg = get_default_config(method="percentile", value=95.0)
    res1 = analyze_degradation(t, c, d, d, d, m, cfg, reference_grid=grid)
    res2 = analyze_degradation(t, c, d, d, d, m, cfg, reference_grid=grid)
    
    assert np.array_equal(res1.degradation_map, res2.degradation_map)
    assert np.array_equal(res1.recovery_map, res2.recovery_map)
    assert np.array_equal(res1.residual_stress, res2.residual_stress)
    assert np.array_equal(res1.canopy_loss, res2.canopy_loss)
    assert res1.statistics == res2.statistics
    assert res1.thresholds == res2.thresholds

# TEST-D6-025: Statistics reconciliation
def test_d6_025_statistics_reconciliation():
    # 16 pixels, covering all 16 transitions once
    t = np.arange(16, dtype=np.uint8)
    # Give them CVA=0.6, absolute threshold=0.5, so stable vegetation has residual stress
    c = np.full(16, 0.6, dtype=np.float32)
    m = np.ones(16, dtype=bool)
    d = np.zeros(16, dtype=np.float32)
    grid = get_dummy_grid(shape=(4, 4))
    
    cfg = get_default_config(method="absolute", value=0.5)
    res = analyze_degradation(t, c, d, d, d, m, cfg, reference_grid=grid)
    stats = res.statistics
    
    # Degradation: H→M (14), H→S (13), M→S (9) = 3 pixels
    assert stats["degradation_pixel_count"] == 3
    # Recovery: M→H (11), S→M (6), S→H (7) = 3 pixels
    assert stats["recovery_pixel_count"] == 3
    # Canopy loss: H→B (12), M→B (8), S→B (4) = 3 pixels
    assert stats["canopy_loss_pixel_count"] == 3
    # Background to vegetation: B→H (3), B→M (2), B→S (1) = 3 pixels
    assert stats["background_to_vegetation_count"] == 3
    # Stable vegetation: H→H (15), M→M (10), S→S (5) = 3 pixels
    assert stats["stable_vegetation_count"] == 3
    # Stable background: B→B (0) = 1 pixel
    assert stats["stable_background_count"] == 1
    # Stable count total = stable veg + stable bg = 4 pixels
    assert stats["stable_count"] == 4
    
    # Check that sum of categories equals total valid pixels (16)
    # Categories: degradation (3) + recovery (3) + canopy loss (3) + bg_to_veg (3) + stable_veg (3) + stable_bg (1) = 16
    total_reconciled = (
        stats["degradation_pixel_count"] +
        stats["recovery_pixel_count"] +
        stats["canopy_loss_pixel_count"] +
        stats["background_to_vegetation_count"] +
        stats["stable_vegetation_count"] +
        stats["stable_background_count"]
    )
    assert total_reconciled == stats["valid_pixel_count"]
    assert stats["valid_pixel_count"] == 16
