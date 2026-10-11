import pytest
import numpy as np
from pathlib import Path
import json
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.gis_engine.metrics.degradation import calculate_metrics, MetricsResult
from src.gis_engine.metrics.statistics import estimate_area_with_reference_samples
from src.gis_engine.alignment.spatial import ReferenceGrid

# Helper to create UTM ReferenceGrid (resolution 10m x 10m -> pixel area = 100 m2)
def get_mock_projected_grid(shape=(4, 4)):
    height, width = shape
    crs = CRS.from_epsg(32632) # UTM 32N (Projected)
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

def get_mock_config(
    healthy=0.0, moderate=0.35, severe=0.80, background=1.0,
    include_canopy_loss_in_swd=False, mmu_enabled=False, mmu_val_m2=0,
    depletion_enabled=False, t1_path=None, t2_path=None
):
    return {
        "wetland": {
            "depletion_claims": {
                "enabled": depletion_enabled
            }
        },
        "metrics": {
            "severity_weights": {
                "healthy": healthy,
                "moderate": moderate,
                "severe": severe,
                "background": background
            },
            "include_canopy_loss_in_swd": include_canopy_loss_in_swd,
            "minimum_change_area": {
                "enabled": mmu_enabled,
                "value_m2": mmu_val_m2
            },
            "ndr": {
                "enabled": True
            },
            "temporal_wetland_extent": {
                "t1_path": t1_path,
                "t2_path": t2_path
            }
        }
    }

def get_mock_arrays(
    shape=(2, 2),
    trans_val=15, # Default H->H
    cva_val=1.0,
    ndvi_val=0.5,
    gndvi_val=0.4,
    ndre_val=0.3,
    deg_val=0,
    rec_val=0,
    res_val=0,
    canopy_val=0,
    wetland_val=1,
    valid_val=1
):
    return (
        np.full(shape, trans_val, dtype=np.uint8),
        np.full(shape, cva_val, dtype=np.float32),
        np.full(shape, ndvi_val, dtype=np.float32),
        np.full(shape, gndvi_val, dtype=np.float32),
        np.full(shape, ndre_val, dtype=np.float32),
        np.full(shape, deg_val, dtype=np.uint8),
        np.full(shape, rec_val, dtype=np.uint8),
        np.full(shape, res_val, dtype=np.uint8),
        np.full(shape, canopy_val, dtype=np.uint8),
        np.full(shape, wetland_val, dtype=np.uint8),
        np.full(shape, valid_val, dtype=np.uint8)
    )

# TEST-M8-001: Known raster area. projected synthetic raster. Verify exact area.
def test_m8_001_known_raster_area():
    grid = get_mock_projected_grid(shape=(4, 4)) # 16 pixels
    arrays = get_mock_arrays(shape=(4, 4))
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    # 16 pixels * 100 m2/pixel = 1600 m2
    assert res.baseline_area["area_m2"] == 1600.0

# TEST-M8-002: Hectare conversion. Verify 10,000 m2 = 1 ha
def test_m8_002_hectare_conversion():
    grid = get_mock_projected_grid(shape=(10, 10)) # 100 pixels * 100 m2 = 10,000 m2
    arrays = get_mock_arrays(shape=(10, 10))
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.baseline_area["area_m2"] == 10000.0
    assert res.baseline_area["area_ha"] == 1.0

# TEST-M8-003: Baseline area. Known wetland mask. Verify A_baseline.
def test_m8_003_baseline_area():
    grid = get_mock_projected_grid(shape=(4, 4))
    # 12 valid pixels, 4 invalid
    valid_mask = np.ones((4, 4), dtype=np.uint8)
    valid_mask[3, :] = 0 # 4 pixels invalid
    arrays = get_mock_arrays(shape=(4, 4), valid_val=1)
    # Replace last argument (valid_analysis_mask) with valid_mask
    arrays = list(arrays)
    arrays[-1] = valid_mask
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.baseline_area["pixel_count"] == 12
    assert res.baseline_area["area_m2"] == 1200.0

# TEST-M8-004: Degradation area. Known degradation map. Verify A_impact.
def test_m8_004_degradation_area():
    grid = get_mock_projected_grid(shape=(2, 2))
    deg_map = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    arrays = list(get_mock_arrays(shape=(2, 2)))
    arrays[5] = deg_map # degradation_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.baseline_area["pixel_count"] == 4
    assert res.impact_area["impact_area_m2"] == 200.0
    assert res.impact_area["impact_area_ha"] == 0.02

# TEST-M8-005: Recovery area. Known recovery map. Verify A_recovery.
def test_m8_005_recovery_area():
    grid = get_mock_projected_grid(shape=(2, 2))
    rec_map = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    arrays = list(get_mock_arrays(shape=(2, 2)))
    arrays[6] = rec_map # recovery_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.recovery_area["area_m2"] == 100.0
    assert res.recovery_area["area_ha"] == 0.01

# TEST-M8-006: Canopy-loss area. Known canopy-loss map. Verify A_canopy_loss.
def test_m8_006_canopy_loss_area():
    grid = get_mock_projected_grid(shape=(2, 2))
    canopy_map = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    arrays = list(get_mock_arrays(shape=(2, 2)))
    arrays[8] = canopy_map # canopy_loss_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.canopy_loss_area["area_m2"] == 300.0
    assert res.canopy_loss_area["ratio"] == 0.75

# TEST-M8-007: Residual-stress area. Known residual map. Verify A_residual_stress.
def test_m8_007_residual_stress_area():
    grid = get_mock_projected_grid(shape=(2, 2))
    res_map = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    # Make sure we have stable veg (trans = 15) to calculate ratio
    arrays = list(get_mock_arrays(shape=(2, 2), trans_val=15))
    arrays[7] = res_map # residual_stress_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.residual_stress_area["area_m2"] == 100.0
    # Stable veg is 4 pixels = 400 m2. Ratio should be 100 / 400 = 0.25
    assert res.residual_stress_area["ratio"] == 0.25

# TEST-M8-008: Transition-specific areas. Known 4x4 synthetic transition raster. Verify all 16 transition areas.
def test_m8_008_transition_specific_areas():
    grid = get_mock_projected_grid(shape=(4, 4))
    # Fill transition map with IDs 0 to 15
    trans_map = np.arange(16, dtype=np.uint8).reshape((4, 4))
    arrays = list(get_mock_arrays(shape=(4, 4)))
    arrays[0] = trans_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    # Each transition should have exactly 1 pixel = 100 m2
    for tid in range(16):
        name = [
            "B_to_B", "B_to_S", "B_to_M", "B_to_H",
            "S_to_B", "S_to_S", "S_to_M", "S_to_H",
            "M_to_B", "M_to_S", "M_to_M", "M_to_H",
            "H_to_B", "H_to_S", "H_to_M", "H_to_H"
        ][tid]
        assert res.transition_metrics[name]["pixel_count"] == 1
        assert res.transition_metrics[name]["area_m2"] == 100.0

# TEST-M8-009: Transition reconciliation. Verify sum transitions = baseline area.
def test_m8_009_transition_reconciliation():
    grid = get_mock_projected_grid(shape=(4, 4))
    arrays = get_mock_arrays(shape=(4, 4))
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    # Sum transitions area must ≈ baseline area
    sum_trans_area = sum(m["area_m2"] for m in res.transition_metrics.values())
    assert sum_trans_area == res.baseline_area["area_m2"]

# TEST-M8-010: Class reconciliation. Verify T1 class areas = baseline and T2 class areas = baseline.
def test_m8_010_class_reconciliation():
    grid = get_mock_projected_grid(shape=(4, 4))
    # Set transition map to mix classes
    trans_map = np.arange(16, dtype=np.uint8).reshape((4, 4))
    arrays = list(get_mock_arrays(shape=(4, 4)))
    arrays[0] = trans_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    # Sum of T1 class areas must ≈ baseline
    t1_sum = sum(info["area_m2"] for info in res.class_area_metrics["T1"].values())
    t2_sum = sum(info["area_m2"] for info in res.class_area_metrics["T2"].values())
    assert t1_sum == 1600.0
    assert t2_sum == 1600.0

# TEST-M8-011: Severity weighting: 10 m2 H->M. Expected 3.5 m2 weighted area.
def test_m8_011_severity_hm():
    grid = get_mock_projected_grid(shape=(1, 1))
    # Make pixel area 10 m2 by using transform size
    grid.transform = from_origin(500000, 5000000, np.sqrt(10), np.sqrt(10))
    # H->M transition is 14
    arrays = list(get_mock_arrays(shape=(1, 1), trans_val=14, deg_val=1))
    res = calculate_metrics(*arrays, get_mock_config(moderate=0.35), reference_grid=grid)
    # 10 m2 * 0.35 = 3.5 m2
    assert np.isclose(res.severity_weighted_area["severity_weighted_area_m2"], 3.5)

# TEST-M8-012: Severity weighting: 10 m2 H->S. Expected 8.0 m2 weighted area.
def test_m8_012_severity_hs():
    grid = get_mock_projected_grid(shape=(1, 1))
    grid.transform = from_origin(500000, 5000000, np.sqrt(10), np.sqrt(10))
    # H->S transition is 13
    arrays = list(get_mock_arrays(shape=(1, 1), trans_val=13, deg_val=1))
    res = calculate_metrics(*arrays, get_mock_config(severe=0.80), reference_grid=grid)
    # 10 m2 * 0.80 = 8.0 m2
    assert np.isclose(res.severity_weighted_area["severity_weighted_area_m2"], 8.0)

# TEST-M8-013: Severity weighting: 10 m2 M->S. Expected 8.0 m2 weighted area.
def test_m8_013_severity_ms():
    grid = get_mock_projected_grid(shape=(1, 1))
    grid.transform = from_origin(500000, 5000000, np.sqrt(10), np.sqrt(10))
    # M->S transition is 9
    arrays = list(get_mock_arrays(shape=(1, 1), trans_val=9, deg_val=1))
    res = calculate_metrics(*arrays, get_mock_config(severe=0.80), reference_grid=grid)
    # 10 m2 * 0.80 = 8.0 m2
    assert np.isclose(res.severity_weighted_area["severity_weighted_area_m2"], 8.0)

# TEST-M8-014: NDR. A_SWD = 100, A_baseline = 1000. Expected NDR = 0.1
def test_m8_014_ndr():
    grid = get_mock_projected_grid(shape=(10, 1)) # 10 pixels * 100 m2 = 1000 m2 baseline
    # 1 pixel is H->S (13), which has severe weight = 1.0 (so 100 m2 SWD area)
    trans_map = np.full((10, 1), 15, dtype=np.uint8)
    trans_map[0, 0] = 13
    deg_map = np.zeros((10, 1), dtype=np.uint8)
    deg_map[0, 0] = 1
    arrays = list(get_mock_arrays(shape=(10, 1)))
    arrays[0] = trans_map
    arrays[5] = deg_map
    res = calculate_metrics(*arrays, get_mock_config(severe=1.0), reference_grid=grid)
    assert res.normalized_degradation_ratio == 0.1

# TEST-M8-015: Zero baseline. Expected NDR = null/NoData and QA warning.
def test_m8_015_zero_baseline():
    grid = get_mock_projected_grid(shape=(2, 2))
    arrays = list(get_mock_arrays(shape=(2, 2), valid_val=0)) # zero baseline
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.normalized_degradation_ratio is None
    assert any("Baseline area is zero" in w for w in res.qa_flags["warnings"])

# TEST-M8-016: CVA statistics. Known synthetic distribution. Verify mean, median, P95, min, max, std.
def test_m8_016_cva_statistics():
    grid = get_mock_projected_grid(shape=(10, 1))
    cva_map = np.arange(10, dtype=np.float32).reshape((10, 1)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    arrays = list(get_mock_arrays(shape=(10, 1)))
    arrays[1] = cva_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    cva_res = res.cva_statistics["all_valid_wetland"]
    assert cva_res["mean"] == 4.5
    assert cva_res["median"] == 4.5
    assert cva_res["min"] == 0.0
    assert cva_res["max"] == 9.0
    assert np.isclose(cva_res["p95"], 8.55)

# TEST-M8-017: Delta statistics. Verify NDVI, GNDVI, NDRE
def test_m8_017_delta_statistics():
    grid = get_mock_projected_grid(shape=(2, 2))
    arrays = list(get_mock_arrays(shape=(2, 2), ndvi_val=0.1, gndvi_val=0.2, ndre_val=0.3))
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert np.isclose(res.delta_statistics["ndvi"]["all_valid_wetland"]["mean"], 0.1)
    assert np.isclose(res.delta_statistics["gndvi"]["all_valid_wetland"]["mean"], 0.2)
    assert np.isclose(res.delta_statistics["ndre"]["all_valid_wetland"]["mean"], 0.3)

# TEST-M8-018: NoData. Verify NoData excluded from all metrics
def test_m8_018_nodata_exclusion():
    grid = get_mock_projected_grid(shape=(2, 2))
    trans_map = np.array([[15, 255], [15, 15]], dtype=np.uint8) # 1 NoData pixel
    arrays = list(get_mock_arrays(shape=(2, 2)))
    arrays[0] = trans_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.baseline_area["pixel_count"] == 3

# TEST-M8-019: Background. Verify Background pixels contribute to class/transition metrics.
def test_m8_019_background_pixels():
    grid = get_mock_projected_grid(shape=(2, 2))
    trans_map = np.array([[0, 0], [15, 15]], dtype=np.uint8) # 2 B_to_B (Background)
    arrays = list(get_mock_arrays(shape=(2, 2)))
    arrays[0] = trans_map
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res.transition_metrics["B_to_B"]["pixel_count"] == 2
    assert res.class_area_metrics["T1"]["background"]["pixel_count"] == 2

# TEST-M8-020: CRS safety. None CRS is handled gracefully (pixel-coordinate fallback).
def test_m8_020_crs_safety():
    grid = get_mock_projected_grid(shape=(2, 2))
    grid.crs = None  # set CRS to None — now handled as pixel-space (no error)
    arrays = get_mock_arrays(shape=(2, 2))
    # With CRS=None the engine falls back to pixel-coordinate area calculation
    result = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert result is not None  # Should succeed without raising

# TEST-M8-021: Geometry mismatch. Expected clear error
def test_m8_021_geometry_mismatch():
    grid = get_mock_projected_grid(shape=(2, 2))
    arrays = list(get_mock_arrays(shape=(2, 2)))
    # Modify shape of one array
    arrays[0] = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape|mismatch|identical"):
        calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)

# TEST-M8-022: Reproducibility. Same inputs + config: identical results.
def test_m8_022_reproducibility():
    grid = get_mock_projected_grid(shape=(4, 4))
    arrays = get_mock_arrays(shape=(4, 4))
    res1 = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    res2 = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    assert res1.to_dict() == res2.to_dict()

# TEST-M8-023: Configuration. Changing severity weights must change A_SWD but not A_impact.
def test_m8_023_config_weights_change():
    grid = get_mock_projected_grid(shape=(1, 1))
    # H->M (14)
    arrays = list(get_mock_arrays(shape=(1, 1), trans_val=14, deg_val=1))
    res1 = calculate_metrics(*arrays, get_mock_config(moderate=0.35), reference_grid=grid)
    res2 = calculate_metrics(*arrays, get_mock_config(moderate=0.55), reference_grid=grid)
    # Impact area remains the same (100 m2)
    assert res1.impact_area["impact_area_m2"] == res2.impact_area["impact_area_m2"]
    # SWD area changes
    assert np.isclose(res1.severity_weighted_area["severity_weighted_area_m2"], 35.0)
    assert np.isclose(res2.severity_weighted_area["severity_weighted_area_m2"], 55.0)

# TEST-M8-024: Canopy-loss exclusion. By default: canopy loss must not alter A_SWD.
def test_m8_024_canopy_loss_exclusion():
    grid = get_mock_projected_grid(shape=(2, 2))
    # 1 pixel degradation H->M (14, deg_val=1)
    # 1 pixel canopy loss H->B (12, canopy_val=1)
    trans_map = np.array([[14, 12], [15, 15]], dtype=np.uint8)
    deg_map = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    canopy_map = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    arrays = list(get_mock_arrays(shape=(2, 2)))
    arrays[0] = trans_map
    arrays[5] = deg_map
    arrays[8] = canopy_map
    
    # 1. By default, include_canopy_loss_in_swd = False
    res_default = calculate_metrics(*arrays, get_mock_config(include_canopy_loss_in_swd=False), reference_grid=grid)
    # SWD uses moderate weight (0.35) for H->M only: 100 * 0.35 = 35.0 m2
    assert res_default.severity_weighted_area["severity_weighted_area_m2"] == 35.0
    
    # 2. When enabled, include_canopy_loss_in_swd = True
    res_enabled = calculate_metrics(*arrays, get_mock_config(include_canopy_loss_in_swd=True, background=1.0), reference_grid=grid)
    # SWD uses 100 * 0.35 + 100 * 1.0 = 135.0 m2
    assert res_enabled.severity_weighted_area["severity_weighted_area_m2"] == 135.0

# TEST-M8-025: Physical depletion unavailable.
def test_m8_025_physical_depletion_unavailable():
    grid = get_mock_projected_grid(shape=(2, 2))
    arrays = get_mock_arrays(shape=(2, 2))
    res = calculate_metrics(*arrays, get_mock_config(depletion_enabled=False), reference_grid=grid)
    assert res.wetland_extent_metrics["physical_wetland_depletion"] == "NOT_AVAILABLE"

# Additional tests for Olofsson hook interface and geographic CRS
def test_olofsson_hook():
    res = estimate_area_with_reference_samples(None)
    assert res["status"] == "NOT_AVAILABLE"
    assert "Model-derived estimate" in res["message"]

def test_geographic_crs():
    # Test geographic CRS (EPSG:4326) ellipsoidal area calculation
    crs = CRS.from_epsg(4326)
    transform = from_origin(100.0, 10.0, 0.0001, 0.0001) # ~11m x ~11m
    grid = ReferenceGrid(
        crs=crs,
        transform=transform,
        width=10,
        height=10,
        resolution=(0.0001, 0.0001),
        bounds=(100.0, 10.0 - 0.001, 100.0 + 0.001, 10.0)
    )
    arrays = get_mock_arrays(shape=(10, 10))
    res = calculate_metrics(*arrays, get_mock_config(), reference_grid=grid)
    # Ellipsoidal area must be calculated and positive
    assert res.baseline_area["area_m2"] > 0
    # Must not be the same as simple cell width * height * shape
    simple_area = 0.0001 * 0.0001 * 100
    assert res.baseline_area["area_m2"] != simple_area
