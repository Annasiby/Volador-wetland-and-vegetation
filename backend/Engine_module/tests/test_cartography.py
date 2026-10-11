"""
tests/test_cartography.py
===========================
Unit tests for cartographic map rendering module (categorical, continuous,
scale bar, north arrow, NoData handling, empty rasters).
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.gis_engine.output.cartography import (
    draw_north_arrow,
    draw_scale_bar,
    generate_report_maps,
    render_categorical_map,
    render_continuous_map,
)
import matplotlib.pyplot as plt


@pytest.fixture
def sample_raster_env(tmp_path):
    """Creates temporary categorical and continuous GeoTIFFs."""
    res_dir = tmp_path / "rasters"
    res_dir.mkdir()

    transform = from_origin(500000, 4000000, 10, 10)  # UTM-like projected
    height, width = 64, 64

    # Categorical raster
    cat_data = np.zeros((height, width), dtype=np.uint8)
    cat_data[10:30, 10:30] = 1  # Severe
    cat_data[30:50, 10:30] = 2  # Moderate
    cat_data[10:50, 30:50] = 3  # Healthy
    cat_data[0:5, 0:5] = 255     # NoData

    cat_path = res_dir / "categorical.tif"
    with rasterio.open(
        cat_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:32632",
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(cat_data, 1)

    # Continuous raster (e.g. CVA)
    cont_data = np.random.uniform(0.0, 1.5, size=(height, width)).astype(np.float32)
    cont_data[0:5, 0:5] = np.nan

    cont_path = res_dir / "continuous.tif"
    with rasterio.open(
        cont_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(cont_data, 1)

    return {"cat_path": cat_path, "cont_path": cont_path, "out_dir": tmp_path / "figures"}


def test_render_categorical_map(sample_raster_env):
    """Verifies rendering categorical GeoTIFF with colormap and legend."""
    out_png = sample_raster_env["out_dir"] / "test_cat.png"
    cmap_rgba = {
        0: (0.5, 0.5, 0.5, 1.0),
        1: (1.0, 0.0, 0.0, 1.0),
        2: (1.0, 0.65, 0.0, 1.0),
        3: (0.0, 0.67, 0.0, 1.0),
    }
    labels = {0: "Background", 1: "Severe", 2: "Moderate", 3: "Healthy"}

    res = render_categorical_map(
        sample_raster_env["cat_path"],
        out_png,
        title="Test Health Map",
        colormap_dict=cmap_rgba,
        class_labels=labels,
        subtitle="Unit test validation",
        dpi=150,
    )

    assert res is not None
    assert out_png.is_file()
    assert out_png.stat().st_size > 1000


def test_render_continuous_map(sample_raster_env):
    """Verifies rendering continuous floating-point GeoTIFF with colorbar."""
    out_png = sample_raster_env["out_dir"] / "test_cont.png"

    res = render_continuous_map(
        sample_raster_env["cont_path"],
        out_png,
        title="Test CVA Magnitude",
        colorbar_label="Magnitude ||ΔV||",
        cmap_name="YlOrRd",
        dpi=150,
    )

    assert res is not None
    assert out_png.is_file()
    assert out_png.stat().st_size > 1000


def test_cartography_helpers():
    """Tests north arrow and scale bar drawing routines."""
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_north_arrow(ax)
    bounds = (500000, 4000000, 505000, 4005000)
    draw_scale_bar(ax, bounds, crs_str="EPSG:32632")
    # Geographic scale bar
    geo_bounds = (12.0, 45.0, 12.1, 45.1)
    draw_scale_bar(ax, geo_bounds, crs_str="EPSG:4326")
    plt.close(fig)


def test_generate_report_maps_hierarchy(tmp_path):
    """Tests generate_report_maps across all standard subdirectories."""
    run_dir = tmp_path / "run_01"
    (run_dir / "inference").mkdir(parents=True)
    (run_dir / "change").mkdir(parents=True)
    (run_dir / "degradation").mkdir(parents=True)
    (run_dir / "hydrology").mkdir(parents=True)
    (run_dir / "fusion").mkdir(parents=True)

    transform = from_origin(500000, 4000000, 10, 10)
    data_u8 = np.ones((16, 16), dtype=np.uint8)
    data_f32 = np.ones((16, 16), dtype=np.float32)

    def _write_tif(p, data, dtype, nodata):
        with rasterio.open(
            p, "w", driver="GTiff", height=16, width=16, count=1,
            dtype=dtype, crs="EPSG:32632", transform=transform, nodata=nodata
        ) as dst:
            dst.write(data, 1)

    _write_tif(run_dir / "inference" / "t1_health.tif", data_u8, "uint8", 255)
    _write_tif(run_dir / "inference" / "t2_health.tif", data_u8, "uint8", 255)
    _write_tif(run_dir / "change" / "transition_map.tif", data_u8, "uint8", 255)
    _write_tif(run_dir / "change" / "cva_magnitude.tif", data_f32, "float32", -9999.0)
    _write_tif(run_dir / "degradation" / "degradation_map.tif", data_u8, "uint8", 255)
    _write_tif(run_dir / "hydrology" / "water_change.tif", data_u8, "uint8", 255)
    _write_tif(run_dir / "fusion" / "evidence_fusion.tif", data_u8, "uint8", 0)

    figs = generate_report_maps(run_dir, dpi=100)
    assert len(figs) >= 6
    for fig_key, fig_path in figs.items():
        assert fig_path.is_file()
        assert fig_path.stat().st_size > 500
