"""
backend/tests/test_dataset_validation_modes.py
==============================================
Regression tests for optional wetland boundary in both analysis modes:
- Vegetation Mode:
  1. Vegetation passes without boundary.
  2. Vegetation passes with boundary.
  3. Vegetation still requires all four bands.
- Wetland Mode:
  4. Wetland passes without boundary.
  5. Wetland passes with boundary.
  6. Wetland still requires T1.
  7. Wetland still requires T2.
- Cross-mode:
  8. Vegetation with no boundary → PASS.
  9. Wetland with no boundary → PASS.
  10. Missing T2 in Wetland → FAIL.
  11. Missing T1 band in Wetland → FAIL.
  12. Missing vegetation band → FAIL.
- Analysis start:
  13. Vegetation without boundary can start.
  14. Wetland without boundary can start.
- Mode switching:
  15. Switching Wetland -> Vegetation removes T2 error.
  16. Switching Vegetation -> Wetland restores T2 requirement.
"""

import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
ENGINE_SRC = BACKEND_DIR / "Engine_module" / "src"
sys.path.insert(0, str(ENGINE_SRC))

from app.main import app
from app.services import ProjectStore

client = TestClient(app)


def make_tiff_bytes(shape=(20, 20), value=0.5):
    """Create in-memory GeoTIFF bytes."""
    memfile = io.BytesIO()
    transform = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    data = np.full(shape, value, dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:32633",
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(memfile, "w", **profile) as dst:
        dst.write(data, 1)
    memfile.seek(0)
    return memfile.getvalue()


def make_boundary_geojson_bytes():
    """Create in-memory GeoJSON boundary bytes."""
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [500010.0, 4999810.0],
                        [500190.0, 4999810.0],
                        [500190.0, 4999990.0],
                        [500010.0, 4999990.0],
                        [500010.0, 4999810.0]
                    ]]
                },
                "properties": {"name": "Test Wetland"}
            }
        ]
    }
    return json.dumps(geojson_data).encode("utf-8")


@pytest.fixture
def clean_project():
    """Creates a temporary project and cleans it up after the test."""
    res = client.post("/api/projects", json={
        "name": "Mode Validation Test",
        "description": "Testing mode-aware validation",
        "analysis_mode": "wetland"
    })
    assert res.status_code == 201
    proj = res.json()
    proj_id = proj["id"]

    yield proj_id

    # Cleanup
    client.delete(f"/api/projects/{proj_id}")


def upload_bands(proj_id: str, time_point: str, bands: list[str]):
    """Uploads specified bands for a time_point."""
    for b in bands:
        tif_bytes = make_tiff_bytes()
        res = client.post(
            f"/api/projects/{proj_id}/datasets/upload",
            data={"time_point": time_point, "band": b},
            files={"file": (f"{b}.tif", io.BytesIO(tif_bytes), "image/tiff")},
        )
        assert res.status_code == 200, f"Failed uploading {time_point} {b}: {res.text}"


def upload_boundary(proj_id: str):
    """Uploads wetland boundary geojson."""
    boundary_bytes = make_boundary_geojson_bytes()
    res = client.post(
        f"/api/projects/{proj_id}/datasets/upload",
        data={"time_point": "t1", "band": "boundary"},
        files={"file": ("wetland_boundary.geojson", io.BytesIO(boundary_bytes), "application/geo+json")},
    )
    assert res.status_code == 200, f"Failed uploading boundary: {res.text}"


# =====================================================================
# VEGETATION TESTS (1-3)
# =====================================================================

def test_1_vegetation_passes_without_boundary(clean_project):
    """1. Vegetation passes without boundary."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=vegetation")
    assert res.status_code == 200
    val_data = res.json()

    assert val_data["valid"] is True
    assert val_data["status"] in ["PASS", "WARNING"]
    assert any("Boundary: OPTIONAL" in m for m in val_data["messages"])
    assert not any("required" in m.lower() and "boundary" in m.lower() for m in val_data["messages"])


def test_2_vegetation_passes_with_boundary(clean_project):
    """2. Vegetation passes with boundary."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])
    upload_boundary(proj_id)

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=vegetation")
    assert res.status_code == 200
    val_data = res.json()

    assert val_data["valid"] is True
    assert val_data["status"] in ["PASS", "WARNING"]
    assert any("Boundary: PASS" in m for m in val_data["messages"])


def test_3_vegetation_requires_all_four_bands(clean_project):
    """3. Vegetation still requires all four bands."""
    proj_id = clean_project
    # Only upload 3 bands (missing NIR)
    upload_bands(proj_id, "t1", ["green", "red", "rededge"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=vegetation")
    assert res.status_code == 200
    val_data = res.json()

    assert val_data["valid"] is False
    assert val_data["status"] == "ERROR"
    assert any("missing bands" in m.lower() and "nir" in m.lower() for m in val_data["messages"])


# =====================================================================
# WETLAND TESTS (4-7)
# =====================================================================

def test_4_wetland_passes_without_boundary(clean_project):
    """4. Wetland passes without boundary."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])
    upload_bands(proj_id, "t2", ["green", "red", "rededge", "nir"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res.status_code == 200
    val_data = res.json()

    assert val_data["valid"] is True
    assert val_data["status"] in ["PASS", "WARNING"]
    assert any("Boundary: OPTIONAL" in m for m in val_data["messages"])
    assert not any("required" in m.lower() and "boundary" in m.lower() for m in val_data["messages"])


def test_5_wetland_passes_with_boundary(clean_project):
    """5. Wetland passes with boundary."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])
    upload_bands(proj_id, "t2", ["green", "red", "rededge", "nir"])
    upload_boundary(proj_id)

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res.status_code == 200
    val_data = res.json()

    assert val_data["valid"] is True
    assert val_data["status"] in ["PASS", "WARNING"]
    assert any("Boundary: PASS" in m for m in val_data["messages"])


def test_6_wetland_still_requires_t1(clean_project):
    """6. Wetland still requires T1."""
    proj_id = clean_project
    # Upload only T2, no T1
    upload_bands(proj_id, "t2", ["green", "red", "rededge", "nir"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res.status_code == 200
    val_data = res.json()

    assert val_data["valid"] is False
    assert val_data["status"] == "ERROR"
    assert any("t1 missing bands" in m.lower() for m in val_data["messages"])


def test_7_wetland_still_requires_t2(clean_project):
    """7. Wetland still requires T2."""
    proj_id = clean_project
    # Upload only T1, no T2
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res.status_code == 200
    val_data = res.json()

    assert val_data["valid"] is False
    assert val_data["status"] == "ERROR"
    assert any("t2 missing bands" in m.lower() for m in val_data["messages"])


# =====================================================================
# CROSS-MODE TESTS (8-12)
# =====================================================================

def test_8_cross_mode_vegetation_no_boundary_pass(clean_project):
    """8. Vegetation with no boundary → PASS."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=vegetation")
    assert res.status_code == 200
    assert res.json()["valid"] is True


def test_9_cross_mode_wetland_no_boundary_pass(clean_project):
    """9. Wetland with no boundary → PASS."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])
    upload_bands(proj_id, "t2", ["green", "red", "rededge", "nir"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res.status_code == 200
    assert res.json()["valid"] is True


def test_10_cross_mode_missing_t2_in_wetland_fail(clean_project):
    """10. Missing T2 in Wetland → FAIL."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res.status_code == 200
    assert res.json()["valid"] is False


def test_11_cross_mode_missing_t1_band_in_wetland_fail(clean_project):
    """11. Missing T1 band in Wetland → FAIL."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge"])  # missing NIR
    upload_bands(proj_id, "t2", ["green", "red", "rededge", "nir"])

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res.status_code == 200
    assert res.json()["valid"] is False


def test_12_cross_mode_missing_vegetation_band_fail(clean_project):
    """12. Missing vegetation band → FAIL."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "nir"])  # missing rededge

    res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=vegetation")
    assert res.status_code == 200
    assert res.json()["valid"] is False


# =====================================================================
# ANALYSIS START TESTS (13-14)
# =====================================================================

def test_13_analysis_start_vegetation_without_boundary(clean_project):
    """13. Vegetation without boundary can start."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])

    # Validate passes
    val_res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=vegetation")
    assert val_res.json()["valid"] is True

    # Analysis start guard allows run
    run_res = client.post("/api/runs", json={
        "project_id": proj_id,
        "config": {
            "analysis_mode": "vegetation"
        }
    })
    assert run_res.status_code == 202
    assert "run_id" in run_res.json()


def test_14_analysis_start_wetland_without_boundary(clean_project):
    """14. Wetland without boundary can start."""
    proj_id = clean_project
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])
    upload_bands(proj_id, "t2", ["green", "red", "rededge", "nir"])

    # Provide observation dates
    client.patch(f"/api/projects/{proj_id}", json={
        "t1_date": "2023-08-14",
        "t2_date": "2023-08-28"
    })

    # Validate passes
    val_res = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert val_res.json()["valid"] is True

    # Analysis start guard allows run
    run_res = client.post("/api/runs", json={
        "project_id": proj_id,
        "config": {
            "analysis_mode": "wetland",
            "use_observation_dates": True,
            "t1_date": "2023-08-14",
            "t2_date": "2023-08-28"
        }
    })
    assert run_res.status_code == 202
    assert "run_id" in run_res.json()


# =====================================================================
# MODE SWITCHING TESTS (15-16)
# =====================================================================

def test_15_switching_wetland_to_vegetation_removes_t2_error(clean_project):
    """15. Switching Wetland -> Vegetation removes T2 validation error and passes."""
    proj_id = clean_project
    # Upload only T1 bands (Green, Red, RedEdge, NIR), no T2, no boundary
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])

    # 1. Validate in Wetland mode -> FAIL due to missing T2
    res_wetland = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res_wetland.status_code == 200
    val_wetland = res_wetland.json()
    assert val_wetland["valid"] is False
    assert val_wetland["status"] == "ERROR"
    assert any("t2 missing bands" in m.lower() for m in val_wetland["messages"])

    # 2. Switch to Vegetation mode -> PASS, no T2 error
    res_veg = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=vegetation")
    assert res_veg.status_code == 200
    val_veg = res_veg.json()
    assert val_veg["valid"] is True
    assert val_veg["status"] in ["PASS", "WARNING"]
    assert not any("t2 missing bands" in m.lower() for m in val_veg["messages"])
    assert any("vegetation scene" in m.lower() for m in val_veg["messages"])
    assert any("Boundary: OPTIONAL" in m for m in val_veg["messages"])


def test_16_switching_vegetation_to_wetland_restores_t2_requirement(clean_project):
    """16. Switching Vegetation -> Wetland restores T2 requirement and fails if T2 absent."""
    proj_id = clean_project
    # Upload only T1 bands, no T2
    upload_bands(proj_id, "t1", ["green", "red", "rededge", "nir"])

    # 1. Validate in Vegetation mode -> PASS
    res_veg = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=vegetation")
    assert res_veg.status_code == 200
    assert res_veg.json()["valid"] is True

    # 2. Switch to Wetland mode -> FAIL because T2 is required in Wetland mode
    res_wetland = client.post(f"/api/projects/{proj_id}/datasets/validate?analysis_mode=wetland")
    assert res_wetland.status_code == 200
    val_wetland = res_wetland.json()
    assert val_wetland["valid"] is False
    assert val_wetland["status"] == "ERROR"
    assert any("t2 missing bands" in m.lower() for m in val_wetland["messages"])
