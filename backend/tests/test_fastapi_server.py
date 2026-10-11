"""
backend/tests/test_fastapi_server.py
====================================
Tests for the FastAPI application layer endpoints.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
ENGINE_SRC = BACKEND_DIR / "Engine_module" / "src"
sys.path.insert(0, str(ENGINE_SRC))

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"


def test_project_crud_flow():
    # 1. Create project
    res = client.post("/api/projects", json={
        "name": "Test Project",
        "description": "Integration test",
        "aoi_name": "Test Wetland"
    })
    assert res.status_code == 201
    proj = res.json()
    project_id = proj["id"]
    assert proj["name"] == "Test Project"

    # 2. Get project
    res = client.get(f"/api/projects/{project_id}")
    assert res.status_code == 200

    # 3. Validate dataset (uses sample dataset fallback)
    res = client.post(f"/api/projects/{project_id}/datasets/validate")
    assert res.status_code == 200
    val_data = res.json()
    assert val_data["status"] in ["PASS", "WARNING"]

    # 4. Launch run
    res = client.post("/api/runs", json={
        "project_id": project_id
    })
    assert res.status_code == 202
    run_data = res.json()
    run_id = run_data["run_id"]
    assert run_id is not None

    # 5. Check run status
    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
    status_data = res.json()
    assert status_data["run_id"] == run_id

    # 6. Delete project
    res = client.delete(f"/api/projects/{project_id}")
    assert res.status_code == 204


def test_real_tiff_upload_and_validation():
    desktop = Path(r"C:\Users\HP\Desktop")
    files_map = {
        "green": desktop / "DJI_20230814123320_0001_MS_G.TIF",
        "red": desktop / "DJI_20230814123320_0001_MS_R.TIF",
        "rededge": desktop / "DJI_20230814123320_0001_MS_RE.TIF",
        "nir": desktop / "DJI_20230814123320_0001_MS_NIR.TIF",
    }
    for p in files_map.values():
        if not p.exists():
            return  # Skip if local desktop test files are missing

    # 1. Create project
    res = client.post("/api/projects", json={"name": "Real TIFF Upload Test"})
    assert res.status_code == 201
    proj_id = res.json()["id"]

    try:
        # 2. Upload bands to T1 & T2
        for tp in ["t1", "t2"]:
            for band, path in files_map.items():
                with open(path, "rb") as f:
                    up_res = client.post(
                        f"/api/projects/{proj_id}/datasets/upload",
                        data={"time_point": tp, "band": band},
                        files={"file": (path.name, f, "image/tiff")},
                    )
                assert up_res.status_code == 200

        # Upload boundary for wetland mode validation
        sample_boundary = ENGINE_SRC.parent / "data" / "sample_scene" / "wetland_boundary.geojson"
        if sample_boundary.exists():
            with open(sample_boundary, "rb") as f:
                client.post(
                    f"/api/projects/{proj_id}/datasets/upload",
                    data={"time_point": "t1", "band": "boundary"},
                    files={"file": (sample_boundary.name, f, "application/geo+json")},
                )

        # 3. Validate project datasets
        val_res = client.post(f"/api/projects/{proj_id}/datasets/validate")
        assert val_res.status_code == 200
        val_data = val_res.json()
        assert val_data["valid"] is True
        assert any("T1: Green PASS" in m for m in val_data["messages"])
        assert any("T2: Green PASS" in m for m in val_data["messages"])
        assert val_data["metadata"]["width"] == 2592
        assert val_data["metadata"]["height"] == 1944

        # 4. Launch run
        run_res = client.post("/api/runs", json={"project_id": proj_id})
        assert run_res.status_code == 202
        assert run_res.json()["run_id"] is not None

    finally:
        client.delete(f"/api/projects/{proj_id}")

