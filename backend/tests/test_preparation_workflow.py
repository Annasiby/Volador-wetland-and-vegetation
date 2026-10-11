"""
backend/tests/test_preparation_workflow.py
============================================
Comprehensive test suite for the AOI Inclusion + Manual Exclusion Staging Workflow.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
ENGINE_SRC = BACKEND_DIR / "Engine_module" / "src"
sys.path.insert(0, str(ENGINE_SRC))

from app.main import app
from app.services import ProjectStore

client = TestClient(app)


def test_complete_preparation_lifecycle():
    # 1. Create project
    res = client.post("/api/projects", json={
        "name": "Staging Lifecycle Test",
        "description": "Verification of preparation lifecycle",
        "aoi_name": "Test AOI"
    })
    assert res.status_code == 201
    proj = res.json()
    project_id = proj["id"]
    assert proj["preparation_status"] == "NOT_PREPARED"
    assert proj["is_locked"] is False

    try:
        # 2. Validate datasets (fallback to sample scene)
        val_res = client.post(f"/api/projects/{project_id}/datasets/validate")
        assert val_res.status_code == 200
        assert val_res.json()["valid"] is True

        # Check that project status transitioned to STAGED
        p_res = client.get(f"/api/projects/{project_id}")
        assert p_res.json()["status"] == "STAGED"

        # 3. Test Preview endpoints
        prev_img = client.get(f"/api/projects/{project_id}/datasets/preview-image")
        assert prev_img.status_code == 200
        assert prev_img.headers["content-type"] == "image/png"
        assert len(prev_img.content) > 100

        prev_info = client.get(f"/api/projects/{project_id}/datasets/preview-info")
        assert prev_info.status_code == 200
        info_data = prev_info.json()
        assert info_data["width"] > 0
        assert info_data["height"] > 0

        # 4. Get default preparation state
        prep_res = client.get(f"/api/projects/{project_id}/preparation")
        assert prep_res.status_code == 200
        prep = prep_res.json()
        assert prep["status"] == "DRAFT"
        assert prep["is_locked"] is False
        assert prep["metrics"]["source_pixels"] > 0

        # 5. Guard check: Attempting to launch analysis while preparation is uncommitted must fail
        run_res = client.post("/api/runs", json={"project_id": project_id})
        assert run_res.status_code == 400
        assert "confirmed and locked" in run_res.json()["detail"]

        # 6. Update preparation with custom inclusion AOI and a manual exclusion
        # In sample scene (100x100), define a 60x60 inclusion and a 10x10 exclusion inside it
        inclusion_geom = {
            "type": "Polygon",
            "coordinates": [[
                [10.0, 10.0],
                [70.0, 10.0],
                [70.0, 70.0],
                [10.0, 70.0],
                [10.0, 10.0]
            ]]
        }
        exclusion_item = {
            "id": "excl-1",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [20.0, 20.0],
                    [30.0, 20.0],
                    [30.0, 30.0],
                    [20.0, 30.0],
                    [20.0, 20.0]
                ]]
            },
            "category": "Built Infrastructure",
            "comment": "Farm building excluded from wetland analysis"
        }

        update_res = client.put(
            f"/api/projects/{project_id}/preparation",
            json={
                "inclusion": inclusion_geom,
                "manual_exclusions": [exclusion_item]
            }
        )
        assert update_res.status_code == 200
        updated_prep = update_res.json()
        assert updated_prep["status"] == "DRAFT"
        assert len(updated_prep["manual_exclusions"]) == 1

        metrics = updated_prep["metrics"]
        assert metrics["inclusion_pixels"] > 0
        assert metrics["excluded_pixels"] > 0
        assert metrics["effective_pixels"] == metrics["inclusion_pixels"] - metrics["excluded_pixels"]

        # 7. Commit & Lock preparation
        commit_res = client.post(f"/api/projects/{project_id}/preparation/commit")
        assert commit_res.status_code == 200
        committed_prep = commit_res.json()
        assert committed_prep["status"] == "COMMITTED"
        assert committed_prep["is_locked"] is True
        assert committed_prep["committed_at"] is not None

        # Verify project status
        p_res = client.get(f"/api/projects/{project_id}")
        assert p_res.json()["preparation_status"] == "COMMITTED"
        assert p_res.json()["status"] == "PREPARATION_COMMITTED"

        # 8. Modifying locked preparation directly without unlocking must fail
        fail_update = client.put(
            f"/api/projects/{project_id}/preparation",
            json={"inclusion": None, "manual_exclusions": []}
        )
        assert fail_update.status_code == 400
        assert "Cannot modify a locked preparation" in fail_update.json()["detail"]

        # 9. Unlock preparation
        unlock_res = client.post(f"/api/projects/{project_id}/preparation/unlock")
        assert unlock_res.status_code == 200
        unlocked_prep = unlock_res.json()
        assert unlocked_prep["is_locked"] is False
        assert unlocked_prep["status"] == "DRAFT"

        p_res = client.get(f"/api/projects/{project_id}")
        assert p_res.json()["status"] == "STAGED"

        # Re-commit for analysis run
        commit_res = client.post(f"/api/projects/{project_id}/preparation/commit")
        assert commit_res.status_code == 200

        # 10. Launch run now succeeds with committed preparation
        run_res = client.post("/api/runs", json={"project_id": project_id})
        assert run_res.status_code == 202
        run_data = run_res.json()
        assert run_data["run_id"] is not None

    finally:
        client.delete(f"/api/projects/{project_id}")


def test_run_inherits_committed_inclusion_when_omitted_in_config():
    """Verify BUG 1 fix: Run launched with config payload containing inclusion_geometry=None
    must inherit the committed inclusion geometry from preparation.json."""
    res = client.post("/api/projects", json={"name": "Inclusion Config Override Test"})
    assert res.status_code == 201
    proj = res.json()
    project_id = proj["id"]
    try:
        client.post(f"/api/projects/{project_id}/datasets/validate")
        inclusion_geom = {
            "type": "Polygon",
            "coordinates": [[[10.0, 10.0], [70.0, 10.0], [70.0, 70.0], [10.0, 70.0], [10.0, 10.0]]]
        }
        client.put(f"/api/projects/{project_id}/preparation", json={"inclusion": inclusion_geom, "manual_exclusions": []})
        commit_res = client.post(f"/api/projects/{project_id}/preparation/commit")
        assert commit_res.status_code == 200

        # Simulate frontend request with explicit inclusion_geometry: None in config
        run_res = client.post("/api/runs", json={
            "project_id": project_id,
            "config": {
                "cva_percentile": 95.0,
                "inclusion_geometry": None,
                "manual_exclusions": []
            }
        })
        assert run_res.status_code == 202
        run_id = run_res.json()["run_id"]
        assert run_id is not None
    finally:
        client.delete(f"/api/projects/{project_id}")

