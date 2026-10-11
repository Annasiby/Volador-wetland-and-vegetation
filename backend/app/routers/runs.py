"""
backend/app/routers/runs.py
============================
Router for initiating analysis runs and checking run execution status.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from ..schemas import RunCreateRequest, RunStatusResponse
from ..services import RunWorker, ProjectStore, DatasetService, ENGINE_SRC

router = APIRouter(prefix="/api/runs", tags=["Runs"])


@router.post("", response_model=RunStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def launch_run(payload: RunCreateRequest):
    project = ProjectStore.get_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {payload.project_id} not found")

    t1_dir = Path(project["t1_dir"])
    t2_dir = Path(project["t2_dir"])
    sample_dir = ENGINE_SRC.parent / "data" / "sample_scene"

    effective_mode = "wetland"
    if payload.config:
        if payload.config.analysis_mode:
            effective_mode = payload.config.analysis_mode.strip().lower()
        if payload.config.analysisMode:
            effective_mode = payload.config.analysisMode.strip().lower()
    elif project.get("analysis_mode"):
        effective_mode = project["analysis_mode"].strip().lower()

    is_veg = (effective_mode == "vegetation")

    user_has_t1 = t1_dir.exists() and bool(list(t1_dir.glob("*.tif")) or list(t1_dir.glob("*.TIF")))
    user_has_t2 = t2_dir.exists() and bool(list(t2_dir.glob("*.tif")) or list(t2_dir.glob("*.TIF")))

    band_mapping = {
        "green": ["*_G.tif", "*_G.TIF", "*green.tif"],
        "red": ["*_R.tif", "*_R.TIF", "*red.tif"],
        "rededge": ["*_RE.tif", "*_RE.TIF", "*rededge.tif"],
        "nir": ["*_NIR.tif", "*_NIR.TIF", "*nir.tif"],
    }

    if is_veg:
        if user_has_t1:
            val_res = DatasetService.inspect_and_validate(
                t1_dir=t1_dir,
                t2_dir=None,
                band_mapping=band_mapping,
                analysis_mode="vegetation",
                wetland_boundary=project.get("wetland_boundary"),
            )
            if not val_res["valid"]:
                msg = "; ".join(val_res["messages"])
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot start analysis: Dataset validation failed. {msg}"
                )
        elif not sample_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot start analysis: No dataset files uploaded and no sample scene available."
            )
    else:
        wetland_boundary = project.get("wetland_boundary")
        if user_has_t1 or user_has_t2:
            val_res = DatasetService.inspect_and_validate(
                t1_dir=t1_dir,
                t2_dir=t2_dir,
                band_mapping=band_mapping,
                analysis_mode="wetland",
                wetland_boundary=wetland_boundary,
            )
            if not val_res["valid"]:
                msg = "; ".join(val_res["messages"])
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot start analysis: Dataset validation failed. {msg}"
                )
        elif not sample_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot start analysis: No dataset files uploaded and no sample scene available."
            )

    print(
        "[GIS DEBUG] backend payload manual_exclusions:",
        payload.config.manual_exclusions if payload.config else None,
    )

    print(
        "[GIS DEBUG] backend config dict:",
        payload.config.dict() if payload.config else None,
    )
    config_dict = payload.config.dict() if payload.config else None
    try:
        run_info = RunWorker.start_run(payload.project_id, config_dict)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
    return run_info


@router.get("/{run_id}", response_model=RunStatusResponse)
def get_run_status(run_id: str):
    run_info = RunWorker.get_run_status(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run_info
