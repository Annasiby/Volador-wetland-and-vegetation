"""
backend/app/routers/datasets.py
================================
Router for dataset registration, band file uploads, and raster validation.
"""

import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response, status, Query, Body
from ..schemas import DatasetValidationResult, DatasetRegisterRequest, DatasetValidateRequest
from ..services import ProjectStore, DatasetService, TileRenderer, ENGINE_SRC

router = APIRouter(prefix="/api/projects/{project_id}/datasets", tags=["Datasets"])


import re

@router.post("/upload")
def upload_dataset_band(
    project_id: str,
    time_point: str = Form(..., example="t1"),  # t1 or t2
    band: str = Form(..., example="green"),     # green, red, rededge, nir, boundary
    file: UploadFile = File(...)
):
    project = ProjectStore.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    proj_dir = Path(project["t1_dir"]).parent

    if band == "boundary":
        dest_dir = proj_dir
        filename = f"wetland_boundary_{file.filename}"
    else:
        dest_dir = proj_dir / time_point
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = {"green": "G", "red": "R", "rededge": "RE", "nir": "NIR"}.get(band, band.upper())
        filename = f"scene_{suffix}.tif"

    target_path = dest_dir / filename
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    updates = {}
    if band == "boundary":
        updates["wetland_boundary"] = str(target_path)
    else:
        # Extract acquisition date from original filename if present (e.g. DJI_20230814...)
        date_pattern = re.compile(r'(?<![0-9])(2[0-9]{3})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(?![0-9])')
        m = date_pattern.search(file.filename or "")
        if m:
            year, month, day = m.group(1), m.group(2), m.group(3)
            if 2000 <= int(year) <= 2100:
                extracted_date = f"{year}-{month}-{day}"
                date_field = "t1_date" if time_point.lower() == "t1" else "t2_date"
                updates[date_field] = extracted_date

    if updates:
        ProjectStore.update_project(project_id, updates)

    return {
        "message": f"Uploaded {band} for {time_point}",
        "path": str(target_path)
    }


@router.post("/validate", response_model=DatasetValidationResult)
def validate_project_datasets(
    project_id: str,
    analysis_mode: Optional[str] = Query(None),
    request_body: Optional[DatasetValidateRequest] = Body(None),
):
    project = ProjectStore.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    effective_mode = "wetland"
    if analysis_mode:
        effective_mode = analysis_mode.strip().lower()
    elif request_body and request_body.analysis_mode:
        effective_mode = request_body.analysis_mode.strip().lower()
    elif project.get("analysis_mode"):
        effective_mode = project["analysis_mode"].strip().lower()

    # Persist the mode on the project
    ProjectStore.update_project(project_id, {"analysis_mode": effective_mode})
    project["analysis_mode"] = effective_mode

    is_veg = (effective_mode == "vegetation")

    t1_dir = Path(project["t1_dir"])
    t2_dir = Path(project["t2_dir"])
    wetland_boundary = project.get("wetland_boundary")

    sample_dir = ENGINE_SRC.parent / "data" / "sample_scene"
    used_sample = False

    # Check whether user has uploaded custom band files
    user_has_t1 = t1_dir.exists() and bool(list(t1_dir.glob("*.tif")) or list(t1_dir.glob("*.TIF")))
    user_has_t2 = t2_dir.exists() and bool(list(t2_dir.glob("*.tif")) or list(t2_dir.glob("*.TIF")))

    if is_veg:
        # In vegetation mode, we only check T1
        if not user_has_t1 and sample_dir.exists():
            t1_dir = sample_dir / "t1"
            used_sample = True
    else:
        # In wetland mode, fallback to sample scene if neither T1 nor T2 uploaded
        if not (user_has_t1 or user_has_t2) and sample_dir.exists():
            t1_dir = sample_dir / "t1"
            t2_dir = sample_dir / "t2"
            if not wetland_boundary:
                sample_boundary = sample_dir / "wetland_boundary.geojson"
                if sample_boundary.exists():
                    wetland_boundary = str(sample_boundary)
            used_sample = True

    band_mapping = {
        "green": ["*_G.tif", "*_G.TIF", "*green.tif"],
        "red": ["*_R.tif", "*_R.TIF", "*red.tif"],
        "rededge": ["*_RE.tif", "*_RE.TIF", "*rededge.tif"],
        "nir": ["*_NIR.tif", "*_NIR.TIF", "*nir.tif"],
    }

    result = DatasetService.inspect_and_validate(
        t1_dir=t1_dir,
        t2_dir=None if is_veg else t2_dir,
        band_mapping=band_mapping,
        analysis_mode=effective_mode,
        wetland_boundary=wetland_boundary,
    )

    if used_sample:
        result["messages"].insert(
            0,
            "Notice: No custom user band files found; validating default sample dataset package."
        )

    t1_date = project.get("t1_date")
    t2_date = project.get("t2_date")

    dates_required = (
        result["valid"]
        and not is_veg
        and not used_sample
        and (not t1_date or not t2_date)
    )

    result["dates_required"] = dates_required

    if dates_required:
        result["messages"].append(
            "Observation dates are required. "
            "Please enter the T1 and T2 acquisition dates."
        )

    current_status = project.get("status", "CREATED")
    if result["valid"]:
        if current_status not in ["PREPARATION_COMMITTED", "ANALYZED"]:
            new_status = "STAGED"
        else:
            new_status = current_status
    else:
        new_status = "INVALID"

    ProjectStore.update_project(project_id, {
        "datasets_ready": result["valid"],
        "status": new_status,
        "analysis_mode": effective_mode,
    })

    return result


@router.get("/preview-image")
def get_dataset_preview_image(project_id: str, time_point: str = "t1"):
    try:
        png_bytes = TileRenderer.render_dataset_preview_png(project_id, time_point=time_point)
        return Response(content=png_bytes, media_type="image/png")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview image: {str(exc)}")


@router.get("/preview-info")
def get_dataset_preview_info(project_id: str):
    try:
        return TileRenderer.get_dataset_preview_info(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

