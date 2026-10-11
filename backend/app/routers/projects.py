"""
backend/app/routers/projects.py
================================
Router for managing project workspaces.
"""

from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status
from ..schemas import Project, ProjectCreate, PreparationState, PreparationUpdateRequest
from ..services import ProjectStore, PreparationService, now_iso

router = APIRouter(prefix="/api/projects", tags=["Projects"])

class ProjectDatesUpdate(BaseModel):
    t1_date: str
    t2_date: str


@router.post("", response_model=Project, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate):
    project = ProjectStore.create_project(
        name=payload.name,
        description=payload.description,
        aoi_name=payload.aoi_name,
        analysis_mode=payload.analysis_mode,
        t1_date=payload.t1_date,
        t2_date=payload.t2_date,
    )
    return project


@router.get("", response_model=List[Project])
def list_projects():
    projects = ProjectStore.load_projects()
    return list(projects.values())


@router.get("/{project_id}", response_model=Project)
def get_project(project_id: str):
    project = ProjectStore.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return project

@router.patch("/{project_id}/dates")
def update_project_dates(project_id: str, payload: ProjectDatesUpdate):
    project = ProjectStore.get_project(project_id)

    if not project:
        raise HTTPException(
            status_code=404,
            detail=f"Project {project_id} not found"
        )

    # Validate date format and chronological order.
    from datetime import datetime

    try:
        t1 = datetime.strptime(payload.t1_date, "%Y-%m-%d")
        t2 = datetime.strptime(payload.t2_date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {exc}"
        )

    if t1 > t2:
        raise HTTPException(
            status_code=400,
            detail="T1 date must not be later than T2 date."
        )

    ProjectStore.update_project(project_id, {
        "t1_date": payload.t1_date,
        "t2_date": payload.t2_date,
    })

    updated_project = ProjectStore.get_project(project_id)

    return updated_project

@router.post("/{project_id}/reset", response_model=Project)
def reset_project(project_id: str):
    project = ProjectStore.reset_project(project_id)

    if not project:
        raise HTTPException(
            status_code=404,
            detail=f"Project {project_id} not found",
        )

    return project

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str):
    success = ProjectStore.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return None


@router.get("/{project_id}/preparation", response_model=PreparationState)
def get_project_preparation(project_id: str):
    project = ProjectStore.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    prep = ProjectStore.load_preparation(project_id)
    if not prep:
        prep = ProjectStore.initialize_default_preparation(project_id)
    return prep


@router.put("/{project_id}/preparation", response_model=PreparationState)
def update_project_preparation(project_id: str, payload: PreparationUpdateRequest):
    project = ProjectStore.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    prep = ProjectStore.load_preparation(project_id)
    if not prep:
        prep = ProjectStore.initialize_default_preparation(project_id)

    if prep.get("is_locked", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify a locked preparation. Unlock it first."
        )

    clean_exclusions = []
    for exc in payload.manual_exclusions:
        exc_dict = exc.dict() if hasattr(exc, "dict") else exc
        if not exc_dict.get("comment", "").strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Exclusion {exc_dict.get('id', 'item')} requires a non-empty comment."
            )
        clean_exclusions.append(exc_dict)

    metrics = PreparationService.calculate_metrics(
        project_id=project_id,
        inclusion_geom=payload.inclusion,
        manual_exclusions=clean_exclusions
    )

    prep["inclusion"] = payload.inclusion
    prep["manual_exclusions"] = clean_exclusions
    prep["metrics"] = metrics
    prep["status"] = "DRAFT"
    prep["updated_at"] = now_iso()

    ProjectStore.save_preparation(project_id, prep)
    ProjectStore.update_project(project_id, {
        "preparation_status": "DRAFT",
        "is_locked": False
    })

    return prep


@router.post("/{project_id}/preparation/commit", response_model=PreparationState)
def commit_project_preparation(project_id: str):
    project = ProjectStore.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    prep = ProjectStore.load_preparation(project_id)
    if not prep:
        prep = ProjectStore.initialize_default_preparation(project_id)

    metrics = prep.get("metrics", {})
    eff_px = metrics.get("effective_pixels", 0)
    if eff_px <= 0:
        metrics = PreparationService.calculate_metrics(
            project_id,
            prep.get("inclusion"),
            prep.get("manual_exclusions")
        )
        prep["metrics"] = metrics
        if metrics.get("effective_pixels", 0) <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot lock preparation: Effective analysis area is 0 pixels. Please adjust inclusion or exclusions."
            )

    prep["status"] = "COMMITTED"
    prep["is_locked"] = True
    prep["committed_at"] = now_iso()
    prep["updated_at"] = now_iso()

    ProjectStore.save_preparation(project_id, prep)
    ProjectStore.update_project(project_id, {
        "preparation_status": "COMMITTED",
        "is_locked": True,
        "status": "PREPARATION_COMMITTED"
    })

    return prep


@router.post("/{project_id}/preparation/unlock", response_model=PreparationState)
def unlock_project_preparation(project_id: str):
    project = ProjectStore.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    prep = ProjectStore.load_preparation(project_id)
    if not prep:
        prep = ProjectStore.initialize_default_preparation(project_id)

    prep["is_locked"] = False
    prep["status"] = "DRAFT"
    prep["updated_at"] = now_iso()

    ProjectStore.save_preparation(project_id, prep)
    proj_updates = {
        "preparation_status": "DRAFT",
        "is_locked": False
    }
    if project.get("status") == "PREPARATION_COMMITTED":
        proj_updates["status"] = "STAGED"

    ProjectStore.update_project(project_id, proj_updates)

    return prep

