"""
backend/app/schemas.py
======================
Pydantic API schemas for the GIS Wetland Degradation & Depletion Engine.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field


# --- Projects ---
class ProjectCreate(BaseModel):
    name: str = Field(..., example="Wetland Degradation Assessment")
    description: Optional[str] = Field(None, example="Sentinel-2 temporal analysis")
    aoi_name: Optional[str] = Field("Central Reserve", example="Central Reserve")
    analysis_mode: Optional[str] = Field("wetland", example="wetland")
    t1_date: Optional[str] = Field(None, example="2023-08-14")
    t2_date: Optional[str] = Field(None, example="2023-08-14")


class Project(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    aoi_name: Optional[str] = None
    created_at: str
    analysis_mode: Optional[str] = "wetland"
    t1_date: Optional[str] = None
    t2_date: Optional[str] = None
    status: str = "CREATED"
    datasets_ready: bool = False
    latest_run_id: Optional[str] = None
    preparation_status: str = "NOT_PREPARED"  # NOT_PREPARED, DRAFT, COMMITTED
    is_locked: bool = False


# --- Datasets ---
class BandPaths(BaseModel):
    green: Optional[str] = None
    red: Optional[str] = None
    rededge: Optional[str] = None
    nir: Optional[str] = None


class DatasetMetadata(BaseModel):
    date: Optional[str] = None
    crs: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    resolution: Optional[str] = None
    transform: Optional[str] = None
    nodata: Optional[float] = None
    dtype: Optional[str] = None


class DatasetValidationResult(BaseModel):
    status: str = "NOT_VALIDATED"  # PASS, WARNING, ERROR, NOT_VALIDATED
    valid: bool = False
    messages: List[str] = []
    metadata: Optional[DatasetMetadata] = None
    dates_required: bool = False


class DatasetValidateRequest(BaseModel):
    analysis_mode: Optional[str] = "wetland"

class DatasetState(BaseModel):
    id: str
    label: str
    metadata: DatasetMetadata
    bands: BandPaths
    validation: str = "NOT_VALIDATED"
    wetland_boundary: Optional[str] = None


class DatasetRegisterRequest(BaseModel):
    t1_dir: Optional[str] = None
    t2_dir: Optional[str] = None
    # Dates must be user-provided (YYYY-MM-DD) or None (extracted from filenames later)
    t1_date: Optional[str] = None
    t2_date: Optional[str] = None
    wetland_boundary: Optional[str] = None

class ManualExclusion(BaseModel):
    id: str
    geometry: Dict[str, Any]
    category: str
    comment: str = Field(..., min_length=1)


# --- Preparation / Staging ---
class InclusionArea(BaseModel):
    geometry: Dict[str, Any]
    label: Optional[str] = "Analysis Area Boundary"


class PreparationMetrics(BaseModel):
    source_pixels: int = 0
    inclusion_pixels: int = 0
    excluded_pixels: int = 0
    effective_pixels: int = 0
    source_ha: Optional[float] = None
    inclusion_ha: Optional[float] = None
    excluded_ha: Optional[float] = None
    effective_ha: Optional[float] = None
    area_basis: str = "PIXEL_COUNT_ONLY"  # PIXEL_COUNT_ONLY or RASTER_CRS_AND_TRANSFORM


class PreparationState(BaseModel):
    project_id: str
    status: str = "NOT_PREPARED"  # NOT_PREPARED, DRAFT, COMMITTED
    is_locked: bool = False
    committed_at: Optional[str] = None
    updated_at: Optional[str] = None
    inclusion: Optional[Dict[str, Any]] = None
    manual_exclusions: List[ManualExclusion] = Field(default_factory=list)
    metrics: PreparationMetrics = Field(default_factory=PreparationMetrics)


class PreparationUpdateRequest(BaseModel):
    inclusion: Optional[Dict[str, Any]] = None
    manual_exclusions: List[ManualExclusion] = Field(default_factory=list)

    
# --- Analysis Run ---
class RunConfig(BaseModel):
    # Analysis mode:
    # - vegetation: vegetation analysis only
    # - wetland: vegetation analysis + wetland-specific analysis
    # Defaults to wetland for backward compatibility.
    analysis_mode: Literal["vegetation", "wetland"] = "wetland"
    analysisMode: Optional[Literal["vegetation", "wetland"]] = None
    
    # When enabled, T1/T2 observation dates are required.
    # When disabled, temporal T1 → T2 analysis may proceed without dates.
    use_observation_dates: bool = True
    
    t1_date: Optional[str] = None
    t2_date: Optional[str] = None

    cva_threshold_method: str = "percentile"
    cva_percentile: float = 95.0
    ndwi_threshold: float = 0.0
    min_change_area_m2: float = 0.0

    # User-defined geographic inclusion area (AOI boundary)
    inclusion_geometry: Optional[Dict[str, Any]] = None

    # User-defined geographic areas excluded from analysis.
    manual_exclusions: List[ManualExclusion] = Field(default_factory=list)

class RunCreateRequest(BaseModel):
    project_id: str
    config: Optional[RunConfig] = None


class RunStatusResponse(BaseModel):
    run_id: str
    project_id: str
    status: str  # QUEUED, VALIDATING, ALIGNING, PREPROCESSING, INFERENCE, CHANGE_ANALYSIS, HYDROLOGY, METRICS, VALIDATION, REPORTING, COMPLETED, FAILED
    progress: int = 0  # 0 to 100
    current_step: str = ""
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


# --- Layer ---
class MapLayerInfo(BaseModel):
    id: str
    name: str
    type: str = "raster"  # raster, vector
    url: str
    format: str = "png"
    bounds: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat] or bbox in EPSG:4326
    crs: str = "EPSG:4326"
    opacity: float = 0.8
    legend: Optional[List[Dict[str, Any]]] = None
    ready: bool = True


# --- API Errors ---
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class APIErrorResponse(BaseModel):
    error: ErrorDetail
