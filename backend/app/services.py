"""
backend/app/services.py
=======================
Application services for project persistence, background job execution,
and GIS layer tile rendering.
"""

import re
import sys
import json
import logging
import io
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image

# Ensure Engine_module/src is in sys.path
ENGINE_SRC = Path(__file__).resolve().parent.parent / "Engine_module" / "src"
if str(ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(ENGINE_SRC))

from gis_engine.api.service import analyze_wetland
from gis_engine.ingestion.loader import RasterLoader
from gis_engine.ingestion.validator import InputValidator
from gis_engine.alignment.spatial import ReferenceGrid
from gis_engine.wetland.exclusion import build_inclusion_mask, build_manual_exclusion_mask

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Shared thread pool for background runs
executor = ThreadPoolExecutor(max_workers=2)

# In-memory run state tracking for fast polling + JSON persistence
RUN_STATES: Dict[str, Dict[str, Any]] = {}


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_date_from_directory(directory: Path) -> Optional[str]:
    """
    Attempts to extract an acquisition date from filenames in a dataset directory.
    Supports DJI-style filenames (e.g., DJI_20230814123320_0001_MS_G.TIF)
    and generic YYYYMMDD patterns anywhere in the filename.

    Returns ISO date string 'YYYY-MM-DD' if found, else None.
    """
    date_pattern = re.compile(r'(?<![0-9])(2[0-9]{3})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(?![0-9])')
    candidates = list(directory.glob('*.tif')) + list(directory.glob('*.TIF'))
    for fpath in candidates:
        m = date_pattern.search(fpath.name)
        if m:
            year, month, day = m.group(1), m.group(2), m.group(3)
            # Sanity check: year in reasonable range
            if 2000 <= int(year) <= 2100:
                return f"{year}-{month}-{day}"
    return None


class ProjectStore:
    """Manages project workspaces and dataset configurations on disk."""

    @staticmethod
    def _projects_file() -> Path:
        return STORAGE_DIR / "projects.json"

    @classmethod
    def load_projects(cls) -> Dict[str, Dict[str, Any]]:
        file_path = cls._projects_file()
        if not file_path.exists():
            return {}
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def save_projects(cls, projects: Dict[str, Dict[str, Any]]):
        with open(cls._projects_file(), "w") as f:
            json.dump(projects, f, indent=2)

    @classmethod
    def create_project(
        cls,
        name: str,
        description: Optional[str],
        aoi_name: Optional[str],
        analysis_mode: Optional[str] = "wetland",
        t1_date: Optional[str] = None,
        t2_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        projects = cls.load_projects()
        now_ts = int(time.time() * 1000)
        project_id = f"prj_{now_ts}"
        while project_id in projects or (STORAGE_DIR / "projects" / project_id).exists():
            now_ts += 1
            project_id = f"prj_{now_ts}"
        proj_dir = STORAGE_DIR / "projects" / project_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "datasets" / "t1").mkdir(parents=True, exist_ok=True)
        (proj_dir / "datasets" / "t2").mkdir(parents=True, exist_ok=True)
        (proj_dir / "runs").mkdir(parents=True, exist_ok=True)

        project_data = {
            "id": project_id,
            "name": name,
            "description": description or "",
            "aoi_name": aoi_name or "Wetland AOI",
            "created_at": now_iso(),
            "analysis_mode": (analysis_mode or "wetland").strip().lower(),
            # Dates are NOT defaulted — they must come from uploaded filenames
            # or be explicitly set by the user.
            "t1_date": t1_date or None,
            "t2_date": t2_date or None,
            "status": "CREATED",
            "datasets_ready": False,
            "t1_dir": str(proj_dir / "datasets" / "t1"),
            "t2_dir": str(proj_dir / "datasets" / "t2"),
            "wetland_boundary": None,
            "latest_run_id": None,
            "preparation_status": "NOT_PREPARED",
            "is_locked": False,
        }
        projects[project_id] = project_data
        cls.save_projects(projects)
        return project_data

    @classmethod
    def get_project(cls, project_id: str) -> Optional[Dict[str, Any]]:
        projects = cls.load_projects()
        return projects.get(project_id)

    @classmethod
    def delete_project(cls, project_id: str) -> bool:
        projects = cls.load_projects()
        if project_id in projects:
            del projects[project_id]
            cls.save_projects(projects)
            proj_dir = STORAGE_DIR / "projects" / project_id
            if proj_dir.exists():
                shutil.rmtree(proj_dir, ignore_errors=True)
            return True
        return False

    @classmethod
    def reset_project(cls, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Reset a project workspace to a blank analysis session.

        The project record itself is preserved.
        Uploaded datasets and previous run outputs are removed.
        Project metadata related to analysis state is reset.
        """

        projects = cls.load_projects()

        if project_id not in projects:
            return None

        project = projects[project_id]

        # --------------------------------------------------------
        # Clear uploaded T1/T2 datasets
        # --------------------------------------------------------

        for key in ("t1_dir", "t2_dir"):
            directory = Path(project[key])

            directory.mkdir(parents=True, exist_ok=True)

            for item in directory.iterdir():
                try:
                    if item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except OSError as exc:
                    logger.warning(
                        "Failed to remove dataset item %s: %s",
                        item,
                        exc,
                    )

        # --------------------------------------------------------
        # Clear previous analysis runs
        # --------------------------------------------------------

        runs_dir = (
            STORAGE_DIR
            / "projects"
            / project_id
            / "runs"
        )

        if runs_dir.exists():
            for item in runs_dir.iterdir():
                try:
                    if item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except OSError as exc:
                    logger.warning(
                        "Failed to remove run item %s: %s",
                        item,
                        exc,
                    )

        runs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Remove in-memory run states belonging to this project
        # --------------------------------------------------------

        run_ids_to_remove = [
            run_id
            for run_id, run_info in RUN_STATES.items()
            if run_info.get("project_id") == project_id
        ]

        for run_id in run_ids_to_remove:
            RUN_STATES.pop(run_id, None)

        # --------------------------------------------------------
        # Reset project metadata
        # --------------------------------------------------------

        project.update(
            {
                "t1_date": None,
                "t2_date": None,
                "status": "CREATED",
                "datasets_ready": False,
                "wetland_boundary": None,
                "latest_run_id": None,
                "preparation_status": "NOT_PREPARED",
                "is_locked": False,
            }
        )

        # Clear preparation file if present
        prep_file = cls._preparation_file(project_id)
        if prep_file.exists():
            try:
                prep_file.unlink()
            except OSError:
                pass

        cls.save_projects(projects)

        return project

    @classmethod
    def update_project(cls, project_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        projects = cls.load_projects()
        if project_id not in projects:
            return None
        projects[project_id].update(updates)
        cls.save_projects(projects)
        return projects[project_id]

    @classmethod
    def _preparation_file(cls, project_id: str) -> Path:
        return STORAGE_DIR / "projects" / project_id / "preparation.json"

    @classmethod
    def load_preparation(cls, project_id: str) -> Optional[Dict[str, Any]]:
        pfile = cls._preparation_file(project_id)
        if not pfile.exists():
            return None
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read preparation file %s: %s", pfile, e)
            return None

    @classmethod
    def save_preparation(cls, project_id: str, prep_data: Dict[str, Any]):
        pfile = cls._preparation_file(project_id)
        pfile.parent.mkdir(parents=True, exist_ok=True)
        with open(pfile, "w", encoding="utf-8") as f:
            json.dump(prep_data, f, indent=2)

    @classmethod
    def initialize_default_preparation(cls, project_id: str) -> Dict[str, Any]:
        existing = cls.load_preparation(project_id)
        if existing:
            return existing
        metrics = PreparationService.calculate_metrics(project_id, None, [])
        default_prep = {
            "project_id": project_id,
            "status": "DRAFT",
            "is_locked": False,
            "committed_at": None,
            "updated_at": now_iso(),
            "inclusion": None,
            "manual_exclusions": [],
            "metrics": metrics,
        }
        cls.save_preparation(project_id, default_prep)
        cls.update_project(project_id, {"preparation_status": "DRAFT", "is_locked": False})
        return default_prep


class PreparationService:
    """Service for managing AOI inclusion, manual exclusions, and staging metrics."""

    @staticmethod
    def _find_reference_raster(project: Dict[str, Any]) -> Optional[Path]:
        t1_dir = Path(project.get("t1_dir", ""))
        if t1_dir.exists():
            candidates = list(t1_dir.glob("*.tif")) + list(t1_dir.glob("*.TIF"))
            if candidates:
                for c in candidates:
                    if "_G." in c.name or "green" in c.name.lower():
                        return c
                return candidates[0]

        # Check sample scene fallback
        sample_dir = ENGINE_SRC.parent / "data" / "sample_scene" / "t1"
        if sample_dir.exists():
            sample_candidates = list(sample_dir.glob("*.tif")) + list(sample_dir.glob("*.TIF"))
            if sample_candidates:
                for c in sample_candidates:
                    if "_G." in c.name or "green" in c.name.lower():
                        return c
                return sample_candidates[0]

        return None

    @classmethod
    def calculate_metrics(
        cls,
        project_id: str,
        inclusion_geom: Optional[Dict[str, Any]] = None,
        manual_exclusions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        project = ProjectStore.get_project(project_id)
        empty_metrics = {
            "source_pixels": 0,
            "inclusion_pixels": 0,
            "excluded_pixels": 0,
            "effective_pixels": 0,
            "source_ha": None,
            "inclusion_ha": None,
            "excluded_ha": None,
            "effective_ha": None,
            "area_basis": "PIXEL_COUNT_ONLY"
        }
        if not project:
            return empty_metrics

        raster_path = cls._find_reference_raster(project)
        if not raster_path or not raster_path.exists():
            return empty_metrics

        try:
            with rasterio.open(raster_path) as src:
                width = src.width
                height = src.height
                transform = src.transform
                crs = src.crs

                res_x, res_y = abs(transform.a), abs(transform.e)
                is_projected = bool(crs and crs.is_projected)
                pixel_area_ha = (res_x * res_y) / 10000.0 if is_projected else None

                bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                grid = ReferenceGrid(
                    crs=crs,
                    transform=transform,
                    width=width,
                    height=height,
                    resolution=(res_x, res_y),
                    bounds=bounds,
                )

            source_pixels = int(width * height)

            # 1. Inclusion mask
            if inclusion_geom:
                inc_mask = build_inclusion_mask(inclusion_geom, grid)
            else:
                inc_mask = np.ones((height, width), dtype=np.uint8)
            inclusion_pixels = int(np.sum(inc_mask == 1))

            # 2. Exclusion mask
            clean_exclusions = []
            if manual_exclusions:
                for item in manual_exclusions:
                    if hasattr(item, "dict"):
                        clean_exclusions.append(item.dict())
                    elif isinstance(item, dict):
                        clean_exclusions.append(item)

            if clean_exclusions:
                excl_mask = build_manual_exclusion_mask(clean_exclusions, grid)
                excluded_pixels = int(np.sum((excl_mask == 1) & (inc_mask == 1)))
                effective_mask = (inc_mask == 1) & (excl_mask == 0)
                effective_pixels = int(np.sum(effective_mask))
            else:
                excluded_pixels = 0
                effective_pixels = inclusion_pixels

            area_basis = "RASTER_CRS_AND_TRANSFORM" if pixel_area_ha is not None else "PIXEL_COUNT_ONLY"

            return {
                "source_pixels": source_pixels,
                "inclusion_pixels": inclusion_pixels,
                "excluded_pixels": excluded_pixels,
                "effective_pixels": effective_pixels,
                "source_ha": round(source_pixels * pixel_area_ha, 4) if pixel_area_ha else None,
                "inclusion_ha": round(inclusion_pixels * pixel_area_ha, 4) if pixel_area_ha else None,
                "excluded_ha": round(excluded_pixels * pixel_area_ha, 4) if pixel_area_ha else None,
                "effective_ha": round(effective_pixels * pixel_area_ha, 4) if pixel_area_ha else None,
                "area_basis": area_basis
            }
        except Exception as e:
            logger.warning("Error calculating preparation metrics for project %s: %s", project_id, e)
            return empty_metrics



class DatasetService:
    """Handles dataset verification, band inspection, and validation."""

    @staticmethod
    def inspect_and_validate(
        t1_dir: Path,
        t2_dir: Optional[Path] = None,
        band_mapping: Optional[Dict[str, List[str]]] = None,
        analysis_mode: str = "wetland",
        wetland_boundary: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if band_mapping is None:
            band_mapping = {
                "green": ["*_G.tif", "*_G.TIF", "*green.tif"],
                "red": ["*_R.tif", "*_R.TIF", "*red.tif"],
                "rededge": ["*_RE.tif", "*_RE.TIF", "*rededge.tif"],
                "nir": ["*_NIR.tif", "*_NIR.TIF", "*nir.tif"],
            }

        loader = RasterLoader(band_mapping=band_mapping)
        messages = []
        status = "PASS"
        meta_dict = {}

        mode = (analysis_mode or "wetland").strip().lower()
        is_veg = (mode == "vegetation")

        t1_paths = {}
        missing_t1 = []

        for band_name in ["green", "red", "rededge", "nir"]:
            try:
                t1_paths[band_name] = loader._find_band_file(t1_dir, band_name)
            except (FileNotFoundError, Exception):
                t1_paths[band_name] = None
                missing_t1.append(band_name)

        if missing_t1:
            scene_label = "Vegetation Scene" if is_veg else "T1"
            messages.append(f"{scene_label} missing bands: {', '.join(missing_t1)}")
            status = "ERROR"
        else:
            scene_label = "Vegetation Scene" if is_veg else "T1"
            messages.append(f"{scene_label}: Green PASS, Red PASS, RedEdge PASS, NIR PASS")

        t2_paths = {}
        if not is_veg:
            missing_t2 = []
            if not t2_dir or not Path(t2_dir).exists():
                missing_t2 = ["green", "red", "rededge", "nir"]
            else:
                for band_name in ["green", "red", "rededge", "nir"]:
                    try:
                        t2_paths[band_name] = loader._find_band_file(Path(t2_dir), band_name)
                    except (FileNotFoundError, Exception):
                        t2_paths[band_name] = None
                        missing_t2.append(band_name)

            if missing_t2:
                messages.append(f"T2 missing bands: {', '.join(missing_t2)}")
                status = "ERROR"
            else:
                messages.append("T2: Green PASS, Red PASS, RedEdge PASS, NIR PASS")

        # Boundary check (OPTIONAL in both modes)
        if wetland_boundary:
            boundary_path = Path(wetland_boundary)
            if not boundary_path.exists() or not boundary_path.is_file():
                messages.append(f"Wetland boundary file not found: {boundary_path}")
                status = "ERROR"
            else:
                messages.append(f"Boundary: PASS ({boundary_path.name})")
        else:
            messages.append("Boundary: OPTIONAL")

        if status != "ERROR":
            try:
                # Validate spatial integrity using InputValidator
                validator = InputValidator(require_crs=False)
                valid_t1 = validator.validate_files(t1_paths)
                if not is_veg:
                    valid_t2 = validator.validate_files(t2_paths)

                # Inspect metadata from T1 Green
                green_path = t1_paths["green"]
                with rasterio.open(green_path) as src:
                    crs_str = src.crs.to_string() if src.crs else "No CRS"
                    res_x, res_y = abs(src.transform.a), abs(src.transform.e)
                    meta_dict = {
                        "crs": crs_str,
                        "width": src.width,
                        "height": src.height,
                        "resolution": f"{res_x:.1f}m x {res_y:.1f}m",
                        "transform": str(src.transform),
                        "nodata": float(src.nodata) if src.nodata is not None else -9999.0,
                        "dtype": str(src.dtypes[0])
                    }

                messages.append("All raster bands successfully loaded and validated.")
                if crs_str == "No CRS":
                    status = "WARNING"
                    messages.append("Dataset lacks spatial CRS projection information.")
            except Exception as e:
                status = "ERROR"
                messages.append(f"Validation failed: {str(e)}")

        messages.append(f"Overall: {status}")

        return {
            "status": status,
            "valid": (status in ["PASS", "WARNING"]),
            "messages": messages,
            "metadata": meta_dict
        }


class RunWorker:
    """Manages background async execution of the GIS engine pipeline."""

    @classmethod
    def start_run(cls, project_id: str, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        project = ProjectStore.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Preparation guard: Check if preparation is committed
        prep = ProjectStore.load_preparation(project_id)
        if prep:
            if prep.get("status") != "COMMITTED" or not prep.get("is_locked", False):
                raise ValueError(
                    "Analysis cannot be started: Preparation area has not been confirmed and locked. "
                    "Please finalize the analysis area in the Prepare Area stage."
                )
            if config_overrides is None:
                config_overrides = {}
            if not config_overrides.get("inclusion_geometry") and prep.get("inclusion"):
                config_overrides["inclusion_geometry"] = prep.get("inclusion")
            if not config_overrides.get("manual_exclusions") and prep.get("manual_exclusions"):
                config_overrides["manual_exclusions"] = prep.get("manual_exclusions")
        else:
            # Fallback for legacy calls/tests without preparation state
            logger.info("No preparation found for project %s; creating committed default preparation.", project_id)
            prep = ProjectStore.initialize_default_preparation(project_id)
            prep["status"] = "COMMITTED"
            prep["is_locked"] = True
            prep["committed_at"] = now_iso()
            ProjectStore.save_preparation(project_id, prep)
            ProjectStore.update_project(project_id, {"preparation_status": "COMMITTED", "is_locked": True, "status": "PREPARATION_COMMITTED"})

        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        runs_dir = STORAGE_DIR / "projects" / project_id / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_output_dir = runs_dir / run_id


        run_info = {
            "run_id": run_id,
            "project_id": project_id,
            "status": "QUEUED",
            "progress": 0,
            "current_step": "QUEUED",
            "created_at": now_iso(),
            "started_at": None,
            "completed_at": None,
            "message": "Analysis queued for background execution",
            "error": None,
            "output_directory": str(run_output_dir)
        }
        RUN_STATES[run_id] = run_info

        # Launch background task
        executor.submit(cls._execute_pipeline, run_id, project, run_output_dir, config_overrides)
        return run_info

    @classmethod
    def _execute_pipeline(cls, run_id: str, project: Dict[str, Any], output_dir: Path, config_overrides: Optional[Dict[str, Any]]):
        run_info = RUN_STATES[run_id]
        run_info["started_at"] = now_iso()
        run_info["status"] = "VALIDATING"
        run_info["progress"] = 10
        run_info["current_step"] = "Input Validation"

        try:
            config_path = ENGINE_SRC.parent / "configs" / "config.json"
            model_path = ENGINE_SRC.parent / "models" / "best_finetuned_model.onnx"
            if not model_path.exists():
                model_path = config_path

            # If caller supplied custom run configurations, write a runtime config snapshot
            if config_overrides and isinstance(config_overrides, dict):
                try:
                    with open(config_path, "r", encoding="utf-8") as cf:
                        base_cfg = json.load(cf)
                    # Apply overrides
                    mode_override = config_overrides.get("analysis_mode") or config_overrides.get("analysisMode")
                    if mode_override:
                        base_cfg["analysis_mode"] = mode_override
                    if "cva_threshold_method" in config_overrides:
                        base_cfg.setdefault("cva", {})["threshold_method"] = config_overrides["cva_threshold_method"]
                    if "cva_percentile" in config_overrides:
                        base_cfg.setdefault("cva", {})["percentile"] = config_overrides["cva_percentile"]
                    if "ndwi_threshold" in config_overrides:
                        base_cfg.setdefault("water", {})["ndwi_threshold"] = config_overrides["ndwi_threshold"]
                    if config_overrides.get("manual_exclusions"):
                        raw_excl = config_overrides["manual_exclusions"]
                        clean_excl = [e.dict() if hasattr(e, "dict") else e for e in raw_excl] if raw_excl else []
                        base_cfg["manual_exclusions"] = clean_excl
                    else:
                        prep = ProjectStore.load_preparation(project["id"])
                        if prep and prep.get("manual_exclusions"):
                            raw_excl = prep.get("manual_exclusions")
                            clean_excl = [e.dict() if hasattr(e, "dict") else e for e in raw_excl] if raw_excl else []
                            base_cfg["manual_exclusions"] = clean_excl

                    if config_overrides.get("inclusion_geometry"):
                        base_cfg["inclusion_geometry"] = config_overrides["inclusion_geometry"]
                    else:
                        prep = ProjectStore.load_preparation(project["id"])
                        if prep and prep.get("inclusion"):
                            base_cfg["inclusion_geometry"] = prep.get("inclusion")

                    custom_cfg_path = output_dir.parent / f".{run_id}_run_config.json"
                    with open(custom_cfg_path, "w", encoding="utf-8") as cf:
                        json.dump(base_cfg, cf, indent=2)
                    config_path = custom_cfg_path
                except Exception as exc:
                    logger.warning("Failed to write config_overrides snapshot: %s", exc)

            t1_dir = Path(project["t1_dir"])
            t2_dir = Path(project["t2_dir"])
            wetland_boundary = project.get("wetland_boundary")

            # Determine effective analysis mode from overrides, project, or default to wetland
            effective_mode = "wetland"
            if config_overrides and isinstance(config_overrides, dict):
                effective_mode = (
                    config_overrides.get("analysis_mode")
                    or config_overrides.get("analysisMode")
                    or project.get("analysis_mode")
                    or "wetland"
                ).lower()
            elif project.get("analysis_mode"):
                effective_mode = project["analysis_mode"].lower()
            is_vegetation_mode = effective_mode == "vegetation"

            # Fallback to sample scene if user directories lack band files.
            # Vegetation mode only requires T1; wetland mode requires T1 + T2.
            sample_dir = ENGINE_SRC.parent / "data" / "sample_scene"
            user_has_t1 = t1_dir.exists() and bool(list(t1_dir.glob("*.tif")) or list(t1_dir.glob("*.TIF")))
            user_has_t2 = t2_dir.exists() and bool(list(t2_dir.glob("*.tif")) or list(t2_dir.glob("*.TIF")))

            if is_vegetation_mode:
                # Vegetation mode — only T1 is needed
                if not user_has_t1:
                    if sample_dir.exists():
                        t1_dir = sample_dir / "t1"
                        logger.info("Vegetation mode — using sample T1 scene fallback for run %s", run_id)
                    else:
                        raise FileNotFoundError(
                            f"No GeoTIFF band files found in T1 ({t1_dir}) directory."
                        )
                # Keep t2_dir as-is; the pipeline will ignore it in vegetation mode
            else:
                # Wetland mode — both T1 and T2 are needed
                if not (user_has_t1 and user_has_t2):
                    if sample_dir.exists():
                        t1_dir = sample_dir / "t1"
                        t2_dir = sample_dir / "t2"
                        if not wetland_boundary:
                            wetland_boundary = str(sample_dir / "wetland_boundary.geojson")
                        logger.info("Using sample scene fallback for run %s", run_id)
                    else:
                        raise FileNotFoundError(
                            f"No GeoTIFF band files found in T1 ({t1_dir}) or T2 ({t2_dir}) directory."
                        )
            # Observation-date requirement is controlled by Analysis configuration.
            # Default remains enabled to preserve existing behavior.
            use_observation_dates = True

            if config_overrides and isinstance(config_overrides, dict):
                use_observation_dates = config_overrides.get(
                    "use_observation_dates",
                    True,
                )

            # Prefer project-stored dates (user-provided or extracted from filenames)
            t1_date = project.get("t1_date") or None
            t2_date = project.get("t2_date") or None

            # If dates are still unset, try extracting from the actual TIFF filenames
            if not t1_date and t1_dir.exists():
                t1_date = _extract_date_from_directory(t1_dir)
                if t1_date:
                    logger.info("Extracted T1 date from filename: %s", t1_date)
                    ProjectStore.update_project(project["id"], {"t1_date": t1_date})
            if not t2_date and t2_dir.exists():
                t2_date = _extract_date_from_directory(t2_dir)
                if t2_date:
                    logger.info("Extracted T2 date from filename: %s", t2_date)
                    ProjectStore.update_project(project["id"], {"t2_date": t2_date})

            # If using sample scene fallback and dates are not specified, apply
            # the sample scene's known acquisition dates.
            # Vegetation mode only needs T1; we do NOT set t2_date in veg mode.
            if is_vegetation_mode:
                if not user_has_t1:
                    t1_date = t1_date or "2023-08-14"
            else:
                if not (user_has_t1 and user_has_t2):
                    t1_date = t1_date or "2023-08-14"
                    t2_date = t2_date or "2023-08-28"

            # Date validation:
            # - Vegetation mode does not require observation dates.
            # - Wetland mode requires both T1 and T2 dates when use_observation_dates is set.
            if use_observation_dates:
                if not is_vegetation_mode:
                    if not t1_date or not t2_date:
                        raise ValueError(
                            "Observation dates could not be determined for T1 and/or T2. "
                            "Please set t1_date and t2_date on the project before running analysis, "
                            "or disable 'Use observation dates' in Analysis."
                        )

            if not use_observation_dates:
                logger.info(
                    "Observation dates disabled for run %s; "
                    "continuing without requiring T1/T2 dates.",
                    run_id,
                )

            run_info["status"] = "PREPROCESSING"
            run_info["progress"] = 25
            run_info["current_step"] = "Spatial Alignment & Preprocessing"

            time.sleep(0.5)
            run_info["status"] = "INFERENCE"
            run_info["progress"] = 45
            run_info["current_step"] = "Model Inference (5-Channel Sentinel-2)"

            time.sleep(0.5)
            run_info["status"] = "CHANGE_ANALYSIS"
            run_info["progress"] = 65
            run_info["current_step"] = "4x4 Transition Analysis & CVA"

            # Execute real GIS analysis engine
            res = analyze_wetland(
                t1_scene=t1_dir,
                t2_scene=None if is_vegetation_mode else t2_dir,
                t1_date=t1_date if t1_date else ("single-date" if is_vegetation_mode else None),
                t2_date=None if is_vegetation_mode else t2_date,
                config_path=config_path,
                model_path=model_path,
                output_dir=output_dir.parent,
                wetland_boundary=wetland_boundary,
                run_id=run_id,
            )

            run_info["status"] = "REPORTING"
            run_info["progress"] = 90
            run_info["current_step"] = "Generating PDF Cartographic Report"

            if res.status in ["SUCCESS", "WARNING"]:
                run_info["status"] = "COMPLETED"
                run_info["progress"] = 100
                run_info["current_step"] = "COMPLETED"
                run_info["completed_at"] = now_iso()
                run_info["message"] = "Analysis completed successfully"
                ProjectStore.update_project(project["id"], {"latest_run_id": run_id, "status": "ANALYZED"})
            else:
                run_info["status"] = "FAILED"
                run_info["completed_at"] = now_iso()
                run_info["error"] = "; ".join(res.errors) if res.errors else "Pipeline execution failed"

        except Exception as e:
            logger.exception("Pipeline execution failed for run %s", run_id)
            run_info["status"] = "FAILED"
            run_info["completed_at"] = now_iso()
            run_info["error"] = str(e)

    @classmethod
    def get_run_status(cls, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Return run status from in-memory state when available.

        Completed runs must remain discoverable after a backend restart/reload,
        so fall back to the persistent run directory and manifest.json.
        """

        # Fast path for currently running / recently completed runs.
        run_info = RUN_STATES.get(run_id)
        if run_info is not None:
            return run_info

        # Persistent recovery path.
        # A run directory is authoritative evidence that the run exists.
        projects_dir = STORAGE_DIR / "projects"

        if not projects_dir.exists():
            return None

        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            run_dir = project_dir / "runs" / run_id
            if not run_dir.is_dir():
                continue

            manifest_path = run_dir / "manifest.json"
            metadata_path = run_dir / "metadata.json"

            persisted: Dict[str, Any] = {}

            # Prefer manifest.json because it is explicitly run-level metadata.
            if manifest_path.exists():
                try:
                    with manifest_path.open("r", encoding="utf-8") as f:
                        manifest = json.load(f)

                    if isinstance(manifest, dict):
                        persisted.update(manifest)
                except (OSError, json.JSONDecodeError):
                    pass

            # Metadata can provide additional persisted information.
            if metadata_path.exists():
                try:
                    with metadata_path.open("r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    if isinstance(metadata, dict):
                        for key, value in metadata.items():
                            persisted.setdefault(key, value)
                except (OSError, json.JSONDecodeError):
                    pass

            # Reconstruct the API run object using persisted information.
            recovered: Dict[str, Any] = {
                "run_id": run_id,
                "project_id": project_dir.name,
                "status": persisted.get("status", "COMPLETED"),
                "progress": persisted.get("progress", 100),
                "current_step": persisted.get("current_step", "COMPLETED"),
                "created_at": persisted.get("created_at", ""),
                "started_at": persisted.get("started_at"),
                "completed_at": persisted.get("completed_at"),
                "message": persisted.get(
                    "message",
                    "Run recovered from persistent storage."
                ),
                "output_directory": str(run_dir),
            }

            if persisted.get("error"):
                recovered["error"] = persisted["error"]

            # If the persistent outputs exist but the manifest does not contain
            # an explicit terminal status, the presence of analysis_result.json
            # and the final report provides a deterministic completion signal.
            analysis_result = run_dir / "analysis_result.json"
            final_report = run_dir / "reports" / "FINAL_ANALYSIS_REPORT.pdf"

            if (
                recovered["status"] not in {"FAILED"}
                and analysis_result.exists()
                and final_report.exists()
            ):
                recovered["status"] = "COMPLETED"
                recovered["progress"] = 100
                recovered["current_step"] = "COMPLETED"

            # Cache the recovered state so subsequent requests are cheap.
            RUN_STATES[run_id] = recovered

            return recovered

        return None


class TileRenderer:
    """Renders GeoTIFF rasters as PNG images for frontend map display."""

    # Standard color maps matching config output
    COLORMAPS = {
        # Classification (0: Background, 1: Severe, 2: Moderate, 3: Healthy)
        "health": {
            0: (128, 128, 128, 255),
            1: (255, 0, 0, 255),
            2: (255, 165, 0, 255),
            3: (0, 170, 0, 255)
        },
        # Transition / Degradation / Recovery
        "change": {
            0: (0, 0, 0, 0),         # transparent
            1: (220, 220, 220, 200), # No change
            2: (255, 50, 50, 255),   # Degradation
            3: (50, 205, 50, 255),   # Recovery
            4: (0, 100, 255, 255)    # Water change
        }
    }

    @classmethod
    def render_raster_as_png(cls, raster_path: Path, layer_type: str = "health") -> bytes:
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster layer not found: {raster_path}")

        with rasterio.open(raster_path) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999.0

            # Scale/Colorize
            if layer_type in ["health", "degradation", "recovery", "transition"]:
                h, w = data.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)

                if layer_type == "health":
                    cmap = cls.COLORMAPS["health"]
                    for val, color in cmap.items():
                        mask = (data == val)
                        rgba[mask] = color
                else:
                    # Generic categorical change
                    mask_deg = (data > 0) & (data != nodata)
                    rgba[mask_deg] = (255, 69, 0, 220)  # Orange-Red highlight
                    rgba[data == 0] = (0, 0, 0, 0)      # Transparent

                img = Image.fromarray(rgba, mode="RGBA")
            else:
                # Continuous scale (CVA, Delta NDVI, etc.)
                valid_mask = (data != nodata) & (~np.isnan(data))
                norm_data = np.zeros_like(data, dtype=np.uint8)
                if np.any(valid_mask):
                    v_min, v_max = data[valid_mask].min(), data[valid_mask].max()
                    if v_max > v_min:
                        scaled = ((data - v_min) / (v_max - v_min) * 255).astype(np.uint8)
                        norm_data[valid_mask] = scaled[valid_mask]

                # Apply heat map palette
                h, w = data.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[..., 0] = norm_data           # Red
                rgba[..., 1] = 255 - norm_data     # Green
                rgba[..., 2] = 128                 # Blue
                rgba[..., 3] = np.where(valid_mask, 200, 0).astype(np.uint8)
                img = Image.fromarray(rgba, mode="RGBA")

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    @classmethod
    def get_raster_bounds_wgs84(cls, raster_path: Path) -> List[float]:
        """Returns [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 for map fitting."""
        with rasterio.open(raster_path) as src:
            if src.crs:
                try:
                    left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    if -180 <= left <= 180 and -180 <= right <= 180 and -90 <= bottom <= 90 and -90 <= top <= 90:
                        return [left, bottom, right, top]
                except Exception:
                    pass
            # Fallback for unprojected/pixel-space rasters (e.g. raw drone images)
            w, h = src.width, src.height
            # Normalize to 0.01 degree box centered around Berlin/default AOI
            return [13.4, 52.5, 13.4 + (w * 0.00001), 52.5 + (h * 0.00001)]

    @classmethod
    def render_dataset_preview_png(cls, project_id: str, time_point: str = "t1") -> bytes:
        project = ProjectStore.get_project(project_id)
        if not project:
            raise FileNotFoundError(f"Project {project_id} not found")

        raster_path = PreparationService._find_reference_raster(project)
        if not raster_path or not raster_path.exists():
            t2_dir = Path(project.get("t2_dir", ""))
            if t2_dir.exists():
                c = list(t2_dir.glob("*.tif")) + list(t2_dir.glob("*.TIF"))
                if c:
                    raster_path = c[0]

        if not raster_path or not raster_path.exists():
            raise FileNotFoundError(f"No GeoTIFF rasters found for project {project_id}")

        with rasterio.open(raster_path) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999.0

            valid = (data != nodata) & (~np.isnan(data))
            norm = np.zeros_like(data, dtype=np.uint8)

            if np.any(valid):
                valid_vals = data[valid]
                p2 = np.percentile(valid_vals, 2)
                p98 = np.percentile(valid_vals, 98)
                if p98 > p2:
                    clipped = np.clip(data, p2, p98)
                    norm[valid] = ((clipped[valid] - p2) / (p98 - p2) * 255).astype(np.uint8)
                else:
                    norm[valid] = np.clip(data[valid], 0, 255).astype(np.uint8)

            h, w = data.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = norm
            rgba[..., 1] = norm
            rgba[..., 2] = norm
            rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

            img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    @classmethod
    def get_dataset_preview_info(cls, project_id: str) -> Dict[str, Any]:
        project = ProjectStore.get_project(project_id)
        if not project:
            raise FileNotFoundError(f"Project {project_id} not found")

        raster_path = PreparationService._find_reference_raster(project)
        if not raster_path or not raster_path.exists():
            return {
                "width": 2592,
                "height": 1944,
                "extent": [0, 0, 2592, 1944],
                "crs": None,
                "is_pixel_space": True
            }

        with rasterio.open(raster_path) as src:
            width = src.width
            height = src.height
            crs_str = src.crs.to_string() if src.crs else None
            is_pixel_space = (src.crs is None)
            extent = [0, 0, width, height] if is_pixel_space else [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            return {
                "width": width,
                "height": height,
                "extent": extent,
                "crs": crs_str,
                "is_pixel_space": is_pixel_space
            }

