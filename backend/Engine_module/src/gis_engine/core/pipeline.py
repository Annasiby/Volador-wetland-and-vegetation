"""
src/gis_engine/core/pipeline.py
==================================
Canonical Production Orchestrator for the GIS Wetland Degradation &
Depletion Engine.

This module owns orchestration ONLY. Scientific calculations remain
in their respective modules. The pipeline calls them in sequence,
handles exceptions, manages the output directory structure, writes
provenance metadata, the output manifest, and the analysis report.

Architecture (Phase 13 locked):
    INPUT → INGESTION → ALIGNMENT → VEGETATION INFERENCE →
    TRANSITIONS → CVA → DEGRADATION/RECOVERY → WETLAND CONTEXT →
    HYDROLOGY → FUSION → METRICS → QA → REPORTING → OUTPUT PACKAGE
"""

import hashlib
import json
import logging
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rasterio

from .config import load_config, save_config_snapshot, config_hash
from .dispatcher import AnalysisDispatcher
from .constants import (
    CHANNEL_ORDER,
    NODATA_CATEGORICAL,
    NODATA_CONTINUOUS,
    PHYSICAL_DEPLETION_DEFAULT,
    SOFTWARE_VERSION,
)
from .exceptions import (
    AlignmentError,
    ConfigurationError,
    FusionError,
    GISProcessingError,
    HydrologyError,
    InferenceError,
    InputValidationError,
    MetricError,
    OutputError,
    ValidationError,
)
from .results import AnalysisResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_run_id(user_run_id: Optional[str] = None) -> str:
    """Generates a unique run ID in format YYYYMMDDTHHMMSSZ_<short-uuid>."""
    if user_run_id:
        return user_run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}"


def _setup_logging(run_id: str, log_dir: Path, log_level: str = "INFO") -> logging.FileHandler:
    """Configures structured logging for a production run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "analysis.log"

    root_logger = logging.getLogger("gis_engine")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove pre-existing handlers to avoid duplication across runs
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=f"[%(levelname)s] [{run_id}] %(name)s — %(message)s",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    return fh


def _file_sha256(path: Path) -> str:
    """Returns the SHA-256 hex digest of a file, or empty string on error."""
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()
    except OSError:
        return ""


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    """Returns a fingerprint dict for an input file/directory."""
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.is_file() else None,
        "sha256": _file_sha256(p) if p.is_file() else None,
    }


def _write_json(path: Path, data: Any) -> None:
    """Writes JSON to path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def _output_entry(run_dir: Path, rel_path: str) -> Dict[str, Any]:
    """Creates a manifest entry for an output file."""
    full = run_dir / rel_path
    return {
        "path": rel_path,
        "type": (
            "GeoTIFF" if rel_path.endswith(".tif") else
            "JSON" if rel_path.endswith(".json") else
            "CSV" if rel_path.endswith(".csv") else
            "Markdown" if rel_path.endswith(".md") else
            "PDF" if rel_path.endswith(".pdf") else
            "PNG" if rel_path.endswith(".png") else "File"
        ),
        "size_bytes": full.stat().st_size if full.is_file() else None,
        "sha256": _file_sha256(full) if full.is_file() else None,
    }


# ---------------------------------------------------------------------------
# ProductionPipeline
# ---------------------------------------------------------------------------

class ProductionPipeline:
    """
    Canonical production orchestrator for the GIS Wetland Degradation
    & Depletion Engine.

    Usage::

        pipeline = ProductionPipeline()
        result = pipeline.run_analysis(
            t1_scene=Path("data/T1"),
            t2_scene=Path("data/T2"),
            t1_date="2023-08-14",
            t2_date="2023-08-28",
            config_path=Path("configs/config.json"),
            model_path=Path("models/best_finetuned_model.onnx"),
            output_base_dir=Path("outputs/runs"),
        )
    """

    def run_analysis(
        self,
        t1_scene: Union[str, Path],
        t2_scene: Optional[Union[str, Path]],
        t1_date: Optional[str],
        t2_date: Optional[str],
        config_path: Union[str, Path],
        model_path: Union[str, Path],
        output_base_dir: Union[str, Path],
        wetland_boundary: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        log_level: str = "INFO",
    ) -> AnalysisResult:
        """
        Executes the complete production analysis pipeline.

        Returns an AnalysisResult regardless of success/failure.
        Check result.status for "SUCCESS", "WARNING", or "FAILED".
        """
        # ------------------------------------------------------------------
        # 0. Run ID & directory setup
        # ------------------------------------------------------------------
        run_id = _generate_run_id(run_id)
        output_base_dir = Path(output_base_dir)
        run_dir = output_base_dir / run_id

        if run_dir.exists():
            raise OutputError(
                f"Run directory already exists: {run_dir}. "
                "Use a unique run_id or remove the existing directory."
            )

        run_dir.mkdir(parents=True, exist_ok=True)

        veg_dir     = run_dir / "vegetation"
        change_dir  = run_dir / "change"
        deg_dir     = run_dir / "degradation"
        wetland_dir = run_dir / "wetland"
        hydro_dir   = run_dir / "hydrology"
        fusion_dir  = run_dir / "fusion"
        metrics_dir = run_dir / "metrics"
        val_dir     = run_dir / "validation"
        reports_dir = run_dir / "reports"
        logs_dir    = run_dir / "logs"

        for d in [veg_dir, change_dir, deg_dir, wetland_dir, hydro_dir,
                  fusion_dir, metrics_dir, val_dir, reports_dir, logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        fh = None
        try:
            fh = _setup_logging(run_id, logs_dir, log_level)
            run_logger = logging.getLogger("gis_engine.pipeline")

            warnings: List[str] = []
            errors:   List[str] = []
            outputs:  Dict[str, str] = {}

            run_start = time.perf_counter()
            run_logger.info("Phase 13 — Production pipeline START. run_id=%s", run_id)

            # Keep refs for later stages (populated as pipeline proceeds)
            t1_health = t2_health = overlap_mask = None
            manual_exclusion_mask = None
            effective_overlap_mask = None
            ref_grid = aligned_pair = pair_result = None
            transition_map = delta_ndvi = delta_gndvi = delta_ndre = cva_magnitude = None
            deg_result = wetland_result = None
            hydro_metrics: Dict[str, Any] = {}
            fusion_metrics: Dict[str, Any] = {}
            metrics_result_dict: Dict[str, Any] = {}
            qa_dict: Dict[str, Any] = {"overall_status": "PASS", "flags": []}

            # ==============================================================
            # STEP 1 — Load & validate configuration
            # ==============================================================
            run_logger.info("Step 1 — Loading configuration: %s", config_path)
            try:
                config = load_config(config_path)
                config["model"]["path"] = str(model_path)

                config["model"]["path"] = str(model_path)

                dispatcher = AnalysisDispatcher.from_config(config)

                run_logger.info(
                    "Analysis mode: %s",
                    dispatcher.analysis_mode,
                )
            except ConfigurationError:
                raise
            except Exception as exc:
                raise ConfigurationError(f"Unexpected config error: {exc}") from exc

            config_snapshot_path = save_config_snapshot(config, run_dir)
            cfg_hash = config_hash(config)
            run_logger.info("Step 1 — Configuration loaded. hash=%s…", cfg_hash[:16])

            # ==============================================================
            # STEP 2 — Validate inputs
            # ==============================================================
            run_logger.info("Step 2 — Validating inputs")

            t1_scene = Path(t1_scene)
            model_path = Path(model_path)

            if not t1_scene.is_dir():
                raise InputValidationError(
                    f"Primary scene directory not found: {t1_scene}"
                )

            if not model_path.is_file():
                raise InputValidationError(
                    f"ONNX model file not found: {model_path}"
                )

            # --------------------------------------------------------------
            # Vegetation mode:
            #   Single-date analysis. Only one scene is required.
            # --------------------------------------------------------------
            if dispatcher.is_vegetation:
                t2_scene = Path(t2_scene) if t2_scene else None

                t1_date = t1_date or "single-date"

                t2_date = t2_date or None
                interval_days = None

                run_logger.info(
                    "Step 2 — Vegetation mode: single scene=%s",
                    t1_scene,
                )

            # --------------------------------------------------------------
            # Wetland mode:
            #   Temporal analysis. T1 + T2 and both dates are required.
            # --------------------------------------------------------------
            else:
                if not t2_scene:
                    raise InputValidationError(
                        "Wetland mode requires a T2 scene directory."
                    )

                if not t1_date:
                    raise InputValidationError(
                        "Wetland mode requires a T1 date."
                    )

                if not t2_date:
                    raise InputValidationError(
                        "Wetland mode requires a T2 date."
                    )

                t2_scene = Path(t2_scene)

                if not t2_scene.is_dir():
                    raise InputValidationError(
                        f"T2 scene directory not found: {t2_scene}"
                    )

                from datetime import datetime as _dt

                try:
                    d1 = _dt.strptime(t1_date, "%Y-%m-%d")
                    d2 = _dt.strptime(t2_date, "%Y-%m-%d")
                except ValueError as exc:
                    raise InputValidationError(
                        f"Invalid date format: {exc}"
                    ) from exc

                if d1 > d2:
                    raise InputValidationError(
                        f"T1 date ({t1_date}) must not be later than "
                        f"T2 date ({t2_date})."
                    )

                if d1 == d2:
                    warnings.append(
                        f"T1 and T2 dates are identical ({t1_date}). "
                        "This is valid for same-acquisition baseline testing. "
                        "Temporal interval is 0 days; no genuine temporal "
                        "change can exist."
                    )

                interval_days = (d2 - d1).days

                run_logger.info(
                    "Step 2 — Wetland inputs valid. "
                    "T1=%s T2=%s interval=%d days",
                    t1_date,
                    t2_date,
                    interval_days,
                )

            # ==============================================================
            # VEGETATION MODE BRANCH — Single-Date Pipeline Execution
            # ==============================================================
            if dispatcher.is_vegetation:
                run_logger.info("Step 3 — Vegetation mode: single-date execution (bypassing temporal alignment)")
                from ..alignment.spatial import ReferenceGrid
                from ..ingestion.loader import RasterLoader
                band_mapping = config.get("ingestion", {}).get("band_mapping", {})
                loader = RasterLoader(band_mapping=band_mapping)
                t1_paths = {}
                for b_name in ["green", "red", "rededge", "nir"]:
                    try:
                        t1_paths[b_name] = loader._find_band_file(t1_scene, b_name)
                    except FileNotFoundError:
                        pass
                if not t1_paths:
                    raise FileNotFoundError(f"No valid spectral bands found in scene: {t1_scene}")
                first_band_path = next(iter(t1_paths.values()))
                with rasterio.open(first_band_path) as src:
                    ref_grid = ReferenceGrid(
                        crs=src.crs,
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                        resolution=src.res,
                        bounds=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                    )
                    h, w = src.height, src.width

                overlap_mask = np.ones((h, w), dtype=bool)
                effective_overlap_mask = overlap_mask
                overlap_pixel_count = int(h * w)
                inclusion_pixel_count = overlap_pixel_count
                excluded_pixel_count = 0
                effective_pixel_count = overlap_pixel_count
                total_overlap_area_ha = None
                inclusion_area_ha = None
                excluded_area_ha = None
                effective_analysis_area_ha = None
                physical_area_available = False
                area_basis = "PIXEL_COUNT_ONLY"
                if ref_grid.crs is not None:
                    try:
                        from ..metrics.area import calculate_pixel_areas
                        pixel_areas = calculate_pixel_areas(
                            ref_grid.transform,
                            (ref_grid.height, ref_grid.width),
                            ref_grid.crs,
                        )
                        effective_analysis_area_ha = float(np.sum(pixel_areas)) / 10000.0
                        total_overlap_area_ha = effective_analysis_area_ha
                        inclusion_area_ha = effective_analysis_area_ha
                        excluded_area_ha = 0.0
                        physical_area_available = True
                        area_basis = "RASTER_CRS_AND_TRANSFORM"
                    except Exception:
                        pass

                band_statistics = {"scene": {}}
                for b_name in ["green", "red", "rededge", "nir"]:
                    p = t1_paths.get(b_name)
                    if p and Path(p).is_file():
                        with rasterio.open(p) as src:
                            arr = src.read(1).astype(np.float32)
                            valid = ~np.isnan(arr) & ~np.isinf(arr) & (arr != (src.nodata or -9999.0))
                            v = arr[valid]
                            if v.size > 0:
                                band_statistics["scene"][b_name] = {
                                    "valid_count": int(v.size),
                                    "min": float(np.min(v)),
                                    "max": float(np.max(v)),
                                    "mean": float(np.mean(v)),
                                    "median": float(np.median(v)),
                                    "std": float(np.std(v)),
                                    "p95": float(np.percentile(v, 95.0)),
                                }

                # Step 4 — Run Single-Date Vegetation Analysis
                run_logger.info("Steps 4–7 — Vegetation analysis (single-date)")
                try:
                    from ..vegetation.pipeline import VegetationAnalysisPipeline
                    veg_pipeline = VegetationAnalysisPipeline()
                    veg_result = veg_pipeline.run(
                        scene_dir=t1_scene,
                        model_path=model_path,
                        veg_dir=veg_dir,
                        config=config,
                        run_logger=run_logger,
                    )
                except Exception as exc:
                    raise GISProcessingError(f"Vegetation analysis failed: {exc}") from exc

                outputs["vegetation/health.tif"] = "vegetation/health.tif"
                outputs["vegetation/confidence.tif"] = "vegetation/confidence.tif"
                outputs["vegetation/statistics.json"] = "vegetation/statistics.json"

                # Spectral indices on single scene
                eps = 1e-7
                index_stats = {}
                nir_p = t1_paths.get("nir")
                red_p = t1_paths.get("red")
                green_p = t1_paths.get("green")
                re_p = t1_paths.get("rededge")

                def _calc_s_idx(arr: np.ndarray, formula_str: str) -> Dict[str, Any]:
                    v = arr[~np.isnan(arr) & ~np.isinf(arr)]
                    return {
                        "formula": formula_str,
                        "mean": float(np.mean(v)) if v.size else 0.0,
                        "median": float(np.median(v)) if v.size else 0.0,
                        "std": float(np.std(v)) if v.size else 0.0,
                        "p95": float(np.percentile(v, 95.0)) if v.size else 0.0,
                    }

                if nir_p and red_p:
                    with rasterio.open(nir_p) as s_nir, rasterio.open(red_p) as s_red:
                        n_arr = s_nir.read(1).astype(np.float32)
                        r_arr = s_red.read(1).astype(np.float32)
                        ndvi = (n_arr - r_arr) / (n_arr + r_arr + eps)
                        index_stats["ndvi"] = _calc_s_idx(ndvi, "(NIR - Red) / (NIR + Red)")

                if nir_p and green_p:
                    with rasterio.open(nir_p) as s_nir, rasterio.open(green_p) as s_green:
                        n_arr = s_nir.read(1).astype(np.float32)
                        g_arr = s_green.read(1).astype(np.float32)
                        gndvi = (n_arr - g_arr) / (n_arr + g_arr + eps)
                        index_stats["gndvi"] = _calc_s_idx(gndvi, "(NIR - Green) / (NIR + Green)")

                if nir_p and re_p:
                    with rasterio.open(nir_p) as s_nir, rasterio.open(re_p) as s_re:
                        n_arr = s_nir.read(1).astype(np.float32)
                        re_arr = s_re.read(1).astype(np.float32)
                        ndre = (n_arr - re_arr) / (n_arr + re_arr + eps)
                        index_stats["ndre"] = _calc_s_idx(ndre, "(NIR - RedEdge) / (NIR + RedEdge)")

                # Single-date QA
                try:
                    from ..validation.quality import QualityAssuranceEngine
                    qa_engine = QualityAssuranceEngine(config)
                    qa_engine.run_uncertainty_qa(config, reference_samples_exist=False)
                    qa_dict = qa_engine.generate_qa_report()
                    _write_json(val_dir / "qa_report.json", qa_dict)
                    outputs["validation/qa_report.json"] = "validation/qa_report.json"
                except Exception as exc:
                    warnings.append(f"QA: {exc}")
                    run_logger.warning("Step 12 — QA issue: %s", exc)

                total_runtime = time.perf_counter() - run_start

                veg_classes = veg_result.class_statistics.get("classes", {})
                healthy_pct = veg_classes.get("3", {}).get("percentage", 0.0)
                moderate_pct = veg_classes.get("2", {}).get("percentage", 0.0)
                severe_pct = veg_classes.get("1", {}).get("percentage", 0.0)
                background_pct = veg_classes.get("0", {}).get("percentage", 0.0)
                veg_coverage_pct = healthy_pct + moderate_pct + severe_pct

                mean_conf = float(np.nanmean(veg_result.confidence[veg_result.valid_mask])) if np.any(veg_result.valid_mask) else 0.0

                result_content = {
                    "run_id": run_id,
                    "software_version": SOFTWARE_VERSION,
                    "analysis_mode": "vegetation",
                    "status": "WARNING" if warnings else "SUCCESS",
                    "inputs": {
                        "scene_path": str(t1_scene),
                        "model_path": str(model_path),
                        "band_statistics": band_statistics,
                    },
                    "temporal": {
                        "date": t1_date,
                    },
                    "spectral_indices": index_stats,
                    "spatial": {
                        "crs": str(ref_grid.crs) if ref_grid and ref_grid.crs is not None else None,
                        "dimensions": [ref_grid.height, ref_grid.width],
                        "resolution": list(ref_grid.resolution),
                        "extent": list(ref_grid.bounds),
                        "total_area_ha": total_overlap_area_ha,
                        "effective_analysis_area_ha": effective_analysis_area_ha,
                        "physical_area_available": physical_area_available,
                        "area_basis": area_basis,
                        "total_pixels": int(ref_grid.width * ref_grid.height),
                        "valid_pixels": int(veg_result.class_statistics.get("total_valid_pixels", 0)),
                    },
                    "vegetation": veg_result.class_statistics,
                    "metrics": {
                        "class_distribution": veg_result.class_statistics,
                        "healthy_vegetation_percentage": healthy_pct,
                        "moderate_stress_percentage": moderate_pct,
                        "severe_stress_percentage": severe_pct,
                        "background_percentage": background_pct,
                        "vegetation_coverage_percentage": veg_coverage_pct,
                        "mean_confidence": mean_conf,
                        "analysis_extent": {
                            "valid_analysis_area_ha": effective_analysis_area_ha or 0.0,
                        },
                    },
                    "validation": qa_dict,
                    "uncertainty": {
                        "mean_confidence": mean_conf,
                    },
                    "warnings": warnings,
                    "errors": errors,
                    "outputs": outputs,
                    "performance": {
                        "total_runtime_seconds": total_runtime,
                        "inference": veg_result.performance,
                    },
                }

                _write_json(run_dir / "analysis_result.json", result_content)
                outputs["analysis_result.json"] = "analysis_result.json"

                # Step 14 — Reports (Vegetation Health Markdown, Maps & PDF)
                run_logger.info("Step 14 — Generating vegetation analysis reports")
                try:
                    from ..output.report import generate_analysis_report
                    generate_analysis_report(
                        result_dict=result_content,
                        output_path=reports_dir / "analysis_report.md",
                        run_id=run_id,
                    )
                    outputs["reports/analysis_report.md"] = "reports/analysis_report.md"
                except Exception as exc:
                    warnings.append(f"Markdown Report: {exc}")
                    run_logger.warning("Step 14 — Markdown report warning: %s", exc)

                map_assets = {}
                try:
                    from ..output.cartography import generate_report_maps
                    map_assets = generate_report_maps(
                        run_dir=run_dir,
                        figures_dir=reports_dir / "figures",
                    )
                    for fig_key, fig_path in map_assets.items():
                        rel_fig = str(fig_path.relative_to(run_dir)).replace("\\", "/")
                        outputs[f"reports/figures/{fig_path.name}"] = rel_fig
                except Exception as exc:
                    warnings.append(f"Map Rendering: {exc}")
                    run_logger.warning("Step 14 — Map rendering warning: %s", exc)

                try:
                    from ..output.pdf import generate_all_reports
                    generated_pdfs = generate_all_reports(
                        result_dict=result_content,
                        run_dir=run_dir,
                        reports_dir=reports_dir,
                        map_assets=map_assets,
                        run_id=run_id,
                    )
                    # Register all generated PDFs in the output manifest
                    for pdf_key, pdf_path in generated_pdfs.items():
                        rel = str(pdf_path.relative_to(run_dir)).replace("\\", "/")
                        outputs[rel] = rel
                except Exception as exc:
                    warnings.append(f"PDF Report: {exc}")
                    run_logger.warning("Step 14 — PDF report warning: %s", exc)

                # Provenance metadata.json & manifest
                model_sha256 = _file_sha256(model_path)
                try:
                    import subprocess
                    git_commit = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                    ).decode().strip()
                except Exception:
                    git_commit = "unavailable"

                metadata = {
                    "run_id": run_id,
                    "software_version": SOFTWARE_VERSION,
                    "analysis_mode": "vegetation",
                    "git_commit": git_commit,
                    "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "python_version": sys.version,
                    "os": platform.platform(),
                    "model_path": str(model_path),
                    "model_sha256": model_sha256,
                    "configuration_hash": cfg_hash,
                    "config_snapshot": str(config_snapshot_path),
                    "input_scene": _file_fingerprint(t1_scene),
                    "date": t1_date,
                    "crs": str(ref_grid.crs) if ref_grid and ref_grid.crs is not None else None,
                    "resolution": list(ref_grid.resolution) if ref_grid else None,
                    "dimensions": [ref_grid.height, ref_grid.width] if ref_grid else None,
                    "total_runtime_seconds": time.perf_counter() - run_start,
                }
                _write_json(run_dir / "metadata.json", metadata)
                outputs["metadata.json"] = "metadata.json"

                manifest = {
                    "run_id": run_id,
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "total_outputs": len(outputs),
                    "outputs": [_output_entry(run_dir, rel) for rel in outputs.values()],
                }
                _write_json(run_dir / "manifest.json", manifest)
                outputs["manifest.json"] = "manifest.json"

                final_status = "WARNING" if warnings else "SUCCESS"
                run_logger.info(
                    "Phase 13 — Vegetation Pipeline COMPLETE. status=%s run_id=%s runtime=%.1fs",
                    final_status, run_id, time.perf_counter() - run_start,
                )

                return AnalysisResult(
                    run_id=run_id,
                    status=final_status,
                    output_directory=str(run_dir),
                    outputs=outputs,
                    metrics=result_content.get("metrics", {}),
                    validation=qa_dict,
                    warnings=warnings,
                    errors=errors,
                    metadata=metadata,
                )

            # ==============================================================
            # STEP 3 — Spatial / Temporal Alignment (Wetland Mode)
            # ==============================================================
            run_logger.info("Step 3 — Spatial/temporal alignment")
            try:
                from ..alignment.temporal import TemporalAligner
                aligner = TemporalAligner(config)
                aligned_pair = aligner.align_pair(t1_scene, t1_date, t2_scene, t2_date)
            except Exception as exc:
                raise AlignmentError(f"Alignment failed: {exc}") from exc

            ref_grid     = aligned_pair.reference_grid
            overlap_mask = aligned_pair.overlap_mask          # bool ndarray (H, W)
            run_logger.info(
                "Step 3 — Alignment complete. overlap=%.2f%%",
                aligned_pair.overlap_percentage,
            )

            # Create alignment output directories
            alignment_dir = run_dir / "alignment"
            t1_aligned_out = alignment_dir / "t1_aligned"
            t2_aligned_out = alignment_dir / "t2_aligned"
            t1_aligned_out.mkdir(parents=True, exist_ok=True)
            t2_aligned_out.mkdir(parents=True, exist_ok=True)

            # Write aligned bands to disk as green.tif, red.tif, rededge.tif, nir.tif
            align_profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "nodata": NODATA_CONTINUOUS,
                "width": ref_grid.width,
                "height": ref_grid.height,
                "count": 1,
                "crs": ref_grid.crs,
                "transform": ref_grid.transform,
                "compress": "lzw",
            }

            for band_name, band_arr in aligned_pair.t1_data.items():
                with rasterio.open(t1_aligned_out / f"{band_name}.tif", "w", **align_profile) as dst:
                    dst.write(band_arr.astype(np.float32), 1)

            for band_name, band_arr in aligned_pair.t2_data.items():
                with rasterio.open(t2_aligned_out / f"{band_name}.tif", "w", **align_profile) as dst:
                    dst.write(band_arr.astype(np.float32), 1)

            # Write overlap mask to alignment folder
            mask_profile = {
                "driver": "GTiff",
                "dtype": "uint8",
                "nodata": 0,
                "width": ref_grid.width,
                "height": ref_grid.height,
                "count": 1,
                "crs": ref_grid.crs,
                "transform": ref_grid.transform,
                "compress": "lzw",
            }
            with rasterio.open(alignment_dir / "overlap_mask.tif", "w", **mask_profile) as dst:
                dst.write(overlap_mask.astype(np.uint8), 1)

            # Record outputs
            outputs["alignment/t1_aligned/green.tif"] = "alignment/t1_aligned/green.tif"
            outputs["alignment/t1_aligned/red.tif"] = "alignment/t1_aligned/red.tif"
            outputs["alignment/t1_aligned/rededge.tif"] = "alignment/t1_aligned/rededge.tif"
            outputs["alignment/t1_aligned/nir.tif"] = "alignment/t1_aligned/nir.tif"
            outputs["alignment/t2_aligned/green.tif"] = "alignment/t2_aligned/green.tif"
            outputs["alignment/t2_aligned/red.tif"] = "alignment/t2_aligned/red.tif"
            outputs["alignment/t2_aligned/rededge.tif"] = "alignment/t2_aligned/rededge.tif"
            outputs["alignment/t2_aligned/nir.tif"] = "alignment/t2_aligned/nir.tif"
            outputs["alignment/overlap_mask.tif"] = "alignment/overlap_mask.tif"

            # ==============================================================
            # STEP 3A — Preparation Masking (Inclusion AOI & Manual Exclusions)
            # ==============================================================
            run_logger.info("Step 3A — Building authoritative preparation mask")

            try:
                from ..wetland.exclusion import build_manual_exclusion_mask, build_inclusion_mask

                inclusion_geometry = config.get("inclusion_geometry")
                manual_exclusions = config.get("manual_exclusions", [])

                exclusion_dir = run_dir / "exclusions"
                exclusion_dir.mkdir(parents=True, exist_ok=True)

                inclusion_mask_path = (
                    exclusion_dir / "inclusion_mask.tif"
                )
                manual_exclusion_mask_path = (
                    exclusion_dir / "manual_exclusion_mask.tif"
                )

                run_logger.info(
                    "[EXCLUSION TRACE] reference grid: "
                    f"crs={ref_grid.crs}, "
                    f"transform={ref_grid.transform}, "
                    f"width={ref_grid.width}, "
                    f"height={ref_grid.height}, "
                    f"bounds={ref_grid.bounds}"
                )

                run_logger.info(
                    "[EXCLUSION TRACE] inclusion geometry: "
                    f"{'Provided' if inclusion_geometry else 'Full Grid'}"
                )
                run_logger.info(
                    "[EXCLUSION TRACE] manual exclusions: "
                    f"{len(manual_exclusions)} exclusions"
                )

                # 1. Build inclusion mask (1 = inside AOI, 0 = outside)
                inclusion_mask = build_inclusion_mask(
                    geometry=inclusion_geometry,
                    reference_grid=ref_grid,
                    output_path=inclusion_mask_path,
                )

                # 2. Build manual exclusion mask (1 = excluded, 0 = not excluded)
                manual_exclusion_mask = build_manual_exclusion_mask(
                    exclusions=manual_exclusions,
                    reference_grid=ref_grid,
                    output_path=manual_exclusion_mask_path,
                )

                # 3. Form authoritative effective analysis mask:
                # effective = overlap & inclusion & ~exclusion
                effective_overlap_mask = (
                    overlap_mask.astype(bool)
                    & inclusion_mask.astype(bool)
                    & ~manual_exclusion_mask.astype(bool)
                )

                overlap_pixel_count = int(
                    np.sum(overlap_mask.astype(bool))
                )

                inclusion_pixel_count = int(
                    np.sum(
                        overlap_mask.astype(bool)
                        & inclusion_mask.astype(bool)
                    )
                )

                excluded_pixel_count = int(
                    np.sum(
                        overlap_mask.astype(bool)
                        & inclusion_mask.astype(bool)
                        & manual_exclusion_mask.astype(bool)
                    )
                )

                effective_pixel_count = int(
                    np.sum(effective_overlap_mask)
                )

                outputs[
                    "exclusions/inclusion_mask.tif"
                ] = "exclusions/inclusion_mask.tif"
                outputs[
                    "exclusions/manual_exclusion_mask.tif"
                ] = "exclusions/manual_exclusion_mask.tif"

                # Physical hectare areas are only authoritative when the
                # analysis grid is georeferenced. For the current pixel-space
                # DJI test imagery (CRS=None + identity transform), pixel counts
                # remain authoritative and physical area is explicitly unavailable.
                total_overlap_area_ha = None
                inclusion_area_ha = None
                excluded_area_ha = None
                effective_analysis_area_ha = None
                physical_area_available = False
                area_basis = "PIXEL_COUNT_ONLY"

                try:
                    if ref_grid.crs is not None:
                        from ..metrics.area import calculate_pixel_areas

                        pixel_areas = calculate_pixel_areas(
                            ref_grid.transform,
                            (ref_grid.height, ref_grid.width),
                            ref_grid.crs,
                        )

                        total_overlap_area_ha = float(
                            np.sum(
                                pixel_areas[
                                    overlap_mask.astype(bool)
                                ]
                            )
                        ) / 10000.0

                        inclusion_area_ha = float(
                            np.sum(
                                pixel_areas[
                                    overlap_mask.astype(bool)
                                    & inclusion_mask.astype(bool)
                                ]
                            )
                        ) / 10000.0

                        excluded_area_ha = float(
                            np.sum(
                                pixel_areas[
                                    overlap_mask.astype(bool)
                                    & inclusion_mask.astype(bool)
                                    & manual_exclusion_mask.astype(bool)
                                ]
                            )
                        ) / 10000.0

                        effective_analysis_area_ha = float(
                            np.sum(
                                pixel_areas[
                                    effective_overlap_mask
                                ]
                            )
                        ) / 10000.0

                        physical_area_available = True
                        area_basis = "RASTER_CRS_AND_TRANSFORM"
                    else:
                        run_logger.warning(
                            "Step 3A — Physical area unavailable: analysis grid "
                            "has no CRS. Reporting pixel counts only."
                        )
                except Exception as exc:
                    run_logger.warning(
                        "Failed to calculate physical exclusion areas: %s",
                        exc,
                    )

                if physical_area_available:
                    run_logger.info(
                        "Step 3A — Analysis footprint: total_overlap=%.4f ha, "
                        "inclusion=%.4f ha, excluded=%.4f ha, effective=%.4f ha",
                        total_overlap_area_ha,
                        inclusion_area_ha,
                        excluded_area_ha,
                        effective_analysis_area_ha,
                    )
                else:
                    run_logger.info(
                        "Step 3A — Analysis footprint: total_overlap_pixels=%d, "
                        "inclusion_pixels=%d, excluded_pixels=%d, effective_pixels=%d; "
                        "physical area unavailable.",
                        overlap_pixel_count,
                        inclusion_pixel_count,
                        excluded_pixel_count,
                        effective_pixel_count,
                    )
                # Preserve the original alignment footprint separately from the
                # effective analysis footprint after manual exclusions.
                effective_mask_path = alignment_dir / "effective_analysis_mask.tif"

                effective_mask_profile = {
                    "driver": "GTiff",
                    "height": int(ref_grid.height),
                    "width": int(ref_grid.width),
                    "count": 1,
                    "dtype": "uint8",
                    "crs": ref_grid.crs,
                    "transform": ref_grid.transform,
                    "nodata": 0,
                    "compress": "lzw",
                }

                with rasterio.open(
                    effective_mask_path,
                    "w",
                    **effective_mask_profile,
                ) as dst:
                    dst.write(
                        effective_overlap_mask.astype(np.uint8),
                        1,
                    )

                outputs[
                    "alignment/effective_analysis_mask.tif"
                ] = "alignment/effective_analysis_mask.tif"
                run_logger.info(
                    "Step 3A — Preparation complete: inclusion=%s, %d exclusions, "
                    "excluded_pixels=%d, effective_pixels=%d",
                    "custom" if inclusion_geometry else "full",
                    len(manual_exclusions),
                    excluded_pixel_count,
                    effective_pixel_count,
                )

            except Exception as exc:
                raise GISProcessingError(
                    f"Manual exclusion / preparation processing failed: {exc}"
                ) from exc

            # Authoritative input band radiometric statistics
            band_statistics: Dict[str, Any] = {"t1": {}, "t2": {}}
            for b_name in ["green", "red", "rededge", "nir"]:
                arr1 = aligned_pair.t1_data.get(b_name)
                if arr1 is not None:
                    v1 = arr1[effective_overlap_mask & ~np.isnan(arr1) & ~np.isinf(arr1)]
                    if v1.size > 0:
                        band_statistics["t1"][b_name] = {
                            "valid_count": int(v1.size),
                            "min": float(np.min(v1)),
                            "max": float(np.max(v1)),
                            "mean": float(np.mean(v1)),
                            "median": float(np.median(v1)),
                            "std": float(np.std(v1)),
                            "p95": float(np.percentile(v1, 95.0)),
                        }
                arr2 = aligned_pair.t2_data.get(b_name)
                if arr2 is not None:
                    v2 = arr2[effective_overlap_mask & ~np.isnan(arr2) & ~np.isinf(arr2)]
                    if v2.size > 0:
                        band_statistics["t2"][b_name] = {
                            "valid_count": int(v2.size),
                            "min": float(np.min(v2)),
                            "max": float(np.max(v2)),
                            "mean": float(np.mean(v2)),
                            "median": float(np.median(v2)),
                            "std": float(np.std(v2)),
                            "p95": float(np.percentile(v2, 95.0)),
                        }
            # ==============================================================
            # STEP 4–7 — Reusable Vegetation Analysis Pipeline (Two-Date Wetland)
            # ==============================================================
            run_logger.info("Steps 4–7 — Vegetation analysis on T1 & T2")

            try:
                from ..vegetation.pipeline import VegetationAnalysisPipeline
                from ..core.results import TwoDateInferenceResult
                from ..core.pair import calculate_class_stats
                from ..change.transition import calculate_transitions
                from ..change.cva import calculate_cva
                from ..change.degradation import analyze_degradation

                vegetation_pipeline = VegetationAnalysisPipeline()

                veg_t1_dir = veg_dir / "T1"
                veg_t2_dir = veg_dir / "T2"

                run_logger.info("Steps 4–7 — Vegetation inference T1: %s", t1_aligned_out)
                t1_veg = vegetation_pipeline.run(
                    scene_dir=t1_aligned_out,
                    model_path=model_path,
                    veg_dir=veg_t1_dir,
                    config=config,
                    run_logger=run_logger,
                )

                run_logger.info("Steps 4–7 — Vegetation inference T2: %s", t2_aligned_out)
                t2_veg = vegetation_pipeline.run(
                    scene_dir=t2_aligned_out,
                    model_path=model_path,
                    veg_dir=veg_t2_dir,
                    config=config,
                    run_logger=run_logger,
                )

                t1_health = t1_veg.health_map
                t2_health = t2_veg.health_map

                veg_profile = {
                    "driver": "GTiff",
                    "dtype": "uint8",
                    "nodata": NODATA_CATEGORICAL,
                    "width": ref_grid.width,
                    "height": ref_grid.height,
                    "count": 1,
                    "crs": ref_grid.crs,
                    "transform": ref_grid.transform,
                    "compress": "lzw",
                }
                with rasterio.open(veg_dir / "t1_health.tif", "w", **veg_profile) as dst:
                    dst.write(t1_health, 1)
                with rasterio.open(veg_dir / "t2_health.tif", "w", **veg_profile) as dst:
                    dst.write(t2_health, 1)

                mask_profile = {**veg_profile, "nodata": 0}
                with rasterio.open(veg_dir / "overlap_mask.tif", "w", **mask_profile) as dst:
                    dst.write(overlap_mask.astype(np.uint8), 1)

                outputs["vegetation/t1_health.tif"] = "vegetation/t1_health.tif"
                outputs["vegetation/t2_health.tif"] = "vegetation/t2_health.tif"
                outputs["vegetation/T1/confidence.tif"] = "vegetation/T1/confidence.tif"
                outputs["vegetation/T2/confidence.tif"] = "vegetation/T2/confidence.tif"
                outputs["vegetation/overlap_mask.tif"] = "vegetation/overlap_mask.tif"

                class_statistics_t1 = calculate_class_stats(t1_health, effective_overlap_mask)
                class_statistics_t2 = calculate_class_stats(t2_health, effective_overlap_mask)

                pair_result = TwoDateInferenceResult(
                    t1_health_map=t1_health,
                    t2_health_map=t2_health,
                    t1_logits=t1_veg.logits,
                    t2_logits=t2_veg.logits,
                    t1_probabilities=t1_veg.probabilities,
                    t2_probabilities=t2_veg.probabilities,
                    t1_confidence=t1_veg.confidence,
                    t2_confidence=t2_veg.confidence,
                    t1_valid_mask=t1_veg.valid_mask,
                    t2_valid_mask=t2_veg.valid_mask,
                    overlap_mask=overlap_mask,
                    reference_grid=ref_grid,
                    t1_metadata=t1_veg.metadata.__dict__ if hasattr(t1_veg.metadata, "__dict__") else {},
                    t2_metadata=t2_veg.metadata.__dict__ if hasattr(t2_veg.metadata, "__dict__") else {},
                    class_statistics_t1=class_statistics_t1,
                    class_statistics_t2=class_statistics_t2,
                    performance={
                        "t1_inference": t1_veg.performance,
                        "t2_inference": t2_veg.performance,
                    },
                )
                perf = pair_result.performance

                from ..output.json import write_pair_metadata_json
                pair_meta_dict = {
                    "pair_name": f"{t1_date}_{t2_date}",
                    "t1_date": t1_date,
                    "t2_date": t2_date,
                    "class_statistics_t1": class_statistics_t1,
                    "class_statistics_t2": class_statistics_t2,
                    "performance": perf,
                }
                write_pair_metadata_json(str(veg_dir / "pair_metadata.json"), pair_meta_dict)
                outputs["vegetation/pair_metadata.json"] = "vegetation/pair_metadata.json"

                # Step 5: Transition Analysis
                run_logger.info("Vegetation analysis — transition analysis")
                trans_result = calculate_transitions(
                    t1_health_map=t1_health,
                    t2_health_map=t2_health,
                    valid_overlap_mask=effective_overlap_mask.astype(np.uint8),
                    reference_grid=ref_grid,
                    output_dir=change_dir,
                    pair_name=f"{t1_date}_{t2_date}",
                )
                transition_map = trans_result.transition_map
                outputs["change/transition_map.tif"] = "change/transition_map.tif"

                # Step 6: CVA
                run_logger.info("Vegetation analysis — CVA")
                cva_result = calculate_cva(
                    t1_indices=t1_aligned_out,
                    t2_indices=t2_aligned_out,
                    valid_overlap_mask=effective_overlap_mask,
                    reference_grid=ref_grid,
                    output_dir=change_dir,
                    pair_name=f"{t1_date}_{t2_date}",
                    transition_map=transition_map,
                    config=config,
                )
                delta_ndvi = cva_result.delta_ndvi
                delta_gndvi = cva_result.delta_gndvi
                delta_ndre = cva_result.delta_ndre
                cva_magnitude = cva_result.cva_magnitude

                for fname in ["delta_ndvi.tif", "delta_gndvi.tif", "delta_ndre.tif", "cva_magnitude.tif"]:
                    outputs[f"change/{fname}"] = f"change/{fname}"

                # Step 7: Degradation
                run_logger.info("Vegetation analysis — degradation analysis")
                deg_result = analyze_degradation(
                    transition_map=transition_map,
                    cva_magnitude=cva_magnitude,
                    delta_ndvi=delta_ndvi,
                    delta_gndvi=delta_gndvi,
                    delta_ndre=delta_ndre,
                    valid_mask=effective_overlap_mask,
                    config=config,
                    reference_grid=ref_grid,
                    output_dir=deg_dir,
                )
                for fname in ["degradation_map.tif", "recovery_map.tif", "residual_stress.tif", "canopy_loss_map.tif"]:
                    outputs[f"degradation/{fname}"] = f"degradation/{fname}"

            except Exception as exc:
                raise GISProcessingError(
                    f"Vegetation analysis failed: {exc}"
                ) from exc

            run_logger.info(
                "Steps 4–7 — Vegetation analysis complete"
            )
            # ==============================================================
            if dispatcher.requires_wetland_context:
                # STEP 8 — Wetland Context
                # ==============================================================
                run_logger.info("Step 8 — Wetland context")
                try:
                    from ..wetland.mask import build_wetland_mask
                    wetland_source = (
                        str(wetland_boundary)
                        if wetland_boundary and Path(wetland_boundary).is_file()
                        else None
                    )
                    wetland_result = build_wetland_mask(
                        source=wetland_source,
                        reference_grid=ref_grid,
                        config=config,
                        output_dir=wetland_dir,
                        # Pass dynamic paths — never hardcoded date strings
                        t1_aligned_dir=str(t1_aligned_out),
                        t2_aligned_dir=str(t2_aligned_out),
                        overlap_mask=effective_overlap_mask.astype(np.uint8),
                    )
                except Exception as exc:
                    warnings.append(f"Wetland context: {exc}")
                    run_logger.warning("Step 8 — Wetland context issue: %s", exc)
                    from ..wetland.mask import WetlandMaskResult
                    zero = np.zeros((ref_grid.height, ref_grid.width), dtype=np.uint8)
                    wetland_result = WetlandMaskResult(
                        wetland_mask=zero,
                        valid_analysis_mask=effective_overlap_mask.astype(np.uint8),
                        source_type="UNAVAILABLE",
                        source_metadata={},
                        wetland_area_m2=0.0, wetland_area_ha=0.0,
                        valid_analysis_area_m2=0.0, valid_analysis_area_ha=0.0,
                        status="WETLAND_BOUNDARY_UNAVAILABLE",
                        physical_wetland_depletion_enabled=False,
                        metadata={"status": "WETLAND_BOUNDARY_UNAVAILABLE"},
                    )

                if (wetland_dir / "wetland_mask.tif").is_file():
                    outputs["wetland/wetland_mask.tif"] = "wetland/wetland_mask.tif"
                run_logger.info("Step 8 — Wetland context: %s", wetland_result.status)

                # ==============================================================
                # STEP 9 — NDWI Hydrology
                # ==============================================================
                run_logger.info("Step 9 — NDWI hydrology")
                try:
                    from ..wetland.water import analyze_hydrology
                    overlap_mask_tif = alignment_dir / "effective_analysis_mask.tif"
                    wetland_mask_tif  = wetland_dir / "wetland_mask.tif"

                    hydro_metrics = analyze_hydrology(
                        t1_aligned_dir=t1_aligned_out,
                        t2_aligned_dir=t2_aligned_out,
                        output_dir=hydro_dir,
                        config=config,
                        wetland_mask_path=wetland_mask_tif if wetland_mask_tif.is_file() else None,
                        overlap_mask_path=overlap_mask_tif if overlap_mask_tif.is_file() else None,
                    )
                    for fname in ["ndwi_t1.tif", "ndwi_t2.tif", "delta_ndwi.tif",
                                  "water_mask_t1.tif", "water_mask_t2.tif", "water_change.tif"]:
                        if (hydro_dir / fname).is_file():
                            outputs[f"hydrology/{fname}"] = f"hydrology/{fname}"
                except Exception as exc:
                    warnings.append(f"Hydrology: {exc}")
                    run_logger.warning("Step 9 — Hydrology issue: %s", exc)

                run_logger.info("Step 9 — Hydrology complete")

                # ==============================================================
                # STEP 10 — Multi-Evidence Fusion
                # ==============================================================
                run_logger.info("Step 10 — Multi-evidence fusion")
                water_change_path = hydro_dir / "water_change.tif"
                if water_change_path.is_file():
                    try:
                        from ..change.fusion import fuse_evidence
                        fusion_metrics = fuse_evidence(
                            transition_map=transition_map,
                            water_change=water_change_path,
                            valid_analysis_mask=wetland_result.valid_analysis_mask,
                            config=config,
                            delta_ndvi=delta_ndvi,
                            delta_gndvi=delta_gndvi,
                            delta_ndre=delta_ndre,
                            cva_magnitude=cva_magnitude,
                            wetland_mask=wetland_result.wetland_mask,
                            output_dir=fusion_dir,
                            reference_grid=ref_grid,
                        )
                        # Copy fusion metrics JSON to metrics dir
                        fusion_metrics_src = fusion_dir / "fusion_metrics.json"
                        if fusion_metrics_src.is_file():
                            import shutil
                            shutil.copy2(fusion_metrics_src, metrics_dir / "fusion_metrics.json")
                            outputs["metrics/fusion_metrics.json"] = "metrics/fusion_metrics.json"

                        # Copy evidence matrix CSV
                        em_src = fusion_dir / "evidence_matrix.csv"
                        if em_src.is_file():
                            shutil.copy2(em_src, metrics_dir / "evidence_matrix.csv")
                            outputs["metrics/evidence_matrix.csv"] = "metrics/evidence_matrix.csv"

                        if (fusion_dir / "evidence_fusion.tif").is_file():
                            outputs["fusion/evidence_fusion.tif"] = "fusion/evidence_fusion.tif"
                    except Exception as exc:
                        warnings.append(f"Fusion: {exc}")
                        run_logger.warning("Step 10 — Fusion issue: %s", exc)
                else:
                    warnings.append("water_change.tif unavailable; fusion skipped.")
                    run_logger.warning("Step 10 — water_change.tif missing; fusion skipped")

                run_logger.info("Step 10 — Fusion complete")

                # ==============================================================
                # STEP 11 — Metrics
                # ==============================================================
                run_logger.info("Step 11 — Metrics calculation")
                deg_map_path    = deg_dir / "degradation_map.tif"
                rec_map_path    = deg_dir / "recovery_map.tif"
                res_map_path    = deg_dir / "residual_stress.tif"
                canopy_map_path = deg_dir / "canopy_loss_map.tif"
                cva_mag_path    = change_dir / "cva_magnitude.tif"
                trans_path      = change_dir / "transition_map.tif"
                d_ndvi_path     = change_dir / "delta_ndvi.tif"
                d_gndvi_path    = change_dir / "delta_gndvi.tif"
                d_ndre_path     = change_dir / "delta_ndre.tif"
                wetland_mask_tif = wetland_dir / "wetland_mask.tif"

                required = [deg_map_path, rec_map_path, res_map_path, canopy_map_path,
                            cva_mag_path, trans_path, d_ndvi_path, d_gndvi_path, d_ndre_path]
                if all(p.is_file() for p in required):
                    try:
                        from ..metrics.degradation import calculate_metrics
                        wm = wetland_mask_tif if wetland_mask_tif.is_file() else wetland_result.wetland_mask
                        vm = wetland_result.valid_analysis_mask

                        metrics_result = calculate_metrics(
                            transition_map=trans_path,
                            cva_magnitude=cva_mag_path,
                            delta_ndvi=d_ndvi_path,
                            delta_gndvi=d_gndvi_path,
                            delta_ndre=d_ndre_path,
                            degradation_map=deg_map_path,
                            recovery_map=rec_map_path,
                            residual_stress_map=res_map_path,
                            canopy_loss_map=canopy_map_path,
                            wetland_mask=wm,
                            valid_analysis_mask=vm,
                            config=config,
                            reference_grid=ref_grid,
                        )
                        metrics_result_dict = metrics_result.to_dict()
                        _write_json(metrics_dir / "metrics.json", metrics_result_dict)
                        outputs["metrics/metrics.json"] = "metrics/metrics.json"

                        # Copy transition matrix CSV if produced by calculate_transitions
                        trans_csv_src = change_dir / "transition_matrix.csv"
                        if trans_csv_src.is_file():
                            import shutil
                            shutil.copy2(trans_csv_src, metrics_dir / "transition_matrix.csv")
                            outputs["metrics/transition_matrix.csv"] = "metrics/transition_matrix.csv"
                    except Exception as exc:
                        warnings.append(f"Metrics: {exc}")
                        run_logger.warning("Step 11 — Metrics issue: %s", exc)
                else:
                    warnings.append("Some raster outputs missing; full metrics skipped.")

                # Water metrics JSON
                water_metrics_src = hydro_dir / "water_metrics.json"
                if water_metrics_src.is_file():
                    import shutil
                    shutil.copy2(water_metrics_src, metrics_dir / "water_metrics.json")
                    outputs["metrics/water_metrics.json"] = "metrics/water_metrics.json"

                run_logger.info("Step 11 — Metrics complete")

                # ==============================================================
                # STEP 12 — Quality Assurance
                # ==============================================================
                run_logger.info("Step 12 — Quality assurance")
                try:
                    from ..validation.quality import QualityAssuranceEngine
                    qa_engine = QualityAssuranceEngine(config)
                    # Run lightweight QA checks available without all intermediate data
                    qa_engine.run_uncertainty_qa(config, reference_samples_exist=False)
                    qa_dict = qa_engine.generate_qa_report()
                    _write_json(val_dir / "qa_report.json", qa_dict)
                    outputs["validation/qa_report.json"] = "validation/qa_report.json"
                except Exception as exc:
                    warnings.append(f"QA: {exc}")
                    run_logger.warning("Step 12 — QA issue: %s", exc)

                run_logger.info("Step 12 — QA: %s", qa_dict.get("overall_status", "UNKNOWN"))

                # ==============================================================

            # STEP 13 — Assemble result package
            # ==============================================================
            run_logger.info("Step 13 — Assembling result package")
            total_runtime = time.perf_counter() - run_start

            veg_stats: Dict[str, Any] = {}
            if pair_result:
                veg_stats = {
                    "t1": pair_result.class_statistics_t1,
                    "t2": pair_result.class_statistics_t2,
                }

            deg_stats: Dict[str, Any] = {}
            rec_stats: Dict[str, Any] = {}
            if deg_result:
                stats = deg_result.statistics
                deg_count = stats.get("degradation_pixel_count", 0)
                rec_count = stats.get("recovery_pixel_count", 0)
                valid_cnt = stats.get("valid_pixel_count", 1) or 1
                deg_pct   = stats.get("degradation_percentage", 0.0)
                rec_pct   = stats.get("recovery_percentage", 0.0)
                # Area from metrics if available — use the actual field names from MetricsResult
                deg_area_ha = metrics_result_dict.get("impact_area", {}).get("impact_area_ha", 0.0) if metrics_result_dict else 0.0
                rec_area_ha = metrics_result_dict.get("recovery_area", {}).get("area_ha", 0.0) if metrics_result_dict else 0.0
                deg_stats = {"pixel_count": deg_count, "area_ha": deg_area_ha, "percentage": deg_pct}
                rec_stats = {"pixel_count": rec_count, "area_ha": rec_area_ha, "percentage": rec_pct}

            cva_stats: Dict[str, Any] = {}
            if cva_result:
                cva_stats = cva_result.statistics

            change_summary: Dict[str, Any] = {}
            if trans_result:
                change_summary = {
                    cat: {
                        "count": trans_result.category_counts.get(cat, 0),
                        "area_ha": trans_result.category_areas_ha.get(cat, 0.0),
                        "percentage": trans_result.category_percentages.get(cat, 0.0),
                    }
                    for cat in trans_result.category_counts
                }

            # Authoritative spectral index statistics (NDVI, GNDVI, NDRE, NDWI)
            index_stats: Dict[str, Any] = {}
            if aligned_pair is not None and effective_overlap_mask is not None:
                nir1, red1 = aligned_pair.t1_data.get("nir"), aligned_pair.t1_data.get("red")
                green1, re1 = aligned_pair.t1_data.get("green"), aligned_pair.t1_data.get("rededge")
                nir2, red2 = aligned_pair.t2_data.get("nir"), aligned_pair.t2_data.get("red")
                green2, re2 = aligned_pair.t2_data.get("green"), aligned_pair.t2_data.get("rededge")

                eps = 1e-7

                def _calc_idx_stats(arr1: np.ndarray, arr2: np.ndarray, formula_str: str) -> Dict[str, Any]:
                    v1 = arr1[effective_overlap_mask & ~np.isnan(arr1) & ~np.isinf(arr1)] if arr1 is not None else np.array([])
                    v2 = arr2[effective_overlap_mask & ~np.isnan(arr2) & ~np.isinf(arr2)] if arr2 is not None else np.array([])
                    d_arr = (arr2 - arr1) if (arr1 is not None and arr2 is not None) else None
                    vd = d_arr[effective_overlap_mask & ~np.isnan(d_arr) & ~np.isinf(d_arr)] if d_arr is not None else np.array([])
                    return {
                        "formula": formula_str,
                        "t1_mean": float(np.mean(v1)) if v1.size else 0.0,
                        "t1_median": float(np.median(v1)) if v1.size else 0.0,
                        "t1_std": float(np.std(v1)) if v1.size else 0.0,
                        "t1_p95": float(np.percentile(v1, 95.0)) if v1.size else 0.0,
                        "t2_mean": float(np.mean(v2)) if v2.size else 0.0,
                        "t2_median": float(np.median(v2)) if v2.size else 0.0,
                        "t2_std": float(np.std(v2)) if v2.size else 0.0,
                        "t2_p95": float(np.percentile(v2, 95.0)) if v2.size else 0.0,
                        "delta_mean": float(np.mean(vd)) if vd.size else 0.0,
                        "delta_median": float(np.median(vd)) if vd.size else 0.0,
                        "delta_std": float(np.std(vd)) if vd.size else 0.0,
                        "delta_p95": float(np.percentile(vd, 95.0)) if vd.size else 0.0,
                    }

                if nir1 is not None and red1 is not None and nir2 is not None and red2 is not None:
                    ndvi1 = (nir1 - red1) / (nir1 + red1 + eps)
                    ndvi2 = (nir2 - red2) / (nir2 + red2 + eps)
                    index_stats["ndvi"] = _calc_idx_stats(ndvi1, ndvi2, "(NIR - Red) / (NIR + Red)")

                if nir1 is not None and green1 is not None and nir2 is not None and green2 is not None:
                    gndvi1 = (nir1 - green1) / (nir1 + green1 + eps)
                    gndvi2 = (nir2 - green2) / (nir2 + green2 + eps)
                    index_stats["gndvi"] = _calc_idx_stats(gndvi1, gndvi2, "(NIR - Green) / (NIR + Green)")

                if nir1 is not None and re1 is not None and nir2 is not None and re2 is not None:
                    ndre1 = (nir1 - re1) / (nir1 + re1 + eps)
                    ndre2 = (nir2 - re2) / (nir2 + re2 + eps)
                    index_stats["ndre"] = _calc_idx_stats(ndre1, ndre2, "(NIR - RedEdge) / (NIR + RedEdge)")

                if dispatcher.is_wetland:
                    if green1 is not None and nir1 is not None and green2 is not None and nir2 is not None:
                        ndwi1 = (green1 - nir1) / (green1 + nir1 + eps)
                        ndwi2 = (green2 - nir2) / (green2 + nir2 + eps)
                        index_stats["ndwi"] = _calc_idx_stats(
                            ndwi1,
                            ndwi2,
                            "(Green - NIR) / (Green + NIR)",
                        )

            # Physical depletion
            indep_hydro = config.get("fusion", {}).get("independent_hydrological_source_accepted", False)
            if indep_hydro and wetland_result and wetland_result.physical_wetland_depletion_enabled:
                phys_depl = "SUPPORTED_BY_INDEPENDENT_EVIDENCE"
            else:
                phys_depl = PHYSICAL_DEPLETION_DEFAULT

            result_content: Dict[str, Any] = {
                "run_id": run_id,
                "software_version": SOFTWARE_VERSION,
                "analysis_mode": dispatcher.analysis_mode,
                "status": "WARNING" if warnings else "SUCCESS",

                "inputs": {
                    "t1_path": str(t1_scene),
                    "t2_path": str(t2_scene),
                    "model_path": str(model_path),
                    "wetland_boundary": str(wetland_boundary) if wetland_boundary else None,
                    "band_statistics": band_statistics,
                },

                "temporal": {
                    "t1_date": t1_date,
                    "t2_date": t2_date,
                    "interval_days": interval_days,
                },

                "spectral_indices": index_stats,

                "spatial": {
                    "crs": str(ref_grid.crs),
                    "dimensions": [ref_grid.height, ref_grid.width],
                    "resolution": list(ref_grid.resolution),
                    "extent": list(ref_grid.bounds),
                    "overlap_percentage": aligned_pair.overlap_percentage if aligned_pair else None,
                    "total_overlap_area_ha": total_overlap_area_ha,
                    "inclusion_area_ha": inclusion_area_ha,
                    "manual_excluded_area_ha": excluded_area_ha,
                    "effective_analysis_area_ha": effective_analysis_area_ha,
                    "physical_area_available": physical_area_available,
                    "area_basis": area_basis,
                    "overlap_pixel_count": overlap_pixel_count,
                    "inclusion_pixel_count": inclusion_pixel_count,
                    "excluded_pixel_count": excluded_pixel_count,
                    "effective_pixel_count": effective_pixel_count,
                },

                # Common vegetation-analysis results.
                "vegetation": veg_stats,
                "change": {
                    "category_summary": change_summary,
                },
                "cva": cva_stats,
                "degradation": deg_stats,
                "recovery": rec_stats,

                # Common analysis metrics.
                "metrics": {
                    "normalized_degradation_ratio": (
                        metrics_result_dict.get("normalized_degradation_ratio")
                        if metrics_result_dict
                        else None
                    ),

                    "severity_weighted_degradation_area_ha": (
                        metrics_result_dict.get("severity_weighted_area", {})
                        .get("severity_weighted_area_ha_deg")
                        if metrics_result_dict
                        else None
                    ),

                    # Stable public API fields.
                    "impact_area": {
                        "degradation_ha": (
                            metrics_result_dict.get("impact_area", {})
                            .get("impact_area_ha", 0.0)
                            if metrics_result_dict
                            else 0.0
                        ),
                    },

                    "recovery_area": {
                        "recovery_ha": (
                            metrics_result_dict.get("recovery_area", {})
                            .get("area_ha", 0.0)
                            if metrics_result_dict
                            else 0.0
                        ),
                    },

                    "cva_magnitude": {
                        "mean": (
                            (
                                metrics_result_dict
                                .get("cva_statistics", {})
                                .get("all_valid_wetland", {})
                                .get("mean")
                            )
                            if (
                                metrics_result_dict
                                and metrics_result_dict
                                .get("cva_statistics", {})
                                .get("all_valid_wetland", {})
                                .get("mean") is not None
                            )
                            else 0.0
                        ),
                    },

                    "analysis_extent": {
                        "valid_analysis_area_ha": (
                            metrics_result_dict.get("baseline_area", {})
                            .get("area_ha", 0.0)
                            if metrics_result_dict
                            else 0.0
                        ),
                    },

                    "transition_matrix": {
                        "matrix_m2": [
                            [float(trans_result.transition_areas_m2.get(i * 4 + j) or 0.0) for j in range(4)]
                            for i in range(4)
                        ],
                        "matrix_pixels": [
                            [int(trans_result.transition_counts.get(i * 4 + j) or 0) for j in range(4)]
                            for i in range(4)
                        ],
                    } if trans_result else None,
                },

                "validation": qa_dict,

                "uncertainty": {
                    "mean_confidence_t1": (
                        float(np.nanmean(pair_result.t1_confidence))
                        if pair_result and hasattr(pair_result, "t1_confidence")
                        else None
                    ),
                },

                "warnings": warnings,
                "errors": errors,
                "outputs": outputs,

                "performance": {
                    "total_runtime_seconds": total_runtime,
                    "inference": perf if pair_result else {},
                },
            }

            _write_json(run_dir / "analysis_result.json", result_content)

            if dispatcher.is_wetland:
                result_content["wetland"] = {
                    "status": wetland_result.status if wetland_result else "UNAVAILABLE",
                    "source_type": wetland_result.source_type if wetland_result else "UNAVAILABLE",
                    "area_ha": wetland_result.wetland_area_ha if wetland_result else 0.0,
                    "physical_depletion_enabled": (
                        wetland_result.physical_wetland_depletion_enabled
                        if wetland_result
                        else False
                    ),
                }

                result_content["hydrology"] = {
                    "ndwi_threshold": config.get("water", {}).get("ndwi_threshold", 0.0),
                    "water_gain_area_ha": hydro_metrics.get("water_gain_ha"),
                    "water_loss_area_ha": hydro_metrics.get("water_loss_ha"),
                    "water_area_t1_ha": hydro_metrics.get("water_area_t1_ha"),
                    "water_area_t2_ha": hydro_metrics.get("water_area_t2_ha"),
                }

                result_content["fusion"] = {
                    "evidence_level": fusion_metrics.get(
                        "physical_depletion_assessment",
                        "N/A",
                    ),
                    "hydro_ecological_degradation_area_ha": fusion_metrics.get(
                        "hydro_ecological_degradation_area_ha"
                    ),
                    "hydro_ecological_recovery_area_ha": fusion_metrics.get(
                        "hydro_ecological_recovery_area_ha"
                    ),
                }

                result_content["physical_depletion_assessment"] = phys_depl
            _write_json(run_dir / "analysis_result.json", result_content)
            outputs["analysis_result.json"] = "analysis_result.json"

            # ==============================================================
            # STEP 14 — Generate Analysis Reports (Markdown, Figures & PDF)
            # ==============================================================
            run_logger.info("Step 14 — Generating analysis reports")
            try:
                from ..output.report import generate_analysis_report
                generate_analysis_report(
                    result_dict=result_content,
                    output_path=reports_dir / "analysis_report.md",
                    run_id=run_id,
                )
                outputs["reports/analysis_report.md"] = "reports/analysis_report.md"
            except Exception as exc:
                warnings.append(f"Markdown Report: {exc}")
                run_logger.warning("Step 14 — Markdown report warning: %s", exc)

            map_assets: Dict[str, Path] = {}
            try:
                from ..output.cartography import generate_report_maps
                map_assets = generate_report_maps(
                    run_dir=run_dir,
                    figures_dir=reports_dir / "figures",
                )
                for fig_key, fig_path in map_assets.items():
                    rel_fig = str(fig_path.relative_to(run_dir)).replace("\\", "/")
                    outputs[f"reports/figures/{fig_path.name}"] = rel_fig
            except Exception as exc:
                warnings.append(f"Map Rendering: {exc}")
                run_logger.warning("Step 14 — Map rendering warning: %s", exc)

            try:
                from ..output.pdf import generate_all_reports
                generated_pdfs = generate_all_reports(
                    result_dict=result_content,
                    run_dir=run_dir,
                    reports_dir=reports_dir,
                    map_assets=map_assets,
                    run_id=run_id,
                )
                # Register all generated PDFs in the output manifest
                for pdf_key, pdf_path in generated_pdfs.items():
                    rel = str(pdf_path.relative_to(run_dir)).replace("\\", "/")
                    outputs[rel] = rel
            except Exception as exc:
                warnings.append(f"PDF Report: {exc}")
                run_logger.warning("Step 14 — PDF report warning: %s", exc)

            # ==============================================================
            # STEP 15 — Provenance metadata.json
            # ==============================================================
            run_logger.info("Step 15 — Writing provenance metadata")
            model_sha256 = _file_sha256(model_path)
            try:
                import subprocess
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                git_commit = "unavailable"

            metadata: Dict[str, Any] = {
                "run_id": run_id,
                "software_version": SOFTWARE_VERSION,
                "analysis_mode": dispatcher.analysis_mode,
                "git_commit": git_commit,
                "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "os": platform.platform(),
                "model_path": str(model_path),
                "model_sha256": model_sha256,
                "configuration_hash": cfg_hash,
                "config_snapshot": str(config_snapshot_path),
                "input_t1": _file_fingerprint(t1_scene),
                "input_t2": _file_fingerprint(t2_scene),
                "t1_date": t1_date,
                "t2_date": t2_date,
                "crs": str(ref_grid.crs) if ref_grid else None,
                "resolution": list(ref_grid.resolution) if ref_grid else None,
                "dimensions": [ref_grid.height, ref_grid.width] if ref_grid else None,
                "total_runtime_seconds": time.perf_counter() - run_start,
            }
            _write_json(run_dir / "metadata.json", metadata)
            outputs["metadata.json"] = "metadata.json"

            # ==============================================================
            # STEP 16 — Output manifest.json
            # ==============================================================
            run_logger.info("Step 16 — Writing output manifest")
            manifest = {
                "run_id": run_id,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "total_outputs": len(outputs),
                "outputs": [_output_entry(run_dir, rel) for rel in outputs.values()],
            }
            _write_json(run_dir / "manifest.json", manifest)
            outputs["manifest.json"] = "manifest.json"

            # ==============================================================
            # Done
            # ==============================================================
            final_status = "WARNING" if warnings else "SUCCESS"
            run_logger.info(
                "Phase 13 — Pipeline COMPLETE. status=%s run_id=%s runtime=%.1fs",
                final_status, run_id, time.perf_counter() - run_start,
            )

            return AnalysisResult(
                run_id=run_id,
                status=final_status,
                output_directory=str(run_dir),
                outputs=outputs,
                metrics=result_content.get("metrics", {}),
                validation=qa_dict,
                warnings=warnings,
                errors=errors,
                metadata=metadata,
            )

        except (InputValidationError, ConfigurationError, AlignmentError,
                InferenceError, HydrologyError, FusionError, MetricError,
                OutputError, ValidationError, GISProcessingError) as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
            run_logger.error("Phase 13 — Pipeline FAILED: %s", exc)
            return AnalysisResult(
                run_id=run_id,
                status="FAILED",
                output_directory=str(run_dir),
                outputs=outputs,
                metrics={},
                validation={},
                warnings=warnings,
                errors=errors,
                metadata={"run_id": run_id, "error": str(exc)},
            )
        except Exception as exc:
            errors.append(f"Internal error: {exc}")
            run_logger.error("Phase 13 — Unexpected error: %s", exc, exc_info=True)
            return AnalysisResult(
                run_id=run_id,
                status="FAILED",
                output_directory=str(run_dir),
                outputs=outputs,
                metrics={},
                validation={},
                warnings=warnings,
                errors=errors,
                metadata={"run_id": run_id, "error": str(exc)},
            )
        finally:
            if fh is not None:
                fh.close()
                logging.getLogger("gis_engine").removeHandler(fh)


def run_analysis(
    t1_scene: Union[str, Path],
    t2_scene: Union[str, Path],
    config_path: Union[str, Path],
    model_path: Union[str, Path],
    output_base_dir: Union[str, Path],
    t1_date: str,
    t2_date: str,
    wetland_boundary: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    log_level: str = "INFO",
) -> AnalysisResult:
    """Convenience helper invoking ProductionPipeline().run_analysis()."""
    pipeline = ProductionPipeline()
    return pipeline.run_analysis(
        t1_scene=t1_scene,
        t2_scene=t2_scene,
        t1_date=t1_date,
        t2_date=t2_date,
        config_path=config_path,
        model_path=model_path,
        output_base_dir=output_base_dir,
        wetland_boundary=wetland_boundary,
        run_id=run_id,
        log_level=log_level,
    )


