"""
src/gis_engine/api/service.py
================================
Public API for the GIS Wetland Degradation & Depletion Engine.

Single entry point: ``analyze_wetland()``.

All internal module complexity is hidden behind this function.
Callers only need to supply paths, dates, and configuration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..core.pipeline import ProductionPipeline
from ..core.results import AnalysisResult

logger = logging.getLogger(__name__)


def analyze_wetland(
    t1_scene: Union[str, Path],
    t2_scene: Optional[Union[str, Path]] = None,
    t1_date: Optional[str] = None,
    t2_date: Optional[str] = None,
    config_path: Union[str, Path] = None,
    model_path: Union[str, Path] = None,
    output_dir: Union[str, Path] = None,
    wetland_boundary: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    log_level: str = "INFO",
) -> AnalysisResult:
    """
    Analyses vegetation health change and surface-water change between two
    Sentinel-2 scene dates and returns a structured result object.

    This is the authoritative public entry point for the GIS Wetland
    Degradation & Depletion Engine V1. It orchestrates the complete
    Phase 1–12 processing chain internally.

    Parameters
    ----------
    t1_scene : str | Path
        Path to the T1 (baseline) scene directory containing GeoTIFF band
        files named according to the band_mapping in config.json.
    t2_scene : str | Path
        Path to the T2 (target) scene directory.
    t1_date : str
        T1 acquisition date in ISO 8601 format ``YYYY-MM-DD``.
    t2_date : str
        T2 acquisition date in ISO 8601 format ``YYYY-MM-DD``.
        Must be strictly later than ``t1_date``.
    config_path : str | Path
        Path to the JSON configuration file (e.g. ``configs/config.json``).
    model_path : str | Path
        Path to the ONNX model file
        (e.g. ``models/best_finetuned_model.onnx``).
    output_dir : str | Path
        Base directory where the run output package will be written.
        A subdirectory ``<run_id>/`` is created automatically.
    wetland_boundary : str | Path | None
        Optional path to a wetland boundary vector file (.geojson or .shp).
        If provided, wetland-restricted metrics are computed. If omitted,
        the wetland context defaults to ``WETLAND_BOUNDARY_UNAVAILABLE``.
    run_id : str | None
        Optional caller-specified run identifier. If ``None``, a unique ID
        is generated automatically as ``YYYYMMDDTHHMMSSZ_<uuid8>``.
    log_level : str
        Logging verbosity. One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``. Defaults to ``"INFO"``.

    Returns
    -------
    AnalysisResult
        Structured result containing:

        - ``run_id`` — unique identifier for this run
        - ``status`` — ``"SUCCESS"``, ``"WARNING"``, or ``"FAILED"``
        - ``output_directory`` — absolute path to the run output package
        - ``outputs`` — mapping of output key → relative path
        - ``metrics`` — computed scalar and area metrics
        - ``validation`` — QA flags and overall status
        - ``warnings`` — non-blocking warning messages
        - ``errors`` — error messages (non-empty if status == ``"FAILED"``)
        - ``metadata`` — provenance and execution metadata

    Notes
    -----
    - The engine **never** infers physical wetland depletion from vegetation
      degradation or NDWI water loss alone. The
      ``physical_depletion_assessment`` field is ``"NOT_ASSESSED"`` by
      default unless an independent hydrological source is accepted via
      configuration.
    - Scientific algorithm implementations are frozen at Phase 12.
      This function is a pure production orchestration wrapper.
    - A ``config_used.json`` snapshot is written inside the run output
      directory for full reproducibility.

    Raises
    ------
    Does **not** raise. All errors are captured in ``result.errors`` and
    ``result.status == "FAILED"``. This guarantees the caller always receives
    a structured response.

    Examples
    --------
    >>> from gis_engine.api.service import analyze_wetland
    >>> result = analyze_wetland(
    ...     t1_scene="data/2023-08-14",
    ...     t2_scene="data/2023-08-28",
    ...     t1_date="2023-08-14",
    ...     t2_date="2023-08-28",
    ...     config_path="configs/config.json",
    ...     model_path="models/best_finetuned_model.onnx",
    ...     output_dir="outputs/runs",
    ... )
    >>> print(result.status)  # "SUCCESS" | "WARNING" | "FAILED"
    >>> print(result.output_directory)
    """
    logger.info(
        "[service] analyze_wetland called: t1=%s t2=%s t1_date=%s t2_date=%s",
        t1_scene, t2_scene, t1_date, t2_date,
    )

    pipeline = ProductionPipeline()
    result = pipeline.run_analysis(
        t1_scene=t1_scene,
        t2_scene=t2_scene,
        t1_date=t1_date,
        t2_date=t2_date,
        config_path=config_path,
        model_path=model_path,
        output_base_dir=output_dir,
        wetland_boundary=wetland_boundary,
        run_id=run_id,
        log_level=log_level,
    )

    logger.info(
        "[service] analyze_wetland complete: run_id=%s status=%s",
        result.run_id, result.status,
    )
    return result
