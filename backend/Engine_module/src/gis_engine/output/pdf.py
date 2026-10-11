"""
src/gis_engine/output/pdf.py
==============================
Production PDF report generation entrypoint for the GIS Engine.

Supports both analysis modes:
1. VEGETATION MODE:
   - Simple Report: Vegetation Health Report (vegetation_report.pdf)
   - Technical Report: Vegetation Health Technical Report (vegetation_technical_report.pdf)
2. WETLAND MODE:
   - Simple Report: Wetland Degradation Report (wetland_report.pdf)
   - Technical Report: Wetland Degradation Technical Report (wetland_technical_report.pdf)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .pdf_common import (
    NumberedCanvas,
    make_numbered_canvas,
    get_report_styles,
    _safe_float,
    _safe_int,
    _calc_pct,
    _fmt_pct,
    _fmt_num,
    _fmt_ha,
    _fmt_spatial_proportion,
    _safe_get,
    _p,
    _image_exists,
    _img,
    _image_pair,
    _table,
    _callout_box,
    _side_by_side_table,
    _load_wetland_metadata,
    _wetland_source_label,
    _extract_class_metrics,
    _veg_class_pct_from_metrics,
    _wetland_assessment,
)
from .pdf_vegetation import (
    generate_vegetation_report,
    generate_vegetation_technical_report,
)
from .pdf_wetland import (
    generate_wetland_report,
    generate_wetland_technical_report,
)

logger = logging.getLogger(__name__)


def generate_all_reports(
    result_dict: Dict[str, Any],
    run_dir: Path,
    reports_dir: Path,
    map_assets: Optional[Dict[str, Path]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Generate both Simple and Technical PDF reports for the completed analysis run.

    ONE analysis run produces ONE result contract, which is passed into:
    1. Simple Report PDF (accessible, plain-language summary)
    2. Technical Report PDF (full technical and scientific audit)
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    map_assets = map_assets or {}
    run_id = run_id or result_dict.get("run_id", "RUN_UNKNOWN")

    analysis_mode = str(result_dict.get("analysis_mode", "")).strip().lower()
    # Auto-detect if not explicitly set
    if not analysis_mode:
        if "temporal" in result_dict and "t2_date" in result_dict.get("temporal", {}):
            analysis_mode = "wetland"
        elif "change" in result_dict or "cva" in result_dict or "hydrology" in result_dict:
            analysis_mode = "wetland"
        else:
            analysis_mode = "vegetation"

    generated: Dict[str, Path] = {}

    if analysis_mode == "vegetation":
        simple_path = reports_dir / "vegetation_report.pdf"
        tech_path = reports_dir / "vegetation_technical_report.pdf"

        generate_vegetation_report(
            result_dict=result_dict,
            output_path=simple_path,
            map_assets=map_assets,
            run_id=run_id,
            run_dir=run_dir,
        )
        generate_vegetation_technical_report(
            result_dict=result_dict,
            output_path=tech_path,
            map_assets=map_assets,
            run_id=run_id,
            run_dir=run_dir,
        )

        generated["report"] = simple_path
        generated["technical_report"] = tech_path

    else:
        simple_path = reports_dir / "wetland_report.pdf"
        tech_path = reports_dir / "wetland_technical_report.pdf"

        generate_wetland_report(
            result_dict=result_dict,
            output_path=simple_path,
            map_assets=map_assets,
            run_id=run_id,
            run_dir=run_dir,
        )
        generate_wetland_technical_report(
            result_dict=result_dict,
            output_path=tech_path,
            map_assets=map_assets,
            run_id=run_id,
            run_dir=run_dir,
        )

        generated["report"] = simple_path
        generated["technical_report"] = tech_path

    # Backward compatibility: write or duplicate as FINAL_ANALYSIS_REPORT.pdf
    legacy_path = reports_dir / "FINAL_ANALYSIS_REPORT.pdf"
    try:
        shutil.copyfile(simple_path, legacy_path)
        generated["legacy_report"] = legacy_path
    except Exception as exc:
        logger.warning("Could not create legacy FINAL_ANALYSIS_REPORT.pdf alias: %s", exc)

    return generated


def generate_pdf_report(
    result_dict: Dict[str, Any],
    output_path: Path,
    map_assets: Optional[Dict[str, Path]] = None,
    run_id: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Backward-compatible single report generator.
    Dispatches to appropriate mode generator based on result_dict.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analysis_mode = str(result_dict.get("analysis_mode", "")).strip().lower()
    if not analysis_mode:
        if "temporal" in result_dict and "t2_date" in result_dict.get("temporal", {}):
            analysis_mode = "wetland"
        elif "change" in result_dict or "cva" in result_dict or "hydrology" in result_dict:
            analysis_mode = "wetland"
        else:
            analysis_mode = "vegetation"

    if analysis_mode == "vegetation":
        return generate_vegetation_report(
            result_dict=result_dict,
            output_path=output_path,
            map_assets=map_assets,
            run_id=run_id,
            run_dir=run_dir,
        )
    else:
        return generate_wetland_report(
            result_dict=result_dict,
            output_path=output_path,
            map_assets=map_assets,
            run_id=run_id,
            run_dir=run_dir,
        )


__all__ = [
    "generate_all_reports",
    "generate_pdf_report",
    "generate_vegetation_report",
    "generate_vegetation_technical_report",
    "generate_wetland_report",
    "generate_wetland_technical_report",
    "NumberedCanvas",
    "make_numbered_canvas",
    "get_report_styles",
]
