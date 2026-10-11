"""
backend/app/routers/reports.py
===============================
Router for serving generated PDF analysis reports.

Endpoints:
  GET /api/runs/{run_id}/report/pdf           — Simple (client-facing) report
  GET /api/runs/{run_id}/report/pdf/technical — Full scientific/technical report
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..services import RunWorker

router = APIRouter(prefix="/api/runs/{run_id}/report", tags=["Reports"])


def _resolve_run_dir(run_id: str) -> Path:
    """Return the run output directory, raising 404 if the run is unknown."""
    run_info = RunWorker.get_run_status(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    out_dir = Path(run_info["output_directory"])
    if (out_dir / run_id).is_dir():
        out_dir = out_dir / run_id
    return out_dir


def _find_simple_pdf(out_dir: Path) -> Path:
    """
    Locate the simple (client-facing) PDF report.
    Priority order:
      1. vegetation_report.pdf  (vegetation mode)
      2. wetland_report.pdf     (wetland mode)
      3. FINAL_ANALYSIS_REPORT.pdf  (legacy alias written by generate_all_reports)
      4. analysis_report.pdf    (older legacy)
    """
    candidates = [
        out_dir / "reports" / "vegetation_report.pdf",
        out_dir / "reports" / "wetland_report.pdf",
        out_dir / "reports" / "FINAL_ANALYSIS_REPORT.pdf",
        out_dir / "reports" / "analysis_report.pdf",
        out_dir / "analysis_report.pdf",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[2]  # will trigger 404 via caller


def _find_technical_pdf(out_dir: Path) -> Path:
    """
    Locate the technical (scientific audit) PDF report.
    Priority order:
      1. vegetation_technical_report.pdf  (vegetation mode)
      2. wetland_technical_report.pdf     (wetland mode)
      3. FINAL_ANALYSIS_REPORT.pdf        (fallback to simple if only one exists)
    """
    candidates = [
        out_dir / "reports" / "vegetation_technical_report.pdf",
        out_dir / "reports" / "wetland_technical_report.pdf",
        out_dir / "reports" / "FINAL_ANALYSIS_REPORT.pdf",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # will trigger 404 via caller


@router.get("/pdf", summary="Download the client-facing summary PDF report")
def get_pdf_report(run_id: str):
    """
    Returns the simple (executive summary) PDF for the given run.
    Works for both VEGETATION and WETLAND analysis modes.
    """
    out_dir = _resolve_run_dir(run_id)
    pdf_path = _find_simple_pdf(out_dir)

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"PDF report not found for run {run_id}. "
                "The analysis may still be running or the reporting step may have failed."
            ),
        )

    filename = f"Analysis_Report_{run_id}.pdf"
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
    )


@router.get("/pdf/technical", summary="Download the full scientific/technical PDF report")
def get_technical_pdf_report(run_id: str):
    """
    Returns the technical (full scientific audit) PDF for the given run.
    Works for both VEGETATION and WETLAND analysis modes.
    """
    out_dir = _resolve_run_dir(run_id)
    pdf_path = _find_technical_pdf(out_dir)

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Technical PDF report not found for run {run_id}. "
                "The analysis may still be running or the reporting step may have failed."
            ),
        )

    filename = f"Technical_Report_{run_id}.pdf"
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
    )
