"""
backend/app/routers/results.py
===============================
Router for fetching scientific run outputs, metrics, and transition matrices.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from ..services import RunWorker, STORAGE_DIR

router = APIRouter(prefix="/api/runs/{run_id}/results", tags=["Results"])


@router.get("")
def get_run_results(run_id: str):
    run_info = RunWorker.get_run_status(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    out_dir = Path(run_info["output_directory"])
    analysis_file = out_dir / "analysis_result.json"
    if not analysis_file.exists() and (out_dir / run_id / "analysis_result.json").exists():
        analysis_file = out_dir / run_id / "analysis_result.json"

    if not analysis_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Analysis results not ready for run {run_id}. Current status: {run_info['status']}"
        )

    try:
        with open(analysis_file, "r") as f:
            data = json.load(f)

        # Enrich transition matrix from transition_matrix.csv if missing
        if "metrics" in data and ("transition_matrix" not in data["metrics"] or not data["metrics"]["transition_matrix"]):
            trans_csv_candidates = [
                out_dir / "metrics" / "transition_matrix.csv",
                out_dir / run_id / "metrics" / "transition_matrix.csv",
                out_dir / "change" / "transition_matrix.csv",
                out_dir / run_id / "change" / "transition_matrix.csv",
            ]
            for trans_csv in trans_csv_candidates:
                if trans_csv.is_file():
                    import csv
                    matrix_m2 = [[0.0] * 4 for _ in range(4)]
                    matrix_pixels = [[0] * 4 for _ in range(4)]
                    try:
                        with open(trans_csv, "r", encoding="utf-8") as cf:
                            reader = csv.DictReader(cf)
                            for row in reader:
                                tid = int(row.get("transition_id", -1))
                                if 0 <= tid < 16:
                                    r = tid // 4
                                    c = tid % 4
                                    matrix_m2[r][c] = float(row.get("area_m2") or 0.0)
                                    matrix_pixels[r][c] = int(row.get("count") or 0)
                        data["metrics"]["transition_matrix"] = {
                            "matrix_m2": matrix_m2,
                            "matrix_pixels": matrix_pixels,
                        }
                        break
                    except Exception:
                        pass

        # Convenience alias for cva mean_magnitude & p95_magnitude if nested
        cva = data.get("cva")
        if isinstance(cva, dict):
            cva_mag = cva.get("cva_magnitude")
            if isinstance(cva_mag, dict):
                if "mean_magnitude" not in cva and "mean" in cva_mag:
                    cva["mean_magnitude"] = cva_mag["mean"]
                if "p95_magnitude" not in cva and "p95" in cva_mag:
                    cva["p95_magnitude"] = cva_mag["p95"]
            for d_name in ["delta_ndvi", "delta_gndvi", "delta_ndre"]:
                d_obj = cva.get(d_name)
                alias_name = f"mean_{d_name}"
                if isinstance(d_obj, dict) and alias_name not in cva and "mean" in d_obj:
                    cva[alias_name] = d_obj["mean"]

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read analysis results: {str(e)}")
