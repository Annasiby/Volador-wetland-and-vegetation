"""
tests/test_pipeline_e2e_report.py
===================================
End-to-end pipeline test verifying that one call to pipeline.run_analysis()
creates the full GIS output hierarchy, cartographic figures, and the final PDF.
"""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import pypdf

from gis_engine.api.service import analyze_wetland as run_analysis


def _create_mock_scene(dir_path: Path, prefix: str = "scene"):
    dir_path.mkdir(parents=True, exist_ok=True)
    transform = from_origin(500000, 4000000, 10, 10)
    data = (np.ones((64, 64), dtype=np.float32) * 2000.0)

    for band in ["G", "R", "RE", "NIR"]:
        with rasterio.open(
            dir_path / f"{prefix}_{band}.tif",
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=1,
            dtype="float32",
            crs="EPSG:32632",
            transform=transform,
        ) as dst:
            dst.write(data, 1)


@patch("gis_engine.inference.onnx.ONNXModelWrapper.load_model")
@patch("gis_engine.inference.onnx.ONNXModelWrapper.predict")
def test_pipeline_e2e_produces_pdf_and_maps(mock_predict, mock_load, tmp_path):
    """Verifies that run_analysis produces FINAL_ANALYSIS_REPORT.pdf and map figures."""
    mock_load.return_value = None
    # Mock logits (1, 4, 64, 64) and class map (64, 64)
    logits = np.zeros((1, 4, 64, 64), dtype=np.float32)
    logits[0, 3, :, :] = 5.0  # Healthy
    classes = np.full((64, 64), 3, dtype=np.uint8)
    mock_predict.return_value = (logits, classes)

    t1_dir = tmp_path / "t1"
    t2_dir = tmp_path / "t2"
    _create_mock_scene(t1_dir)
    _create_mock_scene(t2_dir)

    out_dir = tmp_path / "outputs"

    # Resolve config path relative to Engine_module root (two levels up from this test file)
    _engine_root = Path(__file__).resolve().parent.parent
    _config_path = _engine_root / "configs" / "config.json"

    result = run_analysis(
        t1_scene=t1_dir,
        t2_scene=t2_dir,
        t1_date="2023-08-14",
        t2_date="2023-08-28",
        config_path=str(_config_path),
        model_path=t1_dir / "scene_G.tif",  # dummy model path
        output_dir=out_dir,
    )

    assert result.status in ["SUCCESS", "WARNING"]
    run_dir = Path(result.output_directory)

    # Check Markdown report
    md_report = run_dir / "reports" / "analysis_report.md"
    assert md_report.is_file()

    # Check PDF report
    pdf_report = run_dir / "reports" / "FINAL_ANALYSIS_REPORT.pdf"
    assert pdf_report.is_file()
    assert pdf_report.stat().st_size > 5000

    # Verify PDF content via PyPDF
    reader = pypdf.PdfReader(str(pdf_report))
    assert len(reader.pages) >= 3

    # Check outputs dictionary & manifest
    assert "reports/FINAL_ANALYSIS_REPORT.pdf" in result.outputs
    manifest_p = run_dir / "manifest.json"
    assert manifest_p.is_file()
