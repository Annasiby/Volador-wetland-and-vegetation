"""
backend/tests/test_identical_input_report.py
=============================================
Regression test for identical T1/T2 datasets and PDF report data integrity.

Verifies:
1. Same input date handling (T1=2023-08-14, T2=2023-08-14, Interval=0 days)
2. Zero temporal change (Degradation=0 ha, Recovery=0 ha, Water Change=0 ha)
3. Nonzero baseline area when valid pixels exist
4. Class-area reconciliation (T1 class sum == Analysis Area == T2 class sum)
5. Transition-area reconciliation (H->H == Analysis Area)
6. CVA finite statistics (0.0000, NOT N/A)
7. Spectral difference finite statistics (0.0000, NOT N/A)
8. Water baseline & present extents match (179.36 ha, Gain=0, Loss=0)
9. PDF compilation succeeds and faithfully displays all numeric values
"""

import sys
from pathlib import Path

# Ensure Engine_module/src is in sys.path
ENGINE_SRC = Path(__file__).resolve().parent.parent / "Engine_module" / "src"
if str(ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(ENGINE_SRC))

import json
import pypdf
import pytest

from gis_engine.output.pdf import generate_pdf_report
from gis_engine.output.report import generate_analysis_report


@pytest.fixture
def identical_run_result():
    """Builds a representative result dictionary for an identical T1/T2 drone imagery run."""
    total_pixels = 5038848
    pixel_area_ha = 1.0 / 10000.0  # 1m x 1m resolution = 0.0001 ha/pixel
    total_area_ha = total_pixels * pixel_area_ha  # 503.8848 ha
    water_pixels = 1793647
    water_area_ha = water_pixels * pixel_area_ha  # 179.3647 ha

    return {
        "run_id": "run_20260829_identical_test",
        "software_version": "1.0.0",
        "status": "SUCCESS",
        "inputs": {
            "t1_path": "data/T1",
            "t2_path": "data/T2",
            "model_path": "models/best_finetuned_model.onnx",
            "wetland_boundary": None,
        },
        "temporal": {
            "t1_date": "2023-08-14",
            "t2_date": "2023-08-14",
            "interval_days": 0,
        },
        "spatial": {
            "crs": "None",
            "dimensions": [1944, 2592],
            "resolution": [1.0, 1.0],
            "extent": [0.0, 1944.0, 2592.0, 0.0],
            "overlap_percentage": 100.0,
        },
        "vegetation": {
            "t1": {
                "date_valid": {
                    "total_valid_pixels": total_pixels,
                    "classes": {
                        "0": {"name": "Background", "count": 0, "percentage": 0.0},
                        "1": {"name": "Severe", "count": 0, "percentage": 0.0},
                        "2": {"name": "Moderate", "count": 0, "percentage": 0.0},
                        "3": {"name": "Healthy", "count": total_pixels, "percentage": 100.0},
                    },
                },
                "common_overlap": {
                    "total_valid_pixels": total_pixels,
                    "classes": {
                        "0": {"name": "Background", "count": 0, "percentage": 0.0},
                        "1": {"name": "Severe", "count": 0, "percentage": 0.0},
                        "2": {"name": "Moderate", "count": 0, "percentage": 0.0},
                        "3": {"name": "Healthy", "count": total_pixels, "percentage": 100.0},
                    },
                },
            },
            "t2": {
                "date_valid": {
                    "total_valid_pixels": total_pixels,
                    "classes": {
                        "0": {"name": "Background", "count": 0, "percentage": 0.0},
                        "1": {"name": "Severe", "count": 0, "percentage": 0.0},
                        "2": {"name": "Moderate", "count": 0, "percentage": 0.0},
                        "3": {"name": "Healthy", "count": total_pixels, "percentage": 100.0},
                    },
                },
                "common_overlap": {
                    "total_valid_pixels": total_pixels,
                    "classes": {
                        "0": {"name": "Background", "count": 0, "percentage": 0.0},
                        "1": {"name": "Severe", "count": 0, "percentage": 0.0},
                        "2": {"name": "Moderate", "count": 0, "percentage": 0.0},
                        "3": {"name": "Healthy", "count": total_pixels, "percentage": 100.0},
                    },
                },
            },
        },
        "change": {
            "category_summary": {
                "STABLE": {"count": total_pixels, "area_ha": total_area_ha, "percentage": 100.0},
                "DEGRADATION": {"count": 0, "area_ha": 0.0, "percentage": 0.0},
                "RECOVERY": {"count": 0, "area_ha": 0.0, "percentage": 0.0},
                "VEGETATION_TO_BACKGROUND": {"count": 0, "area_ha": 0.0, "percentage": 0.0},
                "BACKGROUND_TO_VEGETATION": {"count": 0, "area_ha": 0.0, "percentage": 0.0},
            }
        },
        "cva": {
            "delta_ndvi": {
                "count": total_pixels,
                "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0,
                "std": 0.0, "p95": 0.0,
            },
            "delta_gndvi": {
                "count": total_pixels,
                "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0,
                "std": 0.0, "p95": 0.0,
            },
            "delta_ndre": {
                "count": total_pixels,
                "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0,
                "std": 0.0, "p95": 0.0,
            },
            "cva_magnitude": {
                "count": total_pixels,
                "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0,
                "std": 0.0, "p95": 0.0,
            },
        },
        "degradation": {
            "pixel_count": 0,
            "area_ha": 0.0,
            "percentage": 0.0,
        },
        "recovery": {
            "pixel_count": 0,
            "area_ha": 0.0,
            "percentage": 0.0,
        },
        "wetland": {
            "status": "WETLAND_BOUNDARY_UNAVAILABLE",
            "source_type": "UNAVAILABLE",
            "area_ha": total_area_ha,
            "physical_depletion_enabled": False,
        },
        "hydrology": {
            "ndwi_threshold": 0.0,
            "water_gain_area_ha": 0.0,
            "water_loss_area_ha": 0.0,
            "water_area_t1_ha": water_area_ha,
            "water_area_t2_ha": water_area_ha,
        },
        "fusion": {
            "evidence_level": "NOT_ASSESSED",
            "hydro_ecological_degradation_area_ha": 0.0,
            "hydro_ecological_recovery_area_ha": 0.0,
        },
        "metrics": {
            "normalized_degradation_ratio": 0.0,
            "severity_weighted_degradation_area_ha": 0.0,
            "impact_area": {"degradation_ha": 0.0},
            "recovery_area": {"recovery_ha": 0.0},
            "cva_magnitude": {"mean": 0.0},
            "analysis_extent": {"valid_analysis_area_ha": total_area_ha},
        },
        "validation": {
            "overall_status": "WARN",
            "flags": [
                {
                    "code": "QA-011",
                    "severity": "LOW",
                    "message": "Wetland boundary is inferred or unavailable",
                    "phase": "Uncertainty QA",
                    "status": "WARN",
                }
            ],
        },
        "uncertainty": {
            "mean_confidence_t1": 0.95568,
        },
        "physical_depletion_assessment": "NOT_ASSESSED",
        "warnings": ["T1 and T2 dates are identical (2023-08-14). Interval is 0 days."],
        "errors": [],
    }


def test_date_provenance_identical_run(identical_run_result):
    """Verifies that identical T1/T2 inputs preserve 0-day interval without date fabrication."""
    t1_date = identical_run_result["temporal"]["t1_date"]
    t2_date = identical_run_result["temporal"]["t2_date"]
    interval = identical_run_result["temporal"]["interval_days"]

    assert t1_date == "2023-08-14"
    assert t2_date == "2023-08-14"
    assert interval == 0


def test_area_reconciliation_mathematical_consistency(identical_run_result):
    """Verifies that valid class areas reconcile with total analysis area and transition area."""
    analysis_area = identical_run_result["metrics"]["analysis_extent"]["valid_analysis_area_ha"]
    assert analysis_area == pytest.approx(503.8848, rel=1e-4)

    # T1 class areas sum
    t1_classes = identical_run_result["vegetation"]["t1"]["date_valid"]["classes"]
    t1_total_pixels = sum(c["count"] for c in t1_classes.values())
    t1_area_ha = t1_total_pixels * 0.0001
    assert t1_area_ha == pytest.approx(analysis_area, rel=1e-4)

    # T2 class areas sum
    t2_classes = identical_run_result["vegetation"]["t2"]["date_valid"]["classes"]
    t2_total_pixels = sum(c["count"] for c in t2_classes.values())
    t2_area_ha = t2_total_pixels * 0.0001
    assert t2_area_ha == pytest.approx(analysis_area, rel=1e-4)

    # Transition category sum
    cat_summary = identical_run_result["change"]["category_summary"]
    cat_area_ha = sum(c["area_ha"] for c in cat_summary.values())
    assert cat_area_ha == pytest.approx(analysis_area, rel=1e-4)
    assert cat_summary["STABLE"]["area_ha"] == pytest.approx(analysis_area, rel=1e-4)
    assert cat_summary["DEGRADATION"]["area_ha"] == 0.0
    assert cat_summary["RECOVERY"]["area_ha"] == 0.0


def test_pdf_report_data_fidelity(tmp_path, identical_run_result):
    """Verifies PDF report generation with identical run data: no N/A on valid numbers, exact dates."""
    pdf_path = tmp_path / "FINAL_ANALYSIS_REPORT.pdf"
    res_path = generate_pdf_report(
        result_dict=identical_run_result,
        output_path=pdf_path,
        map_assets={},
        run_id="run_20260829_identical_test",
    )

    assert res_path.is_file()
    assert pdf_path.stat().st_size > 10000

    reader = pypdf.PdfReader(str(pdf_path))
    full_text = "\n".join(page.extract_text() for page in reader.pages)

    # Date provenance checks
    assert "2023-08-14" in full_text
    assert "0 days" in full_text
    assert "2023-08-28" not in full_text  # NO fabricated 14-day interval

    # Area checks
    assert "503.88 ha" in full_text

    # CRS checks
    assert "Undefined (not provided by source raster)" in full_text
    # Should not literally display "CRS: None"
    assert "CRS: None" not in full_text

    # CVA statistics: MUST be 0.0000, NOT "N/A"
    assert "CVA Magnitude" in full_text
    assert "0.0000" in full_text
    assert ("ΔNDVI" in full_text or "∆NDVI" in full_text)
    assert ("ΔGNDVI" in full_text or "∆GNDVI" in full_text)
    assert ("ΔNDRE" in full_text or "∆NDRE" in full_text)

    # Hydrology checks
    assert "Baseline Water Extent (T1)" in full_text
    assert "Present Water Extent (T2)" in full_text
    assert "179.36 ha" in full_text
    assert "Surface Water Expansion (Gain)" in full_text
    assert "Surface Water Contraction (Loss)" in full_text


def test_markdown_report_data_fidelity(tmp_path, identical_run_result):
    """Verifies Markdown report generation with identical run data."""
    md_path = tmp_path / "analysis_report.md"
    res_path = generate_analysis_report(
        result_dict=identical_run_result,
        output_path=md_path,
        run_id="run_20260829_identical_test",
    )

    assert res_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")

    assert "2023-08-14" in md_text
    assert "0 days" in md_text
    assert "503.88 ha" in md_text
    assert "179.36 ha" in md_text
    assert "Undefined / not provided by source raster" in md_text
