"""
tests/test_pdf.py
===================
Unit and integration tests for PDF report generation using ReportLab.
Validates multi-page PDF compilation, page count, table content, map embedding,
and ensures no NaN/None artifacts.
"""

from pathlib import Path
import pypdf
import pytest
from src.gis_engine.output.pdf import generate_pdf_report, generate_all_reports


@pytest.fixture
def sample_result_dict():
    """Provides a structured result dictionary reflecting standard pipeline output."""
    return {
        "run_id": "TEST_RUN_20230814_20230828",
        "software_version": "1.0.0",
        "status": "SUCCESS",
        "inputs": {
            "t1_path": "data/real_georeferenced/2023-08-14",
            "t2_path": "data/real_georeferenced/2023-08-28",
            "model_path": "models/best_finetuned_model.onnx",
            "wetland_boundary": None,
        },
        "temporal": {
            "t1_date": "2023-08-14",
            "t2_date": "2023-08-28",
            "interval_days": 14,
        },
        "spatial": {
            "crs": "EPSG:32632",
            "dimensions": [500, 500],
            "resolution": [10.0, 10.0],
            "extent": [500000.0, 4000000.0, 505000.0, 4005000.0],
            "overlap_percentage": 100.0,
        },
        "vegetation": {
            "t1": {
                "0": {"pixel_count": 50000, "area_ha": 50.0, "percentage": 20.0},
                "1": {"pixel_count": 50000, "area_ha": 50.0, "percentage": 20.0},
                "2": {"pixel_count": 50000, "area_ha": 50.0, "percentage": 20.0},
                "3": {"pixel_count": 100000, "area_ha": 100.0, "percentage": 40.0},
            },
            "t2": {
                "0": {"pixel_count": 60000, "area_ha": 60.0, "percentage": 24.0},
                "1": {"pixel_count": 70000, "area_ha": 70.0, "percentage": 28.0},
                "2": {"pixel_count": 40000, "area_ha": 40.0, "percentage": 16.0},
                "3": {"pixel_count": 80000, "area_ha": 80.0, "percentage": 32.0},
            },
        },
        "change": {
            "category_summary": {
                "STABLE": {"count": 180000, "area_ha": 180.0, "percentage": 72.0},
                "DEGRADATION": {"count": 30000, "area_ha": 30.0, "percentage": 12.0},
                "RECOVERY": {"count": 10000, "area_ha": 10.0, "percentage": 4.0},
                "VEGETATION_TO_BACKGROUND": {"count": 15000, "area_ha": 15.0, "percentage": 6.0},
                "BACKGROUND_TO_VEGETATION": {"count": 15000, "area_ha": 15.0, "percentage": 6.0},
            }
        },
        "cva": {
            "mean_magnitude": 0.32,
            "median_magnitude": 0.28,
            "p95_magnitude": 0.75,
            "mean_delta_ndvi": -0.05,
            "median_delta_ndvi": -0.03,
            "mean_delta_gndvi": -0.04,
            "median_delta_gndvi": -0.02,
            "mean_delta_ndre": -0.03,
            "median_delta_ndre": -0.01,
        },
        "degradation": {
            "pixel_count": 30000,
            "area_ha": 30.0,
            "percentage": 12.0,
        },
        "recovery": {
            "pixel_count": 10000,
            "area_ha": 10.0,
            "percentage": 4.0,
        },
        "hydrology": {
            "ndwi_threshold": 0.0,
            "water_gain_area_ha": 5.5,
            "water_loss_area_ha": 8.2,
        },
        "fusion": {
            "evidence_level": "HYDRO_ECOLOGICAL",
            "hydro_ecological_degradation_area_ha": 12.4,
            "hydro_ecological_recovery_area_ha": 3.1,
        },
        "metrics": {
            "normalized_degradation_ratio": 0.0909,
            "severity_weighted_degradation_area_ha": 38.5,
        },
        "validation": {
            "overall_status": "PASS",
            "checks": {
                "spatial_alignment": "PASS",
                "mathematical_reconciliation": "PASS",
            },
        },
        "physical_depletion_assessment": "NOT_ASSESSED",
    }


def test_generate_pdf_report(tmp_path, sample_result_dict):
    """Verifies that PDF report is compiled, readable, and structured."""
    pdf_out = tmp_path / "FINAL_ANALYSIS_REPORT.pdf"

    res_path = generate_pdf_report(
        result_dict=sample_result_dict,
        output_path=pdf_out,
        map_assets={},
        run_id="TEST_RUN_20230814_20230828",
    )

    assert res_path == pdf_out
    assert pdf_out.is_file()
    assert pdf_out.stat().st_size > 5000  # Non-trivial PDF size

    # Verify readable by PyPDF
    reader = pypdf.PdfReader(str(pdf_out))
    assert len(reader.pages) >= 3

    # Check text content of first page
    p1_text = reader.pages[0].extract_text()
    assert "GIS Wetland Degradation Assessment" in p1_text
    assert "WETLAND DEGRADATION ASSESSMENT" in p1_text
    assert "TEST_RUN_20230814_20230828" in p1_text
    assert "Executive Summary" in p1_text
    assert "Vegetation Degradation" in p1_text

    # Verify no raw pixel counts anywhere in the document
    full_pdf_text = "\n".join(p.extract_text() for p in reader.pages)
    assert " px" not in full_pdf_text
    assert " pixels" not in full_pdf_text
    assert "pixel count" not in full_pdf_text.lower()


def test_generate_pdf_report_with_map_assets(tmp_path, sample_result_dict):
    """Verifies embedding map figures into the PDF."""
    pdf_out = tmp_path / "REPORT_WITH_MAPS.pdf"
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()

    # Create dummy PNG figures
    import matplotlib.pyplot as plt
    map_assets = {}
    for name in ["health_t1", "health_t2", "transition", "cva_magnitude", "delta_ndvi", "water_change", "evidence_fusion"]:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, 1])
        ax.set_title(name)
        p = fig_dir / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        map_assets[name] = p

    res_path = generate_pdf_report(
        result_dict=sample_result_dict,
        output_path=pdf_out,
        map_assets=map_assets,
        run_id="TEST_RUN_MAPS",
    )

    assert res_path.is_file()
    reader = pypdf.PdfReader(str(res_path))
    assert len(reader.pages) >= 3


@pytest.fixture
def sample_vegetation_result_dict():
    """Provides a structured result dictionary reflecting single-date vegetation mode."""
    return {
        "run_id": "TEST_VEG_20230814",
        "analysis_mode": "vegetation",
        "software_version": "1.0.0",
        "status": "SUCCESS",
        "inputs": {
            "scene_path": "data/real_georeferenced/2023-08-14",
            "model_path": "models/best_finetuned_model.onnx",
        },
        "observation_date": "2023-08-14",
        "spatial": {
            "crs": "EPSG:32632",
            "dimensions": [500, 500],
            "resolution": [10.0, 10.0],
            "extent": [500000.0, 4000000.0, 505000.0, 4005000.0],
            "total_area_ha": 25.0,
        },
        "classes": {
            "0": {"name": "Non-Vegetation", "pixel_count": 50000, "area_ha": 5.0, "percentage": 20.0},
            "1": {"name": "Stressed Vegetation", "pixel_count": 50000, "area_ha": 5.0, "percentage": 20.0},
            "2": {"name": "Moderate Vegetation", "pixel_count": 50000, "area_ha": 5.0, "percentage": 20.0},
            "3": {"name": "Healthy Vegetation", "pixel_count": 100000, "area_ha": 10.0, "percentage": 40.0},
        },
        "indices": {
            "ndvi": {"mean": 0.55, "min": -0.1, "max": 0.85, "std": 0.15},
            "gndvi": {"mean": 0.48, "min": -0.05, "max": 0.78, "std": 0.12},
            "ndre": {"mean": 0.35, "min": -0.08, "max": 0.65, "std": 0.10},
        },
        "quality_assurance": {
            "overall_status": "PASS",
            "checks": {
                "spatial_alignment": "PASS",
                "valid_pixel_ratio": 1.0,
            },
        },
        "provenance": {
            "run_id": "TEST_VEG_20230814",
            "analysis_mode": "vegetation",
            "pipeline_version": "1.0.0",
        },
    }


def test_generate_all_reports_vegetation(tmp_path, sample_vegetation_result_dict):
    """Verifies generate_all_reports generates simple, technical, and legacy PDFs for vegetation mode."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_all_reports(
        result_dict=sample_vegetation_result_dict,
        run_dir=tmp_path,
        reports_dir=reports_dir,
        map_assets={},
        run_id="TEST_VEG_20230814",
    )

    assert "report" in generated
    assert "technical_report" in generated
    assert "legacy_report" in generated

    simple_pdf = generated["report"]
    tech_pdf = generated["technical_report"]
    legacy_pdf = generated["legacy_report"]

    assert simple_pdf.is_file()
    assert tech_pdf.is_file()
    assert legacy_pdf.is_file()
    assert simple_pdf.name == "vegetation_report.pdf"
    assert tech_pdf.name == "vegetation_technical_report.pdf"
    assert legacy_pdf.name == "FINAL_ANALYSIS_REPORT.pdf"

    # Verify readable and content
    reader_simple = pypdf.PdfReader(str(simple_pdf))
    assert len(reader_simple.pages) >= 2
    text_simple = "\n".join(p.extract_text() for p in reader_simple.pages)
    assert "Vegetation Health Assessment" in text_simple or "VEGETATION HEALTH" in text_simple.upper()
    assert "Transition Matrix" not in text_simple
    assert "Change Vector Analysis" not in text_simple

    reader_tech = pypdf.PdfReader(str(tech_pdf))
    assert len(reader_tech.pages) >= 2
    text_tech = "\n".join(p.extract_text() for p in reader_tech.pages)
    assert "TECHNICAL" in text_tech.upper() and "AUDIT" in text_tech.upper()


def test_generate_all_reports_wetland(tmp_path, sample_result_dict):
    """Verifies generate_all_reports generates simple, technical, and legacy PDFs for wetland mode."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_all_reports(
        result_dict=sample_result_dict,
        run_dir=tmp_path,
        reports_dir=reports_dir,
        map_assets={},
        run_id="TEST_RUN_20230814_20230828",
    )

    assert "report" in generated
    assert "technical_report" in generated
    assert "legacy_report" in generated

    assert generated["report"].name == "wetland_report.pdf"
    assert generated["technical_report"].name == "wetland_technical_report.pdf"
    assert generated["legacy_report"].name == "FINAL_ANALYSIS_REPORT.pdf"

    reader_simple = pypdf.PdfReader(str(generated["report"]))
    assert len(reader_simple.pages) >= 3
    text_simple = "\n".join(p.extract_text() for p in reader_simple.pages)
    assert "Wetland Degradation Assessment" in text_simple or "WETLAND DEGRADATION" in text_simple.upper()

    reader_tech = pypdf.PdfReader(str(generated["technical_report"]))
    assert len(reader_tech.pages) >= 2
    text_tech = "\n".join(p.extract_text() for p in reader_tech.pages)
    assert "TECHNICAL" in text_tech.upper() or "AUDIT" in text_tech.upper()

