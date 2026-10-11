import pytest
import numpy as np
from src.gis_engine.validation.olofsson import estimate_area_with_reference_samples
from src.gis_engine.validation.calibration import check_probability_calibration
from src.gis_engine.validation.quality import QualityAssuranceEngine

def test_olofsson_no_samples():
    """Verify Olofsson returns NOT_AVAILABLE when no reference samples exist."""
    res = estimate_area_with_reference_samples(None)
    assert res["status"] == "NOT_AVAILABLE"
    assert res["model_derived_estimate"] is True
    assert res["independent_area_error_correction"] is False
    assert "Model-derived estimate" in res["message"]

def test_olofsson_with_samples():
    """Verify Olofsson calculation is correct with mock samples and strata."""
    # 4 classes: 0, 1, 2, 3
    # Strata definition area
    strata_def = {
        0: {"area": 40.0},
        1: {"area": 30.0},
        2: {"area": 20.0},
        3: {"area": 10.0}
    }
    # reference samples
    # format: list of dicts with mapped_class and reference_class
    ref_samples = []
    # Strata 0: 10 samples, 8 match class 0, 2 are class 1
    ref_samples.extend([{"mapped_class": 0, "reference_class": 0}] * 8)
    ref_samples.extend([{"mapped_class": 0, "reference_class": 1}] * 2)
    # Strata 1: 10 samples, 9 match class 1, 1 is class 2
    ref_samples.extend([{"mapped_class": 1, "reference_class": 1}] * 9)
    ref_samples.extend([{"mapped_class": 1, "reference_class": 2}] * 1)
    # Strata 2: 10 samples, 7 match class 2, 3 are class 0
    ref_samples.extend([{"mapped_class": 2, "reference_class": 2}] * 7)
    ref_samples.extend([{"mapped_class": 2, "reference_class": 0}] * 3)
    # Strata 3: 10 samples, 10 match class 3
    ref_samples.extend([{"mapped_class": 3, "reference_class": 3}] * 10)
    
    res = estimate_area_with_reference_samples(
        classification_map=None,
        reference_samples=ref_samples,
        strata_definition=strata_def
    )
    
    assert res["status"] == "PASS"
    assert res["model_derived_estimate"] is False
    assert res["independent_area_error_correction"] is True
    assert res["total_area"] == 100.0
    
    strata_res = res["strata_results"]
    assert len(strata_res) == 4
    assert strata_res[3]["adjusted_area_estimate"] == pytest.approx(10.0) # all 10 matched, area weight 10%
    assert strata_res[3]["confidence_interval_95_lower"] == 10.0
    assert strata_res[3]["confidence_interval_95_upper"] == 10.0

def test_calibration_check():
    """Verify calibration check works as expected."""
    # Deferred
    res = check_probability_calibration(None)
    assert res["status"] == "NOT_AVAILABLE"
    assert "deferred" in res["message"].lower()
    
    # Active calibration run
    preds = np.random.rand(10, 4)
    gt = np.random.randint(0, 4, size=(10,))
    res_active = check_probability_calibration(preds, gt)
    assert res_active["status"] == "PASS"
    assert res_active["data_count"] == 10

def test_qa_engine_cva_math():
    """Verify CVA math identity check triggers a CRITICAL flag on mismatch."""
    config = {
        "metrics": {"severity_weights": {"healthy": 0.0, "moderate": 0.35, "severe": 0.8, "background": 1.0}}
    }
    engine = QualityAssuranceEngine(config)
    
    # Correct CVA math
    cva_mag = np.array([5.0], dtype=np.float32)
    d_ndvi = np.array([3.0], dtype=np.float32)
    d_gndvi = np.array([4.0], dtype=np.float32)
    d_ndre = np.array([0.0], dtype=np.float32)
    valid_mask = np.array([True])
    
    res = engine.run_cva_qa(cva_mag, d_ndvi, d_gndvi, d_ndre, valid_mask)
    assert res["status"] == "PASS"
    assert len(engine.flags) == 0
    
    # Incorrect CVA math
    cva_mag_bad = np.array([10.0], dtype=np.float32) # should be 5
    res_bad = engine.run_cva_qa(cva_mag_bad, d_ndvi, d_gndvi, d_ndre, valid_mask)
    assert res_bad["status"] == "FAIL"
    assert len(engine.flags) == 1
    assert engine.flags[0]["code"] == "QA-020"
    assert engine.flags[0]["severity"] == "CRITICAL"
