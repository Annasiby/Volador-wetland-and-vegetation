# GIS Wetland Degradation & Depletion Engine — Analysis Report

**Run ID:** `run_20261001_052913`  
**Generated:** 2026-10-01 05:29:15 UTC  
**Software Version:** 1.0.0  
**Status:** SUCCESS  

---

## 1. Executive Summary

This report presents the results of a two-date vegetation health and surface-water change analysis for the wetland study area.

- **QA Status:** WARN
- **Wetland Boundary Status:** WETLAND_BOUNDARY_UNAVAILABLE

> **Note:** This engine assesses vegetation health change and surface-water change from optical satellite imagery. Results represent vegetation degradation evidence and surface-water change evidence. Physical wetland depletion requires independent supporting evidence beyond the scope of this V1 analysis.

---

## 2. Input Data

| Field | Value |
|---|---|
| T1 Scene | `N/A` |
| T2 Scene | `N/A` |
| Model | `C:\Users\HP\OneDrive\Desktop\wetLand_Depreciation\backend\Engine_module\models\best_finetuned_model.onnx` |

---

## 3. Temporal Information

| Field | Value |
|---|---|
| T1 Acquisition Date | N/A |
| T2 Acquisition Date | N/A |
| Temporal Interval | N/A days |

---

## 4. Spatial Information

| Field | Value |
|---|---|
| CRS | `EPSG:32633` |
| Dimensions (H × W) | 20 × 20 px |
| Resolution | 10.0 × 10.0 m |
| T1/T2 Overlap | N/A |

---

## 5. Vegetation Health

### T1 (Baseline)
| Class | Pixels | Area | Percentage |
|---|---|---|---|
| Class 0 | 0 | 0.00 ha (0 m²) | 0.00% |
| Class 1 | 0 | 0.00 ha (0 m²) | 0.00% |
| Class 2 | 0 | 0.00 ha (0 m²) | 0.00% |
| Class 3 | 0 | 0.00 ha (0 m²) | 0.00% |

### T2 (Target)
| Class | Pixels | Area | Percentage |
|---|---|---|---|
| Class 0 | 0 | 0.00 ha (0 m²) | 0.00% |
| Class 1 | 0 | 0.00 ha (0 m²) | 0.00% |
| Class 2 | 0 | 0.00 ha (0 m²) | 0.00% |
| Class 3 | 0 | 0.00 ha (0 m²) | 0.00% |

---

## 6. Vegetation Change (Post-Classification Transitions)

| Category | Pixel Count | Area |
|---|---|---|
| Vegetation Degradation | N/A | N/A |
| Vegetation Recovery | N/A | N/A |
| Stable | N/A | — |

---

## 7. Change Vector Analysis (CVA)

| Metric | Value |
|---|---|
| Mean CVA Magnitude | N/A |
| Median CVA Magnitude | N/A |
| 95th Percentile CVA Magnitude | N/A |
| Mean ΔNDVI | N/A |
| Mean ΔGNDVI | N/A |
| Mean ΔNDRE | N/A |
| Valid Pixels | N/A |

---

## 8. Degradation

| Metric | Value |
|---|---|
| Degraded Pixels | N/A |
| Degraded Area | N/A |
| Degraded Percentage (valid area) | N/A |

## 9. Recovery

| Metric | Value |
|---|---|
| Recovered Pixels | N/A |
| Recovered Area | N/A |
| Recovered Percentage (valid area) | N/A |

---

## 10. Wetland Context

| Field | Value |
|---|---|
| Boundary Source | UNAVAILABLE |
| Wetland Area | N/A |
| Physical Depletion Assessment Enabled | No |

---

## 11. Surface-Water Change (NDWI Hydrology)

| Metric | Value |
|---|---|
| NDWI Threshold | N/A |
| Baseline Surface-Water Area (T1) | N/A |
| Present Surface-Water Area (T2) | N/A |
| Surface-Water Gain Area | N/A |
| Surface-Water Loss Area | N/A |

---

## 12. Hydro-Ecological Evidence (Multi-Evidence Fusion)

| Metric | Value |
|---|---|
| Evidence Level | N/A |
| Hydro-Ecological Degradation Evidence Area | N/A |
| Hydro-Ecological Recovery Evidence Area | N/A |

---

## 13. Metrics Summary

| Metric | Value |
|---|---|
| Normalized Degradation Ratio (NDR) | N/A |
| Severity-Weighted Degradation Area | N/A |

---

## 14. Quality Assurance

**Overall QA Status:** WARN

| Code | Severity | Phase | Message |
|---|---|---|---|
| QA-011 | LOW | Uncertainty QA | Wetland boundary is inferred or unavailable |
| QA-012 | INFO | Uncertainty QA | Independent reference samples unavailable. Area-estimation uncertainty unavailable. |
| QA-014 | INFO | Uncertainty QA | Olofsson area-error correction unavailable. |

---

## 15. Uncertainty

- **Mean Model Confidence (T1):** N/A
- Confidence calibration: V2 feature (not available in V1).
- No independent reference samples provided; Olofsson area corrections not available.

---

## 16. V1 Scope & Limitations

The following capabilities are **outside the scope of V1** and are planned for future versions:

- MNDWI, LSWI, SWIR, SAR, LST analysis
- Soil chemistry or groundwater modelling
- Field calibration or accuracy assessment with reference samples
- Multi-year trajectory modelling (LandTrendr, BFAST, CCDC)
- Automatic wetland boundary segmentation
- Probability calibration framework (Platt / isotonic regression)

---

## 17. Physical Depletion Assessment

**Assessment Status: NOT_ASSESSED**

Physical wetland depletion has **not** been assessed in this run. The engine detected vegetation degradation evidence and/or surface-water loss evidence from optical imagery, but these indicators alone are insufficient to conclude physical wetland depletion.

Conservative interpretation: *Vegetation degradation associated with surface-water loss* represents hydro-ecological degradation evidence. Independent hydrological survey data would be required to support a physical depletion claim.

---

## 20. Output Inventory

| Key | Relative Path |
|---|---|
| vegetation/health.tif | `vegetation/health.tif` |
| vegetation/confidence.tif | `vegetation/confidence.tif` |
| vegetation/statistics.json | `vegetation/statistics.json` |
| validation/qa_report.json | `validation/qa_report.json` |
| analysis_result.json | `analysis_result.json` |

---

*End of Report*