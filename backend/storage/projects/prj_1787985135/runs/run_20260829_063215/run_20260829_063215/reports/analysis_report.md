# GIS Wetland Degradation & Depletion Engine — Analysis Report

**Run ID:** `run_20260829_063215`  
**Generated:** 2026-08-29 06:32:17 UTC  
**Software Version:** 1.0.0  
**Status:** SUCCESS  

---

## 1. Executive Summary

This report presents the results of a two-date vegetation health and surface-water change analysis for the wetland study area.

- **QA Status:** WARN
- **Wetland Boundary Status:** WETLAND_BOUNDARY_AVAILABLE
- **Vegetation Degradation Area:** 0.00 ha (0 m²)
- **Vegetation Recovery Area:** 0.00 ha (0 m²)

> **Note:** This engine assesses vegetation health change and surface-water change from optical satellite imagery. Results represent vegetation degradation evidence and surface-water change evidence. Physical wetland depletion requires independent supporting evidence beyond the scope of this V1 analysis.

---

## 2. Input Data

| Field | Value |
|---|---|
| T1 Scene | `C:\Users\HP\OneDrive\Desktop\wetLand_Depreciation\backend\Engine_module\data\sample_scene\t1` |
| T2 Scene | `C:\Users\HP\OneDrive\Desktop\wetLand_Depreciation\backend\Engine_module\data\sample_scene\t2` |
| Model | `C:\Users\HP\OneDrive\Desktop\wetLand_Depreciation\backend\Engine_module\models\best_finetuned_model.onnx` |

---

## 3. Temporal Information

| Field | Value |
|---|---|
| T1 Acquisition Date | 2023-08-14 |
| T2 Acquisition Date | 2023-08-28 |
| Temporal Interval | 14 days |

---

## 4. Spatial Information

| Field | Value |
|---|---|
| CRS | `EPSG:32633` |
| Dimensions (H × W) | 100 × 100 px |
| Resolution | 10.0 × 10.0 m |
| T1/T2 Overlap | 100.00% |

---

## 5. Vegetation Health

### T1 (Baseline)
| Class | Pixels | Percentage |
|---|---|---|
| Class 0 | N/A | N/A |
| Class 1 | N/A | N/A |
| Class 2 | N/A | N/A |
| Class 3 | N/A | N/A |

### T2 (Target)
| Class | Pixels | Percentage |
|---|---|---|
| Class 0 | N/A | N/A |
| Class 1 | N/A | N/A |
| Class 2 | N/A | N/A |
| Class 3 | N/A | N/A |

---

## 6. Vegetation Change (Post-Classification Transitions)

| Category | Pixel Count | Area |
|---|---|---|
| Vegetation Degradation | 0 | 0.00 ha (0 m²) |
| Vegetation Recovery | 0 | 0.00 ha (0 m²) |
| Stable | 10000 | — |

---

## 7. Change Vector Analysis (CVA)

| Metric | Value |
|---|---|
| Mean CVA Magnitude | N/A |
| 95th Percentile CVA Magnitude | N/A |
| Valid Pixels | N/A |

---

## 8. Degradation

| Metric | Value |
|---|---|
| Degraded Pixels | 0 |
| Degraded Area | 0.00 ha (0 m²) |
| Degraded Percentage (valid area) | 0.00% |

## 9. Recovery

| Metric | Value |
|---|---|
| Recovered Pixels | 0 |
| Recovered Area | 0.00 ha (0 m²) |
| Recovered Percentage (valid area) | 0.00% |

---

## 10. Wetland Context

| Field | Value |
|---|---|
| Boundary Source | PROJECT_BOUNDARY |
| Wetland Area | 36.00 ha (360,000 m²) |
| Physical Depletion Assessment Enabled | No |

---

## 11. Surface-Water Change (NDWI Hydrology)

| Metric | Value |
|---|---|
| NDWI Threshold | 0.0 |
| Surface-Water Gain Area | 6.72 ha (67,200 m²) |
| Surface-Water Loss Area | 0.00 ha (0 m²) |

---

## 12. Hydro-Ecological Evidence (Multi-Evidence Fusion)

| Metric | Value |
|---|---|
| Evidence Level | NOT_ASSESSED |
| Hydro-Ecological Degradation Evidence Area | 0.00 ha (0 m²) |
| Hydro-Ecological Recovery Evidence Area | 0.00 ha (0 m²) |

---

## 13. Metrics Summary

| Metric | Value |
|---|---|
| Normalized Degradation Ratio (NDR) | 0.0000 |
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

- **Mean Model Confidence (T1):** 0.9270
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
| alignment/t1_aligned/green.tif | `alignment/t1_aligned/green.tif` |
| alignment/t1_aligned/red.tif | `alignment/t1_aligned/red.tif` |
| alignment/t1_aligned/rededge.tif | `alignment/t1_aligned/rededge.tif` |
| alignment/t1_aligned/nir.tif | `alignment/t1_aligned/nir.tif` |
| alignment/t2_aligned/green.tif | `alignment/t2_aligned/green.tif` |
| alignment/t2_aligned/red.tif | `alignment/t2_aligned/red.tif` |
| alignment/t2_aligned/rededge.tif | `alignment/t2_aligned/rededge.tif` |
| alignment/t2_aligned/nir.tif | `alignment/t2_aligned/nir.tif` |
| alignment/overlap_mask.tif | `alignment/overlap_mask.tif` |
| vegetation/t1_health.tif | `vegetation/t1_health.tif` |
| vegetation/t2_health.tif | `vegetation/t2_health.tif` |
| change/transition_map.tif | `change/transition_map.tif` |
| change/cva_magnitude.tif | `change/cva_magnitude.tif` |
| change/delta_ndvi.tif | `change/delta_ndvi.tif` |
| change/delta_gndvi.tif | `change/delta_gndvi.tif` |
| change/delta_ndre.tif | `change/delta_ndre.tif` |
| degradation/degradation_map.tif | `degradation/degradation_map.tif` |
| degradation/recovery_map.tif | `degradation/recovery_map.tif` |
| degradation/canopy_loss_map.tif | `degradation/canopy_loss_map.tif` |
| degradation/residual_stress.tif | `degradation/residual_stress.tif` |
| wetland/wetland_mask.tif | `wetland/wetland_mask.tif` |
| hydrology/ndwi_t1.tif | `hydrology/ndwi_t1.tif` |
| hydrology/ndwi_t2.tif | `hydrology/ndwi_t2.tif` |
| hydrology/delta_ndwi.tif | `hydrology/delta_ndwi.tif` |
| hydrology/water_mask_t1.tif | `hydrology/water_mask_t1.tif` |
| hydrology/water_mask_t2.tif | `hydrology/water_mask_t2.tif` |
| hydrology/water_change.tif | `hydrology/water_change.tif` |
| metrics/fusion_metrics.json | `metrics/fusion_metrics.json` |
| metrics/evidence_matrix.csv | `metrics/evidence_matrix.csv` |
| fusion/evidence_fusion.tif | `fusion/evidence_fusion.tif` |
| metrics/metrics.json | `metrics/metrics.json` |
| metrics/transition_matrix.csv | `metrics/transition_matrix.csv` |
| metrics/water_metrics.json | `metrics/water_metrics.json` |
| validation/qa_report.json | `validation/qa_report.json` |
| analysis_result.json | `analysis_result.json` |

---

*End of Report*