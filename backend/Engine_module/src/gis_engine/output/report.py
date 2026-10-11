"""
src/gis_engine/output/report.py
==================================
Production Markdown analysis report generator for the GIS Wetland
Degradation & Depletion Engine.

Generates a human-readable analysis_report.md that preserves conservative
scientific terminology. The report never claims physical wetland depletion
from vegetation degradation or NDWI water loss alone.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _fmt_area(val_m2: float, val_ha: Optional[float] = None) -> str:
    """Formats an area value as 'X.XX ha (Y,YYY m²)'."""
    if val_ha is None:
        val_ha = val_m2 / 10_000.0
    return f"{val_ha:,.2f} ha ({val_m2:,.0f} m²)"


def _fmt_pct(val: float) -> str:
    return f"{val:.2f}%"


def _safe(d: Dict[str, Any], *keys, default="N/A"):
    """Safely navigate nested dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
        if cur is None:
            return default
    return cur


def generate_analysis_report(
    result_dict: Dict[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """
    Generates the production analysis report as a Markdown file.

    Args:
        result_dict: Flat dictionary containing all analysis results,
                     metrics, validation, and metadata.
        output_path: Full path where the report Markdown file is written.
        run_id:      The unique run identifier.

    Returns:
        Path to the written report file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    meta = result_dict.get("metadata", {})
    inputs = result_dict.get("inputs", {})
    spatial = result_dict.get("spatial", {})
    temporal = result_dict.get("temporal", {})
    veg = result_dict.get("vegetation", {})
    change = result_dict.get("change", {})
    cva_stats = result_dict.get("cva", {})
    degradation = result_dict.get("degradation", {})
    recovery = result_dict.get("recovery", {})
    wetland = result_dict.get("wetland", {})
    hydrology = result_dict.get("hydrology", {})
    fusion = result_dict.get("fusion", {})
    metrics = result_dict.get("metrics", {})
    qa = result_dict.get("validation", {})
    uncertainty = result_dict.get("uncertainty", {})
    outputs_inv = result_dict.get("outputs", {})
    warnings_list = result_dict.get("warnings", [])
    errors_list = result_dict.get("errors", [])
    phys_depl = result_dict.get("physical_depletion_assessment", "NOT_ASSESSED")
    sw_version = meta.get("software_version", "1.0.0")
    status = result_dict.get("status", "UNKNOWN")

    lines = []

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------
    lines += [
        "# GIS Wetland Degradation & Depletion Engine — Analysis Report",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** {ts}  ",
        f"**Software Version:** {sw_version}  ",
        f"**Status:** {status}  ",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Executive Summary
    # -------------------------------------------------------------------------
    lines += ["## 1. Executive Summary", ""]

    qa_overall = _safe(qa, "overall_status", default="UNKNOWN")
    deg_area_ha = _safe(degradation, "area_ha", default=None)
    rec_area_ha = _safe(recovery, "area_ha", default=None)
    wetland_status = _safe(wetland, "status", default="WETLAND_BOUNDARY_UNAVAILABLE")

    lines += [
        f"This report presents the results of a two-date vegetation health and "
        f"surface-water change analysis for the wetland study area.",
        "",
        f"- **QA Status:** {qa_overall}",
        f"- **Wetland Boundary Status:** {wetland_status}",
    ]
    if deg_area_ha is not None:
        lines.append(f"- **Vegetation Degradation Area:** {_fmt_area(deg_area_ha * 10_000, deg_area_ha)}")
    if rec_area_ha is not None:
        lines.append(f"- **Vegetation Recovery Area:** {_fmt_area(rec_area_ha * 10_000, rec_area_ha)}")

    lines += [
        "",
        "> **Note:** This engine assesses vegetation health change and surface-water change "
        "from optical satellite imagery. Results represent vegetation degradation evidence "
        "and surface-water change evidence. Physical wetland depletion requires independent "
        "supporting evidence beyond the scope of this V1 analysis.",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Input Data
    # -------------------------------------------------------------------------
    lines += ["## 2. Input Data", ""]
    t1_path = _safe(inputs, "t1_path")
    t2_path = _safe(inputs, "t2_path")
    model_path = _safe(inputs, "model_path")
    lines += [
        f"| Field | Value |",
        f"|---|---|",
        f"| T1 Scene | `{t1_path}` |",
        f"| T2 Scene | `{t2_path}` |",
        f"| Model | `{model_path}` |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Temporal Information
    # -------------------------------------------------------------------------
    lines += ["## 3. Temporal Information", ""]
    t1_date = _safe(temporal, "t1_date")
    t2_date = _safe(temporal, "t2_date")
    interval_days = _safe(temporal, "interval_days", default="N/A")
    lines += [
        f"| Field | Value |",
        f"|---|---|",
        f"| T1 Acquisition Date | {t1_date} |",
        f"| T2 Acquisition Date | {t2_date} |",
        f"| Temporal Interval | {interval_days} days |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Spatial Information
    # -------------------------------------------------------------------------
    lines += ["## 4. Spatial Information", ""]
    crs_raw = _safe(spatial, "crs")
    if crs_raw is None or str(crs_raw).strip().lower() in ("none", "", "null", "unknown crs"):
        crs = "Undefined / not provided by source raster"
    else:
        crs = str(crs_raw)
    dims = _safe(spatial, "dimensions", default=[])
    res = _safe(spatial, "resolution", default=[1.0, 1.0])
    overlap_pct = _safe(spatial, "overlap_percentage", default="N/A")
    
    # Calculate pixel area
    try:
        pixel_res_x = float(res[0]) if isinstance(res, (list, tuple)) and len(res) > 0 else 1.0
        pixel_res_y = float(res[1]) if isinstance(res, (list, tuple)) and len(res) > 1 else 1.0
        pixel_area_ha = abs(pixel_res_x * pixel_res_y) / 10000.0
    except Exception:
        pixel_area_ha = 1.0 / 10000.0

    lines += [
        f"| Field | Value |",
        f"|---|---|",
        f"| CRS | `{crs}` |",
        f"| Dimensions (H × W) | {dims[0] if dims else 'N/A'} × {dims[1] if len(dims) > 1 else 'N/A'} px |",
        f"| Resolution | {res[0] if res else 'N/A'} × {res[1] if len(res) > 1 else 'N/A'} m |",
        f"| T1/T2 Overlap | {overlap_pct if isinstance(overlap_pct, str) else f'{overlap_pct:.2f}%'} |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Vegetation Health
    # -------------------------------------------------------------------------
    lines += ["## 5. Vegetation Health", ""]
    
    def _extract_classes(veg_side: Any) -> Dict[str, Any]:
        if not isinstance(veg_side, dict):
            return {}
        if "date_valid" in veg_side and isinstance(veg_side["date_valid"], dict) and "classes" in veg_side["date_valid"]:
            return veg_side["date_valid"]["classes"]
        if "common_overlap" in veg_side and isinstance(veg_side["common_overlap"], dict) and "classes" in veg_side["common_overlap"]:
            return veg_side["common_overlap"]["classes"]
        if "classes" in veg_side and isinstance(veg_side["classes"], dict):
            return veg_side["classes"]
        return veg_side

    for date_label, date_stats in [("T1 (Baseline)", _safe(veg, "t1", default={})),
                                    ("T2 (Target)", _safe(veg, "t2", default={}))]:
        lines.append(f"### {date_label}")
        classes = _extract_classes(date_stats)
        lines += [
            "| Class | Pixels | Area | Percentage |",
            "|---|---|---|---|",
        ]
        for cls_id in ["0", "1", "2", "3"]:
            cls_info = classes.get(cls_id, classes.get(int(cls_id), {}))
            name = cls_info.get("name", f"Class {cls_id}")
            count = cls_info.get("count", 0)
            if isinstance(count, (int, float)):
                area_ha = float(cls_info.get("area_ha", count * pixel_area_ha))
                area_str = _fmt_area(area_ha * 10000.0, area_ha)
            else:
                area_str = "N/A"
            pct = cls_info.get("percentage", 0.0)
            pct_str = f"{pct:.2f}%" if isinstance(pct, (int, float)) else str(pct)
            lines.append(f"| {name} | {count} | {area_str} | {pct_str} |")
        lines.append("")

    lines += ["---", ""]

    # -------------------------------------------------------------------------
    # Vegetation Change
    # -------------------------------------------------------------------------
    lines += ["## 6. Vegetation Change (Post-Classification Transitions)", ""]
    cat_summary = _safe(change, "category_summary", default={})
    deg_count = _safe(cat_summary, "DEGRADATION", "count", default="N/A")
    deg_ha = _safe(cat_summary, "DEGRADATION", "area_ha", default=None)
    rec_count = _safe(cat_summary, "RECOVERY", "count", default="N/A")
    rec_ha = _safe(cat_summary, "RECOVERY", "area_ha", default=None)
    stable_count = _safe(cat_summary, "STABLE", "count", default="N/A")
    stable_ha = _safe(cat_summary, "STABLE", "area_ha", default=None)

    lines += [
        "| Category | Pixel Count | Area |",
        "|---|---|---|",
        f"| Vegetation Degradation | {deg_count} | {_fmt_area(deg_ha * 10_000, deg_ha) if deg_ha is not None else 'N/A'} |",
        f"| Vegetation Recovery | {rec_count} | {_fmt_area(rec_ha * 10_000, rec_ha) if rec_ha is not None else 'N/A'} |",
        f"| Stable | {stable_count} | {_fmt_area(stable_ha * 10_000, stable_ha) if stable_ha is not None else '—'} |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # CVA
    # -------------------------------------------------------------------------
    lines += ["## 7. Change Vector Analysis (CVA)", ""]
    cva_mean = _safe(cva_stats, "cva_magnitude", "mean", default=_safe(cva_stats, "cva_magnitude_mean", default="N/A"))
    cva_median = _safe(cva_stats, "cva_magnitude", "median", default="N/A")
    cva_p95 = _safe(cva_stats, "cva_magnitude", "p95", default=_safe(cva_stats, "cva_magnitude_p95", default="N/A"))
    cva_valid = _safe(cva_stats, "cva_magnitude", "count", default=_safe(cva_stats, "valid_pixels", default="N/A"))
    
    d_ndvi_mean = _safe(cva_stats, "delta_ndvi", "mean", default="N/A")
    d_gndvi_mean = _safe(cva_stats, "delta_gndvi", "mean", default="N/A")
    d_ndre_mean = _safe(cva_stats, "delta_ndre", "mean", default="N/A")

    lines += [
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean CVA Magnitude | {cva_mean if isinstance(cva_mean, str) else f'{cva_mean:.4f}'} |",
        f"| Median CVA Magnitude | {cva_median if isinstance(cva_median, str) else f'{cva_median:.4f}'} |",
        f"| 95th Percentile CVA Magnitude | {cva_p95 if isinstance(cva_p95, str) else f'{cva_p95:.4f}'} |",
        f"| Mean ΔNDVI | {d_ndvi_mean if isinstance(d_ndvi_mean, str) else f'{d_ndvi_mean:.4f}'} |",
        f"| Mean ΔGNDVI | {d_gndvi_mean if isinstance(d_gndvi_mean, str) else f'{d_gndvi_mean:.4f}'} |",
        f"| Mean ΔNDRE | {d_ndre_mean if isinstance(d_ndre_mean, str) else f'{d_ndre_mean:.4f}'} |",
        f"| Valid Pixels | {cva_valid} |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Degradation & Recovery
    # -------------------------------------------------------------------------
    lines += ["## 8. Degradation", ""]
    deg_pixels = _safe(degradation, "pixel_count", default="N/A")
    deg_ha_val = _safe(degradation, "area_ha", default=None)
    deg_pct = _safe(degradation, "percentage", default="N/A")
    lines += [
        f"| Metric | Value |",
        f"|---|---|",
        f"| Degraded Pixels | {deg_pixels} |",
        f"| Degraded Area | {_fmt_area(deg_ha_val * 10_000, deg_ha_val) if deg_ha_val is not None else 'N/A'} |",
        f"| Degraded Percentage (valid area) | {_fmt_pct(deg_pct) if isinstance(deg_pct, (int, float)) else deg_pct} |",
        "",
        "## 9. Recovery",
        "",
    ]
    rec_pixels = _safe(recovery, "pixel_count", default="N/A")
    rec_ha_val = _safe(recovery, "area_ha", default=None)
    rec_pct = _safe(recovery, "percentage", default="N/A")
    lines += [
        f"| Metric | Value |",
        f"|---|---|",
        f"| Recovered Pixels | {rec_pixels} |",
        f"| Recovered Area | {_fmt_area(rec_ha_val * 10_000, rec_ha_val) if rec_ha_val is not None else 'N/A'} |",
        f"| Recovered Percentage (valid area) | {_fmt_pct(rec_pct) if isinstance(rec_pct, (int, float)) else rec_pct} |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Wetland Context
    # -------------------------------------------------------------------------
    lines += ["## 10. Wetland Context", ""]
    wetland_src = _safe(wetland, "source_type", default="UNAVAILABLE")
    wetland_area_ha = _safe(wetland, "area_ha", default=None)
    depl_enabled = _safe(wetland, "physical_depletion_enabled", default=False)
    lines += [
        f"| Field | Value |",
        f"|---|---|",
        f"| Boundary Source | {wetland_src} |",
        f"| Wetland Area | {_fmt_area(wetland_area_ha * 10_000, wetland_area_ha) if wetland_area_ha is not None else 'N/A'} |",
        f"| Physical Depletion Assessment Enabled | {'Yes' if depl_enabled else 'No'} |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Surface-Water Change
    # -------------------------------------------------------------------------
    lines += ["## 11. Surface-Water Change (NDWI Hydrology)", ""]
    ndwi_thresh = _safe(hydrology, "ndwi_threshold", default="N/A")
    water_t1_ha = _safe(hydrology, "water_area_t1_ha", default=None)
    water_t2_ha = _safe(hydrology, "water_area_t2_ha", default=None)
    water_gain_ha = _safe(hydrology, "water_gain_area_ha", default=None)
    water_loss_ha = _safe(hydrology, "water_loss_area_ha", default=None)
    lines += [
        f"| Metric | Value |",
        f"|---|---|",
        f"| NDWI Threshold | {ndwi_thresh} |",
        f"| Baseline Surface-Water Area (T1) | {_fmt_area(water_t1_ha * 10_000, water_t1_ha) if water_t1_ha is not None else 'N/A'} |",
        f"| Present Surface-Water Area (T2) | {_fmt_area(water_t2_ha * 10_000, water_t2_ha) if water_t2_ha is not None else 'N/A'} |",
        f"| Surface-Water Gain Area | {_fmt_area(water_gain_ha * 10_000, water_gain_ha) if water_gain_ha is not None else 'N/A'} |",
        f"| Surface-Water Loss Area | {_fmt_area(water_loss_ha * 10_000, water_loss_ha) if water_loss_ha is not None else 'N/A'} |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Hydro-Ecological Evidence (Fusion)
    # -------------------------------------------------------------------------
    lines += ["## 12. Hydro-Ecological Evidence (Multi-Evidence Fusion)", ""]
    hydro_deg_ha = _safe(fusion, "hydro_ecological_degradation_area_ha", default=None)
    hydro_rec_ha = _safe(fusion, "hydro_ecological_recovery_area_ha", default=None)
    fusion_status = _safe(fusion, "evidence_level", default="N/A")
    lines += [
        f"| Metric | Value |",
        f"|---|---|",
        f"| Evidence Level | {fusion_status} |",
        f"| Hydro-Ecological Degradation Evidence Area | {_fmt_area(hydro_deg_ha * 10_000, hydro_deg_ha) if hydro_deg_ha is not None else 'N/A'} |",
        f"| Hydro-Ecological Recovery Evidence Area | {_fmt_area(hydro_rec_ha * 10_000, hydro_rec_ha) if hydro_rec_ha is not None else 'N/A'} |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    lines += ["## 13. Metrics Summary", ""]
    ndr = _safe(metrics, "normalized_degradation_ratio", default="N/A")
    swd = _safe(metrics, "severity_weighted_degradation_area_ha", default=None)
    lines += [
        f"| Metric | Value |",
        f"|---|---|",
        f"| Normalized Degradation Ratio (NDR) | {ndr if isinstance(ndr, str) else f'{ndr:.4f}'} |",
        f"| Severity-Weighted Degradation Area | {_fmt_area(swd * 10_000, swd) if swd is not None else 'N/A'} |",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Quality Assurance
    # -------------------------------------------------------------------------
    lines += ["## 14. Quality Assurance", ""]
    qa_flags = qa.get("flags", [])
    lines.append(f"**Overall QA Status:** {qa_overall}")
    lines.append("")
    if qa_flags:
        lines += [
            "| Code | Severity | Phase | Message |",
            "|---|---|---|---|",
        ]
        for flag in qa_flags:
            code = flag.get("code", "—")
            sev = flag.get("severity", "—")
            phase = flag.get("phase", "—")
            msg = flag.get("message", "—")
            lines.append(f"| {code} | {sev} | {phase} | {msg} |")
    else:
        lines.append("*No QA flags raised.*")

    lines += ["", "---", ""]

    # -------------------------------------------------------------------------
    # Uncertainty
    # -------------------------------------------------------------------------
    lines += ["## 15. Uncertainty", ""]
    conf_mean = _safe(uncertainty, "mean_confidence_t1", default="N/A")
    lines += [
        f"- **Mean Model Confidence (T1):** {conf_mean if isinstance(conf_mean, str) else f'{conf_mean:.4f}'}",
        "- Confidence calibration: V2 feature (not available in V1).",
        "- No independent reference samples provided; Olofsson area corrections not available.",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # V1 Limitations
    # -------------------------------------------------------------------------
    lines += [
        "## 16. V1 Scope & Limitations",
        "",
        "The following capabilities are **outside the scope of V1** and are planned for future versions:",
        "",
        "- MNDWI, LSWI, SWIR, SAR, LST analysis",
        "- Soil chemistry or groundwater modelling",
        "- Field calibration or accuracy assessment with reference samples",
        "- Multi-year trajectory modelling (LandTrendr, BFAST, CCDC)",
        "- Automatic wetland boundary segmentation",
        "- Probability calibration framework (Platt / isotonic regression)",
        "",
        "---",
        "",
    ]

    # -------------------------------------------------------------------------
    # Physical Depletion Assessment
    # -------------------------------------------------------------------------
    lines += [
        "## 17. Physical Depletion Assessment",
        "",
    ]
    if phys_depl == "NOT_ASSESSED":
        lines += [
            "**Assessment Status: NOT_ASSESSED**",
            "",
            "Physical wetland depletion has **not** been assessed in this run. "
            "The engine detected vegetation degradation evidence and/or surface-water loss evidence "
            "from optical imagery, but these indicators alone are insufficient to conclude "
            "physical wetland depletion.",
            "",
            "Conservative interpretation: *Vegetation degradation associated with surface-water loss* "
            "represents hydro-ecological degradation evidence. Independent hydrological survey data "
            "would be required to support a physical depletion claim.",
        ]
    else:
        lines += [
            f"**Assessment Status: {phys_depl}**",
            "",
            "An independent hydrological source has been accepted. See `analysis_result.json` "
            "for the full assessment rationale.",
        ]

    lines += ["", "---", ""]

    # -------------------------------------------------------------------------
    # Warnings & Errors
    # -------------------------------------------------------------------------
    if warnings_list:
        lines += ["## 18. Warnings", ""]
        for w in warnings_list:
            lines.append(f"- {w}")
        lines += ["", "---", ""]

    if errors_list:
        lines += ["## 19. Errors", ""]
        for e in errors_list:
            lines.append(f"- {e}")
        lines += ["", "---", ""]

    # -------------------------------------------------------------------------
    # Output Inventory
    # -------------------------------------------------------------------------
    lines += ["## 20. Output Inventory", ""]
    if outputs_inv:
        lines += [
            "| Key | Relative Path |",
            "|---|---|",
        ]
        for key, rel_path in outputs_inv.items():
            lines.append(f"| {key} | `{rel_path}` |")
    else:
        lines.append("*No outputs recorded.*")

    lines += ["", "---", ""]
    lines.append("*End of Report*")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")

    logger.info("[report] Analysis report written: %s", output_path)
    return output_path
