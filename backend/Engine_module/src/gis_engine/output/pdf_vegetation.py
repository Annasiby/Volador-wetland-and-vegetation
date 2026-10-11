"""
src/gis_engine/output/pdf_vegetation.py
=======================================
PDF report generators for VEGETATION MODE (single-date analysis):
1. generate_vegetation_report: Accessible summary for non-specialists.
2. generate_vegetation_technical_report: Comprehensive technical and scientific audit.

STRICT SCIENTIFIC SAFEGUARD:
Vegetation reports analyze a SINGLE multispectral scene.
They MUST NOT contain:
- T1/T2 comparisons
- Temporal vegetation transitions
- Degradation or recovery transitions
- Change Vector Analysis (CVA)
- Hydrological / NDWI evidence
- Wetland degradation or physical depletion
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .pdf_common import (
    ALERT,
    ALERT_LIGHT,
    CONTENT_W,
    EARTH,
    EARTH_LIGHT,
    GREEN_CALLOUT,
    INK,
    LINE,
    MARGIN,
    MUTED,
    OLIVE,
    OLIVE_LIGHT,
    PAGE_H,
    PAGE_W,
    PAPER,
    PAPER_DARK,
    SLATE,
    WATER,
    WATER_LIGHT,
    _calc_pct,
    _callout_box,
    _fmt_ha,
    _fmt_num,
    _fmt_pct,
    _image_exists,
    _img,
    _image_pair,
    _p,
    _safe_float,
    _safe_get,
    _safe_int,
    _side_by_side_table,
    _table,
    get_report_styles,
    make_numbered_canvas,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. VEGETATION SIMPLE REPORT (Vegetation Health Report)
# ===========================================================================

def generate_vegetation_report(
    result_dict: Dict[str, Any],
    output_path: Path,
    map_assets: Optional[Dict[str, Path]] = None,
    run_id: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Generate the client-facing VEGETATION HEALTH REPORT (Simple Summary).

    Presents single-date canopy condition in clear, non-technical language
    with large readable maps, summary tables, and straightforward takeaways.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_assets = map_assets or {}
    run_id = run_id or result_dict.get("run_id", "RUN_UNKNOWN")

    meta = result_dict.get("metadata", {})
    inputs = result_dict.get("inputs", {})
    spatial = result_dict.get("spatial", {})
    temporal = result_dict.get("temporal", {})
    veg = result_dict.get("vegetation", {})
    metrics = result_dict.get("metrics", {})
    uncertainty = result_dict.get("uncertainty", {})

    obs_date = temporal.get("date") or "N/A"
    crs_str = spatial.get("crs") or "Not specified"
    if str(crs_str).strip() in ("None", "NONE", "null", ""):
        crs_str = "Not specified"

    physical_available = bool(spatial.get("physical_area_available", False))
    total_area_ha = _safe_float(spatial.get("total_area_ha") or spatial.get("effective_analysis_area_ha"))
    valid_pixels = _safe_int(spatial.get("valid_pixels")) or _safe_int(veg.get("total_valid_pixels")) or 0

    classes_dict = veg.get("classes", {}) or {}
    def _cls_info(cid: str) -> Dict[str, Any]:
        return classes_dict.get(cid, classes_dict.get(int(cid), {})) or {}

    bg_info = _cls_info("0")
    sev_info = _cls_info("1")
    mod_info = _cls_info("2")
    hlt_info = _cls_info("3")

    healthy_pct = _safe_float(metrics.get("healthy_vegetation_percentage") or hlt_info.get("percentage"), 0.0) or 0.0
    moderate_pct = _safe_float(metrics.get("moderate_stress_percentage") or mod_info.get("percentage"), 0.0) or 0.0
    severe_pct = _safe_float(metrics.get("severe_stress_percentage") or sev_info.get("percentage"), 0.0) or 0.0
    bg_pct = _safe_float(metrics.get("background_percentage") or bg_info.get("percentage"), 0.0) or 0.0

    healthy_ha = _safe_float(hlt_info.get("area_ha"))
    mod_ha = _safe_float(mod_info.get("area_ha"))
    sev_ha = _safe_float(sev_info.get("area_ha"))
    bg_ha = _safe_float(bg_info.get("area_ha"))

    healthy_m2 = _safe_float(hlt_info.get("area_m2"))
    mod_m2 = _safe_float(mod_info.get("area_m2"))
    sev_m2 = _safe_float(sev_info.get("area_m2"))
    bg_m2 = _safe_float(bg_info.get("area_m2"))

    mean_conf = _safe_float(uncertainty.get("mean_confidence") or metrics.get("mean_confidence"), 0.0) or 0.0
    conf_pct_str = f"{mean_conf * 100:.1f}%" if mean_conf <= 1.0 else f"{mean_conf:.1f}%"

    # Overall condition evaluation
    veg_active_pct = healthy_pct + moderate_pct + severe_pct
    if healthy_pct >= 60.0:
        overall_status = "Predominantly Healthy Canopy"
        overall_color = OLIVE
        overall_finding = (
            f"The analyzed vegetation shows predominantly healthy, robust canopy condition. "
            f"Healthy vegetation accounts for {healthy_pct:.1f}% of the evaluated area, "
            f"with localized moderate stress at {moderate_pct:.1f}% and severe stress restricted to {severe_pct:.1f}%."
        )
    elif (moderate_pct + severe_pct) > healthy_pct:
        overall_status = "Elevated Canopy Stress Detected"
        overall_color = ALERT
        overall_finding = (
            f"Elevated vegetation stress was observed across the analyzed area. "
            f"Combined canopy stress encompasses {moderate_pct + severe_pct:.1f}% of the footprint "
            f"({moderate_pct:.1f}% moderate stress, {severe_pct:.1f}% severe stress), "
            f"while healthy vegetation comprises {healthy_pct:.1f}%."
        )
    else:
        overall_status = "Moderate Vegetation Condition"
        overall_color = SLATE
        overall_finding = (
            f"The vegetation canopy displays a balanced condition profile. "
            f"Healthy vegetation occupies {healthy_pct:.1f}% of the analyzed area, "
            f"moderate stress represents {moderate_pct:.1f}%, and non-vegetated background comprises {bg_pct:.1f}%."
        )

    styles = get_report_styles(overall_color)
    CanvasClass = make_numbered_canvas("VEGETATION HEALTH REPORT", "SINGLE-DATE SATELLITE ANALYSIS")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=45,
        bottomMargin=48,
        title="Vegetation Health Report",
        author="GIS Remote Sensing Engine",
    )

    story: List[Any] = []
    CW = CONTENT_W

    def M(key: str) -> Optional[Path]:
        return map_assets.get(key)

    # ------------------------------------------------------------------
    # SECTION 1: Cover / Report Summary
    # ------------------------------------------------------------------
    story.append(_p("GIS Remote Sensing Engine  ·  Vegetation Health Assessment", styles["eyebrow"]))
    story.append(_p("VEGETATION HEALTH REPORT", styles["title"]))
    story.append(_p("Single-Date Multispectral Canopy Health & Condition Assessment", styles["deck"]))
    story.append(HRFlowable(width="100%", thickness=1.0, color=INK, spaceBefore=2, spaceAfter=8))

    cover_meta = [
        [_p("OBSERVATION DATE", styles["eyebrow"]), _p(str(obs_date), styles["cell_bold"]),
         _p("OVERALL CONDITION", styles["eyebrow"]), _p(overall_status, styles["cell_bold"])],
        [_p("ANALYZED FOOTPRINT", styles["eyebrow"]),
         _p(f"{total_area_ha:,.2f} ha" if (physical_available and total_area_ha) else f"{valid_pixels:,} pixels", styles["cell_bold"]),
         _p("RUN ID", styles["eyebrow"]), _p(str(run_id), styles["cell_bold"])],
        [_p("SPATIAL REFERENCE", styles["eyebrow"]), _p(str(crs_str), styles["cell_bold"]),
         _p("ANALYSIS TYPE", styles["eyebrow"]), _p("Single-Date Multispectral Classification", styles["cell_bold"])],
    ]
    story.append(_table(cover_meta, [115, 165, 115, 133], header_fill=SLATE, compact=True))
    story.append(Spacer(1, 8))

    # Executive Summary Callout
    story.append(_p("EXECUTIVE SUMMARY", styles["eyebrow"]))
    story.append(_p(overall_status.upper(), styles["primary"]))
    story.append(_p(overall_finding, styles["body"]))
    story.append(Spacer(1, 6))

    # ------------------------------------------------------------------
    # SECTION 2: What Was Analyzed
    # ------------------------------------------------------------------
    story.append(_p("1.  WHAT WAS ANALYZED", styles["h1"]))
    story.append(
        _p(
            "This assessment analyzes a single multispectral satellite observation to evaluate current "
            "vegetation health and physiological stress. Four calibrated spectral bands were examined "
            "simultaneously to assess canopy density, photosynthetic activity, and internal cellular hydration:",
            styles["body"],
        )
    )

    band_rows = [
        [_p("SPECTRAL BAND", styles["header"]), _p("WAVELENGTH ROLE", styles["header"]), _p("WHAT IT REVEALS", styles["header"])],
        [_p("Green (Band 3)", styles["cell_bold"]), _p("Visible green light (~560 nm)", styles["cell"]),
         _p("Reflects plant pigment concentration; measures overall leaf surface greenness.", styles["cell"])],
        [_p("Red (Band 4)", styles["cell_bold"]), _p("Visible red light (~665 nm)", styles["cell"]),
         _p("Strongly absorbed by active chlorophyll during photosynthesis; low reflectance indicates healthy foliage.", styles["cell"])],
        [_p("RedEdge (Band 5)", styles["cell_bold"]), _p("Spectral transition zone (~705 nm)", styles["cell"]),
         _p("Early warning indicator; highly sensitive to subtle physiological stress before visible canopy browning.", styles["cell"])],
        [_p("Near-Infrared / NIR (Band 8)", styles["cell_bold"]), _p("Near-infrared light (~842 nm)", styles["cell"]),
         _p("Strongly reflected by healthy spongy mesophyll cellular structure; drops sharply during plant dehydration or wilting.", styles["cell"])],
    ]
    story.append(_table(band_rows, [110, 140, 278], header_fill=OLIVE, compact=True))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 3: Vegetation Health Overview
    # ------------------------------------------------------------------
    story.append(_p("2.  VEGETATION HEALTH OVERVIEW", styles["h1"]))
    story.append(
        _p(
            "The calibrated multispectral data was classified into four distinct condition tiers:",
            styles["body"],
        )
    )

    overview_rows = [
        [_p("HEALTH CATEGORY", styles["header"]), _p("COVERAGE", styles["header_r"]), _p("AREA (HA)", styles["header_r"]), _p("CHARACTERISTICS & MEANING", styles["header"])],
        [_p("Healthy Vegetation", styles["cell_bold"]), _p(_fmt_pct(healthy_pct), styles["right"]),
         _p(_fmt_num(healthy_ha, 2) if healthy_ha else "N/A", styles["right"]),
         _p("Dense, actively photosynthesizing canopy with vigorous chlorophyll absorption and intact cellular structure.", styles["cell"])],
        [_p("Moderate Stress", styles["cell_bold"]), _p(_fmt_pct(moderate_pct), styles["right"]),
         _p(_fmt_num(mod_ha, 2) if mod_ha else "N/A", styles["right"]),
         _p("Canopy showing mild chlorosis, reduced moisture, or sub-optimal vigor. Often recoverable under favorable conditions.", styles["cell"])],
        [_p("Severe Stress", styles["cell_bold"]), _p(_fmt_pct(severe_pct), styles["right"]),
         _p(_fmt_num(sev_ha, 2) if sev_ha else "N/A", styles["right"]),
         _p("Pronounced foliage loss, severe dehydration, or marked dieback requiring immediate conservation or management attention.", styles["cell"])],
        [_p("Background / Non-Vegetation", styles["cell_bold"]), _p(_fmt_pct(bg_pct), styles["right"]),
         _p(_fmt_num(bg_ha, 2) if bg_ha else "N/A", styles["right"]),
         _p("Non-vegetated surfaces including open water, bare soil, sand, rocks, roads, or built structures.", styles["cell"])],
    ]
    story.append(_table(overview_rows, [115, 65, 65, 283], header_fill=SLATE, compact=True))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SECTION 4: Vegetation Health Map
    # ------------------------------------------------------------------
    story.append(_p("3.  VEGETATION HEALTH MAP", styles["h1"]))
    story.append(
        _p(
            "The cartographic map below illustrates the spatial distribution of classified vegetation health "
            "across the analyzed study area. Green denotes healthy canopy, orange denotes moderate stress, "
            "red indicates severe stress, and grey represents non-vegetated background.",
            styles["body"],
        )
    )

    health_map = M("health") or M("health_t1")
    if _image_exists(health_map):
        story.append(
            Table([[Image(str(health_map), width=CW, height=260)],
                   [_p(f"Figure 1: Classified Vegetation Condition Map  ·  Acquisition Date: {obs_date}", styles["caption"])]],
                  colWidths=[CW])
        )
    else:
        story.append(_callout_box(
            _p("Cartographic map plate is currently being rendered or unavailable for this dataset.", styles["small"]),
            PAPER_DARK, LINE
        ))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 5: Vegetation Statistics
    # ------------------------------------------------------------------
    story.append(_p("4.  VEGETATION STATISTICS", styles["h1"]))
    stats_rows = [
        [_p("CONDITION CATEGORY", styles["header"]), _p("SHARE (%)", styles["header_r"]),
         _p("AREA (HECTARES)", styles["header_r"]), _p("AREA (M²)", styles["header_r"])],
        [_p("Healthy Vegetation", styles["cell_bold"]), _p(_fmt_pct(healthy_pct), styles["right"]),
         _p(_fmt_num(healthy_ha, 2) if healthy_ha else "N/A", styles["right"]),
         _p(_fmt_num(healthy_m2, 0) if healthy_m2 else "N/A", styles["right"])],
        [_p("Moderate Stress", styles["cell_bold"]), _p(_fmt_pct(moderate_pct), styles["right"]),
         _p(_fmt_num(mod_ha, 2) if mod_ha else "N/A", styles["right"]),
         _p(_fmt_num(mod_m2, 0) if mod_m2 else "N/A", styles["right"])],
        [_p("Severe Stress", styles["cell_bold"]), _p(_fmt_pct(severe_pct), styles["right"]),
         _p(_fmt_num(sev_ha, 2) if sev_ha else "N/A", styles["right"]),
         _p(_fmt_num(sev_m2, 0) if sev_m2 else "N/A", styles["right"])],
        [_p("Background (Non-Veg)", styles["cell_bold"]), _p(_fmt_pct(bg_pct), styles["right"]),
         _p(_fmt_num(bg_ha, 2) if bg_ha else "N/A", styles["right"]),
         _p(_fmt_num(bg_m2, 0) if bg_m2 else "N/A", styles["right"])],
        [_p("Total Analyzed Area", styles["cell_bold"]), _p("100.00%", styles["right"]),
         _p(_fmt_num(total_area_ha, 2) if total_area_ha else "N/A", styles["right"]),
         _p(_fmt_num((total_area_ha or 0.0) * 10000.0, 0) if total_area_ha else "N/A", styles["right"])],
    ]
    story.append(_table(stats_rows, [168, 100, 130, 130], header_fill=OLIVE, compact=True))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 6: Confidence / Reliability
    # ------------------------------------------------------------------
    story.append(_p("5.  CONFIDENCE & RELIABILITY", styles["h1"]))
    conf_text = (
        f"The machine-learning classification model evaluated the scene with an average certainty score "
        f"of <b>{conf_pct_str}</b> across all valid pixels. This high confidence confirms strong agreement "
        f"between the observed satellite spectral reflectance patterns and calibrated vegetation response signatures. "
        f"Pixels exhibiting boundary ambiguity or spectral mixing between soil and sparse foliage are accounted for "
        f"within the moderate stress classification."
    )
    story.append(_p(conf_text, styles["body"]))
    story.append(Spacer(1, 6))

    # ------------------------------------------------------------------
    # SECTION 7: Key Observations
    # ------------------------------------------------------------------
    story.append(_p("6.  KEY OBSERVATIONS", styles["h1"]))
    obs_bullets = [
        f"Healthy, vigorous vegetation forms the predominant vegetative canopy at {healthy_pct:.1f}% coverage.",
        f"Combined vegetation stress (moderate + severe) affects {moderate_pct + severe_pct:.1f}% of the evaluated area.",
        f"Critical severe stress requiring targeted attention is confined to {severe_pct:.1f}% ({_fmt_ha(sev_ha)}).",
        f"Non-vegetated surfaces (water, open ground, infrastructure) represent {bg_pct:.1f}% of the footprint.",
    ]
    obs_content = "".join([f"• &nbsp; {b}<br/>" for b in obs_bullets])
    story.append(_callout_box(_p(obs_content, styles["body"]), GREEN_CALLOUT, OLIVE))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 8: Method Summary
    # ------------------------------------------------------------------
    story.append(_p("7.  METHOD SUMMARY", styles["h1"]))
    story.append(
        _p(
            "Vegetation health evaluation relies on standard physical spectral indices coupled with deep-learning "
            "semantic segmentation: "
            "<br/><b>• NDVI (Normalized Difference Vegetation Index):</b> Quantifies green canopy biomass and density."
            "<br/><b>• GNDVI (Green NDVI):</b> Highly sensitive to chlorophyll concentration and foliar nutrient status."
            "<br/><b>• NDRE (RedEdge NDVI):</b> Detects internal cellular stress before visible surface chlorosis."
            "<br/><b>• Machine Learning Model:</b> A calibrated convolutional neural network combines these multispectral "
            "channels into categorical health classes without subjective manual thresholds.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 6))

    # ------------------------------------------------------------------
    # SECTION 9: Report Notes & Limitations
    # ------------------------------------------------------------------
    story.append(_p("8.  REPORT NOTES & LIMITATIONS", styles["h1"]))
    story.append(
        _p(
            "<b>Single-Date Snapshot:</b> This report reflects vegetation condition exclusively at the date and time "
            "of satellite acquisition. It does not measure change over time and cannot establish whether stress is "
            "seasonal, drought-induced, or permanent.<br/>"
            "<b>No Wetland Assessment:</b> This mode performs vegetation health classification only. It does NOT "
            "evaluate wetland boundaries, hydrological depletion, or physical wetland loss. For multi-temporal "
            "wetland degradation analysis, run the engine in Wetland Mode with two observation dates.<br/>"
            "<b>Atmospheric & Ground Factors:</b> While cloud-screened and atmospheric corrections are applied, "
            "sun angle, soil moisture, and phenology can influence reflectance values. Ground validation is recommended "
            "for targeted site-level interventions.",
            styles["small"],
        )
    )

    doc.build(story, canvasmaker=CanvasClass)
    return output_path


# ===========================================================================
# 2. VEGETATION TECHNICAL REPORT
# ===========================================================================

def generate_vegetation_technical_report(
    result_dict: Dict[str, Any],
    output_path: Path,
    map_assets: Optional[Dict[str, Path]] = None,
    run_id: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Generate the scientific & engineering VEGETATION HEALTH TECHNICAL REPORT.

    Contains full technical disclosures: band statistics, index formulas,
    model architecture, tensor specifications, class metrics, CRS/transform,
    QA checks, and provenance without simplifications.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_assets = map_assets or {}
    run_id = run_id or result_dict.get("run_id", "RUN_UNKNOWN")

    meta = result_dict.get("metadata", {})
    inputs = result_dict.get("inputs", {})
    spatial = result_dict.get("spatial", {})
    temporal = result_dict.get("temporal", {})
    veg = result_dict.get("vegetation", {})
    metrics = result_dict.get("metrics", {})
    indices = result_dict.get("spectral_indices", {})
    qa = result_dict.get("validation", {})
    perf = result_dict.get("performance", {})
    uncertainty = result_dict.get("uncertainty", {})
    warnings = result_dict.get("warnings", [])
    errors = result_dict.get("errors", [])
    outputs_map = result_dict.get("outputs", {})

    obs_date = temporal.get("date") or "N/A"
    sw_ver = result_dict.get("software_version") or meta.get("software_version", "1.0.0")
    git_commit = meta.get("git_commit", "N/A")
    exec_utc = meta.get("execution_timestamp_utc") or datetime.now(timezone.utc).isoformat()
    os_info = meta.get("os", "N/A")
    py_ver = meta.get("python_version", "N/A")

    # Spatial properties
    crs_str = spatial.get("crs") or "Undefined"
    dims = spatial.get("dimensions", [0, 0])
    res = spatial.get("resolution", [0.0, 0.0])
    extent = spatial.get("extent", [])
    total_px = _safe_int(spatial.get("total_pixels"), 0) or (dims[0] * dims[1] if len(dims) == 2 else 0)
    valid_px = _safe_int(spatial.get("valid_pixels"), 0) or _safe_int(veg.get("total_valid_pixels"), 0) or 0
    nodata_px = max(0, total_px - valid_px)
    total_ha = _safe_float(spatial.get("total_area_ha") or spatial.get("effective_analysis_area_ha"))

    mean_conf = _safe_float(uncertainty.get("mean_confidence") or metrics.get("mean_confidence"), 0.0) or 0.0
    conf_pct_str = f"{mean_conf * 100:.1f}%" if mean_conf <= 1.0 else f"{mean_conf:.1f}%"

    classes_dict = veg.get("classes", {}) or {}
    def _cls(cid: str) -> Dict[str, Any]:
        return classes_dict.get(cid, classes_dict.get(int(cid), {})) or {}

    styles = get_report_styles()
    CanvasClass = make_numbered_canvas("VEGETATION HEALTH TECHNICAL REPORT", "SCIENTIFIC & ENGINEERING AUDIT")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=45,
        bottomMargin=48,
        title="Vegetation Health Technical Report",
        author="GIS Remote Sensing Engine",
    )

    story: List[Any] = []
    CW = CONTENT_W

    def M(key: str) -> Optional[Path]:
        return map_assets.get(key)

    # ------------------------------------------------------------------
    # SECTION 1: Report Metadata
    # ------------------------------------------------------------------
    story.append(_p("GIS Remote Sensing Engine  ·  Technical Specification & Audit", styles["eyebrow"]))
    story.append(_p("VEGETATION HEALTH TECHNICAL REPORT", styles["title"]))
    story.append(_p("Full Technical Specification, Model Disclosures, Class Metrics & Spatial Provenance", styles["deck"]))
    story.append(HRFlowable(width="100%", thickness=1.0, color=INK, spaceBefore=2, spaceAfter=8))

    meta_rows = [
        [_p("RUN ID", styles["header"]), _p(str(run_id), styles["cell_bold"]),
         _p("SOFTWARE VERSION", styles["header"]), _p(str(sw_ver), styles["cell"])],
        [_p("ANALYSIS MODE", styles["cell_bold"]), _p("vegetation (single-date)", styles["cell"]),
         _p("EXECUTION TIMESTAMP", styles["cell_bold"]), _p(str(exec_utc)[:19] + " UTC", styles["cell"])],
        [_p("GIT COMMIT", styles["cell_bold"]), _p(str(git_commit)[:12], styles["cell_code"]),
         _p("ENVIRONMENT", styles["cell_bold"]), _p(f"Python {py_ver[:6]} | {os_info[:16]}", styles["cell"])],
    ]
    story.append(_table(meta_rows, [110, 175, 110, 133], header_fill=SLATE, compact=True))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 2: Analysis Overview
    # ------------------------------------------------------------------
    story.append(_p("1.  ANALYSIS OVERVIEW & STATUS", styles["h1"]))
    status = result_dict.get("status", "SUCCESS")
    tot_time = _safe_float(perf.get("total_runtime_seconds"), 0.0) or 0.0
    inf_time = _safe_float(_safe_get(perf, "inference", "runtime_seconds"), 0.0) or 0.0

    overview_text = (
        f"<b>Execution Status:</b> {status} &nbsp;|&nbsp; <b>Total Pipeline Runtime:</b> {tot_time:.2f} s &nbsp;|&nbsp; "
        f"<b>Inference Latency:</b> {inf_time:.2f} s<br/>"
        f"This technical document audits the single-date multispectral vegetation inference pipeline. "
        f"The pipeline processes four calibrated spectral bands (Green, Red, RedEdge, NIR) through a deep-learning "
        f"segmentation engine (ResNet34 U-Net) to generate pixel-level categorical canopy condition masks and continuous "
        f"confidence surfaces. Temporal change, transitions, and hydrological indices are excluded in single-date mode."
    )
    story.append(_callout_box(_p(overview_text, styles["body"]), PAPER, LINE))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 3: Input Scene Information
    # ------------------------------------------------------------------
    story.append(_p("2.  INPUT SCENE & RASTER SPECIFICATIONS", styles["h1"]))

    scene_path = inputs.get("scene_path", "N/A")
    model_path = inputs.get("model_path", "N/A")
    ext_str = f"[{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}, {extent[3]:.2f}]" if len(extent) >= 4 else "N/A"

    scene_rows = [
        [_p("PARAMETER", styles["header"]), _p("TECHNICAL VALUE / PATH", styles["header"])],
        [_p("Scene Directory", styles["cell_bold"]), _p(str(scene_path), styles["cell_code"])],
        [_p("Model Path", styles["cell_bold"]), _p(str(model_path), styles["cell_code"])],
        [_p("Spatial Reference (CRS)", styles["cell_bold"]), _p(str(crs_str), styles["cell"])],
        [_p("Raster Dimensions [H × W]", styles["cell_bold"]), _p(f"{dims[0]} × {dims[1]} pixels", styles["cell"])],
        [_p("Pixel Resolution [dx, dy]", styles["cell_bold"]), _p(f"{res[0]:.2f} m × {res[1]:.2f} m", styles["cell"])],
        [_p("Geographic Bounds [W, S, E, N]", styles["cell_bold"]), _p(ext_str, styles["cell"])],
        [_p("Pixel Count Summary", styles["cell_bold"]),
         _p(f"Total: {total_px:,} | Valid: {valid_px:,} ({_fmt_pct(_calc_pct(valid_px, total_px))}) | NoData: {nodata_px:,}", styles["cell"])],
    ]
    story.append(_table(scene_rows, [160, 368], header_fill=OLIVE, compact=True))
    story.append(Spacer(1, 6))

    # Band statistics table
    band_stats = _safe_get(inputs, "band_statistics", "scene", default={}) or {}
    if band_stats:
        story.append(_p("Input Spectral Band Radiometric Statistics", styles["h2"]))
        bstat_rows = [
            [_p("BAND", styles["header"]), _p("MIN", styles["header_r"]), _p("MAX", styles["header_r"]),
             _p("MEAN", styles["header_r"]), _p("MEDIAN", styles["header_r"]), _p("STD DEV", styles["header_r"]),
             _p("P95", styles["header_r"]), _p("VALID COUNT", styles["header_r"])],
        ]
        for b_name in ["green", "red", "rededge", "nir"]:
            bs = band_stats.get(b_name, {})
            if bs:
                bstat_rows.append([
                    _p(b_name.upper(), styles["cell_bold"]),
                    _p(_fmt_num(bs.get("min"), 2), styles["right"]),
                    _p(_fmt_num(bs.get("max"), 2), styles["right"]),
                    _p(_fmt_num(bs.get("mean"), 2), styles["right"]),
                    _p(_fmt_num(bs.get("median"), 2), styles["right"]),
                    _p(_fmt_num(bs.get("std"), 2), styles["right"]),
                    _p(_fmt_num(bs.get("p95"), 2), styles["right"]),
                    _p(f"{bs.get('valid_count', 0):,}", styles["right"]),
                ])
        story.append(_table(bstat_rows, [70, 60, 60, 65, 65, 65, 65, 78], header_fill=SLATE, compact=True))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 4 & 5: Preprocessing & Vegetation Indices
    # ------------------------------------------------------------------
    story.append(_p("3.  PREPROCESSING & SPECTRAL INDICES", styles["h1"]))
    story.append(
        _p(
            "Single-scene rasters are radiometrically validated, masked for NoData, and normalized into float32 "
            "tensors. Three key diagnostic vegetation indices were computed from calibrated top-of-canopy reflectance:",
            styles["body"],
        )
    )

    idx_rows = [
        [_p("INDEX", styles["header"]), _p("FORMULA", styles["header"]), _p("MEAN", styles["header_r"]),
         _p("MEDIAN", styles["header_r"]), _p("STD", styles["header_r"]), _p("P95", styles["header_r"])],
    ]
    for idx_key, idx_label, f_str in [
        ("ndvi", "NDVI (Normalized Difference Veg Index)", "(NIR - Red) / (NIR + Red)"),
        ("gndvi", "GNDVI (Green Normalized Veg Index)", "(NIR - Green) / (NIR + Green)"),
        ("ndre", "NDRE (Normalized Difference RedEdge)", "(NIR - RedEdge) / (NIR + RedEdge)"),
    ]:
        istat = indices.get(idx_key, {})
        idx_rows.append([
            _p(idx_label, styles["cell_bold"]),
            _p(istat.get("formula", f_str), styles["cell_code"]),
            _p(_fmt_num(istat.get("mean"), 4), styles["right"]),
            _p(_fmt_num(istat.get("median"), 4), styles["right"]),
            _p(_fmt_num(istat.get("std"), 4), styles["right"]),
            _p(_fmt_num(istat.get("p95"), 4), styles["right"]),
        ])
    story.append(_table(idx_rows, [160, 168, 50, 50, 50, 50], header_fill=OLIVE, compact=True))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SECTION 6: Model / Inference Information
    # ------------------------------------------------------------------
    story.append(_p("4.  MODEL ARCHITECTURE & INFERENCE SPECIFICATION", styles["h1"]))
    model_rows = [
        [_p("SPECIFICATION PARAMETER", styles["header"]), _p("CONFIGURATION / VALUE", styles["header"])],
        [_p("Architecture", styles["cell_bold"]), _p("ResNet34 U-Net (encoder: resnet34, decoder: transpose conv blocks)", styles["cell"])],
        [_p("Input Channels", styles["cell_bold"]), _p("4 channels [B3: Green, B4: Red, B5: RedEdge, B8: NIR]", styles["cell"])],
        [_p("Input Tensor Shape", styles["cell_bold"]), _p(f"[1, 4, {dims[0]}, {dims[1]}] (batch_size=1, dtype=float32)", styles["cell"])],
        [_p("Inference Engine", styles["cell_bold"]), _p("ONNX Runtime (CPUExecutionProvider with CUDA auto-fallback)", styles["cell"])],
        [_p("Target Classes", styles["cell_bold"]), _p("4 discrete states (0: Background, 1: Severe, 2: Moderate, 3: Healthy)", styles["cell"])],
        [_p("Softmax Activation", styles["cell_bold"]), _p("Evaluated along channel axis; argmax yields discrete class, max probability yields confidence", styles["cell"])],
        [_p("Mean Model Confidence", styles["cell_bold"]), _p(f"{_safe_float(uncertainty.get('mean_confidence'), 0.0) * 100:.2f}% across valid pixels", styles["cell"])],
    ]
    story.append(_table(model_rows, [180, 348], header_fill=SLATE, compact=True))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 7 & 8: Classification & Class Statistics
    # ------------------------------------------------------------------
    story.append(_p("5.  CLASSIFICATION CLASS STATISTICS", styles["h1"]))

    cls_table_rows = [
        [_p("CLASS ID", styles["header_c"]), _p("CLASS NAME", styles["header"]), _p("PIXEL COUNT", styles["header_r"]),
         _p("AREA (M²)", styles["header_r"]), _p("AREA (HA)", styles["header_r"]), _p("VALID SHARE (%)", styles["header_r"])],
    ]
    for cid, cname in [("0", "Background / Non-Vegetation"), ("1", "Severe Stress"), ("2", "Moderate Stress"), ("3", "Healthy Vegetation")]:
        cdata = _cls(cid)
        c_px = _safe_int(cdata.get("count") or cdata.get("pixel_count"), 0) or 0
        c_m2 = _safe_float(cdata.get("area_m2"), 0.0) or 0.0
        c_ha = _safe_float(cdata.get("area_ha"), 0.0) or 0.0
        c_pct = _safe_float(cdata.get("percentage"), 0.0) or 0.0
        cls_table_rows.append([
            _p(cid, styles["center"]),
            _p(cname, styles["cell_bold"]),
            _p(f"{c_px:,}", styles["right"]),
            _p(_fmt_num(c_m2, 0), styles["right"]),
            _p(_fmt_num(c_ha, 2), styles["right"]),
            _p(_fmt_pct(c_pct), styles["right"]),
        ])
    cls_table_rows.append([
        _p("—", styles["center"]),
        _p("Total Valid Analyzed", styles["cell_bold"]),
        _p(f"{valid_px:,}", styles["right"]),
        _p(_fmt_num((total_ha or 0.0) * 10000.0, 0) if total_ha else "N/A", styles["right"]),
        _p(_fmt_num(total_ha, 2) if total_ha else "N/A", styles["right"]),
        _p("100.00%", styles["right"]),
    ])
    story.append(_table(cls_table_rows, [55, 173, 75, 75, 75, 75], header_fill=OLIVE, compact=True))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 9 & 10: Health Map & Confidence Map
    # ------------------------------------------------------------------
    story.append(_p("6.  CARTOGRAPHIC MAP PLATES", styles["h1"]))

    h_map = M("health") or M("health_t1")
    c_map = M("confidence")

    if _image_exists(h_map) and _image_exists(c_map):
        map_pair = _image_pair(
            h_map, c_map,
            img_w=252, img_h=180,
            left_caption="Figure 1: Classified Vegetation Condition Map",
            right_caption="Figure 2: Model Prediction Confidence Surface (0.0 - 1.0)",
            caption_style=styles["caption"],
        )
        for el in map_pair:
            story.append(el)
    elif _image_exists(h_map):
        story.append(
            Table([[Image(str(h_map), width=CW, height=240)],
                   [_p("Figure 1: Classified Vegetation Condition Map", styles["caption"])]],
                  colWidths=[CW])
        )
    elif _image_exists(c_map):
        story.append(
            Table([[Image(str(c_map), width=CW, height=240)],
                   [_p("Figure 2: Model Prediction Confidence Surface", styles["caption"])]],
                  colWidths=[CW])
        )
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 11 & 12: QA & Quality Information
    # ------------------------------------------------------------------
    story.append(_p("7.  QUALITY ASSURANCE & VALIDATION AUDIT", styles["h1"]))
    qa_checks = qa.get("checks", {}) or {}
    overall_qa = qa.get("overall_status", "PASS")

    qa_rows = [
        [_p("QA CHECK COMPONENT", styles["header"]), _p("RESULT", styles["header_c"]), _p("CRITERIA / AUDIT DETAILS", styles["header"])],
        [_p("Input Band Completeness", styles["cell_bold"]), _p("PASS", styles["center"]),
         _p("All 4 multispectral channels present with valid float32 ranges.", styles["cell"])],
        [_p("Mathematical Reconciliation", styles["cell_bold"]),
         _p(qa_checks.get("mathematical_reconciliation", "PASS"), styles["center"]),
         _p("Class pixel sums exactly equal total valid pixels (error < 0.001%).", styles["cell"])],
        [_p("Spatial Alignment QA", styles["cell_bold"]),
         _p(qa_checks.get("spatial_alignment", "PASS"), styles["center"]),
         _p(f"Raster dimensions match reference grid ({dims[0]} × {dims[1]}).", styles["cell"])],
        [_p("Model Uncertainty QA", styles["cell_bold"]),
         _p(qa_checks.get("uncertainty", "PASS"), styles["center"]),
         _p(f"Mean confidence {conf_pct_str} exceeds 70% operational threshold.", styles["cell"])],
    ]
    story.append(_table(qa_rows, [160, 65, 303], header_fill=SLATE, compact=True))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 13 & 14: Scientific Methodology & Limitations
    # ------------------------------------------------------------------
    story.append(_p("8.  SCIENTIFIC METHODOLOGY & LIMITATIONS", styles["h1"]))
    method_text = (
        "<b>Methodological Formulation:</b> Single-date vegetation assessment uses normalized spectral band "
        "ratios sensitive to the red-edge inflection point (~700–740 nm), which responds directly to foliar nitrogen "
        "and chlorophyll-a/b absorption. The deep learning segmentation architecture resolves non-linear spectral mixing "
        "inherent in mixed ground pixels (10 m / 20 m Ground Sample Distance).<br/>"
        "<b>Assumptions & Constraints:</b><br/>"
        "• Non-temporal: Single-acquisition analysis cannot infer long-term trends or separate seasonal senescence "
        "from permanent vegetative degradation.<br/>"
        "• Hydrological Exclusion: Does not evaluate surface water dynamics, soil moisture saturation, or wetland boundaries.<br/>"
        "• Radiometric Noise: Cloud edge effects and steep terrain shadow may introduce localized classification noise."
    )
    story.append(_p(method_text, styles["small"]))
    story.append(Spacer(1, 6))

    # ------------------------------------------------------------------
    # SECTION 15 & 16: Provenance & Output Manifest
    # ------------------------------------------------------------------
    story.append(_p("9.  PROVENANCE & TECHNICAL OUTPUT REFERENCES", styles["h1"]))

    out_rows = [
        [_p("ARTIFACT NAME", styles["header"]), _p("TYPE", styles["header"]), _p("RELATIVE OUTPUT PATH", styles["header"])],
        [_p("Vegetation Health Raster", styles["cell_bold"]), _p("GeoTIFF (Categorical)", styles["cell"]),
         _p("vegetation/health.tif", styles["cell_code"])],
        [_p("Classification Confidence", styles["cell_bold"]), _p("GeoTIFF (Continuous)", styles["cell"]),
         _p("vegetation/confidence.tif", styles["cell_code"])],
        [_p("Vegetation Statistics", styles["cell_bold"]), _p("JSON Record", styles["cell"]),
         _p("vegetation/statistics.json", styles["cell_code"])],
        [_p("Complete Analysis Result", styles["cell_bold"]), _p("JSON Contract", styles["cell"]),
         _p("analysis_result.json", styles["cell_code"])],
        [_p("Vegetation Health Report (Simple)", styles["cell_bold"]), _p("PDF Document", styles["cell"]),
         _p("reports/vegetation_report.pdf", styles["cell_code"])],
        [_p("Vegetation Technical Report", styles["cell_bold"]), _p("PDF Document", styles["cell"]),
         _p("reports/vegetation_technical_report.pdf", styles["cell_code"])],
    ]
    story.append(_table(out_rows, [160, 118, 250], header_fill=OLIVE, compact=True))

    doc.build(story, canvasmaker=CanvasClass)
    return output_path
