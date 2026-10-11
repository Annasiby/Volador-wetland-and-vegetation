"""
src/gis_engine/output/pdf_common.py
===================================
Shared styling, ReportLab flowable helpers, palette, and canvas numbering
used across simple and technical PDF reporting layers.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
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

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = letter
MARGIN = 42
CONTENT_W = PAGE_W - 2 * MARGIN  # ≈ 528 pt

# Refined editorial palette
INK = colors.HexColor("#20241F")
SLATE = colors.HexColor("#586057")
MUTED = colors.HexColor("#7A8178")
PAPER = colors.HexColor("#F7F6F1")
PAPER_DARK = colors.HexColor("#ECEAE1")
LINE = colors.HexColor("#D6D4CA")
OLIVE = colors.HexColor("#5E6B4F")
OLIVE_LIGHT = colors.HexColor("#E5E8DE")
WATER = colors.HexColor("#607B78")
WATER_LIGHT = colors.HexColor("#E1E9E7")
EARTH = colors.HexColor("#8A735A")
EARTH_LIGHT = colors.HexColor("#EEE8DF")
ALERT = colors.HexColor("#9A6A3A")
ALERT_LIGHT = colors.HexColor("#FBEFE6")
GREEN_CALLOUT = colors.HexColor("#EBF3EB")
DIVIDER_COLOR = colors.HexColor("#C8D0C0")


# ---------------------------------------------------------------------------
# Canvas with running header/footer
# ---------------------------------------------------------------------------

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for Page X of Y numbering and consistent headers."""
    _default_header_title = "GIS REMOTE SENSING ANALYSIS"
    _default_header_sub = "REMOTE-SENSING ANALYSIS"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header_footer(total)
            super().showPage()
        super().save()

    def _draw_header_footer(self, total: int):
        self.saveState()
        self.setStrokeColor(LINE)
        self.setLineWidth(0.5)

        h_title = getattr(self, "_header_title", self._default_header_title)
        h_sub = getattr(self, "_header_sub", self._default_header_sub)

        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 7)
            self.setFillColor(SLATE)
            self.drawString(MARGIN, PAGE_H - 27, h_title)
            self.setFont("Helvetica", 7)
            self.setFillColor(MUTED)
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - 27, h_sub)
            self.line(MARGIN, PAGE_H - 33, PAGE_W - MARGIN, PAGE_H - 33)

        self.line(MARGIN, 35, PAGE_W - MARGIN, 35)
        self.setFont("Helvetica", 6.8)
        self.setFillColor(MUTED)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.drawString(MARGIN, 21, f"GIS Remote Sensing Engine  ·  {stamp}")
        self.drawRightString(
            PAGE_W - MARGIN, 21, f"Page {self._pageNumber} of {total}"
        )
        self.restoreState()


def make_numbered_canvas(header_title: str, subheader: str = "REMOTE-SENSING ANALYSIS"):
    class CustomNumberedCanvas(NumberedCanvas):
        _default_header_title = header_title
        _default_header_sub = subheader
    return CustomNumberedCanvas


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _calc_pct(count: Any, denom: Any) -> Optional[float]:
    """Calculate percentage. Returns None if either value is unavailable."""
    c = _safe_float(count)
    d = _safe_float(denom)
    if c is None or d is None or d <= 0:
        return None
    return (c / d) * 100.0


def _fmt_pct(value: Any, default: str = "Not available") -> str:
    """Format floating percentage with 2 decimal places."""
    val = _safe_float(value)
    if val is None:
        return default
    return f"{val:.2f}%"


def _fmt_num(value: Any, decimals: int = 2, default: str = "N/A") -> str:
    val = _safe_float(value)
    return default if val is None else f"{val:,.{decimals}f}"


def _fmt_ha(value: Any, default: str = "Not available") -> str:
    val = _safe_float(value)
    return default if val is None else f"{val:,.2f} ha"


def _fmt_spatial_proportion(
    pct: Optional[float],
    area_ha: Optional[float] = None,
    physical_available: bool = False,
    default: str = "Not available",
) -> str:
    """Format client-facing spatial extent using ONLY percentages and hectares."""
    if pct is None and area_ha is None:
        return default
    ha_str = f"{area_ha:,.2f} ha" if (area_ha is not None and area_ha > 0) else None
    if pct is not None and ha_str is not None:
        return f"{pct:.2f}% ({ha_str})"
    if pct is not None:
        return f"{pct:.2f}%"
    if ha_str is not None:
        return ha_str
    return default


def _safe_get(data: Any, *keys: str, default: Any = None) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def _image_exists(path: Any) -> bool:
    try:
        return bool(path) and Path(path).is_file()
    except Exception:
        return False


def _img(path: Any, width: float, height: float) -> Optional[Image]:
    """Return an Image if it exists, else None."""
    if _image_exists(path):
        return Image(str(path), width=width, height=height)
    return None


def _image_pair(
    left: Any,
    right: Any,
    img_w: float = 248,
    img_h: float = 168,
    gap: float = 8,
    left_caption: str = "",
    right_caption: str = "",
    caption_style: Optional[ParagraphStyle] = None,
) -> List[Any]:
    """
    Return a list of flowables: a 2-column image table, optionally with captions.
    Works even if only one image exists.
    """
    left_exists = _image_exists(left)
    right_exists = _image_exists(right)
    if not left_exists and not right_exists:
        return []

    col_w = img_w + gap
    left_cell: Any = Image(str(left), width=img_w, height=img_h) if left_exists else Spacer(img_w, img_h)
    right_cell: Any = Image(str(right), width=img_w, height=img_h) if right_exists else Spacer(img_w, img_h)

    img_table = Table([[left_cell, right_cell]], colWidths=[col_w, col_w])
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    result: List[Any] = [img_table]
    if (left_caption or right_caption) and caption_style:
        cap_left = _p(left_caption, caption_style) if left_caption else Spacer(1, 1)
        cap_right = _p(right_caption, caption_style) if right_caption else Spacer(1, 1)
        cap_table = Table([[cap_left, cap_right]], colWidths=[col_w, col_w])
        cap_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        result.append(cap_table)
    return result


def _table(
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    *,
    header_fill=OLIVE,
    compact: bool = False,
    row_height: Optional[float] = None,
) -> Table:
    table = Table(rows, colWidths=list(widths), repeatRows=1,
                  rowHeights=row_height)
    pad = 3.5 if compact else 5
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.45, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _callout_box(
    content: Any,
    bg_color: Any,
    border_color: Any,
    width: float = CONTENT_W,
    h_pad: int = 12,
    v_pad: int = 9,
) -> Table:
    """A styled single-cell callout box."""
    t = Table([[content]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("BOX", (0, 0), (-1, -1), 0.8, border_color),
        ("LEFTPADDING", (0, 0), (-1, -1), h_pad),
        ("RIGHTPADDING", (0, 0), (-1, -1), h_pad),
        ("TOPPADDING", (0, 0), (-1, -1), v_pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), v_pad),
    ]))
    return t


def _side_by_side_table(
    left_content: Any,
    right_content: Any,
    left_w: float,
    right_w: float,
) -> Table:
    """Generic 2-column layout table."""
    t = Table([[left_content, right_content]], colWidths=[left_w, right_w])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _load_wetland_metadata(run_dir: Optional[Path]) -> Dict[str, Any]:
    if not run_dir:
        return {}
    path = Path(run_dir) / "wetland" / "wetland_metadata.json"
    if not path.is_file():
        return {}
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Wetland metadata unavailable: %s", exc)
        return {}


def _wetland_source_label(source_type: Any) -> str:
    labels = {
        "PROJECT_BOUNDARY": "Project-provided wetland boundary",
        "OFFICIAL_DATASET": "Official wetland dataset",
        "JRC_WATER_CONTEXT": "JRC surface-water context",
        "OPTICAL_WATER_INFERENCE": "Optical water inference",
        "SAR_INFERENCE": "SAR-derived inference",
        "UNAVAILABLE": "User-selected analysis area",
    }
    return labels.get(str(source_type), str(source_type or "User-selected analysis area"))


def _extract_class_metrics(
    metrics: Dict[str, Any],
    time_key: str,  # "T1" or "T2"
) -> Dict[str, Dict[str, Any]]:
    cam = _safe_get(metrics, "class_area_metrics", time_key, default={}) or {}
    return cam


def _veg_class_pct_from_metrics(
    class_metrics: Dict[str, Any],
    class_name: str,
    effective_ha: Optional[float] = None,
    physical: bool = False,
) -> Tuple[Optional[float], Optional[str]]:
    cls = class_metrics.get(class_name, {}) or {}
    pct = _safe_float(cls.get("percentage"))
    ha_v = _safe_float(cls.get("area_ha"))
    ha_str = f"{ha_v:,.2f} ha" if (ha_v is not None and physical and ha_v >= 0) else None
    return pct, ha_str


def _wetland_assessment(
    wetland: Dict[str, Any],
    wetland_meta: Dict[str, Any],
    hydro: Dict[str, Any],
    deg: Dict[str, Any],
    rec: Dict[str, Any],
    cva: Dict[str, Any],
    fusion: Dict[str, Any],
    phys_depl: str,
    deg_pct: Optional[float],
    rec_pct: Optional[float],
    water_loss_pct: Optional[float],
    water_gain_pct: Optional[float],
    hydro_deg_pct: Optional[float],
    water_available: bool,
) -> Dict[str, Any]:
    source_type = str(
        wetland.get("source_type")
        or wetland_meta.get("source_type")
        or "UNAVAILABLE"
    )
    status = str(
        wetland.get("status")
        or wetland_meta.get("confidence_status")
        or "WETLAND_BOUNDARY_UNAVAILABLE"
    )

    cva_mean = _safe_float(_safe_get(cva, "cva_magnitude", "mean", default=None))
    if cva_mean is None:
        cva_mean = _safe_float(cva.get("mean_magnitude"))

    d_ndvi = _safe_float(_safe_get(cva, "delta_ndvi", "mean", default=None))
    if d_ndvi is None:
        d_ndvi = _safe_float(cva.get("mean_delta_ndvi"))

    d_gndvi = _safe_float(_safe_get(cva, "delta_gndvi", "mean", default=None))
    if d_gndvi is None:
        d_gndvi = _safe_float(cva.get("mean_delta_gndvi"))

    d_ndre = _safe_float(_safe_get(cva, "delta_ndre", "mean", default=None))
    if d_ndre is None:
        d_ndre = _safe_float(cva.get("mean_delta_ndre"))

    if source_type == "OPTICAL_WATER_INFERENCE":
        context = (
            "The wetland context is derived from optical surface-water observations, "
            "identifying areas with documented surface-water presence."
        )
    elif source_type == "JRC_WATER_CONTEXT":
        context = (
            "The wetland context is derived from multi-temporal surface-water history "
            "(JRC Global Surface Water), providing hydrological context."
        )
    elif source_type in ("PROJECT_BOUNDARY", "OFFICIAL_DATASET"):
        context = (
            "The analysis is spatially bounded by the configured wetland boundary. "
            "All ecological condition indicators are evaluated within this defined area."
        )
    else:
        context = (
            "This assessment examines changes within the area selected for analysis. "
            "The selected area defines where the assessment is performed and should not "
            "automatically be interpreted as a legal or statutory wetland boundary."
        )

    deg_val = deg_pct if deg_pct is not None else 0.0
    rec_val = rec_pct if rec_pct is not None else 0.0
    cva_val = cva_mean if cva_mean is not None else 0.0
    hydro_deg_val = hydro_deg_pct if hydro_deg_pct is not None else 0.0

    if water_available and hydro_deg_val > 0.0:
        primary_title = "WETLAND DEGRADATION SIGNAL DETECTED"
        primary_color = ALERT
        primary_bg = ALERT_LIGHT
        finding = (
            f"A concurrent wetland degradation signal was detected across {hydro_deg_val:.2f}% "
            "of the effective analysis area, evidenced by concurrent vegetation condition deterioration "
            "and surface-water contraction. Continuous spectral indicators corroborate localized canopy stress."
        )
        conclusion_body = (
            f"Available evidence indicates measurable wetland degradation across {hydro_deg_val:.2f}% "
            "of the analyzed area during the observation period."
        )
    elif deg_val == 0.0 and (cva_mean is None or abs(cva_val) < 1e-4):
        primary_title = "NO MEASURABLE WETLAND DEGRADATION DETECTED"
        primary_color = OLIVE
        primary_bg = OLIVE_LIGHT
        if not water_available:
            finding = (
                "No measurable wetland degradation was detected. Vegetation condition remained "
                "stable across the selected area. Surface-water extent was unavailable for this run."
            )
            conclusion_body = (
                "Available evidence does not indicate measurable wetland degradation. "
                "Surface-water extent was not quantitatively available, so physical wetland loss is not established."
            )
        else:
            finding = (
                "No measurable wetland degradation was detected from the available evidence. "
                "Classified vegetation condition remained stable, and surface-water dynamics "
                "showed no corroborating drying or water loss."
            )
            conclusion_body = (
                "Available evidence indicates no measurable wetland degradation within the selected analysis area. "
                "Vegetation condition and spectral metrics remained stable; no concurrent water loss was observed."
            )
    elif deg_val > 0.0 and not water_available:
        primary_title = "INSUFFICIENT EVIDENCE TO ASSESS WETLAND DEGRADATION"
        primary_color = ALERT
        primary_bg = ALERT_LIGHT
        finding = (
            f"Vegetation condition shifts were detected across {deg_val:.2f}% of the analysis area. "
            "However, independent surface-water extent was unavailable for this run. "
            "Vegetation change alone is not sufficient to establish wetland degradation."
        )
        conclusion_body = (
            f"While {deg_val:.2f}% of the area showed vegetation stress pathways, surface-water change "
            "was not quantitatively available to confirm hydro-ecological degradation."
        )
    elif deg_val > 0.0 and water_available and (water_loss_pct is None or water_loss_pct == 0.0):
        primary_title = "VEGETATION STRESS DETECTED (NO HYDROLOGICAL LOSS)"
        primary_color = SLATE
        primary_bg = PAPER_DARK
        finding = (
            f"Vegetation stress pathways were detected across {deg_val:.2f}% of the analyzed area. "
            "However, independent hydrological evidence demonstrates no concurrent surface-water loss. "
            "This reflects a vegetation-only condition response rather than systemic wetland degradation."
        )
        conclusion_body = (
            f"Vegetation condition declined across {deg_val:.2f}% of the footprint without concurrent "
            "surface-water contraction. This pattern represents vegetation canopy dynamics rather than "
            "hydro-ecological wetland degradation."
        )
    else:
        primary_title = "NO MEASURABLE WETLAND DEGRADATION DETECTED"
        primary_color = OLIVE
        primary_bg = OLIVE_LIGHT
        finding = (
            "Available multi-stream remote-sensing evidence demonstrates no systemic wetland degradation "
            "within the selected analysis area across the observation window."
        )
        conclusion_body = (
            "Multi-evidence synthesis does not identify concurrent vegetation and hydrological decline "
            "sufficient to establish wetland degradation."
        )

    water_loss_ha = _safe_float(hydro.get("water_loss_area_ha"))
    water_gain_ha = _safe_float(hydro.get("water_gain_area_ha"))
    if not water_available:
        water_sentence = (
            "Quantitative surface-water extent was unavailable for this run; "
            "hydrological presence could not be independently quantified."
        )
    elif water_loss_pct is not None and water_gain_pct is not None:
        water_sentence = (
            f"Surface-water dynamics indicate {water_loss_pct:.2f}% contraction (drying) and "
            f"{water_gain_pct:.2f}% expansion (wetting) across the analyzed area."
        )
    elif water_loss_pct is not None:
        water_sentence = f"Surface-water dynamics indicate {water_loss_pct:.2f}% contraction across the analyzed area."
    elif water_gain_pct is not None:
        water_sentence = f"Surface-water dynamics indicate {water_gain_pct:.2f}% expansion across the analyzed area."
    elif water_loss_ha is not None and water_gain_ha is not None:
        water_sentence = (
            f"Surface-water dynamics: {water_loss_ha:,.2f} ha contraction and "
            f"{water_gain_ha:,.2f} ha expansion across the analysis area."
        )
    else:
        water_sentence = "Surface-water dynamics were evaluated across the analysis footprint."

    if d_ndvi is not None and d_gndvi is not None and d_ndre is not None:
        spectral_sentence = (
            f"Continuous spectral indicators (mean ΔNDVI: {d_ndvi:+.4f}, ΔGNDVI: {d_gndvi:+.4f}, "
            f"ΔNDRE: {d_ndre:+.4f}; CVA magnitude: {_fmt_num(cva_mean, 4)}) "
            "provide continuous radiometric evidence of canopy physiology."
        )
    elif cva_mean is not None:
        spectral_sentence = (
            f"Continuous spectral indicators record a mean CVA magnitude of {_fmt_num(cva_mean, 4)}."
        )
    else:
        spectral_sentence = "Continuous spectral indicators evaluate canopy physiological reflectance shifts."

    if phys_depl == "SUPPORTED_BY_INDEPENDENT_EVIDENCE":
        physical_loss_status = "Established"
        depletion_sentence = "Physical wetland depletion is corroborated by configured independent ground/external evidence."
    else:
        physical_loss_status = "Not established"
        depletion_sentence = (
            "Physical wetland loss is not established. In accordance with strict scientific safeguards, "
            "optical vegetation change alone is not interpreted as permanent physical wetland loss."
        )

    return {
        "context": context,
        "finding": finding,
        "water": water_sentence,
        "spectral": spectral_sentence,
        "depletion": depletion_sentence,
        "status": status,
        "primary_title": primary_title,
        "primary_color": primary_color,
        "primary_bg": primary_bg,
        "conclusion_body": conclusion_body,
        "physical_loss_status": physical_loss_status,
        "cva_mean": cva_mean,
        "d_ndvi": d_ndvi,
        "d_gndvi": d_gndvi,
        "d_ndre": d_ndre,
    }


def get_report_styles(primary_color=None) -> Dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=20, leading=23,
        textColor=INK, alignment=TA_LEFT, spaceAfter=4
    )
    deck = ParagraphStyle(
        "ReportDeck", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=13.5,
        textColor=SLATE, spaceAfter=7
    )
    eyebrow = ParagraphStyle(
        "ReportEyebrow", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=7, leading=9,
        textColor=OLIVE, spaceAfter=3
    )
    h1 = ParagraphStyle(
        "ReportH1", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=13.0, leading=16.0,
        textColor=INK, spaceBefore=6, spaceAfter=4, keepWithNext=True
    )
    h2 = ParagraphStyle(
        "ReportH2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=9.5, leading=12.0,
        textColor=SLATE, spaceBefore=6, spaceAfter=3, keepWithNext=True
    )
    body = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=8.2, leading=11.6,
        textColor=INK, spaceAfter=5
    )
    caption = ParagraphStyle(
        "ReportCaption", parent=body,
        fontName="Helvetica-Oblique", fontSize=7.1, leading=9.2,
        textColor=SLATE, alignment=TA_CENTER, spaceBefore=2, spaceAfter=5
    )
    caption_l = ParagraphStyle("ReportCaptionLeft", parent=caption, alignment=TA_LEFT)
    small = ParagraphStyle(
        "ReportSmall", parent=body,
        fontSize=6.8, leading=9.0, textColor=SLATE, spaceAfter=3
    )
    cell = ParagraphStyle("ReportCell", parent=body, fontSize=7.1, leading=9.2, spaceAfter=0)
    cell_bold = ParagraphStyle("ReportCellBold", parent=cell, fontName="Helvetica-Bold")
    cell_code = ParagraphStyle("ReportCellCode", parent=cell, fontName="Courier", fontSize=6.8, leading=8.5)
    right = ParagraphStyle("ReportRight", parent=cell, alignment=TA_RIGHT)
    center = ParagraphStyle("ReportCenter", parent=cell, alignment=TA_CENTER)
    header = ParagraphStyle(
        "ReportHeader", parent=cell, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_LEFT
    )
    header_r = ParagraphStyle("ReportHeaderRight", parent=header, alignment=TA_RIGHT)
    header_c = ParagraphStyle("ReportHeaderCenter", parent=header, alignment=TA_CENTER)
    primary_s = ParagraphStyle(
        "ReportPrimaryResult", parent=body,
        fontName="Helvetica-Bold", fontSize=11.5, leading=15.0,
        textColor=primary_color or INK, spaceAfter=2
    )
    finding_s = ParagraphStyle(
        "ReportFinding", parent=body,
        fontName="Helvetica-Bold", fontSize=9.0, leading=12.5,
        textColor=INK, spaceAfter=0
    )
    label_s = ParagraphStyle(
        "ReportLabel", parent=body,
        fontName="Helvetica-Bold", fontSize=7.5, leading=10,
        textColor=SLATE, spaceAfter=1
    )
    return {
        "title": title_s,
        "deck": deck,
        "eyebrow": eyebrow,
        "h1": h1,
        "h2": h2,
        "body": body,
        "caption": caption,
        "caption_l": caption_l,
        "small": small,
        "cell": cell,
        "cell_bold": cell_bold,
        "cell_code": cell_code,
        "right": right,
        "center": center,
        "header": header,
        "header_r": header_r,
        "header_c": header_c,
        "primary": primary_s,
        "finding": finding_s,
        "label": label_s,
    }
