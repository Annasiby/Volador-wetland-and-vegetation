"""
src/gis_engine/output/cartography.py
======================================
Cartographic Map Rendering Module for the GIS Wetland Engine.

Converts GeoTIFF raster outputs into high-resolution (300 DPI) publication-quality
map plates with actual multispectral source imagery backgrounds, semi-transparent
analytical overlays, legends, north arrows, scale bars, and coordinate information
for embedding in final PDF analysis reports. Manual exclusion and effective-analysis masks are rendered directly from the authoritative raster artifacts used by the engine.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import numpy as np
import rasterio

from ..core.constants import (
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    NODATA_CATEGORICAL,
    NODATA_CONTINUOUS,
    FUSION_CATEGORIES,
)

logger = logging.getLogger(__name__)

# Standard Colormaps (RGBA 0.0 - 1.0)
HEALTH_COLORMAP_RGBA = {
    0: (0.45, 0.45, 0.45, 1.0),   # Gray - Background / Non-Veg
    1: (0.88, 0.15, 0.15, 1.0),   # Red - Severe Stress
    2: (0.95, 0.60, 0.05, 1.0),   # Orange - Moderate Stress
    3: (0.12, 0.72, 0.12, 1.0),   # Green - Healthy Vegetation
}

WATER_CLASS_NAMES = {
    0: "Stable Non-Water",
    1: "Stable Water",
    2: "Water Gain (Wetting)",
    3: "Water Loss (Drying)",
}

WATER_COLORMAP_RGBA = {
    0: (0.75, 0.75, 0.75, 1.0),   # Neutral Gray - Land
    1: (0.10, 0.35, 0.85, 1.0),   # Deep Blue - Stable Water
    2: (0.15, 0.80, 0.85, 1.0),   # Cyan - Water Gain
    3: (0.90, 0.50, 0.15, 1.0),   # Orange/Amber - Water Loss
}

FUSION_CATEGORY_NAMES = {
    0: "NoData",
    1: "Stable Landscape",
    2: "Vegetation Degradation",
    3: "Vegetation Recovery",
    4: "Surface Water Gain",
    5: "Surface Water Loss",
    6: "Hydro-Ecological Degradation",
    7: "Hydro-Ecological Recovery",
    8: "Vegetation → Background (Canopy Loss)",
    9: "Ambiguous / Mixed",
}

FUSION_COLORMAP_RGBA = {
    0: (0.80, 0.80, 0.80, 1.0),   # NoData - Gray
    1: (0.20, 0.68, 0.38, 1.0),   # Stable - Vibrant Emerald Green
    2: (0.90, 0.20, 0.20, 1.0),   # Veg Deg - Red
    3: (0.20, 0.80, 0.20, 1.0),   # Veg Rec - Light Green
    4: (0.20, 0.70, 0.90, 1.0),   # Water Gain - Cyan
    5: (0.90, 0.60, 0.10, 1.0),   # Water Loss - Orange
    6: (0.75, 0.00, 0.20, 1.0),   # Hydro-Eco Deg - Crimson
    7: (0.00, 0.60, 0.50, 1.0),   # Hydro-Eco Rec - Teal
    8: (0.85, 0.45, 0.15, 1.0),   # Veg to Bg - Ochre
    9: (0.60, 0.40, 0.70, 1.0),   # Ambiguous - Purple
}

DEGRADATION_COLORS = {
    0: (0.75, 0.75, 0.75, 0.0),   # Transparent for unchanged
    1: (0.88, 0.15, 0.15, 1.0),   # Red - Degradation
}

RECOVERY_COLORS = {
    0: (0.75, 0.75, 0.75, 0.0),   # Transparent for unchanged
    1: (0.15, 0.72, 0.15, 1.0),   # Green - Recovery
}

CANOPY_LOSS_COLORS = {
    0: (0.75, 0.75, 0.75, 0.0),   # Transparent for retained
    1: (0.90, 0.50, 0.10, 1.0),   # Amber - Canopy Loss
}


# -----------------------------------------------------------------------------
# Cartographic Adornments (North Arrow, Scale Bar)
# -----------------------------------------------------------------------------

def _raster_crs_kind(crs: Optional[Any]) -> str:
    """
    Classifies a raster CRS for cartographic annotation.

    Returns:
        "geographic" for angular geographic CRSs,
        "projected" for projected CRSs,
        "pixel" when no CRS is defined,
        "unknown" otherwise.
    """
    if crs is None:
        return "pixel"
    try:
        if getattr(crs, "is_geographic", False):
            return "geographic"
        if getattr(crs, "is_projected", False):
            return "projected"
    except Exception:
        pass
    return "unknown"


def _is_identity_pixel_transform(transform: Optional[Any]) -> bool:
    """Returns True for the unreferenced pixel-space transform used by V1 inputs."""
    if transform is None:
        return False
    try:
        return (
            math.isclose(transform.a, 1.0)
            and math.isclose(transform.b, 0.0)
            and math.isclose(transform.c, 0.0)
            and math.isclose(transform.d, 0.0)
            and math.isclose(transform.e, 1.0)
            and math.isclose(transform.f, 0.0)
        )
    except Exception:
        return False


def _get_aoi_plot_bounds(
    effective_mask_path: Path,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Derive the display bounding box for maps from the effective analysis mask.

    Opens effective_analysis_mask.tif, finds the tight row/column bounding
    box of non-zero pixels, and converts it to the raster's coordinate system.
    Returns (xlim_min, xlim_max, ylim_min, ylim_max) suitable for
    ax.set_xlim / ax.set_ylim — or None if the mask cannot be read.

    Coordinate safety
    -----------------
    - This function reads the raster-aligned backend mask directly.
    - It does NOT use frontend pixel coordinates.
    - For pixel-space rasters (CRS=None), the extent is
      BoundingBox(left=0, bottom=H, right=W, top=0), so matplotlib's
      extent is [0, W, H, 0].  The row indices map naturally to
      ylim = (row_min, row_max) on the flipped y-axis.
    - A 5% margin is added so the AOI edge is not flush with the frame.
    """
    effective_mask_path = Path(effective_mask_path)
    if not effective_mask_path.is_file():
        return None
    try:
        with rasterio.open(effective_mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            height, width = src.height, src.width

        valid_rows = np.any(mask > 0, axis=1)
        valid_cols = np.any(mask > 0, axis=0)
        if not np.any(valid_rows) or not np.any(valid_cols):
            return None

        row_min = int(np.where(valid_rows)[0][0])
        row_max = int(np.where(valid_rows)[0][-1])
        col_min = int(np.where(valid_cols)[0][0])
        col_max = int(np.where(valid_cols)[0][-1])

        # Add 3% padding around the tight bounding box
        pad_rows = max(5, int((row_max - row_min) * 0.03))
        pad_cols = max(5, int((col_max - col_min) * 0.03))
        row_min = max(0, row_min - pad_rows)
        row_max = min(height - 1, row_max + pad_rows)
        col_min = max(0, col_min - pad_cols)
        col_max = min(width - 1, col_max + pad_cols)

        # Convert pixel row/col to the raster coordinate system.
        # rasterio uses: x = transform.c + col * transform.a
        #                y = transform.f + row * transform.e
        # For pixel-space: transform = identity-like (a=1,e=1,c=0,f=0)
        #   but rasterio's pixel-space bounds are (left=0,top=0,right=W,bottom=H).
        # We return x limits and y limits for matplotlib set_xlim/set_ylim.
        x_left = transform.c + col_min * transform.a
        x_right = transform.c + (col_max + 1) * transform.a
        y_top_raster = transform.f + row_min * transform.e   # raster top = smaller y for geographic
        y_bottom_raster = transform.f + (row_max + 1) * transform.e

        # For pixel-space (e > 0, y increases downward in raster coords):
        #   y_top_raster < y_bottom_raster
        # For geographic (e < 0, y increases upward):
        #   y_top_raster > y_bottom_raster
        # matplotlib ylim needs (bottom_value, top_value) for the render.
        # We return a 4-tuple: (x_min, x_max, y_min, y_max)
        # where min/max correspond to matplotlib's data coordinates.
        x_min = min(x_left, x_right)
        x_max = max(x_left, x_right)
        y_min = min(y_top_raster, y_bottom_raster)
        y_max = max(y_top_raster, y_bottom_raster)

        return (x_min, x_max, y_min, y_max)

    except Exception as exc:
        logger.debug("Could not compute AOI plot bounds: %s", exc)
        return None


def _apply_aoi_mask_to_rgb(
    rgb: np.ndarray,
    effective_mask_path: Optional[Path],
    outside_value: float = 0.88,
) -> np.ndarray:
    """
    Darken / neutralize RGB source pixels that lie outside the effective AOI.

    Parameters
    ----------
    rgb : (H, W, 3) float32 array in [0, 1]
    effective_mask_path : path to effective_analysis_mask.tif
    outside_value : gray value to fill outside-AOI pixels (0.88 = light gray)

    Returns the modified RGB (same array, modified in-place copy).
    """
    if effective_mask_path is None or not Path(effective_mask_path).is_file():
        return rgb
    try:
        with rasterio.open(effective_mask_path) as src:
            aoi = src.read(1)
        if aoi.shape != rgb.shape[:2]:
            return rgb
        out = rgb.copy()
        outside = aoi == 0
        out[outside, 0] = outside_value
        out[outside, 1] = outside_value
        out[outside, 2] = outside_value
        return out
    except Exception:
        return rgb


def draw_north_arrow(
    ax: plt.Axes,
    loc: Tuple[float, float] = (0.93, 0.88),
    size: float = 0.08,
    available: bool = True,
) -> None:
    """
    Draw a north arrow only when the raster orientation is geographically
    interpretable.

    For unreferenced/pixel-space imagery, a north direction cannot be inferred
    from the source raster. In that case a compact "N/A" annotation is used
    instead of presenting a false north arrow.
    """
    x, y = loc

    if not available:
        ax.text(
            x,
            y,
            "N/A",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#4A5568",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="#CBD5E0",
                linewidth=0.5,
                alpha=0.92,
            ),
            zorder=25,
        )
        return

    ax.annotate(
        "N",
        xy=(x, y + size),
        xytext=(x, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            facecolor="black",
            edgecolor="black",
            width=2.0,
            headwidth=6.5,
            headlength=8.5,
            shrink=0.05,
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        zorder=25,
    )


def draw_scale_bar(
    ax: plt.Axes,
    bounds: Tuple[float, float, float, float],
    crs_str: Optional[str] = None,
    loc: Tuple[float, float] = (0.06, 0.06),
    crs: Optional[Any] = None,
    transform: Optional[Any] = None,
) -> None:
    """
    Draw a scale bar only when physical map units are defensible.

    V1 imagery can be deliberately unreferenced (CRS=None, identity transform)
    and represented in pixel coordinates. Such a raster has no defensible metre
    scale. The map therefore reports scale as unavailable rather than treating
    one pixel as one metre.

    For projected CRSs, the rasterio CRS linear unit is respected when available.
    For geographic CRSs, a latitude-dependent metres-per-degree approximation is
    used for the horizontal map width.
    """
    left, bottom, right, top = bounds
    width_units = abs(right - left)

    if width_units <= 0:
        return

    crs_kind = _raster_crs_kind(crs)
    if crs is None and crs_str:
        low_crs = str(crs_str).lower()
        if any(token in low_crs for token in ("4326", "wgs 84", "degree", "longlat", "latlong")):
            crs_kind = "geographic"
        elif "unknown" not in low_crs and "none" not in low_crs:
            # A named projected CRS (for example EPSG:326xx/327xx) is
            # normally metre-based. Exact unit handling is available when a
            # rasterio CRS object is supplied.
            crs_kind = "projected"

    # Explicit pixel-space safeguard.
    if crs_kind == "pixel" or (
        crs_kind == "unknown" and _is_identity_pixel_transform(transform)
    ):
        ax.text(
            loc[0],
            loc[1],
            "Scale unavailable\n(pixel-space raster)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.8,
            fontweight="bold",
            color="#4A5568",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#CBD5E0",
                linewidth=0.5,
                alpha=0.92,
            ),
            zorder=24,
        )
        return

    is_geographic = crs_kind == "geographic"
    mid_lat = (bottom + top) / 2.0

    if is_geographic:
        meters_per_deg = 111_320.0 * max(
            0.01, math.cos(math.radians(mid_lat))
        )
        total_meters = width_units * meters_per_deg
    elif crs_kind == "projected":
        # Rasterio exposes the linear unit name on most projected CRSs.
        unit_name = ""
        try:
            unit_name = str(getattr(crs, "linear_units", "") or "").lower()
        except Exception:
            pass

        if "foot" in unit_name or unit_name in {"ft", "feet"}:
            total_meters = width_units * 0.3048
        else:
            # Standard projected metric CRSs are metres.
            total_meters = width_units
    else:
        # Unknown non-pixel unit system is not safe to label as metres.
        ax.text(
            loc[0],
            loc[1],
            "Scale unavailable\n(unknown map units)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.8,
            fontweight="bold",
            color="#4A5568",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#CBD5E0",
                linewidth=0.5,
                alpha=0.92,
            ),
            zorder=24,
        )
        return

    if not math.isfinite(total_meters) or total_meters <= 0:
        return

    target_bar_meters = total_meters * 0.20

    if target_bar_meters >= 50_000:
        bar_meters = round(target_bar_meters / 50_000) * 50_000
    elif target_bar_meters >= 10_000:
        bar_meters = round(target_bar_meters / 10_000) * 10_000
    elif target_bar_meters >= 1_000:
        bar_meters = round(target_bar_meters / 1_000) * 1_000
    elif target_bar_meters >= 100:
        bar_meters = round(target_bar_meters / 100) * 100
    elif target_bar_meters >= 10:
        bar_meters = round(target_bar_meters / 10) * 10
    else:
        bar_meters = max(1.0, round(target_bar_meters))

    if bar_meters <= 0:
        return

    bar_fraction = bar_meters / total_meters
    if bar_fraction > 0.6:
        bar_fraction = 0.5
        bar_meters = total_meters * 0.5

    x0, y0 = loc
    x1 = x0 + bar_fraction

    if bar_meters >= 1000:
        label = f"{bar_meters / 1000:.1f} km".replace(".0 km", " km")
    else:
        label = f"{bar_meters:.0f} m"

    rect = mpatches.Rectangle(
        (x0 - 0.015, y0 - 0.02),
        bar_fraction + 0.03,
        0.055,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.90,
        zorder=24,
    )
    ax.add_patch(rect)

    ax.plot(
        [x0, x1],
        [y0, y0],
        transform=ax.transAxes,
        color="black",
        linewidth=2.5,
        solid_capstyle="butt",
        zorder=25,
    )
    ax.plot(
        [x0, x0],
        [y0 - 0.008, y0 + 0.008],
        transform=ax.transAxes,
        color="black",
        linewidth=1.5,
        zorder=25,
    )
    ax.plot(
        [x1, x1],
        [y0 - 0.008, y0 + 0.008],
        transform=ax.transAxes,
        color="black",
        linewidth=1.5,
        zorder=25,
    )

    ax.text(
        (x0 + x1) / 2.0,
        y0 + 0.012,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
        color="black",
        zorder=26,
    )


# -----------------------------------------------------------------------------
# Helper: Load Normalized RGB Multispectral Composite
# -----------------------------------------------------------------------------

def load_rgb_composite(
    band_paths: Dict[str, Path],
    band_order: Tuple[str, str, str] = ("nir", "red", "green"),
) -> Tuple[Optional[np.ndarray], Optional[Any], str]:
    """
    Loads and contrast-stretches (2%-98%) 3 multispectral bands into a float32 RGB array.

    Returns:
        (rgb_array [H, W, 3], bounds, crs_string) or (None, None, "Unknown CRS")
    """
    r_key, g_key, b_key = band_order
    if not (r_key in band_paths and g_key in band_paths and b_key in band_paths):
        return None, None, "Unknown CRS"

    loaded_bands = []
    bounds = None
    crs_str = "Unknown CRS"

    for key in [r_key, g_key, b_key]:
        p = Path(band_paths[key])
        if not p.is_file():
            return None, None, "Unknown CRS"
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
            if bounds is None:
                bounds = src.bounds
                crs_str = src.crs.to_string() if src.crs else "Unknown CRS"
            nodata_val = src.nodata
            if nodata_val is not None:
                arr[arr == nodata_val] = np.nan
            loaded_bands.append(arr)

    rgb_norm = []
    for arr in loaded_bands:
        valid = arr[~np.isnan(arr) & ~np.isinf(arr)]
        if valid.size > 0:
            p2, p98 = float(np.percentile(valid, 2.0)), float(np.percentile(valid, 98.0))
            if p98 > p2:
                stretched = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
            else:
                stretched = np.zeros_like(arr)
        else:
            stretched = np.zeros_like(arr)
        stretched = np.nan_to_num(stretched, nan=0.0)
        rgb_norm.append(stretched)

    rgb_img = np.stack(rgb_norm, axis=-1)
    return rgb_img, bounds, crs_str


# -----------------------------------------------------------------------------
# Standalone Multispectral Source Composite Renderer
# -----------------------------------------------------------------------------

def render_multispectral_composite(
    band_paths: Dict[str, Path],
    output_image_path: Path,
    title: str,
    band_order: Tuple[str, str, str] = ("nir", "red", "green"),
    subtitle: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.5, 4.8),
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
    effective_mask_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Renders a standalone Color-Infrared (CIR) false-color composite map plate.

    Parameters
    ----------
    aoi_bounds : (x_min, x_max, y_min, y_max) from _get_aoi_plot_bounds().
                 When supplied the plot axes are restricted to the AOI extent.
    effective_mask_path : path to effective_analysis_mask.tif; when supplied
                          outside-AOI source pixels are greyed out.
    """
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_img, bounds, crs_str = load_rgb_composite(band_paths, band_order)
    if rgb_img is None or bounds is None:
        logger.warning("Could not load bands for composite %s", output_image_path)
        return None

    # Apply AOI mask to source image (grey out outside-AOI pixels)
    if effective_mask_path is not None:
        rgb_img = _apply_aoi_mask_to_rgb(rgb_img, effective_mask_path)

    raster_crs = None
    raster_transform = None
    first_path = next(iter(band_paths.values()), None)
    if first_path and Path(first_path).is_file():
        try:
            with rasterio.open(first_path) as s:
                raster_crs = s.crs
                raster_transform = s.transform
        except Exception:
            pass

    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(
            rgb_img,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            interpolation="bilinear",
        )

        # Restrict display to AOI bounding box
        if aoi_bounds is not None:
            ax.set_xlim(aoi_bounds[0], aoi_bounds[1])
            ax.set_ylim(aoi_bounds[2], aoi_bounds[3])

        full_title = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(full_title, fontsize=8.5, fontweight="bold", pad=6, color="#1A365D")

        ax.tick_params(axis="both", which="major", labelsize=6.0)
        ax.grid(True, linestyle="--", alpha=0.30, color="gray")

        north_available = (
            _raster_crs_kind(raster_crs) in {"geographic", "projected"}
            and not (
                raster_transform is not None
                and (
                    not math.isclose(getattr(raster_transform, "b", 0.0), 0.0)
                    or not math.isclose(getattr(raster_transform, "d", 0.0), 0.0)
                )
            )
        )
        draw_north_arrow(ax, available=north_available)
        draw_scale_bar(
            ax,
            bounds,
            crs_str,
            crs=raster_crs,
            transform=raster_transform,
        )

        r_k, g_k, b_k = band_order
        clip_note = " | Clipped to AOI" if aoi_bounds is not None else ""
        fig.text(
            0.02,
            0.01,
            f"CRS: {crs_str} | Composite: R={r_k.upper()}, G={g_k.upper()}, B={b_k.upper()} (2-98% Stretch){clip_note}",
            fontsize=5.5,
            color="#4A5568",
        )

        fig.tight_layout()
        fig.savefig(output_image_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Rendered source multispectral composite: %s", output_image_path)
        return output_image_path

    except Exception as exc:
        logger.error("Failed to render multispectral composite %s: %s", output_image_path, exc)
        if "fig" in locals():
            plt.close(fig)
        return None


# -----------------------------------------------------------------------------
# Categorical Map Renderer with Source Imagery Background & Transparency
# -----------------------------------------------------------------------------

def render_categorical_map(
    raster_path: Path,
    output_image_path: Path,
    title: str,
    colormap_dict: Dict[int, Tuple[float, float, float, float]],
    class_labels: Dict[int, str],
    source_band_paths: Optional[Dict[str, Path]] = None,
    alpha: float = 0.55,
    filter_active_classes: bool = True,
    subtitle: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.5, 4.8),
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
    effective_mask_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Renders a discrete categorical GeoTIFF overlaid semi-transparently on the
    actual multispectral source imagery background, restricted to the AOI.

    Parameters
    ----------
    aoi_bounds : (x_min, x_max, y_min, y_max) restricts the display axes.
    effective_mask_path : greyed-out outside-AOI pixels in source imagery.
    """
    raster_path = Path(raster_path)
    if not raster_path.is_file():
        logger.warning("Raster not found: %s", raster_path)
        return None

    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds
            raster_crs = src.crs
            raster_transform = src.transform
            crs_str = src.crs.to_string() if src.crs else "Unknown CRS"
            nodata_val = src.nodata if src.nodata is not None else NODATA_CATEGORICAL

        # Load background source composite if available
        rgb_bg = None
        if source_band_paths:
            rgb_bg, bg_bounds, _ = load_rgb_composite(source_band_paths)
            if bg_bounds is not None:
                bounds = bg_bounds

        # Apply AOI mask to source imagery (grey outside-AOI pixels)
        if rgb_bg is not None and effective_mask_path is not None:
            rgb_bg = _apply_aoi_mask_to_rgb(rgb_bg, effective_mask_path)

        sorted_keys = sorted(colormap_dict.keys())
        colors = [colormap_dict[k] for k in sorted_keys]
        cmap = ListedColormap(colors)

        boundaries = [k - 0.5 for k in sorted_keys] + [sorted_keys[-1] + 0.5]
        norm = BoundaryNorm(boundaries, len(colors))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # 1. Render actual source imagery background first
        if rgb_bg is not None:
            ax.imshow(rgb_bg, extent=extent, interpolation="bilinear")

        # 2. Render categorical analytical overlay with alpha
        #    Mask pixels outside the effective AOI to make them transparent
        nodata_for_mask = nodata_val if nodata_val is not None else NODATA_CATEGORICAL
        outside_mask = (data == nodata_for_mask) | np.isnan(data) | (~np.isin(data, sorted_keys))
        if effective_mask_path is not None and Path(effective_mask_path).is_file():
            try:
                with rasterio.open(effective_mask_path) as _ms:
                    _aoi_arr = _ms.read(1)
                if _aoi_arr.shape == data.shape:
                    outside_mask = outside_mask | (_aoi_arr == 0)
            except Exception:
                pass
        masked_data = np.ma.masked_where(outside_mask, data)

        overlay_alpha = alpha if rgb_bg is not None else 1.0
        ax.imshow(
            masked_data,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=extent,
            alpha=overlay_alpha,
        )

        # Restrict axes to AOI extent
        if aoi_bounds is not None:
            ax.set_xlim(aoi_bounds[0], aoi_bounds[1])
            ax.set_ylim(aoi_bounds[2], aoi_bounds[3])

        full_title = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(full_title, fontsize=8.5, fontweight="bold", pad=6, color="#1A365D")

        ax.tick_params(axis="both", which="major", labelsize=6.0)
        ax.grid(True, linestyle="--", alpha=0.30, color="gray")

        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32)
            bounds = src.bounds
            raster_crs = src.crs
            raster_transform = src.transform
            crs_str = src.crs.to_string() if src.crs else "Unknown CRS"
            nodata_val = src.nodata

        # 3. Legend Construction
        valid_pixels = data[(data != nodata_val) & ~np.isnan(data) & np.isin(data, sorted_keys)]
        present_keys = np.unique(valid_pixels).tolist() if valid_pixels.size > 0 else []

        if filter_active_classes and present_keys:
            keys_to_legend = [k for k in sorted_keys if k in present_keys]
        else:
            keys_to_legend = sorted_keys

        legend_patches = []
        for k in keys_to_legend:
            label_text = class_labels.get(k, f"Class {k}")
            color = colormap_dict[k]
            patch = mpatches.Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label_text)
            legend_patches.append(patch)

        if legend_patches:
            ax.legend(
                handles=legend_patches,
                loc="lower right",
                fontsize=6.5,
                framealpha=0.92,
                edgecolor="#CBD5E0",
                title="Active Classes" if (filter_active_classes and len(keys_to_legend) < len(sorted_keys)) else "Classes",
                title_fontsize=7.0,
            )

        bg_text = "CIR Source Composite (R=NIR, G=Red, B=Green)" if rgb_bg is not None else "No background"
        clip_note = " | Clipped to AOI" if aoi_bounds is not None else ""
        fig.text(
            0.02,
            0.01,
            f"CRS: {crs_str} | Base: {bg_text} | Overlay α={overlay_alpha:.2f}{clip_note}",
            fontsize=5.5,
            color="#4A5568",
        )

        fig.tight_layout()
        fig.savefig(output_image_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Rendered categorical map with source background: %s", output_image_path)
        return output_image_path

    except Exception as exc:
        logger.error("Failed to render categorical map %s: %s", raster_path, exc)
        if "fig" in locals():
            plt.close(fig)
        return None


# -----------------------------------------------------------------------------
# Continuous Map Renderer with Source Imagery Background & Diverging Colorbar
# -----------------------------------------------------------------------------

def render_continuous_map(
    raster_path: Path,
    output_image_path: Path,
    title: str,
    colorbar_label: str,
    source_band_paths: Optional[Dict[str, Path]] = None,
    alpha: float = 0.60,
    cmap_name: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    subtitle: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.5, 4.8),
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
    effective_mask_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Renders a continuous floating-point GeoTIFF overlaid semi-transparently on the
    actual multispectral source imagery background with a calibrated colorbar.
    """
    raster_path = Path(raster_path)
    if not raster_path.is_file():
        logger.warning("Raster not found: %s", raster_path)
        return None

    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32)
            bounds = src.bounds
            raster_crs = src.crs
            raster_transform = src.transform
            crs_str = src.crs.to_string() if src.crs else "Unknown CRS"
            nodata_val = src.nodata

        # Load background source composite if available
        rgb_bg = None
        if source_band_paths:
            rgb_bg, bg_bounds, _ = load_rgb_composite(source_band_paths)
            if bg_bounds is not None:
                bounds = bg_bounds

        # Apply AOI mask to source imagery
        if rgb_bg is not None and effective_mask_path is not None:
            rgb_bg = _apply_aoi_mask_to_rgb(rgb_bg, effective_mask_path)

        # Build display mask: nodata + outside-AOI
        mask = np.isnan(data) | np.isinf(data)
        if nodata_val is not None:
            mask |= (data == nodata_val)
        if effective_mask_path is not None and Path(effective_mask_path).is_file():
            try:
                with rasterio.open(effective_mask_path) as _ms:
                    _aoi_arr = _ms.read(1)
                if _aoi_arr.shape == data.shape:
                    mask = mask | (_aoi_arr == 0)
            except Exception:
                pass
        masked_data = np.ma.masked_where(mask, data)

        valid_vals = data[~mask]
        if valid_vals.size > 0:
            if vmin is None:
                vmin = float(np.percentile(valid_vals, 2.0))
            if vmax is None:
                vmax = float(np.percentile(valid_vals, 98.0))
            if vmin == vmax:
                vmin -= 0.1
                vmax += 0.1
        else:
            vmin, vmax = -1.0, 1.0

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # 1. Render source imagery background
        if rgb_bg is not None:
            ax.imshow(rgb_bg, extent=extent, interpolation="bilinear")

        # 2. Render continuous analytical overlay with alpha
        overlay_alpha = alpha if rgb_bg is not None else 1.0
        im = ax.imshow(
            masked_data,
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            extent=extent,
            alpha=overlay_alpha,
        )

        # Restrict axes to AOI extent
        if aoi_bounds is not None:
            ax.set_xlim(aoi_bounds[0], aoi_bounds[1])
            ax.set_ylim(aoi_bounds[2], aoi_bounds[3])

        full_title = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(full_title, fontsize=8.5, fontweight="bold", pad=6, color="#1A365D")

        ax.tick_params(axis="both", which="major", labelsize=6.0)
        ax.grid(True, linestyle="--", alpha=0.30, color="gray")

        # Continuous Colorbar
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.03)
        cbar.set_label(colorbar_label, fontsize=7.0, fontweight="bold")
        cbar.ax.tick_params(labelsize=6.0)

        source_north_available = (
            crs_str != "Unknown CRS"
            and crs_str != "None"
            and "pixel" not in crs_str.lower()
        )
        draw_north_arrow(ax, available=source_north_available)
        draw_scale_bar(
            ax,
            bounds,
            crs_str,
            crs=None,
            transform=None,
        )

        bg_text = "CIR Source Composite (R=NIR, G=Red, B=Green)" if rgb_bg is not None else "No background"
        clip_note = " | Clipped to AOI" if aoi_bounds is not None else ""
        fig.text(
            0.02,
            0.01,
            f"CRS: {crs_str} | Base: {bg_text} | Range: [{vmin:.3f}, {vmax:.3f}]{clip_note}",
            fontsize=5.5,
            color="#4A5568",
        )

        fig.tight_layout()
        fig.savefig(output_image_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Rendered continuous map with source background: %s", output_image_path)
        return output_image_path

    except Exception as exc:
        logger.error("Failed to render continuous map %s: %s", raster_path, exc)
        if "fig" in locals():
            plt.close(fig)
        return None


# -----------------------------------------------------------------------------
# Manual Exclusion Audit Renderer
# -----------------------------------------------------------------------------

def render_manual_exclusion_audit_map(
    exclusion_mask_path: Path,
    output_image_path: Path,
    source_band_paths: Optional[Dict[str, Path]] = None,
    title: str = "Manual Exclusion Audit",
    subtitle: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.5, 4.8),
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Path]:
    """
    Render the authoritative manual-exclusion raster mask over the
    corresponding source landscape.

    Pixel-space convention
    ----------------------
    For the current unreferenced DJI imagery:

        array row 0 = image top
        array column 0 = image left
        x increases left -> right
        y increases top -> bottom

    The PDF renderer must preserve that exact image orientation.

    The exclusion mask itself is never modified. Only its cartographic
    presentation is normalized to the same pixel-space convention as
    the source imagery.
    """
    exclusion_mask_path = Path(exclusion_mask_path)
    output_image_path = Path(output_image_path)

    if not exclusion_mask_path.is_file():
        logger.warning(
            "Manual exclusion mask not found: %s",
            exclusion_mask_path,
        )
        return None

    output_image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        # ------------------------------------------------------------
        # Read authoritative exclusion raster
        # ------------------------------------------------------------
        with rasterio.open(exclusion_mask_path) as src:
            mask = src.read(1)
            bounds = src.bounds
            raster_crs = src.crs
            raster_transform = src.transform
            crs_str = (
                src.crs.to_string()
                if src.crs
                else "Undefined (pixel-space)"
            )

            height = int(src.height)
            width = int(src.width)

        excluded = mask > 0

        # ------------------------------------------------------------
        # Load source imagery
        # ------------------------------------------------------------
        rgb_bg = None

        if source_band_paths:
            rgb_bg, bg_bounds, _ = load_rgb_composite(
                source_band_paths
            )

            if rgb_bg is not None:

                # Background must be exactly the same raster dimensions
                # as the authoritative exclusion mask.
                if rgb_bg.shape[:2] != (height, width):
                    logger.warning(
                        "Exclusion audit background shape %s does not "
                        "match mask shape (%d, %d); rendering without "
                        "background.",
                        rgb_bg.shape[:2],
                        height,
                        width,
                    )
                    rgb_bg = None

                elif bg_bounds is not None:

                    if not (
                        math.isclose(
                            bg_bounds.left,
                            bounds.left,
                        )
                        and math.isclose(
                            bg_bounds.right,
                            bounds.right,
                        )
                        and math.isclose(
                            bg_bounds.bottom,
                            bounds.bottom,
                        )
                        and math.isclose(
                            bg_bounds.top,
                            bounds.top,
                        )
                    ):
                        logger.warning(
                            "Source background bounds differ from "
                            "authoritative exclusion mask bounds; "
                            "retaining authoritative mask extent."
                        )

        # ------------------------------------------------------------
        # Pixel-space vs referenced-space
        # ------------------------------------------------------------
        pixel_space = (
            raster_crs is None
            and _is_identity_pixel_transform(
                raster_transform
            )
        )

        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        if pixel_space:
            # ========================================================
            # PIXEL-SPACE RENDERING
            # ========================================================
            #
            # The raster arrays are image arrays:
            #
            #   row 0        -> top
            #   row height-1 -> bottom
            #
            # imshow(..., origin="upper") already places array row 0 at
            # the TOP of the extent box on a normal (non-inverted)
            # Matplotlib axis — that is exactly the browser/image
            # orientation we want (array top = screen top), and needs
            # no further axis manipulation. Do NOT additionally invert
            # ylim here: doing so undoes the correct mapping that
            # origin="upper" already provides and flips the mask
            # vertically relative to the source imagery and the
            # frontend preview.
            #
            pixel_extent = [
                0,
                width,
                0,
                height,
            ]

            if rgb_bg is not None:
                ax.imshow(
                    rgb_bg,
                    extent=pixel_extent,
                    origin="upper",
                    interpolation="bilinear",
                    zorder=1,
                )

            # --------------------------------------------------------
            # Exact authoritative exclusion mask
            # --------------------------------------------------------
            exclusion_rgba = np.zeros(
                (height, width, 4),
                dtype=np.float32,
            )

            exclusion_rgba[excluded, 0] = 0.82
            exclusion_rgba[excluded, 1] = 0.00
            exclusion_rgba[excluded, 2] = 0.70
            exclusion_rgba[excluded, 3] = 0.45

            ax.imshow(
                exclusion_rgba,
                extent=pixel_extent,
                origin="upper",
                interpolation="nearest",
                zorder=20,
            )

            # --------------------------------------------------------
            # Boundary from the SAME raster mask
            # --------------------------------------------------------
            if np.any(excluded):
                try:
                    ax.contour(
                        excluded.astype(np.uint8),
                        levels=[0.5],
                        extent=pixel_extent,
                        origin="upper",
                        colors=["#C000A8"],
                        linewidths=1.5,
                        zorder=22,
                    )
                except Exception as contour_exc:
                    logger.debug(
                        "Could not draw exclusion boundary: %s",
                        contour_exc,
                    )

            # Non-inverted axis: y=0 at bottom, y=height at top, which
            # matches origin="upper" placing row 0 (image top) at the
            # top of the figure.
            if aoi_bounds is not None:
                # aoi_bounds is (x_min, x_max, y_min, y_max) in raster coords.
                # For pixel-space the raster y increases downward (row -> y),
                # so ylim in matplotlib (non-inverted) maps to (y_min, y_max).
                ax.set_xlim(aoi_bounds[0], aoi_bounds[1])
                ax.set_ylim(aoi_bounds[2], aoi_bounds[3])
            else:
                ax.set_xlim(0, width)
                ax.set_ylim(0, height)

            coordinate_mode = "PIXEL-SPACE (CRS undefined)"

        else:
            # ========================================================
            # GEOREFERENCED RENDERING
            # ========================================================
            extent = [
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top,
            ]

            if rgb_bg is not None:
                ax.imshow(
                    rgb_bg,
                    extent=extent,
                    origin="upper",
                    interpolation="bilinear",
                    zorder=1,
                )

            exclusion_rgba = np.zeros(
                (height, width, 4),
                dtype=np.float32,
            )

            exclusion_rgba[excluded, 0] = 0.82
            exclusion_rgba[excluded, 1] = 0.00
            exclusion_rgba[excluded, 2] = 0.70
            exclusion_rgba[excluded, 3] = 0.45

            ax.imshow(
                exclusion_rgba,
                extent=extent,
                origin="upper",
                interpolation="nearest",
                zorder=20,
            )

            if np.any(excluded):
                try:
                    ax.contour(
                        excluded.astype(np.uint8),
                        levels=[0.5],
                        extent=extent,
                        origin="upper",
                        colors=["#C000A8"],
                        linewidths=1.5,
                        zorder=22,
                    )
                except Exception as contour_exc:
                    logger.debug(
                        "Could not draw exclusion boundary: %s",
                        contour_exc,
                    )

            coordinate_mode = "REFERENCED MAP COORDINATES"

        # ------------------------------------------------------------
        # Titles
        # ------------------------------------------------------------
        full_title = (
            f"{title}\n{subtitle}"
            if subtitle
            else title
        )

        ax.set_title(
            full_title,
            fontsize=8.5,
            fontweight="bold",
            pad=6,
            color="#1A365D",
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=6.0,
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.30,
            color="gray",
        )

        # ------------------------------------------------------------
        # North arrow / scale bar
        # ------------------------------------------------------------
        north_available = (
            _raster_crs_kind(raster_crs)
            in {"geographic", "projected"}
            and math.isclose(
                getattr(raster_transform, "b", 0.0),
                0.0,
            )
            and math.isclose(
                getattr(raster_transform, "d", 0.0),
                0.0,
            )
        )

        draw_north_arrow(
            ax,
            available=north_available,
        )

        draw_scale_bar(
            ax,
            bounds,
            crs_str,
            crs=raster_crs,
            transform=raster_transform,
        )

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        legend_patch = mpatches.Patch(
            facecolor=(0.82, 0.00, 0.70, 0.45),
            edgecolor="#C000A8",
            linewidth=1.0,
            label="Manual exclusion (excluded from analysis)",
        )

        ax.legend(
            handles=[legend_patch],
            loc="upper right",
            fontsize=6.5,
            framealpha=0.94,
            edgecolor="#CBD5E0",
            title="Exclusion Audit",
            title_fontsize=7.0,
        )

        # ------------------------------------------------------------
        # Footer
        # ------------------------------------------------------------
        background_text = (
            "CIR source composite"
            if rgb_bg is not None
            else "Source composite unavailable"
        )

        fig.text(
            0.02,
            0.012,
            f"Grid: {width} × {height} | "
            f"CRS: {crs_str} | "
            f"Coordinate mode: {coordinate_mode} | "
            f"Base: {background_text} | "
            f"Excluded pixels: {int(excluded.sum()):,}",
            fontsize=5.4,
            color="#4A5568",
        )

        fig.tight_layout()

        fig.savefig(
            output_image_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)

        logger.info(
            "Rendered manual exclusion audit map: %s (%d excluded pixels)",
            output_image_path,
            int(excluded.sum()),
        )

        return output_image_path

    except Exception as exc:
        logger.error(
            "Failed to render manual exclusion audit map %s: %s",
            exclusion_mask_path,
            exc,
        )

        if "fig" in locals():
            plt.close(fig)

        return None

# -----------------------------------------------------------------------------
# Effective Analysis Footprint Renderer
# -----------------------------------------------------------------------------

def render_effective_analysis_mask(
    effective_mask_path: Path,
    output_image_path: Path,
    source_band_paths: Optional[Dict[str, Path]] = None,
    title: str = "Effective Analysis Footprint",
    subtitle: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.5, 4.8),
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Path]:
    """
    Render the final effective analysis mask.

    This plate is a diagnostic/provenance map: pixels retained for analysis are
    shown as a restrained green overlay while excluded/non-analysis pixels remain
    visually distinct. The source raster itself is never modified.
    """
    effective_mask_path = Path(effective_mask_path)
    output_image_path = Path(output_image_path)

    if not effective_mask_path.is_file():
        logger.warning(
            "Effective analysis mask not found: %s",
            effective_mask_path,
        )
        return None

    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(effective_mask_path) as src:
            mask = src.read(1)
            bounds = src.bounds
            raster_crs = src.crs
            raster_transform = src.transform
            crs_str = src.crs.to_string() if src.crs else "Undefined (pixel-space)"
            height, width = src.height, src.width

        effective = mask > 0

        rgb_bg = None
        if source_band_paths:
            rgb_bg, bg_bounds, _ = load_rgb_composite(source_band_paths)
            if rgb_bg is not None:
                if rgb_bg.shape[:2] != (height, width):
                    rgb_bg = None
                elif bg_bounds is not None and not (
                    math.isclose(bg_bounds.left, bounds.left)
                    and math.isclose(bg_bounds.right, bounds.right)
                    and math.isclose(bg_bounds.bottom, bounds.bottom)
                    and math.isclose(bg_bounds.top, bounds.top)
                ):
                    rgb_bg = None

        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if rgb_bg is not None:
            ax.imshow(
                rgb_bg,
                extent=extent,
                interpolation="bilinear",
            )
        else:
            ax.set_facecolor("#F7FAFC")

        rgba = np.zeros((height, width, 4), dtype=np.float32)
        rgba[effective, 0] = 0.15
        rgba[effective, 1] = 0.62
        rgba[effective, 2] = 0.25
        rgba[effective, 3] = 0.25

        ax.imshow(
            rgba,
            extent=extent,
            interpolation="nearest",
        )

        if np.any(effective) and np.any(~effective):
            try:
                ax.contour(
                    effective.astype(np.uint8),
                    levels=[0.5],
                    colors=["#176B3A"],
                    linewidths=1.2,
                    extent=extent,
                    origin="upper",
                    zorder=22,
                )
            except Exception:
                pass

        # Restrict display to AOI extent
        if aoi_bounds is not None:
            ax.set_xlim(aoi_bounds[0], aoi_bounds[1])
            ax.set_ylim(aoi_bounds[2], aoi_bounds[3])

        full_title = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(
            full_title,
            fontsize=8.5,
            fontweight="bold",
            pad=6,
            color="#1A365D",
        )
        ax.tick_params(axis="both", which="major", labelsize=6.0)
        ax.grid(True, linestyle="--", alpha=0.30, color="gray")

        north_available = (
            _raster_crs_kind(raster_crs) in {"geographic", "projected"}
            and math.isclose(getattr(raster_transform, "b", 0.0), 0.0)
            and math.isclose(getattr(raster_transform, "d", 0.0), 0.0)
        )
        draw_north_arrow(ax, available=north_available)
        draw_scale_bar(
            ax,
            bounds,
            crs_str,
            crs=raster_crs,
            transform=raster_transform,
        )

        legend_patch = mpatches.Patch(
            facecolor=(0.15, 0.62, 0.25, 0.25),
            edgecolor="#176B3A",
            linewidth=1.0,
            label="Effective analysis area",
        )
        ax.legend(
            handles=[legend_patch],
            loc="upper right",
            fontsize=6.5,
            framealpha=0.94,
            edgecolor="#CBD5E0",
            title="Analysis Footprint",
            title_fontsize=7.0,
        )

        fig.text(
            0.02,
            0.012,
            f"Grid: {width} × {height} | CRS: {crs_str} | "
            f"Effective pixels: {int(effective.sum()):,}",
            fontsize=5.4,
            color="#4A5568",
        )

        fig.tight_layout()
        fig.savefig(
            output_image_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

        logger.info(
            "Rendered effective analysis footprint map: %s (%d effective pixels)",
            output_image_path,
            int(effective.sum()),
        )
        return output_image_path

    except Exception as exc:
        logger.error(
            "Failed to render effective analysis footprint %s: %s",
            effective_mask_path,
            exc,
        )
        if "fig" in locals():
            plt.close(fig)
        return None


# -----------------------------------------------------------------------------
# Master Map Generation Pipeline
# -----------------------------------------------------------------------------

def generate_report_maps(
    run_dir: Path,
    figures_dir: Optional[Path] = None,
    dpi: int = 300,
) -> Dict[str, Path]:
    """
    Renders all available GeoTIFF rasters in a run directory into high-resolution
    cartographic map plates clipped to the effective analysis AOI.

    Cartographic AOI clipping
    -------------------------
    The effective_analysis_mask.tif (alignment/) is used to:
    1. Derive the AOI bounding box for axis limits (no source data modified).
    2. Grey out outside-AOI pixels in source composites.
    3. Mask outside-AOI pixels as NoData in categorical/continuous overlays.
    """
    run_dir = Path(run_dir)
    if figures_dir is None:
        figures_dir = run_dir / "reports" / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated_maps: Dict[str, Path] = {}

    # ---------------------------------------------------------------------------
    # Compute AOI bounds from the authoritative raster-aligned effective mask.
    # This is the single source of truth for cartographic extent restriction.
    # ---------------------------------------------------------------------------
    effective_mask_tif = run_dir / "alignment" / "effective_analysis_mask.tif"
    aoi_bounds = _get_aoi_plot_bounds(effective_mask_tif)
    if aoi_bounds is not None:
        logger.info(
            "AOI plot bounds (x_min=%.1f, x_max=%.1f, y_min=%.1f, y_max=%.1f)",
            *aoi_bounds,
        )
    else:
        logger.info("No effective analysis mask found — maps will show full extent")

    eff_mask_arg = effective_mask_tif if effective_mask_tif.is_file() else None

    # Extract source multispectral bands for T1 and T2
    t1_aligned_dir = run_dir / "alignment" / "t1_aligned"
    t2_aligned_dir = run_dir / "alignment" / "t2_aligned"

    t1_bands = {}
    for b in ["nir", "red", "green", "rededge"]:
        f = t1_aligned_dir / f"{b}.tif"
        if f.is_file():
            t1_bands[b] = f

    t2_bands = {}
    for b in ["nir", "red", "green", "rededge"]:
        f = t2_aligned_dir / f"{b}.tif"
        if f.is_file():
            t2_bands[b] = f

    # -------------------------------------------------------------------------
    # SECTION 2: Standalone Multispectral Source Composites (NIR / Red / Green CIR)
    # -------------------------------------------------------------------------
    if all(k in t1_bands for k in ["nir", "red", "green"]):
        p = render_multispectral_composite(
            t1_bands,
            figures_dir / "source_t1.png",
            title="Source Imagery — Baseline (T1)",
            band_order=("nir", "red", "green"),
            subtitle="CIR: R=NIR, G=Red, B=Green  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["source_t1"] = p

    if all(k in t2_bands for k in ["nir", "red", "green"]):
        p = render_multispectral_composite(
            t2_bands,
            figures_dir / "source_t2.png",
            title="Source Imagery — Present (T2)",
            band_order=("nir", "red", "green"),
            subtitle="CIR: R=NIR, G=Red, B=Green  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["source_t2"] = p

    # -------------------------------------------------------------------------
    # SECTION 1A: Manual Exclusion / Effective Footprint Audit
    # -------------------------------------------------------------------------
    # The exclusion mask is the authoritative rasterized representation consumed
    # by the engine. Render it directly so the final PDF can show exactly which
    # pixels were removed from analysis.
    exclusion_mask_tif = run_dir / "exclusions" / "manual_exclusion_mask.tif"
    if exclusion_mask_tif.is_file():
        p = render_manual_exclusion_audit_map(
            exclusion_mask_tif,
            figures_dir / "manual_exclusions.png",
            source_band_paths=t2_bands if t2_bands else t1_bands if t1_bands else None,
            title="Manual Exclusion Audit",
            subtitle="Authoritative rasterized exclusion mask over the analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
        )
        if p:
            generated_maps["manual_exclusions"] = p

    if effective_mask_tif.is_file():
        p = render_effective_analysis_mask(
            effective_mask_tif,
            figures_dir / "effective_analysis.png",
            source_band_paths=t2_bands if t2_bands else t1_bands if t1_bands else None,
            title="Area Selected for Analysis",
            subtitle="Effective analysis footprint — inclusion minus manual exclusions",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
        )
        if p:
            generated_maps["effective_analysis"] = p

    # -------------------------------------------------------------------------
    # SECTION 4: Vegetation Health Condition (T1 Source + Overlay, T2 Source + Overlay)
    # -------------------------------------------------------------------------
    t1_health = run_dir / "vegetation" / "t1_health.tif"
    if not t1_health.is_file() and (run_dir / "vegetation" / "health.tif").is_file():
        t1_health = run_dir / "vegetation" / "health.tif"
    if not t1_health.is_file() and (run_dir / "inference" / "t1_health.tif").is_file():
        t1_health = run_dir / "inference" / "t1_health.tif"
    if t1_health.is_file():
        p = render_categorical_map(
            t1_health,
            figures_dir / "health_t1.png",
            title="Vegetation Condition — Baseline (T1)" if (run_dir / "vegetation" / "t2_health.tif").is_file() else "Vegetation Health Condition",
            colormap_dict=HEALTH_COLORMAP_RGBA,
            class_labels=CLASS_NAMES,
            source_band_paths=t1_bands if t1_bands else None,
            alpha=0.55,
            filter_active_classes=False,
            subtitle="ResNet34 U-Net classification  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["health_t1"] = p
            generated_maps["health"] = p

    t2_health = run_dir / "vegetation" / "t2_health.tif"
    if not t2_health.is_file() and (run_dir / "inference" / "t2_health.tif").is_file():
        t2_health = run_dir / "inference" / "t2_health.tif"
    if t2_health.is_file():
        p = render_categorical_map(
            t2_health,
            figures_dir / "health_t2.png",
            title="Vegetation Condition — Present (T2)",
            colormap_dict=HEALTH_COLORMAP_RGBA,
            class_labels=CLASS_NAMES,
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            filter_active_classes=False,
            subtitle="ResNet34 U-Net classification  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["health_t2"] = p

    # Continuous Confidence surface
    conf_tif = run_dir / "vegetation" / "confidence.tif"
    if not conf_tif.is_file() and (run_dir / "inference" / "confidence.tif").is_file():
        conf_tif = run_dir / "inference" / "confidence.tif"
    if conf_tif.is_file():
        p = render_continuous_map(
            conf_tif,
            figures_dir / "confidence.png",
            title="Classification Confidence",
            colorbar_label="Confidence",
            source_band_paths=t1_bands if t1_bands else None,
            alpha=0.60,
            cmap_name="viridis",
            vmin=0.0,
            vmax=1.0,
            subtitle="Model prediction confidence  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["confidence"] = p

    # -------------------------------------------------------------------------
    # SECTION 5: 16-State Post-Classification Transitions (T2 Source + Transition Overlay)
    # -------------------------------------------------------------------------
    trans_tif = run_dir / "change" / "transition_map.tif"
    if trans_tif.is_file():
        trans_labels = {
            0: "B→B (Stable Bg)", 1: "B→S (Bg→Severe)", 2: "B→M (Bg→Mod)", 3: "B→H (Bg→Healthy)",
            4: "S→B (Canopy Loss)", 5: "S→S (Stable Sev)", 6: "S→M (Recovery)", 7: "S→H (Recovery)",
            8: "M→B (Canopy Loss)", 9: "M→S (Degradation)", 10: "M→M (Stable Mod)", 11: "M→H (Recovery)",
            12: "H→B (Canopy Loss)", 13: "H→S (Degradation)", 14: "H→M (Degradation)", 15: "H→H (Stable Healthy)",
        }
        trans_colors = {
            0: (0.6, 0.6, 0.6, 1.0),
            1: (0.4, 0.7, 0.9, 1.0), 2: (0.3, 0.6, 0.8, 1.0), 3: (0.2, 0.5, 0.9, 1.0),
            4: (0.9, 0.6, 0.2, 1.0), 5: (0.8, 0.3, 0.3, 1.0), 6: (0.4, 0.8, 0.4, 1.0), 7: (0.2, 0.7, 0.2, 1.0),
            8: (0.9, 0.5, 0.1, 1.0), 9: (0.85, 0.2, 0.2, 1.0), 10: (0.9, 0.7, 0.2, 1.0), 11: (0.3, 0.8, 0.3, 1.0),
            12: (0.95, 0.4, 0.1, 1.0), 13: (0.9, 0.1, 0.1, 1.0), 14: (0.95, 0.3, 0.2, 1.0), 15: (0.1, 0.65, 0.1, 1.0),
        }
        p = render_categorical_map(
            trans_tif,
            figures_dir / "transition.png",
            title="Vegetation Change Transitions",
            colormap_dict=trans_colors,
            class_labels=trans_labels,
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            filter_active_classes=True,
            subtitle="Post-classification transitions  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["transition"] = p

    # Degradation, Recovery, Canopy Loss
    deg_tif = run_dir / "degradation" / "degradation_map.tif"
    if deg_tif.is_file():
        p = render_categorical_map(
            deg_tif,
            figures_dir / "degradation.png",
            title="Vegetation Degradation (H→M, H→S, M→S)",
            colormap_dict=DEGRADATION_COLORS,
            class_labels={0: "Other / Unchanged", 1: "Degradation"},
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.60,
            filter_active_classes=False,
            subtitle="Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["degradation"] = p

    rec_tif = run_dir / "degradation" / "recovery_map.tif"
    if rec_tif.is_file():
        p = render_categorical_map(
            rec_tif,
            figures_dir / "recovery.png",
            title="Vegetation Recovery (M→H, S→M, S→H)",
            colormap_dict=RECOVERY_COLORS,
            class_labels={0: "Other / Unchanged", 1: "Recovery"},
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.60,
            filter_active_classes=False,
            subtitle="Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["recovery"] = p

    canopy_tif = run_dir / "degradation" / "canopy_loss_map.tif"
    if canopy_tif.is_file():
        p = render_categorical_map(
            canopy_tif,
            figures_dir / "canopy_loss.png",
            title="Canopy Loss (Vegetation → Background)",
            colormap_dict=CANOPY_LOSS_COLORS,
            class_labels={0: "Other / Retained", 1: "Canopy Loss"},
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.60,
            filter_active_classes=False,
            subtitle="Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["canopy_loss"] = p

    # -------------------------------------------------------------------------
    # SECTION 6: Continuous Change Vector Analysis (CVA) & Spectral Differences
    # -------------------------------------------------------------------------
    cva_tif = run_dir / "change" / "cva_magnitude.tif"
    if cva_tif.is_file():
        p = render_continuous_map(
            cva_tif,
            figures_dir / "cva_magnitude.png",
            title="Change Vector Analysis (CVA) Magnitude",
            colorbar_label="||ΔV|| Magnitude",
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            cmap_name="YlOrRd",
            subtitle="Spectral trajectory magnitude  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["cva_magnitude"] = p

    d_ndvi = run_dir / "change" / "delta_ndvi.tif"
    if d_ndvi.is_file():
        p = render_continuous_map(
            d_ndvi,
            figures_dir / "delta_ndvi.png",
            title="ΔNDVI  (T2 − T1)",
            colorbar_label="ΔNDVI",
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            cmap_name="RdYlGn",
            vmin=-0.5,
            vmax=0.5,
            subtitle="Photosynthetic vigour change  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["delta_ndvi"] = p

    d_gndvi = run_dir / "change" / "delta_gndvi.tif"
    if d_gndvi.is_file():
        p = render_continuous_map(
            d_gndvi,
            figures_dir / "delta_gndvi.png",
            title="ΔGNDVI  (T2 − T1)",
            colorbar_label="ΔGNDVI",
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            cmap_name="RdYlGn",
            vmin=-0.5,
            vmax=0.5,
            subtitle="Green reflectance / canopy density change  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["delta_gndvi"] = p

    d_ndre = run_dir / "change" / "delta_ndre.tif"
    if d_ndre.is_file():
        p = render_continuous_map(
            d_ndre,
            figures_dir / "delta_ndre.png",
            title="ΔNDRE  (T2 − T1)",
            colorbar_label="ΔNDRE",
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            cmap_name="RdYlGn",
            vmin=-0.5,
            vmax=0.5,
            subtitle="Red-edge / foliar condition change  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["delta_ndre"] = p

    # -------------------------------------------------------------------------
    # SECTION 7: Hydrological Evidence (NDWI & Surface Water Dynamics)
    # -------------------------------------------------------------------------
    ndwi_t1 = run_dir / "hydrology" / "ndwi_t1.tif"
    if ndwi_t1.is_file():
        p = render_continuous_map(
            ndwi_t1,
            figures_dir / "ndwi_t1.png",
            title="NDWI — Baseline (T1)",
            colorbar_label="NDWI",
            source_band_paths=t1_bands if t1_bands else None,
            alpha=0.55,
            cmap_name="Blues",
            vmin=-0.5,
            vmax=0.5,
            subtitle="Surface moisture  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["ndwi_t1"] = p

    ndwi_t2 = run_dir / "hydrology" / "ndwi_t2.tif"
    if ndwi_t2.is_file():
        p = render_continuous_map(
            ndwi_t2,
            figures_dir / "ndwi_t2.png",
            title="NDWI — Present (T2)",
            colorbar_label="NDWI",
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            cmap_name="Blues",
            vmin=-0.5,
            vmax=0.5,
            subtitle="Surface moisture  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["ndwi_t2"] = p

    water_change_tif = run_dir / "hydrology" / "water_change.tif"
    if water_change_tif.is_file():
        p = render_categorical_map(
            water_change_tif,
            figures_dir / "water_change.png",
            title="Surface Water Dynamics (NDWI Change)",
            colormap_dict=WATER_COLORMAP_RGBA,
            class_labels=WATER_CLASS_NAMES,
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            filter_active_classes=True,
            subtitle="Surface water shifts  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["water_change"] = p

    # -------------------------------------------------------------------------
    # SECTION 8: Multi-Evidence Fusion (T2 Source + Fusion Overlay)
    # -------------------------------------------------------------------------
    fusion_tif = run_dir / "fusion" / "evidence_fusion.tif"
    if fusion_tif.is_file():
        p = render_categorical_map(
            fusion_tif,
            figures_dir / "evidence_fusion.png",
            title="Multi-Evidence Fusion Map",
            colormap_dict=FUSION_COLORMAP_RGBA,
            class_labels=FUSION_CATEGORY_NAMES,
            source_band_paths=t2_bands if t2_bands else None,
            alpha=0.55,
            filter_active_classes=True,
            subtitle="Vegetation + water evidence combined  ·  Clipped to analysis area",
            dpi=dpi,
            aoi_bounds=aoi_bounds,
            effective_mask_path=eff_mask_arg,
        )
        if p:
            generated_maps["evidence_fusion"] = p

    logger.info("Generated %d cartographic map plates in %s", len(generated_maps), figures_dir)
    return generated_maps