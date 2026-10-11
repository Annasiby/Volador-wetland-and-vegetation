"""
Physical pixel-area utilities for the GIS Wetland Engine.

Important scientific rule:
- A raster with no CRS is treated as pixel-space, not metres.
- Physical area in square metres/hectares is therefore unavailable unless
  the raster has an authoritative CRS and transform.
- This module never silently assumes 1 pixel = 1 m².
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from typing import Tuple, Optional


class PhysicalAreaUnavailableError(ValueError):
    """Raised when physical pixel area cannot be established reliably."""


def _is_identity_transform(transform: rasterio.transform.Affine) -> bool:
    """Return True for an identity affine transform."""
    if transform is None:
        return True
    identity = rasterio.transform.Affine.identity()
    return transform == identity


def _projected_pixel_area_m2(
    transform: rasterio.transform.Affine,
    crs: CRS,
) -> float:
    """
    Calculate projected pixel area in m².

    The affine determinant gives the area of one pixel in the CRS coordinate
    units, including rotated/sheared grids.
    """
    determinant = abs(
        transform.a * transform.e
        - transform.b * transform.d
    )

    if not np.isfinite(determinant) or determinant <= 0:
        raise PhysicalAreaUnavailableError(
            "Raster transform does not define a valid positive pixel area."
        )

    # Rasterio/PROJ CRS axis units. Avoid a pyproj dependency.
    units = str(getattr(crs, "linear_units", "") or "").lower()

    if units in ("metre", "meter", "metres", "meters", "m"):
        return float(determinant)

    if units in ("foot", "feet", "ft"):
        return float(determinant) * 0.3048 ** 2

    if units in (
        "us survey foot",
        "us survey feet",
        "us_survey_foot",
        "us_survey_feet",
    ):
        return float(determinant) * 0.3048006096012192 ** 2

    # Do not guess unknown projected units.
    raise PhysicalAreaUnavailableError(
        f"Unsupported projected CRS linear units: {units or 'unknown'}."
    )


def _geographic_pixel_areas(
    transform: rasterio.transform.Affine,
    shape: Tuple[int, int],
    crs: CRS,
) -> np.ndarray:
    """
    Calculate ellipsoidal pixel areas for a geographic CRS.

    Uses CRS ellipsoid parameters exposed by Rasterio/GDAL when available.
    Falls back to WGS84 only when the CRS explicitly identifies WGS84.
    """
    height, width = shape

    # Rasterio CRS exposes ellipsoid-related information on supported builds.
    # Keep this dependency-free: no pyproj import.
    try:
        semi_major = float(crs.ellipsoid.semi_major_metre)
        semi_minor = float(crs.ellipsoid.semi_minor_metre)
    except Exception:
        # Explicit WGS84 fallback only.
        try:
            is_wgs84 = bool(crs.to_epsg() == 4326)
        except Exception:
            is_wgs84 = False

        if not is_wgs84:
            raise PhysicalAreaUnavailableError(
                "Cannot determine ellipsoid parameters for the geographic CRS."
            )

        semi_major = 6378137.0
        semi_minor = 6356752.314245179

    if not (
        np.isfinite(semi_major)
        and np.isfinite(semi_minor)
        and semi_major > 0
        and semi_minor > 0
    ):
        raise PhysicalAreaUnavailableError(
            "Geographic CRS has invalid ellipsoid parameters."
        )

    # This area routine assumes north-up geographic pixels. A rotated/sheared
    # geographic grid requires a geodesic polygon calculation, which this
    # lightweight utility intentionally does not approximate.
    if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
        raise PhysicalAreaUnavailableError(
            "Rotated/sheared geographic grids are not supported for physical "
            "area calculation."
        )

    dx_deg = abs(float(transform.a))
    dy_deg = abs(float(transform.e))

    if dx_deg <= 0 or dy_deg <= 0:
        raise PhysicalAreaUnavailableError(
            "Geographic raster has invalid pixel dimensions."
        )

    # Authalic/ellipsoidal surface strip formulation.
    e2 = 1.0 - (semi_minor / semi_major) ** 2
    if e2 <= 0 or e2 >= 1:
        raise PhysicalAreaUnavailableError(
            "Invalid ellipsoid eccentricity."
        )

    e = np.sqrt(e2)
    dx_rad = dx_deg * np.pi / 180.0

    row_indices = np.arange(height, dtype=np.float64)
    lat_top = float(transform.f) + row_indices * float(transform.e)
    lat_bottom = lat_top + float(transform.e)

    # Geographic latitude must stay inside the valid range.
    if (
        np.any(lat_top < -90.0)
        or np.any(lat_top > 90.0)
        or np.any(lat_bottom < -90.0)
        or np.any(lat_bottom > 90.0)
    ):
        raise PhysicalAreaUnavailableError(
            "Geographic raster latitude is outside the valid [-90, 90] range."
        )

    lat_top_rad = lat_top * np.pi / 180.0
    lat_bottom_rad = lat_bottom * np.pi / 180.0

    def q(lat_rad: np.ndarray) -> np.ndarray:
        sin_lat = np.sin(lat_rad)
        term1 = sin_lat / (1.0 - e2 * sin_lat ** 2)
        term2 = (
            1.0 / (2.0 * e)
        ) * np.log(
            (1.0 + e * sin_lat) /
            (1.0 - e * sin_lat)
        )
        return term1 + term2

    row_areas = (
        (semi_minor ** 2)
        * dx_rad
        / 2.0
        * np.abs(q(lat_top_rad) - q(lat_bottom_rad))
    )

    if not np.all(np.isfinite(row_areas)) or np.any(row_areas <= 0):
        raise PhysicalAreaUnavailableError(
            "Calculated geographic pixel areas are invalid."
        )

    return np.tile(row_areas[:, np.newaxis], (1, width))


def calculate_pixel_areas(
    transform: rasterio.transform.Affine,
    shape: Tuple[int, int],
    crs: Optional[CRS] = None,
) -> np.ndarray:
    """
    Calculate the physical area of every raster pixel in square metres.

    Parameters
    ----------
    transform:
        Raster affine transform.
    shape:
        (height, width).
    crs:
        Raster CRS.

    Returns
    -------
    np.ndarray
        2D float64 array containing pixel areas in m².

    Raises
    ------
    PhysicalAreaUnavailableError
        If physical area cannot be established from authoritative spatial
        reference information.

    Notes
    -----
    CRS=None is intentionally NOT interpreted as metres.
    Therefore a CRS-less identity/pixel-space raster cannot produce hectares.
    """
    height, width = shape

    if height <= 0 or width <= 0:
        raise ValueError("Raster shape must contain positive dimensions.")

    if transform is None:
        raise PhysicalAreaUnavailableError(
            "Physical area unavailable: raster has no affine transform."
        )

    if crs is None:
        raise PhysicalAreaUnavailableError(
            "Physical area unavailable: raster has no CRS. "
            "Pixel-space data cannot be converted to m² without authoritative "
            "ground sampling information."
        )

    if _is_identity_transform(transform):
        raise PhysicalAreaUnavailableError(
            "Physical area unavailable: raster uses an identity transform. "
            "This indicates pixel-space/unreferenced data unless an explicit "
            "ground sampling distance has been supplied separately."
        )

    if getattr(crs, "is_geographic", False):
        return _geographic_pixel_areas(transform, (height, width), crs)

    return np.full(
        (height, width),
        _projected_pixel_area_m2(transform, crs),
        dtype=np.float64,
    )
