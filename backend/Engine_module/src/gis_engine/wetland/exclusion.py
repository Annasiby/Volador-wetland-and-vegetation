"""
GIS manual exclusion handling.

Converts user-defined GeoJSON exclusion polygons into a raster mask
on the authoritative aligned analysis grid.

Mask convention:
    0 = not excluded
    1 = excluded
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom, transform_bounds
from affine import Affine


def _flip_y_coords(coords, height: float):
    """
    Recursively flip the Y axis of a GeoJSON coordinate array.

    Why this is needed:
        The frontend draws exclusions in an OpenLayers view that uses a
        custom "GIS-PIXEL" projection with extent [0, 0, width, height].
        OpenLayers map coordinates are always Cartesian (Y increases
        upward), so a feature drawn near the top of the on-screen raster
        has a *large* Y value in the GeoJSON that gets sent to the API.

        Raster/rasterio pixel space is the opposite: row 0 is the top of
        the image and row index increases downward.

        Without this flip, rasterize() (called with the grid's identity
        transform) reads the OL Y value directly as a row index, placing
        the exclusion mask vertically mirrored relative to where the user
        actually drew it.

    Parameters
    ----------
    coords:
        A GeoJSON `coordinates` array (arbitrarily nested — this handles
        Polygon rings and MultiPolygon ring-lists alike).
    height:
        The authoritative raster grid height (pixels), used as the axis
        to flip around.

    Returns
    -------
    The same nested structure, with every [x, y, ...] pair replaced by
    [x, height - y, ...].
    """
    if not coords:
        return coords

    # A coordinate pair looks like [x, y] (occasionally [x, y, z]).
    # Detect this base case by checking whether the first element is a
    # number rather than another nested list.
    if isinstance(coords[0], (int, float)):
        x, y = coords[0], coords[1]
        return [x, height - y, *coords[2:]]

    return [_flip_y_coords(ring, height) for ring in coords]


def _flip_geometry_y(geometry: Dict[str, Any], height: float) -> Dict[str, Any]:
    """Return a copy of a GeoJSON geometry with its Y axis flipped."""
    return {
        **geometry,
        "coordinates": _flip_y_coords(geometry["coordinates"], height),
    }

def _save_exclusion_debug_image(
    geometry: Dict[str, Any],
    output_path: Path,
    width: int,
    height: int,
    title: str,
) -> None:
    """Save a visual diagnostic of exclusion geometry in pixel space."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(
            figsize=(8, 6),
            dpi=150,
        )

        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")

        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            plt.close(fig)
            return

        for polygon in polygons:
            for ring in polygon:
                if not ring:
                    continue

                xs = [point[0] for point in ring]
                ys = [point[1] for point in ring]

                ax.plot(xs, ys, linewidth=2)
                ax.fill(xs, ys, alpha=0.30)

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_title(title)

        # Pixel-space diagnostic:
        # x increases left -> right.
        # y increases top -> bottom.
        ax.invert_yaxis()

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.tight_layout()
        fig.savefig(
            output_path,
            bbox_inches="tight",
        )
        plt.close(fig)

    except Exception:
        logger.exception(
            "Failed to save exclusion debug image: %s",
            output_path,
        )


def _extract_first_coord(coords: Any) -> Optional[Tuple[float, float]]:
    if not coords:
        return None
    if isinstance(coords[0], (int, float)):
        return float(coords[0]), float(coords[1])
    return _extract_first_coord(coords[0])


def _transform_pixel_coords_to_grid(coords: Any, height: float, transform: Any) -> Any:
    if not coords:
        return coords
    if isinstance(coords[0], (int, float)):
        x, y = float(coords[0]), float(coords[1])
        y_flipped = height - y
        if transform is not None:
            tx, ty = transform * (x, y_flipped)
            return [tx, ty, *coords[2:]]
        return [x, y_flipped, *coords[2:]]
    return [_transform_pixel_coords_to_grid(ring, height, transform) for ring in coords]


def prepare_geometry_for_grid(
    geometry: Dict[str, Any],
    reference_grid: Any,
) -> Dict[str, Any]:
    """
    Prepares a GeoJSON geometry for rasterization on reference_grid.
    Detects whether geometry coordinates are:
    1. Already in reference_grid.bounds (CRS coordinates)
    2. In EPSG:4326 (WGS84 geographic)
    3. In OpenLayers pixel coordinates [0, 0, width, height] (Y-up Cartesian)
    and maps them to reference_grid coordinates appropriately.
    """
    height = int(reference_grid.height)
    first_pt = _extract_first_coord(geometry.get("coordinates"))

    if reference_grid.crs is None:
        return _flip_geometry_y(geometry, height)

    if first_pt is not None:
        x0, y0 = first_pt
        # Check if already in reference_grid bounds
        bounds = getattr(reference_grid, "bounds", None)
        if bounds is not None:
            b_left, b_bottom, b_right, b_top = bounds
            min_x, max_x = min(b_left, b_right), max(b_left, b_right)
            min_y, max_y = min(b_bottom, b_top), max(b_bottom, b_top)
            if min_x <= x0 <= max_x and min_y <= y0 <= max_y:
                return geometry

        # Check if WGS84 geographic coordinates
        if -180.0 <= x0 <= 180.0 and -90.0 <= y0 <= 90.0:
            try:
                w_left, w_bottom, w_right, w_top = transform_bounds(
                    reference_grid.crs, "EPSG:4326", *bounds
                )
                pad_x = max(0.05, abs(w_right - w_left) * 0.2)
                pad_y = max(0.05, abs(w_top - w_bottom) * 0.2)
                if (min(w_left, w_right) - pad_x <= x0 <= max(w_left, w_right) + pad_x) and \
                   (min(w_bottom, w_top) - pad_y <= y0 <= max(w_bottom, w_top) + pad_y):
                    return transform_geom(
                        "EPSG:4326",
                        reference_grid.crs,
                        geometry,
                        precision=15,
                    )
            except Exception:
                pass

    # Pixel coordinates from OpenLayers canvas:
    # Y-up Cartesian pixel -> Y-down raster row -> rasterio transform to CRS
    return {
        **geometry,
        "coordinates": _transform_pixel_coords_to_grid(
            geometry["coordinates"],
            height,
            reference_grid.transform,
        ),
    }


def build_manual_exclusion_mask(
    exclusions: Optional[List[Dict[str, Any]]],
    reference_grid: Any,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Rasterize user-defined exclusion geometries onto the reference grid.

    Parameters
    ----------
    exclusions:
        List of exclusion dictionaries. Each item must contain:
            - geometry: GeoJSON geometry dictionary
        Optional fields such as id, category and comment are preserved
        by the caller but are not needed for rasterization.

    reference_grid:
        Authoritative aligned analysis grid. Must provide:
            - width
            - height
            - transform
            - crs

    output_path:
        Optional path for writing manual_exclusion_mask.tif.

    Returns
    -------
    np.ndarray
        uint8 mask:
            0 = included
            1 = manually excluded
    """
    height = int(reference_grid.height)
    width = int(reference_grid.width)

    pixel_space_mode = reference_grid.crs is None

    if pixel_space_mode:
        if reference_grid.transform != Affine.identity():
            raise ValueError(
                "Manual exclusions cannot use an unreferenced analysis grid "
                "unless its transform is identity/pixel-space."
            )

    exclusion_mask = np.zeros((height, width), dtype=np.uint8)

    if not exclusions:
        _write_mask_if_requested(
            exclusion_mask,
            reference_grid,
            output_path,
        )
        return exclusion_mask

    shapes = []

    for exclusion in exclusions:
        if not isinstance(exclusion, dict):
            raise ValueError("Each manual exclusion must be an object.")

        geometry = exclusion.get("geometry")

        if not isinstance(geometry, dict):
            raise ValueError(
                f"Manual exclusion '{exclusion.get('id', 'unknown')}' "
                "does not contain a valid GeoJSON geometry object."
            )

        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")

        if not geometry_type or coordinates is None:
            raise ValueError(
                f"Manual exclusion '{exclusion.get('id', 'unknown')}' "
                "contains incomplete GeoJSON geometry."
            )

        if geometry_type not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError(
                f"Manual exclusion '{exclusion.get('id', 'unknown')}' "
                f"uses unsupported geometry type '{geometry_type}'. "
                "Only Polygon and MultiPolygon are supported."
            )

        debug_dir = None
        if output_path is not None:
            debug_dir = (
                Path(output_path).parent.parent
                / "debug_exclusion"
            )
            _save_exclusion_debug_image(
                geometry=geometry,
                output_path=debug_dir / "01_before_backend_y_flip.png",
                width=width,
                height=height,
                title="01 - Geometry Before Backend Y Flip",
            )

        transformed_geometry = prepare_geometry_for_grid(geometry, reference_grid)

        if debug_dir is not None:
            _save_exclusion_debug_image(
                geometry=transformed_geometry,
                output_path=debug_dir / "02_after_backend_y_flip.png",
                width=width,
                height=height,
                title="02 - Geometry After Backend Y Flip",
            )

        shapes.append((transformed_geometry, 1))

    exclusion_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=reference_grid.transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=False,
    )

    # ------------------------------------------------------------
    # DEBUG CHECKPOINT 03
    # Actual authoritative rasterized exclusion mask.
    # ------------------------------------------------------------
    if output_path is not None:
        debug_dir = (
            Path(output_path).parent.parent
            / "debug_exclusion"
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig, ax = plt.subplots(
            figsize=(8, 6),
            dpi=150,
        )

        ax.imshow(
            exclusion_mask,
            origin="upper",
            interpolation="nearest",
        )

        ax.set_title(
            "03 - Manual Exclusion Mask After Rasterization"
        )

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        fig.tight_layout()

        fig.savefig(
            debug_dir / "03_manual_exclusion_mask.png",
            bbox_inches="tight",
        )

        plt.close(fig)

    _write_mask_if_requested(
        exclusion_mask,
        reference_grid,
        output_path,
    )

    return exclusion_mask


def build_inclusion_mask(
    geometry: Optional[Dict[str, Any]],
    reference_grid: Any,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Rasterize user-defined inclusion geometry onto the reference grid.

    Returns
    -------
    np.ndarray
        uint8 mask:
            1 = inside inclusion AOI
            0 = outside inclusion AOI
        If geometry is None, returns an all-ones mask (full grid included).
    """
    height = int(reference_grid.height)
    width = int(reference_grid.width)

    if geometry is None:
        inclusion_mask = np.ones((height, width), dtype=np.uint8)
        _write_mask_if_requested(
            inclusion_mask,
            reference_grid,
            output_path,
        )
        return inclusion_mask

    if not isinstance(geometry, dict) or "type" not in geometry or "coordinates" not in geometry:
        raise ValueError("Invalid GeoJSON geometry for inclusion area.")

    geom_type = geometry.get("type")
    if geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(
            f"Unsupported inclusion geometry type '{geom_type}'. "
            "Only Polygon and MultiPolygon are supported."
        )

    pixel_space_mode = reference_grid.crs is None
    if pixel_space_mode:
        if reference_grid.transform != Affine.identity():
            raise ValueError(
                "Inclusion area cannot use an unreferenced analysis grid "
                "unless its transform is identity/pixel-space."
            )

    transformed_geometry = prepare_geometry_for_grid(geometry, reference_grid)

    inclusion_mask = rasterize(
        shapes=[(transformed_geometry, 1)],
        out_shape=(height, width),
        transform=reference_grid.transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=False,
    )

    _write_mask_if_requested(
        inclusion_mask,
        reference_grid,
        output_path,
    )

    return inclusion_mask


def _write_mask_if_requested(
    mask: np.ndarray,
    reference_grid: Any,
    output_path: Optional[Union[str, Path]],
) -> None:
    """Write the mask GeoTIFF using the authoritative reference grid."""
    if output_path is None:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": int(reference_grid.height),
        "width": int(reference_grid.width),
        "count": 1,
        "dtype": "uint8",
        "crs": reference_grid.crs,
        "transform": reference_grid.transform,
        "nodata": 0,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)