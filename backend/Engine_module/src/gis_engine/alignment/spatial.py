import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from ..ingestion.metadata import RasterMetadata

class SpatialOverlapError(ValueError):
    """Raised when there is no geographic overlap between T1 and T2 rasters."""
    pass

@dataclass
class ReferenceGrid:
    """
    Represents the common spatial grid geometry.
    """
    crs: Optional[CRS]
    transform: Affine
    width: int
    height: int
    resolution: Tuple[float, float]
    bounds: Tuple[float, float, float, float]  # left, bottom, right, top

    @classmethod
    def from_metadata(cls, metadata: RasterMetadata) -> "ReferenceGrid":
        if metadata.transform is None:
            raise ValueError("Reference metadata is missing transform.")
        return cls(
            crs=metadata.crs,
            transform=metadata.transform,
            width=metadata.width,
            height=metadata.height,
            resolution=metadata.resolution,
            bounds=metadata.extent
        )

    def to_dict(self) -> Dict[str, Any]:
        transform_list = [
            self.transform.a, self.transform.b, self.transform.c,
            self.transform.d, self.transform.e, self.transform.f
        ] if self.transform is not None else None
        return {
            "crs": str(self.crs) if self.crs is not None else None,
            "transform": transform_list,
            "width": self.width,
            "height": self.height,
            "resolution": list(self.resolution) if self.resolution is not None else None,
            "bounds": list(self.bounds) if self.bounds is not None else None
        }

class SpatialAligner:
    """
    Handles spatial validation, comparison, overlap checks, and reprojection/resampling.
    """
    def __init__(self, resampling_method: str = "bilinear"):
        self.resampling_method = resampling_method.lower()
        self._resampling_enum = self._get_resampling_enum(self.resampling_method)

    def _get_resampling_enum(self, name: str) -> Resampling:
        mapping = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "cubic_spline": Resampling.cubic_spline,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
            "mode": Resampling.mode,
            "gauss": Resampling.gauss
        }
        if name not in mapping:
            raise ValueError(
                f"Unsupported resampling method: '{name}'. "
                f"Supported methods: {list(mapping.keys())}"
            )
        return mapping[name]

    def validate_date_bands(self, bands_metadata: Dict[str, RasterMetadata]) -> bool:
        """
        Validates that all bands within a single date share the exact same spatial grid.
        Raises ValueError if there is any mismatch in dimensions, CRS, transform, resolution, or bounds.
        """
        if not bands_metadata:
            raise ValueError("No band metadata supplied for verification.")

        ref_name = next(iter(bands_metadata.keys()))
        ref = bands_metadata[ref_name]

        for name, meta in bands_metadata.items():
            if name == ref_name:
                continue

            # CRS check
            if ref.crs != meta.crs:
                raise ValueError(
                    f"Within-date CRS mismatch: '{ref_name}' ({ref.crs}) vs '{name}' ({meta.crs})"
                )

            # Dimensions check
            if ref.width != meta.width or ref.height != meta.height:
                raise ValueError(
                    f"Within-date shape mismatch: '{ref_name}' ({ref.height}x{ref.width}) vs "
                    f"'{name}' ({meta.height}x{meta.width})"
                )

            # Transform check
            if not np.allclose(ref.transform, meta.transform, atol=1e-5):
                raise ValueError(
                    f"Within-date transform mismatch: '{ref_name}' vs '{name}'"
                )

            # Resolution check
            if not np.allclose(ref.resolution, meta.resolution, atol=1e-5):
                raise ValueError(
                    f"Within-date resolution mismatch: '{ref_name}' vs '{name}'"
                )

            # Bounds check
            if not np.allclose(ref.extent, meta.extent, atol=1e-5):
                raise ValueError(
                    f"Within-date bounds mismatch: '{ref_name}' vs '{name}'"
                )

        return True

    def check_grids_equal(self, grid1: ReferenceGrid, grid2: ReferenceGrid) -> bool:
        """
        Compares two ReferenceGrids for exact spatial equivalence.
        """
        if grid1.crs != grid2.crs:
            return False
        if grid1.width != grid2.width or grid1.height != grid2.height:
            return False
        if not np.allclose(grid1.transform, grid2.transform, atol=1e-5):
            return False
        if not np.allclose(grid1.resolution, grid2.resolution, atol=1e-5):
            return False
        if not np.allclose(grid1.bounds, grid2.bounds, atol=1e-5):
            return False
        return True

    def calculate_overlap(self, t1_grid: ReferenceGrid, t2_grid: ReferenceGrid) -> Tuple[float, Tuple[float, float, float, float]]:
        """
        Calculates spatial overlap bounds and percentage of T1 grid coverage in T1's CRS.
        Raises SpatialOverlapError if there is no intersection.
        """
        # Project T2 bounds into T1 CRS
        if t1_grid.crs == t2_grid.crs:
            t2_bounds_in_t1 = t2_grid.bounds
        elif t1_grid.crs is None or t2_grid.crs is None:
            raise ValueError("CRS mismatch: one grid has CRS defined while the other does not.")
        else:
            try:
                t2_bounds_in_t1 = transform_bounds(
                    t2_grid.crs,
                    t1_grid.crs,
                    *t2_grid.bounds
                )
            except Exception as e:
                raise ValueError(f"Failed to transform bounds between CRSs: {e}")

        # Compute geographic intersection
        t1_left, t1_bottom, t1_right, t1_top = t1_grid.bounds
        t2_left, t2_bottom, t2_right, t2_top = t2_bounds_in_t1

        t1_y1, t1_y2 = min(t1_bottom, t1_top), max(t1_bottom, t1_top)
        t2_y1, t2_y2 = min(t2_bottom, t2_top), max(t2_bottom, t2_top)

        intersect_left = max(t1_left, t2_left)
        intersect_right = min(t1_right, t2_right)
        intersect_bottom = max(t1_y1, t2_y1)
        intersect_top = min(t1_y2, t2_y2)

        if intersect_left >= intersect_right or intersect_bottom >= intersect_top:
            raise SpatialOverlapError("T1 and T2 have no geographic intersection.")

        # Overlap percentage of T1's grid
        t1_area = (t1_right - t1_left) * (t1_y2 - t1_y1)
        overlap_area = (intersect_right - intersect_left) * (intersect_top - intersect_bottom)

        if t1_area <= 0:
            raise ValueError("Reference grid T1 has zero or negative area bounds.")

        overlap_pct = (overlap_area / t1_area) * 100.0
        # Cap overlap_pct at 100% just in case of float inaccuracy
        overlap_pct = min(100.0, max(0.0, overlap_pct))

        return overlap_pct, (intersect_left, intersect_bottom, intersect_right, intersect_top)

    def reproject_band(
        self,
        src_array: np.ndarray,
        src_transform: Affine,
        src_crs: CRS,
        dst_transform: Affine,
        dst_crs: CRS,
        dst_shape: Tuple[int, int],
        src_nodata: Optional[float] = None,
        dst_nodata: Optional[float] = None
    ) -> np.ndarray:
        """
        Reprojects a 2D numpy array to the destination coordinate grid.
        """
        dst_array = np.zeros(dst_shape, dtype=src_array.dtype)

        # Use explicitly provided nodata, default to np.nan if float, otherwise 0
        if dst_nodata is None:
            if np.issubdtype(src_array.dtype, np.floating):
                dst_nodata = np.nan
            else:
                dst_nodata = 0

        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            resampling=self._resampling_enum
        )

        return dst_array

    @staticmethod
    def get_validity_mask(array: np.ndarray, nodata: Optional[float]) -> np.ndarray:
        """
        Generates a boolean mask indicating where raster data is valid (i.e. not NoData and is finite).
        """
        mask = np.isfinite(array)
        if nodata is not None:
            if np.isnan(nodata):
                mask &= ~np.isnan(array)
            else:
                mask &= (array != nodata)
        return mask
