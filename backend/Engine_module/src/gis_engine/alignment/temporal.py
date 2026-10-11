import re
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import rasterio

from ..ingestion.loader import RasterLoader
from ..ingestion.metadata import RasterMetadata
from .spatial import ReferenceGrid, SpatialAligner, SpatialOverlapError

@dataclass
class AlignedPair:
    """
    Represents the output structure for aligned T1/T2 pairs.
    """
    t1_data: Dict[str, np.ndarray]
    t2_data: Dict[str, np.ndarray]
    t1_valid_mask: np.ndarray
    t2_valid_mask: np.ndarray
    overlap_mask: np.ndarray
    reference_grid: ReferenceGrid
    t1_metadata: Dict[str, Any]
    t2_original_metadata: Dict[str, Any]
    t2_aligned_metadata: Dict[str, Any]
    alignment_required: bool
    alignment_operations: List[str]
    overlap_percentage: float

logger = logging.getLogger(__name__)


class TemporalAligner:
    """
    Orchestrates the spatial-temporal alignment of two multispectral acquisitions.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.band_mapping = config["ingestion"]["band_mapping"]
        self.loader = RasterLoader(self.band_mapping)
        # Honor require_crs from preprocessing config (default False)
        self.require_crs = config.get("preprocessing", {}).get("require_crs", False)

        # Get resampling method from config, default to bilinear
        alignment_config = config.get("alignment", {})
        self.resampling_method = alignment_config.get("resampling_method", "bilinear")
        self.spatial_aligner = SpatialAligner(resampling_method=self.resampling_method)

    def _validate_date(self, date_str: str) -> None:
        """
        Validates date format is YYYY-MM-DD and is a valid calendar date.
        """
        if not isinstance(date_str, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            raise ValueError(f"Invalid date format: '{date_str}'. Expected 'YYYY-MM-DD'.")
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid calendar date: '{date_str}'. {e}")

    def load_and_validate_date(self, scene_dir: Path) -> Tuple[Dict[str, np.ndarray], RasterMetadata]:
        """
        Independently loads and validates a single date scene.
        Performs within-date band alignment checks.
        """
        scene_dir = Path(scene_dir)
        if not scene_dir.is_dir():
            raise NotADirectoryError(f"Provided scene directory is not a directory: {scene_dir}")

        # 1. Verify and map band paths
        bands_paths = {}
        bands_metadata = {}
        bands_data = {}

        for band_name in ["green", "red", "rededge", "nir"]:
            band_file = self.loader._find_band_file(scene_dir, band_name)
            bands_paths[band_name] = band_file

            # Open to read metadata
            with rasterio.open(band_file) as src:
                # CRS check — honour require_crs config setting
                if src.crs is None:
                    if self.require_crs:
                        raise ValueError(f"CRS is missing in band '{band_name}' at: {band_file}")
                    else:
                        logger.warning(
                            "Band '%s' at %s has no CRS. Processing in pixel-coordinate space "
                            "(require_crs=False). Ensure all bands share the same pixel grid.",
                            band_name, band_file,
                        )
                if src.width <= 0 or src.height <= 0:
                    raise ValueError(f"Invalid dimensions for band '{band_name}' at: {band_file}")
                
                bands_metadata[band_name] = RasterMetadata(
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    resolution=src.res,
                    extent=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                    nodata=src.nodata
                )
                bands_data[band_name] = src.read(1).astype(np.float32)

        # 2. Within-date band alignment check
        self.spatial_aligner.validate_date_bands(bands_metadata)

        # Since all bands are verified to be aligned, we can use the first one's metadata as representative
        representative_metadata = bands_metadata["green"]
        return bands_data, representative_metadata

    def align_pair(
        self,
        t1_dir: Path,
        t1_date: str,
        t2_dir: Path,
        t2_date: str
    ) -> AlignedPair:
        """
        Main entry point for temporal / spatial alignment.
        """
        # Validate input dates
        self._validate_date(t1_date)
        self._validate_date(t2_date)

        # 1. Load and validate dates
        t1_bands, t1_meta = self.load_and_validate_date(t1_dir)
        t2_bands, t2_meta = self.load_and_validate_date(t2_dir)

        # 2. Build grids
        t1_grid = ReferenceGrid.from_metadata(t1_meta)
        t2_grid = ReferenceGrid.from_metadata(t2_meta)

        # 3. Overlap check
        overlap_pct, overlap_bounds = self.spatial_aligner.calculate_overlap(t1_grid, t2_grid)

        # 4. Check if alignment is required
        alignment_required = not self.spatial_aligner.check_grids_equal(t1_grid, t2_grid)

        aligned_t2_bands = {}
        alignment_operations = []

        if alignment_required:
            # Determine specific alignment operations
            if t1_grid.crs != t2_grid.crs:
                alignment_operations.append("reproject_crs")
            if not np.allclose(t1_grid.resolution, t2_grid.resolution, atol=1e-5):
                alignment_operations.append("resample_resolution")
            # If not exact transform/extent match:
            if (not np.allclose(t1_grid.transform, t2_grid.transform, atol=1e-5) or
                    t1_grid.width != t2_grid.width or
                    t1_grid.height != t2_grid.height or
                    not np.allclose(t1_grid.bounds, t2_grid.bounds, atol=1e-5)):
                alignment_operations.append("align_grid")

            # Reproject T2 to T1 reference grid
            for band_name, band_arr in t2_bands.items():
                aligned_arr = self.spatial_aligner.reproject_band(
                    src_array=band_arr,
                    src_transform=t2_grid.transform,
                    src_crs=t2_grid.crs,
                    dst_transform=t1_grid.transform,
                    dst_crs=t1_grid.crs,
                    dst_shape=(t1_grid.height, t1_grid.width),
                    src_nodata=t2_meta.nodata,
                    dst_nodata=t2_meta.nodata
                )
                aligned_t2_bands[band_name] = aligned_arr
        else:
            aligned_t2_bands = {name: arr.copy() for name, arr in t2_bands.items()}

        # 5. Build validity masks
        t1_valid_mask = np.ones((t1_grid.height, t1_grid.width), dtype=bool)
        for name, arr in t1_bands.items():
            t1_valid_mask &= self.spatial_aligner.get_validity_mask(arr, t1_meta.nodata)

        t2_valid_mask = np.ones((t1_grid.height, t1_grid.width), dtype=bool)
        for name, arr in aligned_t2_bands.items():
            # Use T2's nodata since T2's nodata was propagated during reprojection
            t2_valid_mask &= self.spatial_aligner.get_validity_mask(arr, t2_meta.nodata)

        overlap_mask = t1_valid_mask & t2_valid_mask

        # 6. Aligned T2 metadata representation
        t2_aligned_metadata_dict = {
            "crs": str(t1_grid.crs) if t1_grid.crs is not None else None,
            "transform": [t1_grid.transform.a, t1_grid.transform.b, t1_grid.transform.c,
                          t1_grid.transform.d, t1_grid.transform.e, t1_grid.transform.f],
            "dimensions": [t1_grid.height, t1_grid.width],
            "resolution": list(t1_grid.resolution),
            "extent": list(t1_grid.bounds),
            "nodata": t2_meta.nodata
        }

        # Build output structure
        return AlignedPair(
            t1_data=t1_bands,
            t2_data=aligned_t2_bands,
            t1_valid_mask=t1_valid_mask,
            t2_valid_mask=t2_valid_mask,
            overlap_mask=overlap_mask,
            reference_grid=t1_grid,
            t1_metadata=t1_meta.__dict__,
            t2_original_metadata=t2_meta.__dict__,
            t2_aligned_metadata=t2_aligned_metadata_dict,
            alignment_required=alignment_required,
            alignment_operations=alignment_operations,
            overlap_percentage=overlap_pct
        )
