import os
import glob
from pathlib import Path
from typing import Dict, Tuple, List, Any
import numpy as np
import rasterio
import re
from .metadata import RasterMetadata

class RasterLoader:
    """
    Handles finding and loading geospatial raster bands from a scene directory
    using configurable band name mappings.
    """
    def __init__(self, band_mapping: Dict[str, List[str]]):
        self.band_mapping = band_mapping

    def _find_band_file(self, scene_dir: Path, band_name: str) -> Path:
        patterns = self.band_mapping.get(band_name, [])
        for pattern in patterns:
            # Standard glob search
            files = list(scene_dir.glob(pattern))
            if files:
                return files[0]
            
            # Case-insensitive robust fallback search
            pattern_regex = pattern.replace(".", "\\.").replace("*", ".*")
            regex = re.compile(f"^{pattern_regex}$", re.IGNORECASE)
            for file_path in scene_dir.iterdir():
                if file_path.is_file() and regex.match(file_path.name):
                    return file_path
        
        raise FileNotFoundError(
            f"Missing required band file for '{band_name}' in {scene_dir} "
            f"with matching patterns: {patterns}"
        )

    def load_bands(self, scene_dir: Path) -> Tuple[Dict[str, np.ndarray], RasterMetadata]:
        """
        Loads the four required band rasters and extracts spatial metadata.
        """
        scene_dir = Path(scene_dir)
        if not scene_dir.is_dir():
            raise NotADirectoryError(f"Provided scene path is not a directory: {scene_dir}")

        bands_data = {}
        metadata = None

        for band_name in ["green", "red", "rededge", "nir"]:
            band_file = self._find_band_file(scene_dir, band_name)
            
            with rasterio.open(band_file) as src:
                # Read first band as float32
                array = src.read(1).astype(np.float32)
                
                # Extract metadata profile from the first band
                if metadata is None:
                    metadata = RasterMetadata(
                        crs=src.crs,
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                        resolution=src.res,
                        extent=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                        nodata=src.nodata
                    )
                
                bands_data[band_name] = array

        if metadata is None:
            raise ValueError(f"Failed to load any bands from directory: {scene_dir}")

        return bands_data, metadata
