from dataclasses import dataclass
from typing import Tuple, Optional, Any

@dataclass
class RasterMetadata:
    """
    Data class representing the spatial metadata and profile details of loaded raster bands.
    """
    crs: Optional[Any]
    transform: Optional[Any]
    width: int
    height: int
    resolution: Tuple[float, float]
    extent: Tuple[float, float, float, float]  # left, bottom, right, top
    nodata: Optional[float]
