import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Any
from .metadata import RasterMetadata

class InputValidator:
    """
    Implements ingestion checks to ensure band files exist, are readable,
    and are spatially and numerically compatible.
    """
    def __init__(self, require_crs: bool = True):
        self.require_crs = require_crs

    def validate_files(self, bands_paths: Dict[str, Path]) -> bool:
        """
        Validates the raw band files' geometry and spatial references.
        """
        profiles = {}
        shapes = {}
        
        for name, path in bands_paths.items():
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Band file for '{name}' does not exist at: {path}")
            
            try:
                with rasterio.open(path) as src:
                    profiles[name] = src.profile
                    shapes[name] = (src.height, src.width)
                    
                    # CRS checks
                    if self.require_crs and src.crs is None:
                        raise ValueError(f"CRS is missing in band '{name}' at: {path}")
                    
                    # Empty check
                    if src.width <= 0 or src.height <= 0:
                        raise ValueError(
                            f"Invalid dimensions for band '{name}' at {path}: "
                            f"{src.width}x{src.height}"
                        )
            except Exception as e:
                if isinstance(e, (FileNotFoundError, ValueError)):
                    raise e
                raise ValueError(f"Failed to open or read band file '{name}' at {path}: {e}")

        # Dimension checks
        ref_shape = next(iter(shapes.values()))
        for name, shp in shapes.items():
            if shp != ref_shape:
                raise ValueError(
                    f"Dimension mismatch between bands. Reference shape: {ref_shape}, "
                    f"but '{name}' shape is: {shp}"
                )

        # Transform and CRS alignment checks
        ref_name = next(iter(profiles.keys()))
        ref_profile = profiles[ref_name]
        
        for name, prof in profiles.items():
            if name == ref_name:
                continue
                
            # Transform match within tolerance
            t1 = ref_profile['transform']
            t2 = prof['transform']
            if not np.allclose(t1, t2, atol=1e-5):
                raise ValueError(
                    f"Transform mismatch between '{ref_name}' and '{name}': "
                    f"{t1} vs {t2}"
                )
                
            # CRS match
            c1 = ref_profile.get('crs')
            c2 = prof.get('crs')
            if c1 != c2:
                raise ValueError(
                    f"CRS mismatch between '{ref_name}' and '{name}': "
                    f"{c1} vs {c2}"
                )

        return True

    def validate_arrays(self, bands: Dict[str, np.ndarray]) -> bool:
        """
        Validates the loaded band arrays for numerical soundness and non-emptiness.
        """
        if not bands:
            raise ValueError("No bands loaded to validate.")

        ref_shape = None
        for name, arr in bands.items():
            if arr is None or arr.size == 0:
                raise ValueError(f"Band '{name}' is empty or None.")
            
            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch on loaded arrays. Reference shape: {ref_shape}, "
                    f"but '{name}' shape is: {arr.shape}"
                )
                
            # Content checks
            if np.all(np.isnan(arr)) or np.all(np.isinf(arr)):
                raise ValueError(f"Band '{name}' contains only invalid (NaN/Inf) values.")

        return True
