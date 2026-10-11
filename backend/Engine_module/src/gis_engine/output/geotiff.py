import os
import json
import numpy as np
import rasterio
from typing import Dict, Any, List
from ..ingestion.metadata import RasterMetadata

class GeoTIFFWriter:
    """
    Saves the spatial classification output map as a georeferenced GeoTIFF
    embedded with the official crop-stress colormap, and exports logits and metadata.
    """
    def __init__(self, colormap: Dict[str, List[int]]):
        self.colormap = colormap

    def write_health_map(self, output_path: str, class_map: np.ndarray, metadata: RasterMetadata):
        """
        Saves the predicted class map as a uint8 GeoTIFF with coordinate reference systems,
        bounding transforms, and GIS color tables.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        height, width = class_map.shape
        
        # Configure GeoTIFF write profile
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.uint8,
            'crs': metadata.crs,
            'transform': metadata.transform,
            'nodata': 255,
            'compress': 'lzw'
        }
        
        # On Windows, ignore rasterio not georeferenced warnings when saving un-georeferenced sample files
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(class_map.astype(np.uint8), 1)
            
            # Formulate the GIS colormap lookup
            colormap_dict = {}
            for k, v in self.colormap.items():
                colormap_dict[int(k)] = tuple(v)
                
            # Fill empty slot ranges
            for idx in range(len(colormap_dict), 256):
                colormap_dict[idx] = (0, 0, 0, 0)
                
            dst.write_colormap(1, colormap_dict)

    @staticmethod
    def write_logits(output_path: str, logits: np.ndarray):
        """
        Saves raw float32 logits array in NumPy format.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.save(output_path, logits.astype(np.float32))

    @staticmethod
    def write_metadata_json(
        output_path: str,
        input_scene_dir: str,
        model_path: str,
        input_shape: tuple,
        output_shape: tuple,
        class_mapping: Dict[int, str],
        channel_order: List[str],
        preproc_config: Dict[str, Any],
        metadata: RasterMetadata,
        inference_provider: str,
        inference_time_ms: float
    ):
        """
        Saves the pipeline execution log metadata to a JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Affine transform serialization
        transform_list = None
        if metadata.transform is not None:
            transform_list = [
                metadata.transform.a, metadata.transform.b, metadata.transform.c,
                metadata.transform.d, metadata.transform.e, metadata.transform.f
            ]
            
        metadata_dict = {
            "input_scene_directory": input_scene_dir,
            "model_path": model_path,
            "model_input_shape": list(input_shape),
            "model_output_shape": list(output_shape),
            "class_mapping": {str(k): v for k, v in class_mapping.items()},
            "channel_order": channel_order,
            "preprocessing_configuration": preproc_config,
            "spatial_metadata": {
                "crs": str(metadata.crs) if metadata.crs is not None else None,
                "transform": transform_list,
                "dimensions": [metadata.height, metadata.width],
                "resolution": list(metadata.resolution),
                "extent": list(metadata.extent)
            },
            "performance": {
                "inference_provider": inference_provider,
                "inference_time_ms": float(inference_time_ms)
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(metadata_dict, f, indent=4)

    def write_confidence_map(self, output_path: str, confidence_map: np.ndarray, metadata: RasterMetadata):
        """
        Saves the predicted confidence map as a float32 GeoTIFF.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        height, width = confidence_map.shape
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.float32,
            'crs': metadata.crs,
            'transform': metadata.transform,
            'nodata': np.nan,
            'compress': 'lzw'
        }
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(confidence_map.astype(np.float32), 1)

    def write_mask_map(self, output_path: str, mask: np.ndarray, metadata: RasterMetadata):
        """
        Saves a boolean mask as a uint8 GeoTIFF (1 for valid, 0 for invalid).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        height, width = mask.shape
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.uint8,
            'crs': metadata.crs,
            'transform': metadata.transform,
            'nodata': 0,
            'compress': 'lzw'
        }
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)

