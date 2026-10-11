import pytest
import numpy as np
from pathlib import Path
import tempfile
import rasterio
from rasterio.transform import from_origin

from src.gis_engine.ingestion.metadata import RasterMetadata
from src.gis_engine.ingestion.loader import RasterLoader
from src.gis_engine.ingestion.validator import InputValidator
from src.gis_engine.preprocessing.bands import BandProcessor
from src.gis_engine.preprocessing.indices import VegetationIndexProcessor
from src.gis_engine.preprocessing.normalization import Normalizer
from src.gis_engine.preprocessing.tensor import TensorBuilder

@pytest.fixture
def temp_scene_dir():
    """
    Creates a temporary scene folder containing mock TIFF rasters.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        shape = (10, 10)
        transform = from_origin(100.0, 10.0, 1.0, 1.0)
        
        bands = {
            "G": np.full(shape, 10000, dtype=np.uint16),
            "R": np.full(shape, 8000, dtype=np.uint16),
            "RE": np.full(shape, 12000, dtype=np.uint16),
            "NIR": np.full(shape, 18000, dtype=np.uint16)
        }
        
        for suffix, arr in bands.items():
            path = tmpdir_path / f"scene_{suffix}.tif"
            profile = {
                'driver': 'GTiff',
                'height': shape[0],
                'width': shape[1],
                'count': 1,
                'dtype': rasterio.uint16,
                'crs': 'EPSG:4326',
                'transform': transform
            }
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(arr, 1)
                
        yield tmpdir_path

def test_raster_loading(temp_scene_dir):
    """
    Test 1: Verifies that loader successfully finds and reads rasters.
    """
    band_mapping = {
        "green": ["*G.tif"],
        "red": ["*R.tif"],
        "rededge": ["*RE.tif"],
        "nir": ["*NIR.tif"]
    }
    loader = RasterLoader(band_mapping)
    bands_data, metadata = loader.load_bands(temp_scene_dir)
    
    assert "green" in bands_data
    assert "red" in bands_data
    assert "rededge" in bands_data
    assert "nir" in bands_data
    
    assert bands_data["green"].shape == (10, 10)
    assert metadata.width == 10
    assert metadata.height == 10
    assert metadata.crs.to_epsg() == 4326

def test_input_validation(temp_scene_dir):
    """
    Test 2: Verifies that validator passes on valid matched rasters.
    """
    validator = InputValidator(require_crs=True)
    
    bands_paths = {
        "green": temp_scene_dir / "scene_G.tif",
        "red": temp_scene_dir / "scene_R.tif",
        "rededge": temp_scene_dir / "scene_RE.tif",
        "nir": temp_scene_dir / "scene_NIR.tif"
    }
    
    assert validator.validate_files(bands_paths) is True

def test_shape_mismatch(temp_scene_dir):
    """
    Test 3: Verifies dimension mismatch is correctly caught and raises error.
    """
    mismatch_path = temp_scene_dir / "scene_mismatch.tif"
    profile = {
        'driver': 'GTiff',
        'height': 5,
        'width': 5,
        'count': 1,
        'dtype': rasterio.uint16,
        'crs': 'EPSG:4326',
        'transform': from_origin(100.0, 10.0, 1.0, 1.0)
    }
    with rasterio.open(mismatch_path, 'w', **profile) as dst:
        dst.write(np.zeros((5, 5), dtype=np.uint16), 1)
        
    validator = InputValidator(require_crs=True)
    bands_paths = {
        "green": temp_scene_dir / "scene_G.tif",
        "red": mismatch_path,
        "rededge": temp_scene_dir / "scene_RE.tif",
        "nir": temp_scene_dir / "scene_NIR.tif"
    }
    
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validator.validate_files(bands_paths)

def test_missing_band(temp_scene_dir):
    """
    Test 4: Verifies missing band is correctly caught and raises error.
    """
    validator = InputValidator(require_crs=True)
    bands_paths = {
        "green": temp_scene_dir / "scene_G.tif",
        "red": temp_scene_dir / "scene_missing.tif",
        "rededge": temp_scene_dir / "scene_RE.tif",
        "nir": temp_scene_dir / "scene_NIR.tif"
    }
    
    with pytest.raises(FileNotFoundError):
        validator.validate_files(bands_paths)

def test_index_calculations():
    """
    Test 5, 6, 7: Verifies index calculations (NDVI, GNDVI, NDRE).
    """
    # nir = 0.8, red = 0.2
    # expected ndvi = (0.8 - 0.2) / (0.8 + 0.2 + 1e-6) = 0.6 / 1.000001
    nir = np.full((3, 3), 0.8, dtype=np.float32)
    red = np.full((3, 3), 0.2, dtype=np.float32)
    green = np.full((3, 3), 0.3, dtype=np.float32)
    rededge = np.full((3, 3), 0.4, dtype=np.float32)
    
    bands = {"nir": nir, "red": red, "green": green, "rededge": rededge}
    
    processor = VegetationIndexProcessor(mode="training", gaussian_sigma=0.0)
    indices = processor.compute_indices(bands)
    
    assert np.allclose(indices["ndvi"], (0.8 - 0.2) / (1.0 + 1e-6), atol=1e-5)
    assert np.allclose(indices["gndvi"], (0.8 - 0.3) / (1.1 + 1e-6), atol=1e-5)
    assert np.allclose(indices["ndre"], (0.8 - 0.4) / (1.2 + 1e-6), atol=1e-5)

def test_min_max_normalization():
    """
    Test 8: Verifies min-max normalization.
    """
    arr = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    norm = Normalizer.normalize_channel(arr)
    expected = np.array([0.0, 0.333333, 0.666667, 1.0], dtype=np.float32)
    assert np.allclose(norm, expected, atol=1e-4)

def test_tensor_builder():
    """
    Test 9, 10: Verifies 5-channel ordering and shape.
    """
    ndvi = np.full((4, 4), 0.5, dtype=np.float32)
    gndvi = np.full((4, 4), 0.6, dtype=np.float32)
    ndre = np.full((4, 4), 0.7, dtype=np.float32)
    nir = np.full((4, 4), 0.8, dtype=np.float32)
    rededge = np.full((4, 4), 0.9, dtype=np.float32)
    
    indices = {"ndvi": ndvi, "gndvi": gndvi, "ndre": ndre}
    bands = {"nir": nir, "rededge": rededge}
    
    builder = TensorBuilder()
    tensor = builder.build_tensor(indices, bands)
    
    assert tensor.shape == (1, 5, 4, 4)
    assert tensor.dtype == np.float32
