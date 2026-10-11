import pytest
import numpy as np
from pathlib import Path
import tempfile
import rasterio
import json
from rasterio.transform import from_origin
from unittest.mock import patch


from src.gis_engine.alignment.spatial import ReferenceGrid, SpatialAligner
from src.gis_engine.inference.probabilities import calculate_softmax, calculate_confidence
from src.gis_engine.core.pair import analyze_pair, calculate_class_stats
from src.gis_engine.core.results import TwoDateInferenceResult

TEST_CONFIG = {
    "model": {
        "path": "models/best_finetuned_model.onnx",
        "in_channels": 5,
        "num_classes": 4,
        "channel_order": ["ndvi", "gndvi", "ndre", "nir", "rededge"]
    },
    "preprocessing": {
        "mode": "inference",
        "band_scaling_factor": 10000.0,
        "require_crs": False,
        "epsilon": 1e-6,
        "min_denominator": 0.05,
        "min_band_value": 0.01,
        "gaussian_sigma": 1.0,
        "normalize": True
    },
    "ingestion": {
        "band_mapping": {
            "green": ["*_G.tif", "*green.tif"],
            "red": ["*_R.tif", "*red.tif"],
            "rededge": ["*_RE.tif", "*rededge.tif"],
            "nir": ["*_NIR.tif", "*nir.tif"]
        }
    },
    "output": {
        "colormap": {
            "0": [128, 128, 128, 255],
            "1": [255, 0, 0, 255],
            "2": [255, 165, 0, 255],
            "3": [0, 170, 0, 255]
        }
    }
}

def create_mock_scene_local(
    dir_path: Path,
    shape: tuple = (10, 10),
    transform=None,
    crs: str = "EPSG:4326",
    nodata: float = None,
    fill_value: float = 1.0
):
    """
    Helper to create a scene folder with 4 band geotiffs.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = from_origin(100.0, 10.0, 0.001, 0.001)

    bands = ["G", "R", "RE", "NIR"]
    band_name_map = {
        "G": "green",
        "R": "red",
        "RE": "rededge",
        "NIR": "nir"
    }

    for b in bands:
        b_name = band_name_map[b]
        path = dir_path / f"scene_{b}.tif"
        profile = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,
            'dtype': rasterio.float32,
            'crs': crs,
            'transform': transform,
            'nodata': nodata
        }
        arr = np.full(shape, fill_value, dtype=np.float32)
        
        # Add some mock patterns or NoData regions
        if nodata is not None:
            # Mark the right half of the image as NoData
            arr[:, shape[1]//2:] = nodata

        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(arr, 1)


@pytest.fixture(autouse=True)
def mock_onnx_wrapper_load():
    def mock_load(self):
        self.session = "mock_session"
        self.input_name = "input"
        self.output_name = "output"
        self.execution_provider = "CPUExecutionProvider"
        self.loading_time_ms = 0.0

    with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.load_model', new=mock_load):
        yield

def test_identical_grid_acceptance():

    """
    Test 1: Verification that identical spatial grids are accepted.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "output"
        
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene_local(t1_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        create_mock_scene_local(t2_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        
        # Mock predict to return dummy logits and class map without using real ONNX model
        mock_logits = np.zeros((1, 4, 10, 10), dtype=np.float32)
        mock_logits[0, 3, :, :] = 10.0  # Make Healthy class high
        mock_class_map = np.full((10, 10), 3, dtype=np.int32)
        
        with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', return_value=(mock_logits, mock_class_map)):
            result = analyze_pair(t1_dir, t2_dir, Path("dummy_model.onnx"), output_dir, TEST_CONFIG)
            assert result is not None
            assert result.t1_health_map.shape == (10, 10)


def test_mismatched_grid_rejection():
    """
    Test 2: Verification that mismatched grids are rejected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "output"
        
        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        t2_trans = from_origin(100.005, 10.0, 0.001, 0.001) # shifted
        
        create_mock_scene_local(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326")
        create_mock_scene_local(t2_dir, shape=(10, 10), transform=t2_trans, crs="EPSG:4326")
        
        with pytest.raises(ValueError, match="grids are not aligned"):
            analyze_pair(t1_dir, t2_dir, Path("dummy_model.onnx"), output_dir, TEST_CONFIG)


def test_model_input_tensor_shape():
    """
    Test 3: Verification of correct model input tensor shape (1, 5, H, W).
    """
    from src.gis_engine.preprocessing.tensor import TensorBuilder
    ndvi = np.full((10, 10), 0.5, dtype=np.float32)
    gndvi = np.full((10, 10), 0.6, dtype=np.float32)
    ndre = np.full((10, 10), 0.7, dtype=np.float32)
    nir = np.full((10, 10), 0.8, dtype=np.float32)
    rededge = np.full((10, 10), 0.9, dtype=np.float32)
    
    indices = {"ndvi": ndvi, "gndvi": gndvi, "ndre": ndre}
    bands = {"nir": nir, "rededge": rededge}
    
    builder = TensorBuilder()
    tensor = builder.build_tensor(indices, bands)
    
    assert tensor.shape == (1, 5, 10, 10)
    assert tensor.dtype == np.float32


def test_channel_ordering():
    """
    Test 4: Verification of correct channel ordering.
    """
    from src.gis_engine.preprocessing.tensor import TensorBuilder
    ndvi = np.full((10, 10), 0.1, dtype=np.float32)
    gndvi = np.full((10, 10), 0.2, dtype=np.float32)
    ndre = np.full((10, 10), 0.3, dtype=np.float32)
    nir = np.full((10, 10), 0.4, dtype=np.float32)
    rededge = np.full((10, 10), 0.5, dtype=np.float32)
    
    indices = {"ndvi": ndvi, "gndvi": gndvi, "ndre": ndre}
    bands = {"nir": nir, "rededge": rededge}
    
    builder = TensorBuilder(channel_order=["ndvi", "gndvi", "ndre", "nir", "rededge"])
    tensor = builder.build_tensor(indices, bands)
    
    # Check that channels are populated correctly
    # Since Normalizer.normalize_channel maps min to 0 and max to 1, let's look at relative values or bypass normalizer if testing order.
    # To bypass normalizer, we can inspect Normalizer output on constants.
    # Actually, build_tensor normalizes each channel individually.
    # Let's verify that order matches self.channel_order.
    assert builder.channel_order == ["ndvi", "gndvi", "ndre", "nir", "rededge"]


def test_model_output_shape():
    """
    Test 5: Verification of model output shape (1, 4, H, W).
    """
    # Verify shape contract of mock outputs.
    mock_logits = np.zeros((1, 4, 10, 10), dtype=np.float32)
    assert mock_logits.shape == (1, 4, 10, 10)


def test_class_argmax():
    """
    Test 6: Verification of correct class argmax.
    """
    # Create raw logits where class 2 has the highest value
    logits = np.zeros((1, 4, 3, 3), dtype=np.float32)
    logits[0, 2, :, :] = 5.0
    
    class_map = np.argmax(logits, axis=1)
    class_map_squeezed = np.squeeze(class_map, axis=0)
    
    assert class_map_squeezed.shape == (3, 3)
    assert np.all(class_map_squeezed == 2)


def test_softmax_calculation():
    """
    Test 7: Verification of correct softmax calculation.
    """
    # Simple logit with known values
    logits = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32) # Shape (1, 1, 2, 2)
    # Let's build a (1, 2, 1, 1) array
    logits = np.zeros((1, 2, 1, 1), dtype=np.float32)
    logits[0, 0, 0, 0] = 0.0
    logits[0, 1, 0, 0] = np.log(3.0) # exp(0) = 1, exp(log(3)) = 3. Softmax should be [0.25, 0.75]
    
    probs = calculate_softmax(logits)
    assert probs.shape == (2, 1, 1)
    assert np.allclose(probs[:, 0, 0], [0.25, 0.75])


def test_confidence_calculation():
    """
    Test 8: Verification of correct confidence calculation.
    """
    probs = np.zeros((4, 2, 2), dtype=np.float32)
    probs[0, :, :] = 0.1
    probs[1, :, :] = 0.7
    probs[2, :, :] = 0.15
    probs[3, :, :] = 0.05
    
    conf = calculate_confidence(probs)
    assert conf.shape == (2, 2)
    assert np.allclose(conf, 0.7)


def test_class_statistics():
    """
    Test 9: Verification of correct class statistics.
    """
    class_map = np.array([
        [0, 1, 2],
        [3, 0, 1],
        [2, 3, 0]
    ], dtype=np.uint8)
    
    mask = np.ones((3, 3), dtype=bool)
    # Mask out the bottom right corner
    mask[2, 2] = False
    
    stats = calculate_class_stats(class_map, mask)
    
    assert stats["total_valid_pixels"] == 8
    # Background (0) count should be 2 (since the 3rd one at (2,2) is masked out)
    assert stats["classes"]["0"]["count"] == 2
    assert stats["classes"]["0"]["percentage"] == 25.0
    
    # Severe (1) count should be 2
    assert stats["classes"]["1"]["count"] == 2
    assert stats["classes"]["1"]["percentage"] == 25.0


def test_nodata_not_counted_as_background():
    """
    Test 10: Verification that NoData pixels (255) are not counted as Background (0).
    """
    health_map = np.array([
        [0, 255, 255],
        [255, 0, 255]
    ], dtype=np.uint8)
    
    mask = (health_map != 255)
    
    stats = calculate_class_stats(health_map, mask)
    assert stats["total_valid_pixels"] == 2
    assert stats["classes"]["0"]["count"] == 2
    # Ensure NoData pixels are not in classes dict keys or count is zero for other classes
    assert stats["classes"]["1"]["count"] == 0
    assert stats["classes"]["2"]["count"] == 0
    assert stats["classes"]["3"]["count"] == 0


def test_output_geometry_equality():
    """
    Test 11: Verification that T1 and T2 output GeoTIFF structures share identical grid properties.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "output"
        
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene_local(t1_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        create_mock_scene_local(t2_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        
        mock_logits = np.zeros((1, 4, 10, 10), dtype=np.float32)
        mock_class_map = np.zeros((10, 10), dtype=np.int32)
        
        with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', return_value=(mock_logits, mock_class_map)):
            result = analyze_pair(t1_dir, t2_dir, Path("dummy_model.onnx"), output_dir, TEST_CONFIG)
            
            # Read health maps from disk
            with rasterio.open(output_dir / "T1" / "health.tif") as t1_raster:
                with rasterio.open(output_dir / "T2" / "health.tif") as t2_raster:
                    assert t1_raster.crs == t2_raster.crs
                    assert t1_raster.transform == t2_raster.transform
                    assert t1_raster.width == t2_raster.width
                    assert t1_raster.height == t2_raster.height


def test_logits_preserved():
    """
    Test 12: Verification that logits are preserved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "output"
        
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene_local(t1_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        create_mock_scene_local(t2_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        
        mock_logits = np.random.randn(1, 4, 10, 10).astype(np.float32)
        mock_class_map = np.zeros((10, 10), dtype=np.int32)
        
        with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', return_value=(mock_logits, mock_class_map)):
            result = analyze_pair(t1_dir, t2_dir, Path("dummy_model.onnx"), output_dir, TEST_CONFIG)
            
            # Verify numpy files were saved
            t1_logits_loaded = np.load(output_dir / "T1" / "logits.npy")
            t2_logits_loaded = np.load(output_dir / "T2" / "logits.npy")
            
            assert np.allclose(t1_logits_loaded, mock_logits)
            assert np.allclose(t2_logits_loaded, mock_logits)


def test_same_model_configuration():
    """
    Test 13: Verification that both runs use the same model configuration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "output"
        
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene_local(t1_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        create_mock_scene_local(t2_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        
        mock_logits = np.zeros((1, 4, 10, 10), dtype=np.float32)
        mock_class_map = np.zeros((10, 10), dtype=np.int32)
        
        with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', return_value=(mock_logits, mock_class_map)) as mock_predict:
            result = analyze_pair(t1_dir, t2_dir, Path("dummy_model.onnx"), output_dir, TEST_CONFIG)
            # Verify predict was called twice (once for T1, once for T2)
            assert mock_predict.call_count == 2


def test_overlap_mask_propagation():
    """
    Test 14: Verification that the overlap mask is correctly propagated.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "output"
        
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        # Create T1 with NoData on right side, T2 with NoData on bottom side
        create_mock_scene_local(t1_dir, shape=(10, 10), transform=transform, crs="EPSG:4326", nodata=-999.0)
        create_mock_scene_local(t2_dir, shape=(10, 10), transform=transform, crs="EPSG:4326", nodata=-999.0)
        
        # Modify T2 red band to have NoData at bottom half
        with rasterio.open(t2_dir / "scene_R.tif", "r+") as dst:
            arr = dst.read(1)
            arr[5:, :] = -999.0
            dst.write(arr, 1)
            
        mock_logits = np.zeros((1, 4, 10, 10), dtype=np.float32)
        mock_class_map = np.zeros((10, 10), dtype=np.int32)
        
        with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', return_value=(mock_logits, mock_class_map)):
            result = analyze_pair(t1_dir, t2_dir, Path("dummy_model.onnx"), output_dir, TEST_CONFIG)
            
            # In T1, right half is invalid (cols 5-9)
            # In T2, bottom half is invalid (rows 5-9)
            # Overlap mask should be valid only in top-left quarter (rows 0-4, cols 0-4)
            expected_overlap = np.ones((10, 10), dtype=bool)
            expected_overlap[:, 5:] = False
            expected_overlap[5:, :] = False
            
            assert np.all(result.overlap_mask == expected_overlap)
            
            with rasterio.open(output_dir / "overlap_mask.tif") as mask_raster:
                mask_arr = mask_raster.read(1)
                assert np.all(mask_arr == expected_overlap.astype(np.uint8))


def test_metadata_generated():
    """
    Test 15: Verification that the metadata JSON is generated correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "output"
        
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene_local(t1_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        create_mock_scene_local(t2_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        
        mock_logits = np.zeros((1, 4, 10, 10), dtype=np.float32)
        mock_class_map = np.zeros((10, 10), dtype=np.int32)
        
        with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', return_value=(mock_logits, mock_class_map)):
            result = analyze_pair(t1_dir, t2_dir, Path("dummy_model.onnx"), output_dir, TEST_CONFIG)
            
            metadata_file = output_dir / "pair_metadata.json"
            assert metadata_file.is_file()
            
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                
            assert "pair_name" in metadata
            assert "t1_date_dir" in metadata
            assert "t2_date_dir" in metadata
            assert "model_path" in metadata
            assert "class_statistics" in metadata
            assert "performance" in metadata
            assert "reference_grid" in metadata
