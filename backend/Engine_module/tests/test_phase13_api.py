"""
tests/test_phase13_api.py
===========================
Phase 13 tests for the public API service.py interface.

Verifies:
1. The analyze_wetland() service function never raises, always returning AnalysisResult.
2. Structured result fields are correctly populated.
3. Behavior with missing configuration or missing model file.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.gis_engine.api.service import analyze_wetland
from src.gis_engine.core.results import AnalysisResult


TEST_CONFIG = {
    "model": {
        "path": "dummy_model.onnx",
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
    "water": {
        "ndwi_threshold": 0.0,
        "epsilon": 1e-6
    },
    "change": {
        "residual_stress": {
            "enabled": True,
            "threshold": {"method": "percentile", "value": 95.0}
        }
    },
    "fusion": {
        "independent_hydrological_source_accepted": False
    },
    "wetland": {
        "source": "none"
    },
    "metrics": {
        "severity_weights": {
            "healthy": 0.0,
            "moderate": 0.35,
            "severe": 0.8,
            "background": 1.0
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


def create_mock_bands(dir_path: Path, shape=(10, 10), crs="EPSG:4326", transform=None, fill_val=1000.0):
    dir_path.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = from_origin(100.0, 10.0, 0.001, 0.001)

    bands = ["G", "R", "RE", "NIR"]
    for b in bands:
        path = dir_path / f"scene_{b}.tif"
        profile = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,
            'dtype': rasterio.float32,
            'crs': crs,
            'transform': transform,
            'nodata': -9999.0
        }
        arr = np.full(shape, fill_val, dtype=np.float32)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(arr, 1)


@pytest.fixture(autouse=True)
def mock_onnx_wrapper():
    def mock_load(self):
        self.session = "mock_session"
        self.input_name = "input"
        self.output_name = "output"
        self.execution_provider = "CPUExecutionProvider"
        self.loading_time_ms = 0.0

    def mock_predict(self, tensor):
        logits = np.zeros((1, 4, 10, 10), dtype=np.float32)
        logits[0, 3, :, :] = 10.0
        class_map = np.full((10, 10), 3, dtype=np.uint8)
        return logits, class_map

    with patch('src.gis_engine.inference.onnx.ONNXModelWrapper.load_model', new=mock_load), \
         patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', new=mock_predict):
        yield


def test_api_successful_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "outputs"
        config_path = tmp_path / "config.json"

        # Setup paths
        config_path.write_text(json.dumps(TEST_CONFIG))
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_bands(t1_dir, shape=(10, 10), crs="EPSG:4326", transform=transform, fill_val=1500.0)
        create_mock_bands(t2_dir, shape=(10, 10), crs="EPSG:4326", transform=transform, fill_val=2200.0)

        # Call public API
        result = analyze_wetland(
            t1_scene=t1_dir,
            t2_scene=t2_dir,
            t1_date="2026-08-01",
            t2_date="2026-08-15",
            config_path=config_path,
            model_path=t1_dir / "scene_G.tif",  # dummy model path
            output_dir=output_dir,
            run_id="api_success_run"
        )

        assert isinstance(result, AnalysisResult)
        assert result.status in ["SUCCESS", "WARNING"]
        assert result.run_id == "api_success_run"
        assert result.output_directory == str(output_dir / "api_success_run")
        assert len(result.errors) == 0


def test_api_no_raise_on_failure():
    """Verifies that public API never raises, but returns failed AnalysisResult."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "outputs"
        config_path = tmp_path / "config.json"

        # Missing configuration file completely
        result = analyze_wetland(
            t1_scene=t1_dir,
            t2_scene=t2_dir,
            t1_date="2026-08-01",
            t2_date="2026-08-15",
            config_path=config_path / "does_not_exist.json",
            model_path=t1_dir / "does_not_exist.onnx",
            output_dir=output_dir,
            run_id="api_fail_run"
        )

        # API must capture error in result, not raise exception
        assert isinstance(result, AnalysisResult)
        assert result.status == "FAILED"
        assert len(result.errors) > 0
        assert "Configuration file not found" in result.errors[0]
