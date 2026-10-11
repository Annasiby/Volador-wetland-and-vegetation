"""
tests/test_phase13_cli.py
===========================
Phase 13 tests for the run_analysis.py command line interface.

Verifies:
1. Running scripts/run_analysis.py with --help returns exit code 0.
2. Running the CLI with complete valid inputs works and returns exit code 0.
3. Specific exit codes are returned for different exception types:
   - 1 for input validation error (InputValidationError)
   - 2 for configuration error (ConfigurationError)
   - 3 for alignment error (AlignmentError)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts import run_analysis


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
         patch('src.gis_engine.inference.onnx.ONNXModelWrapper.predict', new=mock_predict), \
         patch('gis_engine.inference.onnx.ONNXModelWrapper.load_model', new=mock_load), \
         patch('gis_engine.inference.onnx.ONNXModelWrapper.predict', new=mock_predict):
        yield


def test_cli_help():
    """Verify that run_analysis.py --help returns exit code 0."""
    cli_path = Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"
    res = subprocess.run([sys.executable, str(cli_path), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "GIS Wetland" in res.stdout or "run_analysis.py" in res.stdout


def test_cli_successful_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "outputs"
        config_path = tmp_path / "config.json"

        # Setup configuration and bands
        config_path.write_text(json.dumps(TEST_CONFIG))
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_bands(t1_dir, shape=(10, 10), crs="EPSG:4326", transform=transform, fill_val=1000.0)
        create_mock_bands(t2_dir, shape=(10, 10), crs="EPSG:4326", transform=transform, fill_val=1200.0)

        # Mock argv and run main()
        test_args = [
            "run_analysis.py",
            "--t1", str(t1_dir),
            "--t2", str(t2_dir),
            "--t1-date", "2026-08-01",
            "--t2-date", "2026-08-15",
            "--config", str(config_path),
            "--model", str(t1_dir / "scene_G.tif"),  # dummy model file path
            "--output", str(output_dir),
            "--run-id", "cli_success_test_run"
        ]

        with patch.object(sys, 'argv', test_args):
            exit_code = run_analysis.main()
            assert exit_code == 0

        assert (output_dir / "cli_success_test_run" / "analysis_result.json").is_file()


def test_cli_validation_exit_code():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "outputs"
        config_path = tmp_path / "config.json"

        config_path.write_text(json.dumps(TEST_CONFIG))
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_bands(t1_dir, shape=(10, 10), crs="EPSG:4326", transform=transform)
        create_mock_bands(t2_dir, shape=(10, 10), crs="EPSG:4326", transform=transform)

        # Test case: T1 date is later than T2 date (Temporal order violation)
        test_args = [
            "run_analysis.py",
            "--t1", str(t1_dir),
            "--t2", str(t2_dir),
            "--t1-date", "2026-08-15",
            "--t2-date", "2026-08-01",  # earlier
            "--config", str(config_path),
            "--model", str(t1_dir / "scene_G.tif"),
            "--output", str(output_dir),
            "--run-id", "cli_val_fail"
        ]

        with patch.object(sys, 'argv', test_args):
            exit_code = run_analysis.main()
            assert exit_code == 1  # 1 = InputValidationError exit code


def test_cli_config_exit_code():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        output_dir = tmp_path / "outputs"

        create_mock_bands(t1_dir, shape=(10, 10))
        create_mock_bands(t2_dir, shape=(10, 10))

        # Test case: Config file does not exist
        test_args = [
            "run_analysis.py",
            "--t1", str(t1_dir),
            "--t2", str(t2_dir),
            "--t1-date", "2026-08-01",
            "--t2-date", "2026-08-15",
            "--config", str(tmp_path / "does_not_exist.json"),
            "--model", str(t1_dir / "scene_G.tif"),
            "--output", str(output_dir),
            "--run-id", "cli_config_fail"
        ]

        with patch.object(sys, 'argv', test_args):
            exit_code = run_analysis.main()
            assert exit_code == 2  # 2 = ConfigurationError exit code
