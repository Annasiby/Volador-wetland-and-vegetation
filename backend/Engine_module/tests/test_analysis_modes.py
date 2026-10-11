from pathlib import Path
import json
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.gis_engine.core.pipeline import ProductionPipeline


TEST_CONFIG = {
    "model": {
        "path": "dummy_model.onnx",
        "in_channels": 5,
        "num_classes": 4,
        "channel_order": ["ndvi", "gndvi", "ndre", "nir", "rededge"],
    },
    "preprocessing": {
        "mode": "inference",
        "band_scaling_factor": 10000.0,
        "require_crs": False,
        "epsilon": 1e-6,
        "min_denominator": 0.05,
        "min_band_value": 0.01,
        "gaussian_sigma": 1.0,
        "normalize": True,
    },
    "ingestion": {
        "band_mapping": {
            "green": ["*_G.tif", "*green.tif"],
            "red": ["*_R.tif", "*red.tif"],
            "rededge": ["*_RE.tif", "*rededge.tif"],
            "nir": ["*_NIR.tif", "*nir.tif"],
        }
    },
    "water": {
        "ndwi_threshold": 0.0,
        "epsilon": 1e-6,
    },
    "change": {
        "residual_stress": {
            "enabled": True,
            "threshold": {"method": "percentile", "value": 95.0},
        }
    },
    "fusion": {
        "independent_hydrological_source_accepted": False,
    },
    "wetland": {
        "source": "none",
    },
    "metrics": {
        "severity_weights": {
            "healthy": 0.0,
            "moderate": 0.35,
            "severe": 0.8,
            "background": 1.0,
        }
    },
    "output": {
        "colormap": {
            "0": [128, 128, 128, 255],
            "1": [255, 0, 0, 255],
            "2": [255, 165, 0, 255],
            "3": [0, 170, 0, 255],
        }
    },
}


def create_mock_bands(dir_path: Path, fill_val: float):
    dir_path.mkdir(parents=True, exist_ok=True)

    transform = from_origin(100.0, 10.0, 0.001, 0.001)

    for band in ["G", "R", "RE", "NIR"]:
        path = dir_path / f"scene_{band}.tif"

        profile = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": -9999.0,
        }

        arr = np.full((10, 10), fill_val, dtype=np.float32)

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)


def _monkeypatch_onnx(monkeypatch):
    """Apply ONNX mock patches for all tests."""

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

    monkeypatch.setattr(
        "src.gis_engine.inference.onnx.ONNXModelWrapper.load_model",
        mock_load,
    )
    monkeypatch.setattr(
        "src.gis_engine.inference.onnx.ONNXModelWrapper.predict",
        mock_predict,
    )


def run_vegetation_mode(tmp_path: Path, monkeypatch):
    """
    Run VEGETATION mode: single-date, single scene.
    t2_scene is not passed — vegetation mode does not require it.
    """
    t1_dir = tmp_path / "t1"
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config_vegetation.json"

    config = dict(TEST_CONFIG)
    config["analysis_mode"] = "vegetation"
    config_path.write_text(json.dumps(config))

    create_mock_bands(t1_dir, 2000.0)

    _monkeypatch_onnx(monkeypatch)

    pipeline = ProductionPipeline()
    return pipeline.run_analysis(
        t1_scene=t1_dir,
        t2_scene=None,  # Vegetation mode: NO T2
        t1_date=None,   # Vegetation mode: date optional (single-date)
        t2_date=None,
        config_path=config_path,
        model_path=t1_dir / "scene_G.tif",
        output_base_dir=output_dir,
        run_id="run_vegetation",
    )


def run_wetland_mode(tmp_path: Path, monkeypatch):
    """
    Run WETLAND mode: two-date temporal analysis.
    Both t1_scene and t2_scene are required.
    """
    t1_dir = tmp_path / "t1"
    t2_dir = tmp_path / "t2"
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config_wetland.json"

    config = dict(TEST_CONFIG)
    config["analysis_mode"] = "wetland"
    config_path.write_text(json.dumps(config))

    create_mock_bands(t1_dir, 2000.0)
    create_mock_bands(t2_dir, 2500.0)

    _monkeypatch_onnx(monkeypatch)

    pipeline = ProductionPipeline()
    return pipeline.run_analysis(
        t1_scene=t1_dir,
        t2_scene=t2_dir,  # Wetland mode: T2 REQUIRED
        t1_date="2026-08-01",
        t2_date="2026-08-15",
        config_path=config_path,
        model_path=t1_dir / "scene_G.tif",
        output_base_dir=output_dir,
        run_id="run_wetland",
    )


# ==============================================================================
# SCIENTIFIC CONTRACT: VEGETATION MODE
# ==============================================================================
# Vegetation mode is SINGLE-DATE vegetation health analysis.
# It does NOT perform temporal comparison, CVA, degradation, hydrology, or fusion.
# Output must contain: analysis_mode="vegetation", spectral_indices (ndvi/gndvi/ndre),
#                      vegetation class stats, spatial info, metrics, validation.
# Output must NOT contain: wetland, hydrology, fusion, physical_depletion_assessment.
# ==============================================================================

def test_vegetation_mode_runs_without_wetland_outputs(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_vegetation_mode(Path(tmpdir), monkeypatch)

        assert result.status in ["SUCCESS", "WARNING"], (
            f"Expected SUCCESS or WARNING, got: {result.status}\n"
            f"Errors: {result.errors}"
        )

        result_path = Path(result.output_directory) / "analysis_result.json"
        assert result_path.exists(), "analysis_result.json not written"

        data = json.loads(result_path.read_text())

        # --- Core identity ---
        assert data["analysis_mode"] == "vegetation", (
            f"Expected analysis_mode='vegetation', got: {data.get('analysis_mode')}"
        )

        # --- Required vegetation outputs (single-date) ---
        assert "vegetation" in data, "Missing 'vegetation' key in result"
        assert "spectral_indices" in data, "Missing 'spectral_indices' in result"
        assert "spatial" in data, "Missing 'spatial' in result"
        assert "metrics" in data, "Missing 'metrics' in result"
        assert "validation" in data, "Missing 'validation' in result"
        assert "uncertainty" in data, "Missing 'uncertainty' in result"
        assert "temporal" in data, "Missing 'temporal' key in result"

        # --- Spectral indices: NDVI, GNDVI, NDRE — no NDWI (no water analysis) ---
        spectral_indices = data.get("spectral_indices", {})
        assert "ndvi" in spectral_indices, "Missing 'ndvi' in spectral_indices"
        assert "gndvi" in spectral_indices, "Missing 'gndvi' in spectral_indices"
        assert "ndre" in spectral_indices, "Missing 'ndre' in spectral_indices"
        assert "ndwi" not in spectral_indices, (
            "Vegetation mode must NOT compute NDWI (water index)"
        )

        # --- Absent keys: vegetation mode is NOT temporal/wetland analysis ---
        absent_keys = ["wetland", "hydrology", "fusion", "physical_depletion_assessment",
                       "change", "cva", "degradation", "recovery"]
        for key in absent_keys:
            assert key not in data, (
                f"Vegetation mode must NOT contain '{key}' — "
                f"that belongs to wetland temporal analysis"
            )


# ==============================================================================
# SCIENTIFIC CONTRACT: WETLAND MODE
# ==============================================================================
# Wetland mode is TWO-DATE temporal degradation analysis.
# It requires T1 + T2 scenes and both acquisition dates.
# Output must contain: all vegetation outputs PLUS change/CVA/degradation/
#                      recovery/hydrology/fusion/wetland keys.
# ==============================================================================

def test_wetland_mode_runs_with_wetland_outputs(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_wetland_mode(Path(tmpdir), monkeypatch)

        assert result.status in ["SUCCESS", "WARNING"], (
            f"Expected SUCCESS or WARNING, got: {result.status}\n"
            f"Errors: {result.errors}"
        )

        result_path = Path(result.output_directory) / "analysis_result.json"
        assert result_path.exists(), "analysis_result.json not written"

        data = json.loads(result_path.read_text())

        # --- Core identity ---
        assert data["analysis_mode"] == "wetland", (
            f"Expected analysis_mode='wetland', got: {data.get('analysis_mode')}"
        )

        # --- Required temporal/vegetation outputs ---
        required_keys = [
            "vegetation", "change", "cva", "degradation", "recovery",
            "metrics", "validation", "uncertainty",
            "wetland", "hydrology", "fusion", "physical_depletion_assessment",
        ]
        for key in required_keys:
            assert key in data, (
                f"Wetland mode must contain '{key}' in analysis result"
            )

        # --- Spectral indices: NDVI, GNDVI, NDRE, NDWI ---
        spectral_indices = data.get("spectral_indices", {})
        assert "ndvi" in spectral_indices, "Missing 'ndvi' in spectral_indices"
        assert "gndvi" in spectral_indices, "Missing 'gndvi' in spectral_indices"
        assert "ndre" in spectral_indices, "Missing 'ndre' in spectral_indices"
        assert "ndwi" in spectral_indices, (
            "Wetland mode must compute NDWI (water index)"
        )


# ==============================================================================
# MODE SEPARATION BOUNDARY TESTS
# ==============================================================================

def test_vegetation_mode_does_not_require_t2_scene(tmp_path, monkeypatch):
    """
    Vegetation mode analyzes ONE date only.
    Passing t2_scene=None must NOT raise an error.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_vegetation_mode(Path(tmpdir), monkeypatch)
        assert result.status in ["SUCCESS", "WARNING"], (
            f"Vegetation mode must succeed with only T1 scene. "
            f"Status: {result.status}, Errors: {result.errors}"
        )


def test_wetland_mode_requires_t2_scene(tmp_path, monkeypatch):
    """
    Wetland mode requires both T1 and T2 scenes.
    Missing T2 must result in a FAILED status (not a Python exception).
    """
    t1_dir = tmp_path / "t1"
    output_dir = tmp_path / "outputs_no_t2"
    config_path = tmp_path / "config_wetland_no_t2.json"

    config = dict(TEST_CONFIG)
    config["analysis_mode"] = "wetland"
    config_path.write_text(json.dumps(config))
    create_mock_bands(t1_dir, 2000.0)

    _monkeypatch_onnx(monkeypatch)

    pipeline = ProductionPipeline()
    result = pipeline.run_analysis(
        t1_scene=t1_dir,
        t2_scene=None,  # Missing T2 — must fail gracefully
        t1_date="2026-08-01",
        t2_date=None,
        config_path=config_path,
        model_path=t1_dir / "scene_G.tif",
        output_base_dir=output_dir,
        run_id="run_wetland_no_t2",
    )

    assert result.status == "FAILED", (
        "Wetland mode with no T2 scene must return status='FAILED'"
    )
    assert result.errors, "FAILED result must include error messages"


def test_dispatcher_mode_field_controls_branch():
    """
    The AnalysisDispatcher must correctly interpret the analysis_mode
    config field and expose the correct boolean flags.
    """
    from src.gis_engine.core.dispatcher import AnalysisDispatcher

    veg = AnalysisDispatcher.from_config({"analysis_mode": "vegetation"})
    assert veg.is_vegetation is True
    assert veg.is_wetland is False

    wet = AnalysisDispatcher.from_config({"analysis_mode": "wetland"})
    assert wet.is_vegetation is False
    assert wet.is_wetland is True

    # Default (no analysis_mode key) must be "wetland" for backward compatibility
    default = AnalysisDispatcher.from_config({})
    assert default.is_wetland is True
