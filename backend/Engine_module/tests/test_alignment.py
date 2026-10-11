import pytest
import numpy as np
from pathlib import Path
import tempfile
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from src.gis_engine.alignment.spatial import ReferenceGrid, SpatialAligner, SpatialOverlapError
from src.gis_engine.alignment.temporal import TemporalAligner, AlignedPair

TEST_CONFIG = {
    "ingestion": {
        "band_mapping": {
            "green": ["*_G.tif"],
            "red": ["*_R.tif"],
            "rededge": ["*_RE.tif"],
            "nir": ["*_NIR.tif"]
        }
    },
    "alignment": {
        "resampling_method": "bilinear"
    }
}

def create_mock_scene(
    dir_path: Path,
    shape: tuple = (10, 10),
    transform=None,
    crs: str = "EPSG:4326",
    nodata: float = None,
    fill_value: float = 1.0,
    mismatch_band: str = None,
    mismatch_attr: str = None,
    mismatch_val = None
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
        b_shape = shape
        b_transform = transform
        b_crs = crs
        b_nodata = nodata

        if mismatch_band == b_name:
            if mismatch_attr == "shape":
                b_shape = mismatch_val if mismatch_val else (5, 5)
            elif mismatch_attr == "crs":
                b_crs = mismatch_val if mismatch_val else "EPSG:3857"
            elif mismatch_attr == "transform":
                b_transform = mismatch_val if mismatch_val else from_origin(105.0, 15.0, 0.002, 0.002)
            elif mismatch_attr == "missing_crs":
                b_crs = None

        path = dir_path / f"scene_{b}.tif"
        profile = {
            'driver': 'GTiff',
            'height': b_shape[0],
            'width': b_shape[1],
            'count': 1,
            'dtype': rasterio.float32,
            'crs': b_crs,
            'transform': b_transform,
            'nodata': b_nodata
        }
        arr = np.full(b_shape, fill_value, dtype=np.float32)
        
        # Add some mock patterns or NoData regions
        if nodata is not None:
            # Mark the right half of the image as NoData
            arr[:, b_shape[1]//2:] = nodata

        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(arr, 1)


def test_identical_grids():
    """
    Scenario 1: Identical T1/T2 grids.
    Expected: alignment_required = false
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"
        
        transform = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene(t1_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")
        create_mock_scene(t2_dir, shape=(10, 10), transform=transform, crs="EPSG:4326")

        aligner = TemporalAligner(TEST_CONFIG)
        pair = aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")

        assert pair.alignment_required is False
        assert len(pair.alignment_operations) == 0


def test_different_resolution():
    """
    Scenario 2: Same CRS but different resolution.
    Expected: T2 aligned to T1 resolution.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        # T1 has 0.001 resolution (10x10 pixels)
        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        # T2 has 0.002 resolution (5x5 pixels) but covers same extent (approx)
        t2_trans = from_origin(100.0, 10.0, 0.002, 0.002)

        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", fill_value=10.0)
        create_mock_scene(t2_dir, shape=(5, 5), transform=t2_trans, crs="EPSG:4326", fill_value=20.0)

        aligner = TemporalAligner(TEST_CONFIG)
        pair = aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")

        assert pair.alignment_required is True
        assert "resample_resolution" in pair.alignment_operations
        assert pair.t2_data["green"].shape == (10, 10)
        # Check alignment resolution matches T1
        assert pair.t2_aligned_metadata["resolution"] == list(pair.reference_grid.resolution)


def test_different_dimensions_extent():
    """
    Scenario 3: Same CRS but different dimensions/extent.
    Expected: T2 aligned to T1 grid.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        # T2 is shifted to the right by 0.002 degrees (2 pixels shift)
        t2_trans = from_origin(100.002, 10.0, 0.001, 0.001)

        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", fill_value=1.0)
        create_mock_scene(t2_dir, shape=(10, 10), transform=t2_trans, crs="EPSG:4326", fill_value=2.0)

        aligner = TemporalAligner(TEST_CONFIG)
        pair = aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")

        assert pair.alignment_required is True
        assert "align_grid" in pair.alignment_operations
        assert pair.t2_data["green"].shape == (10, 10)


def test_different_crs():
    """
    Scenario 4: Different CRS.
    Expected: T2 reprojected to T1 CRS/grid.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        # T1 in EPSG:4326
        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", fill_value=1.0)

        # T2 in EPSG:3857
        t2_trans = from_origin(11131949.0, 1118889.0, 100.0, 100.0)
        create_mock_scene(t2_dir, shape=(10, 10), transform=t2_trans, crs="EPSG:3857", fill_value=2.0)

        aligner = TemporalAligner(TEST_CONFIG)
        pair = aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")

        assert pair.alignment_required is True
        assert "reproject_crs" in pair.alignment_operations
        assert pair.t2_data["green"].shape == (10, 10)
        assert pair.t2_aligned_metadata["crs"] == "EPSG:4326"


def test_different_transform():
    """
    Scenario 5: Different transform but same dimensions.
    Expected: alignment detected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        t2_trans = from_origin(100.0, 9.999, 0.001, 0.001) # shifted slightly vertically

        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326")
        create_mock_scene(t2_dir, shape=(10, 10), transform=t2_trans, crs="EPSG:4326")

        aligner = TemporalAligner(TEST_CONFIG)
        pair = aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")

        assert pair.alignment_required is True
        assert "align_grid" in pair.alignment_operations


def test_no_spatial_overlap():
    """
    Scenario 6: No spatial overlap.
    Expected: clear failure (SpatialOverlapError).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        # T2 is far away (shifted by 1 degree, which is 1000 pixels)
        t2_trans = from_origin(101.0, 10.0, 0.001, 0.001)

        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326")
        create_mock_scene(t2_dir, shape=(10, 10), transform=t2_trans, crs="EPSG:4326")

        aligner = TemporalAligner(TEST_CONFIG)
        with pytest.raises(SpatialOverlapError, match="no geographic intersection"):
            aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")


def test_missing_crs():
    """
    Scenario 7: Missing CRS.
    Expected: clear validation failure (ValueError).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326")
        
        # Write T2 with no CRS on one of the bands
        create_mock_scene(t2_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", mismatch_band="red", mismatch_attr="missing_crs")

        aligner = TemporalAligner(TEST_CONFIG)
        with pytest.raises(ValueError, match="CRS mismatch|CRS is missing|no CRS"):
            aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")


def test_mismatched_bands_t1():
    """
    Scenario 8: Mismatched bands within T1.
    Expected: clear validation failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        # Red band has mismatching shape
        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", mismatch_band="red", mismatch_attr="shape", mismatch_val=(5, 5))
        create_mock_scene(t2_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326")

        aligner = TemporalAligner(TEST_CONFIG)
        with pytest.raises(ValueError, match="Within-date shape mismatch"):
            aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")


def test_mismatched_bands_t2():
    """
    Scenario 9: Mismatched bands within T2.
    Expected: clear validation failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326")
        # Red band in T2 has mismatching CRS
        create_mock_scene(t2_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", mismatch_band="red", mismatch_attr="crs", mismatch_val="EPSG:3857")

        aligner = TemporalAligner(TEST_CONFIG)
        with pytest.raises(ValueError, match="Within-date CRS mismatch"):
            aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")


def test_nodata_outside_t2_coverage():
    """
    Scenario 10: NoData outside T2 coverage.
    Expected: aligned T2 contains invalid/NoData pixels there.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", nodata=-999.0)

        # T2 is shifted to the right by 5 pixels (0.005 degrees), meaning left 5 columns of T1 have no T2 coverage
        t2_trans = from_origin(100.005, 10.0, 0.001, 0.001)
        create_mock_scene(t2_dir, shape=(10, 10), transform=t2_trans, crs="EPSG:4326", nodata=-999.0, fill_value=5.0)

        aligner = TemporalAligner(TEST_CONFIG)
        pair = aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")

        # The left 5 columns (0 to 4) should be NoData (-999.0)
        aligned_green = pair.t2_data["green"]
        assert np.all(aligned_green[:, :5] == -999.0)
        assert np.all(aligned_green[:, 5:] == 5.0)

        # The validity mask should be False for those 5 columns
        assert np.all(pair.t2_valid_mask[:, :5] == False)
        assert np.all(pair.t2_valid_mask[:, 5:] == True)


def test_identical_grid_reproducibility():
    """
    Scenario 11: Identical grid reproducibility.
    Expected: T1 and T2 geometry remain unchanged.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", fill_value=1.0)
        create_mock_scene(t2_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326", fill_value=2.0)

        aligner = TemporalAligner(TEST_CONFIG)
        pair = aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")

        assert np.allclose(pair.t1_data["green"], 1.0)
        assert np.allclose(pair.t2_data["green"], 2.0)
        assert pair.alignment_required is False


def test_reference_grid_integrity():
    """
    Scenario 12: Reference grid integrity.
    Expected: aligned T2 metadata exactly matches T1 metadata.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        t1_dir = tmp_path / "t1"
        t2_dir = tmp_path / "t2"

        t1_trans = from_origin(100.0, 10.0, 0.001, 0.001)
        create_mock_scene(t1_dir, shape=(10, 10), transform=t1_trans, crs="EPSG:4326")

        t2_trans = from_origin(11131949.0, 1118889.0, 100.0, 100.0)
        create_mock_scene(t2_dir, shape=(5, 5), transform=t2_trans, crs="EPSG:3857")

        aligner = TemporalAligner(TEST_CONFIG)
        pair = aligner.align_pair(t1_dir, "2025-01-15", t2_dir, "2026-01-20")

        assert pair.t2_aligned_metadata["crs"] == str(pair.reference_grid.crs)
        t1_trans_list = [
            pair.reference_grid.transform.a, pair.reference_grid.transform.b, pair.reference_grid.transform.c,
            pair.reference_grid.transform.d, pair.reference_grid.transform.e, pair.reference_grid.transform.f
        ]
        assert pair.t2_aligned_metadata["transform"] == t1_trans_list
        assert pair.t2_aligned_metadata["dimensions"] == [pair.reference_grid.height, pair.reference_grid.width]
        assert pair.t2_aligned_metadata["resolution"] == list(pair.reference_grid.resolution)
        assert pair.t2_aligned_metadata["extent"] == list(pair.reference_grid.bounds)
