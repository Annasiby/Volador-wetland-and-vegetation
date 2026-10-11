"""
tests/test_phase13_paths.py
============================
Phase 13 hardcoded path regression tests.

Critical requirement: The engine must NEVER use the hardcoded
2023-08-14_2023-08-28 path when processing a different date pair.

Tests verify:
1. No source files contain the hardcoded date string as a fallback path
2. mask.py raises ValueError when OPTICAL_WATER_INFERENCE is called
   without explicit aligned_dir arguments
3. mask.py overlap_mask fallback no longer references the hardcoded date dir
"""

import ast
import re
from pathlib import Path

import numpy as np
import pytest

# Root of the GIS engine source tree
SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "gis_engine"

# The hardcoded date pair that must NOT appear as a fallback path
FORBIDDEN_DATE_PATH = "2023-08-14_2023-08-28"


class TestNoHardcodedDateFallbacks:
    """Scan all .py source files for the forbidden hardcoded date path."""

    def _source_files(self):
        return list(SRC_ROOT.rglob("*.py"))

    def test_no_hardcoded_date_in_source_files(self):
        """
        Scans every .py file in src/gis_engine/ for the hardcoded
        2023-08-14_2023-08-28 date string used in path construction.
        """
        violations = []
        for src_file in self._source_files():
            text = src_file.read_text(encoding="utf-8", errors="replace")
            # Look for the date string in any context
            if FORBIDDEN_DATE_PATH in text:
                # Allow it in comments/docstrings only
                lines_with_date = [
                    (i + 1, line.strip())
                    for i, line in enumerate(text.splitlines())
                    if FORBIDDEN_DATE_PATH in line
                ]
                for lineno, line in lines_with_date:
                    stripped = line.lstrip()
                    # Skip pure comment lines and docstring lines
                    if not (stripped.startswith("#") or stripped.startswith('"""') or
                            stripped.startswith("'''") or stripped.startswith("*")):
                        violations.append(f"{src_file.relative_to(SRC_ROOT)}:{lineno}  {line}")

        if violations:
            msg = (
                f"Hardcoded date path '{FORBIDDEN_DATE_PATH}' found in production code:\n"
                + "\n".join(violations)
            )
            pytest.fail(msg)

    def test_mask_py_no_hardcoded_fallback(self):
        """mask.py must not contain the hardcoded date string in executable code."""
        mask_py = SRC_ROOT / "wetland" / "mask.py"
        assert mask_py.is_file(), "mask.py not found"

        text = mask_py.read_text(encoding="utf-8")
        lines = text.splitlines()

        for i, line in enumerate(lines, 1):
            if FORBIDDEN_DATE_PATH in line:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    pytest.fail(
                        f"mask.py line {i} contains hardcoded date path in executable code:\n"
                        f"  {line}"
                    )


class TestMaskOpticalWaterInferencePaths:
    """
    Verifies that build_wetland_mask OPTICAL_WATER_INFERENCE raises ValueError
    when aligned dirs are not explicitly provided (no silent fallback).
    """

    def _make_ref_grid(self):
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
        from src.gis_engine.alignment.spatial import ReferenceGrid

        transform = from_bounds(0, 0, 1, 1, width=16, height=16)
        return ReferenceGrid(
            crs=CRS.from_epsg(4326),
            transform=transform,
            width=16,
            height=16,
            resolution=(1.0 / 16, 1.0 / 16),
            bounds=(0.0, 0.0, 1.0, 1.0),
        )

    def test_optical_water_inference_requires_explicit_t1_dir(self):
        """
        If source='OPTICAL_WATER_INFERENCE' and t1_aligned_dir is None/empty,
        mask.py must raise ValueError — NOT silently fall back to a hardcoded path.
        """
        from src.gis_engine.wetland.mask import build_wetland_mask

        ref_grid = self._make_ref_grid()
        overlap = np.ones((16, 16), dtype=np.uint8)

        with pytest.raises(ValueError, match="t1_aligned_dir"):
            build_wetland_mask(
                source="OPTICAL_WATER_INFERENCE",
                reference_grid=ref_grid,
                config={"water": {"ndwi_threshold": 0.0}, "wetland": {}},
                output_dir=None,
                t1_aligned_dir=None,        # Deliberately missing
                t2_aligned_dir=None,
                overlap_mask=overlap,
            )

    def test_optical_water_inference_requires_explicit_t2_dir(self):
        """t2_aligned_dir is also required explicitly."""
        import tempfile
        from src.gis_engine.wetland.mask import build_wetland_mask

        ref_grid = self._make_ref_grid()
        overlap = np.ones((16, 16), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="t2_aligned_dir"):
                build_wetland_mask(
                    source="OPTICAL_WATER_INFERENCE",
                    reference_grid=ref_grid,
                    config={"water": {"ndwi_threshold": 0.0}, "wetland": {}},
                    output_dir=None,
                    t1_aligned_dir=tmpdir,   # T1 provided
                    t2_aligned_dir=None,     # T2 deliberately missing
                    overlap_mask=overlap,
                )


class TestOverlapMaskFallback:
    """
    When overlap_mask=None is passed to build_wetland_mask, the function
    must use an all-ones fallback — NOT read the hardcoded
    2023-08-14_2023-08-28/overlap_mask.tif path.
    """

    def test_none_overlap_mask_does_not_access_hardcoded_path(
        self, tmp_path, monkeypatch
    ):
        """
        Patches rasterio.open to fail if the hardcoded path is ever accessed.
        If the code no longer references the hardcoded path, no call should fail.
        """
        import rasterio
        from src.gis_engine.wetland.mask import build_wetland_mask
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
        from src.gis_engine.alignment.spatial import ReferenceGrid

        # Sentinel: if rasterio is asked to open the forbidden path → fail
        original_open = rasterio.open

        def guarded_open(path, *args, **kwargs):
            if FORBIDDEN_DATE_PATH in str(path):
                raise AssertionError(
                    f"Accessed hardcoded path '{FORBIDDEN_DATE_PATH}' inside rasterio.open!"
                )
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(rasterio, "open", guarded_open)

        transform = from_bounds(0, 0, 1, 1, width=8, height=8)
        ref_grid = ReferenceGrid(
            crs=CRS.from_epsg(4326),
            transform=transform,
            width=8, height=8,
            resolution=(0.125, 0.125),
            bounds=(0.0, 0.0, 1.0, 1.0),
        )

        # Call with source=None (no wetland boundary) and overlap_mask=None
        # Should use all-ones fallback, never touch hardcoded path
        try:
            build_wetland_mask(
                source=None,
                reference_grid=ref_grid,
                config={"water": {"ndwi_threshold": 0.0}, "wetland": {}},
                output_dir=None,
                t1_aligned_dir=None,
                t2_aligned_dir=None,
                overlap_mask=None,
            )
        except AssertionError:
            pytest.fail("build_wetland_mask accessed the hardcoded 2023-08-14_2023-08-28 path!")
        except Exception:
            # Other exceptions (e.g. missing bands) are acceptable —
            # what matters is no AssertionError from our guard
            pass
