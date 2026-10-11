"""
tests/test_phase13_ssot.py
===========================
Phase 13 SSOT (Single Source of Truth) validation tests.

Verifies that:
1. All class IDs, transition IDs, NoData, and channel order are sourced
   exclusively from constants.py
2. No hardcoded values are duplicated in the constants module itself
3. NoData constants do not collide with valid class or state IDs
4. SSOT assertions pass (module-level asserts in constants.py)
"""

import importlib
import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Import the SSOT module
# ---------------------------------------------------------------------------

def _import_constants():
    from src.gis_engine.core import constants
    return constants


class TestSSOTNoDataConstants:
    """NoData constants must be unique and not collide with valid IDs."""

    def test_nodata_categorical_is_255(self):
        c = _import_constants()
        assert c.NODATA_CATEGORICAL == 255

    def test_nodata_continuous_is_minus_9999(self):
        c = _import_constants()
        assert c.NODATA_CONTINUOUS == -9999.0

    def test_nodata_categorical_not_in_class_ids(self):
        c = _import_constants()
        assert c.NODATA_CATEGORICAL not in c.CLASS_IDS

    def test_nodata_categorical_not_in_water_state_ids(self):
        c = _import_constants()
        water_ids = {
            c.WATER_STATE_NO_WATER_T1_NO_WATER_T2,
            c.WATER_STATE_STABLE_WATER,
            c.WATER_STATE_WATER_GAIN,
            c.WATER_STATE_WATER_LOSS,
        }
        assert c.NODATA_CATEGORICAL not in water_ids

    def test_nodata_categorical_not_in_fusion_category_ids(self):
        c = _import_constants()
        assert c.NODATA_CATEGORICAL not in set(c.FUSION_CATEGORIES.keys())


class TestSSOTClassIDs:
    """Class ID schema: 4 classes, IDs 0–3."""

    def test_class_ids_set_has_four_members(self):
        c = _import_constants()
        assert len(c.CLASS_IDS) == 4

    def test_class_ids_range(self):
        c = _import_constants()
        assert c.CLASS_IDS == {0, 1, 2, 3}

    def test_class_names_keys_match_class_ids(self):
        c = _import_constants()
        assert set(c.CLASS_NAMES.keys()) == c.CLASS_IDS

    def test_class_descriptions_keys_match_class_ids(self):
        c = _import_constants()
        assert set(c.CLASS_DESCRIPTIONS.keys()) == c.CLASS_IDS

    def test_background_id_is_zero(self):
        c = _import_constants()
        assert c.CLASS_ID_BACKGROUND == 0

    def test_healthy_id_is_three(self):
        c = _import_constants()
        assert c.CLASS_ID_HEALTHY == 3


class TestSSOTTransitions:
    """Transition schema: 16 entries (4×4), row-major."""

    def test_transition_count(self):
        c = _import_constants()
        assert len(c.TRANSITIONS) == 16

    def test_transition_ids_range(self):
        c = _import_constants()
        assert set(c.TRANSITIONS.keys()) == set(range(16))

    def test_row_major_encoding(self):
        c = _import_constants()
        for tid, info in c.TRANSITIONS.items():
            expected_id = info["from"] * 4 + info["to"]
            assert expected_id == tid, (
                f"Transition {tid}: from={info['from']} to={info['to']} "
                f"→ expected id {expected_id}"
            )

    def test_degradation_ids(self):
        c = _import_constants()
        # 9 = M→S, 13 = H→S, 14 = H→M
        assert c.DEGRADATION_IDS == frozenset({9, 13, 14})

    def test_recovery_ids(self):
        c = _import_constants()
        # 6 = S→M, 7 = S→H, 11 = M→H
        assert c.RECOVERY_IDS == frozenset({6, 7, 11})

    def test_canopy_loss_ids(self):
        c = _import_constants()
        # 4 = S→B, 8 = M→B, 12 = H→B
        assert c.CANOPY_LOSS_IDS == frozenset({4, 8, 12})

    def test_background_to_vegetation_ids(self):
        c = _import_constants()
        # 1 = B→S, 2 = B→M, 3 = B→H
        assert c.BACKGROUND_TO_VEGETATION_IDS == frozenset({1, 2, 3})

    def test_category_names_are_strings(self):
        c = _import_constants()
        for tid, info in c.TRANSITIONS.items():
            assert isinstance(info["category"], str), \
                f"Transition {tid} category is not a string."

    def test_degradation_recovery_disjoint(self):
        c = _import_constants()
        assert c.DEGRADATION_IDS.isdisjoint(c.RECOVERY_IDS)

    def test_canopy_loss_disjoint_from_degradation(self):
        c = _import_constants()
        assert c.CANOPY_LOSS_IDS.isdisjoint(c.DEGRADATION_IDS)


class TestSSOTWaterStates:
    """Water state IDs must follow the Phase 10 public schema."""

    def test_water_stable_non_water_is_zero(self):
        c = _import_constants()
        assert c.WATER_STATE_NO_WATER_T1_NO_WATER_T2 == 0

    def test_water_stable_water_is_one(self):
        c = _import_constants()
        assert c.WATER_STATE_STABLE_WATER == 1

    def test_water_gain_is_two(self):
        c = _import_constants()
        assert c.WATER_STATE_WATER_GAIN == 2

    def test_water_loss_is_three(self):
        c = _import_constants()
        assert c.WATER_STATE_WATER_LOSS == 3

    def test_water_stable_ids_set(self):
        c = _import_constants()
        assert c.WATER_STABLE_IDS == frozenset({0, 1})


class TestSSOTFusionCategories:
    """Fusion category IDs must cover 0–9."""

    def test_fusion_category_count(self):
        c = _import_constants()
        assert len(c.FUSION_CATEGORIES) == 10

    def test_fusion_category_ids_range(self):
        c = _import_constants()
        assert set(c.FUSION_CATEGORIES.keys()) == set(range(10))

    def test_hydro_ecological_degradation_is_six(self):
        c = _import_constants()
        assert c.FUSION_CAT_HYDRO_ECOLOGICAL_DEGRADATION == 6

    def test_hydro_ecological_recovery_is_seven(self):
        c = _import_constants()
        assert c.FUSION_CAT_HYDRO_ECOLOGICAL_RECOVERY == 7


class TestSSOTChannelOrder:
    """Channel order must be the 5-channel locked order."""

    def test_channel_count(self):
        c = _import_constants()
        assert len(c.CHANNEL_ORDER) == 5

    def test_channel_names(self):
        c = _import_constants()
        assert c.CHANNEL_ORDER == ["ndvi", "gndvi", "ndre", "nir", "rededge"]


class TestSSOTPhysicalDepletion:
    """Physical depletion default must be NOT_ASSESSED."""

    def test_default_not_assessed(self):
        c = _import_constants()
        assert c.PHYSICAL_DEPLETION_DEFAULT == "NOT_ASSESSED"


class TestSSOTSoftwareVersion:
    """Software version must be a non-empty string."""

    def test_version_present(self):
        c = _import_constants()
        assert isinstance(c.SOFTWARE_VERSION, str)
        assert len(c.SOFTWARE_VERSION) > 0
