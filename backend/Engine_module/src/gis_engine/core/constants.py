"""
src/gis_engine/core/constants.py
=================================
Single Source of Truth (SSOT) for all authoritative constants used
across the GIS Wetland Degradation & Depletion Engine.

DO NOT modify scientific values. These constants are locked at Phase 13.
All dependent modules must import from here rather than defining local copies.
"""

# ---------------------------------------------------------------------------
# Software Version
# ---------------------------------------------------------------------------
SOFTWARE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# NoData Constants
# ---------------------------------------------------------------------------
# Categorical rasters (uint8): health maps, transition maps, water maps, fusion maps
NODATA_CATEGORICAL: int = 255

# Continuous rasters (float32): NDWI, CVA magnitude, delta indices
NODATA_CONTINUOUS: float = -9999.0

# ---------------------------------------------------------------------------
# Class IDs — U-Net 4-class vegetation health schema
# ---------------------------------------------------------------------------
CLASS_ID_BACKGROUND: int = 0
CLASS_ID_SEVERE: int = 1
CLASS_ID_MODERATE: int = 2
CLASS_ID_HEALTHY: int = 3

CLASS_IDS = {
    CLASS_ID_BACKGROUND,
    CLASS_ID_SEVERE,
    CLASS_ID_MODERATE,
    CLASS_ID_HEALTHY,
}

# Short labels used in statistics dictionaries
CLASS_NAMES = {
    0: "Background",
    1: "Severe",
    2: "Moderate",
    3: "Healthy",
}

# Descriptive labels used in metadata and reports
CLASS_DESCRIPTIONS = {
    0: "Background / Bare Soil",
    1: "Severe Stress",
    2: "Moderate Stress",
    3: "Healthy Vegetation",
}

# ---------------------------------------------------------------------------
# Transition IDs — 4x4 row-major post-classification transition matrix
# Each ID = from_class * 4 + to_class  (range 0–15)
# ---------------------------------------------------------------------------
TRANSITIONS = {
    0:  {"from": 0, "to": 0, "name": "B_to_B",  "category": "STABLE"},
    1:  {"from": 0, "to": 1, "name": "B_to_S",  "category": "BACKGROUND_TO_VEGETATION"},
    2:  {"from": 0, "to": 2, "name": "B_to_M",  "category": "BACKGROUND_TO_VEGETATION"},
    3:  {"from": 0, "to": 3, "name": "B_to_H",  "category": "BACKGROUND_TO_VEGETATION"},

    4:  {"from": 1, "to": 0, "name": "S_to_B",  "category": "VEGETATION_TO_BACKGROUND"},
    5:  {"from": 1, "to": 1, "name": "S_to_S",  "category": "STABLE"},
    6:  {"from": 1, "to": 2, "name": "S_to_M",  "category": "RECOVERY"},
    7:  {"from": 1, "to": 3, "name": "S_to_H",  "category": "RECOVERY"},

    8:  {"from": 2, "to": 0, "name": "M_to_B",  "category": "VEGETATION_TO_BACKGROUND"},
    9:  {"from": 2, "to": 1, "name": "M_to_S",  "category": "DEGRADATION"},
    10: {"from": 2, "to": 2, "name": "M_to_M",  "category": "STABLE"},
    11: {"from": 2, "to": 3, "name": "M_to_H",  "category": "RECOVERY"},

    12: {"from": 3, "to": 0, "name": "H_to_B",  "category": "VEGETATION_TO_BACKGROUND"},
    13: {"from": 3, "to": 1, "name": "H_to_S",  "category": "DEGRADATION"},
    14: {"from": 3, "to": 2, "name": "H_to_M",  "category": "DEGRADATION"},
    15: {"from": 3, "to": 3, "name": "H_to_H",  "category": "STABLE"},
}

# Transition category sets (IDs only — exact values locked by Phase 4 validation)
DEGRADATION_IDS: frozenset = frozenset({9, 13, 14})
RECOVERY_IDS: frozenset = frozenset({6, 7, 11})
CANOPY_LOSS_IDS: frozenset = frozenset({4, 8, 12})
BACKGROUND_TO_VEGETATION_IDS: frozenset = frozenset({1, 2, 3})
STABLE_VEGETATION_IDS: frozenset = frozenset({5, 10, 15})
STABLE_BACKGROUND_IDS: frozenset = frozenset({0})

# ---------------------------------------------------------------------------
# Water State IDs — NDWI-based surface-water classification (Phase 10)
# ---------------------------------------------------------------------------
WATER_STATE_NO_WATER_T1_NO_WATER_T2: int = 0   # Stable non-water
WATER_STATE_STABLE_WATER: int = 1               # Persistent water T1 + T2
WATER_STATE_WATER_GAIN: int = 2                 # New water in T2 (not in T1)
WATER_STATE_WATER_LOSS: int = 3                 # Water in T1, absent in T2

WATER_GAIN_ID: int = WATER_STATE_WATER_GAIN
WATER_LOSS_ID: int = WATER_STATE_WATER_LOSS
WATER_STABLE_IDS: frozenset = frozenset({
    WATER_STATE_NO_WATER_T1_NO_WATER_T2,
    WATER_STATE_STABLE_WATER,
})

# ---------------------------------------------------------------------------
# Fusion Category IDs — Multi-evidence synthesis (Phase 11)
# ---------------------------------------------------------------------------
FUSION_CAT_NODATA: int = 0
FUSION_CAT_STABLE: int = 1
FUSION_CAT_VEGETATION_DEGRADATION: int = 2
FUSION_CAT_VEGETATION_RECOVERY: int = 3
FUSION_CAT_SURFACE_WATER_GAIN: int = 4
FUSION_CAT_SURFACE_WATER_LOSS: int = 5
FUSION_CAT_HYDRO_ECOLOGICAL_DEGRADATION: int = 6
FUSION_CAT_HYDRO_ECOLOGICAL_RECOVERY: int = 7
FUSION_CAT_VEGETATION_TO_BACKGROUND: int = 8
FUSION_CAT_AMBIGUOUS_MIXED: int = 9

FUSION_CATEGORIES = {
    0: "nodata",
    1: "stable",
    2: "vegetation_degradation",
    3: "vegetation_recovery",
    4: "surface_water_gain",
    5: "surface_water_loss",
    6: "hydro_ecological_degradation",
    7: "hydro_ecological_recovery",
    8: "vegetation_to_background",
    9: "ambiguous_mixed_evidence",
}

# ---------------------------------------------------------------------------
# Channel Order — 5-channel U-Net model input (SSOT, locked)
# ---------------------------------------------------------------------------
CHANNEL_ORDER = ["ndvi", "gndvi", "ndre", "nir", "rededge"]

# ---------------------------------------------------------------------------
# NDWI Threshold (default — configuration may override)
# ---------------------------------------------------------------------------
NDWI_THRESHOLD_DEFAULT: float = 0.0

# ---------------------------------------------------------------------------
# Physical Depletion Assessment Default
# ---------------------------------------------------------------------------
PHYSICAL_DEPLETION_DEFAULT: str = "NOT_ASSESSED"

# ---------------------------------------------------------------------------
# NoData Integrity Verification Set
# Confirms NODATA_CATEGORICAL does not coincide with any valid class/state ID
# ---------------------------------------------------------------------------
_ALL_VALID_CLASS_IDS = {
    CLASS_ID_BACKGROUND,
    CLASS_ID_SEVERE,
    CLASS_ID_MODERATE,
    CLASS_ID_HEALTHY,
}

_ALL_VALID_WATER_IDS = {
    WATER_STATE_NO_WATER_T1_NO_WATER_T2,
    WATER_STATE_STABLE_WATER,
    WATER_STATE_WATER_GAIN,
    WATER_STATE_WATER_LOSS,
}

_ALL_VALID_FUSION_IDS = set(FUSION_CATEGORIES.keys())

assert NODATA_CATEGORICAL not in _ALL_VALID_CLASS_IDS, \
    f"NODATA_CATEGORICAL={NODATA_CATEGORICAL} collides with a class ID!"
assert NODATA_CATEGORICAL not in _ALL_VALID_WATER_IDS, \
    f"NODATA_CATEGORICAL={NODATA_CATEGORICAL} collides with a water state ID!"
assert NODATA_CATEGORICAL not in _ALL_VALID_FUSION_IDS, \
    f"NODATA_CATEGORICAL={NODATA_CATEGORICAL} collides with a fusion category ID!"
