"""
src/gis_engine/core/exceptions.py
===================================
Production exception hierarchy for the GIS Wetland Degradation & Depletion Engine.

All engine-specific exceptions inherit from GISProcessingError so callers can
catch the entire family with a single except clause when needed, while also
being able to catch specific subtypes for targeted error handling.
"""


class GISProcessingError(Exception):
    """
    Base exception for all GIS Wetland Engine processing failures.
    All engine-specific exceptions inherit from this class.
    """


class InputValidationError(GISProcessingError):
    """
    Raised when T1/T2 input data fails validation checks.

    Examples:
        - Scene directory not found or not a directory
        - Required band file missing
        - Band CRS missing or invalid
        - Dimension mismatch between bands
        - T1 date >= T2 date (invalid temporal order)
    """


class ConfigurationError(GISProcessingError):
    """
    Raised when the configuration file is missing, malformed, or
    specifies invalid/inconsistent values.

    Examples:
        - config.json not found
        - Missing required configuration key
        - model path not specified
        - Invalid channel order length
    """


class AlignmentError(GISProcessingError):
    """
    Raised when spatial or temporal alignment between T1 and T2 fails.

    Examples:
        - T1 and T2 have no geographic overlap
        - CRS reprojection fails
        - Grid alignment cannot reconcile extents
    """


class InferenceError(GISProcessingError):
    """
    Raised when the ONNX model inference step fails.

    Examples:
        - Model file not found
        - ONNX session initialization fails on all providers
        - Input tensor shape does not match model expectations
        - Model output shape is unexpected
    """


class HydrologyError(GISProcessingError):
    """
    Raised when the NDWI hydrology or water-change computation fails.

    Examples:
        - Green or NIR bands missing for NDWI computation
        - Water classification produces invalid output dimensions
    """


class FusionError(GISProcessingError):
    """
    Raised when the multi-evidence fusion step fails.

    Examples:
        - Transition map and water-change map have incompatible dimensions
        - Fusion LUT produces unexpected output
    """


class MetricError(GISProcessingError):
    """
    Raised when metric computation fails.

    Examples:
        - Invalid pixel area values (e.g., non-positive areas)
        - Metric computation receives mismatched array dimensions
    """


class OutputError(GISProcessingError):
    """
    Raised when output writing fails.

    Examples:
        - Cannot create output directory (permissions)
        - GeoTIFF write fails
        - JSON serialization fails for a metric or report field
    """


class ValidationError(GISProcessingError):
    """
    Raised when the quality assurance or scientific validation step
    identifies a blocking problem.

    Examples:
        - NoData propagation contract violated (NoData converted to a valid class)
        - Output raster dimensions do not match reference grid
    """
